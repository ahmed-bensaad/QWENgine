/// @file q38_eval.cpp
/// Correctness checks and benchmarks of the engine (most ctest entries run this tool).
///
///  q38_eval <model.gguf> golden <golden.bin> [--ref ref.bin] [--batched-prefill [--chunk N]] [--save out.bin]
///         [--min-top1 N] [--max-kl X]
///      Feeds the golden prompt (decode, or batched prefill), then teacher-forced decode of the golden's generated
///      tokens; compares the full logits of every stored row with the llama.cpp golden and the FP32 reference.
///      --save writes the engine's rows in the golden format (bit-identity checks between builds).
///  q38_eval <model.gguf> bench [--depth N | --synthetic-depth N] [--gen N]
///      Plain decode speed (optionally at a real or synthetic context depth); Q38_PROFILE_WAITS=1|2 adds the
///      CPU/GPU wait profile.
///  q38_eval <model.gguf> pbench --depth N [--chunk N] [--gen N]
///      Batched prefill of N tokens, then decode at that depth; Q38_PROFILE_PREFILL=1 adds the phase profile.
///  q38_eval <model.gguf> batchtest [--depth N]
///      decode_batch (2-4 tokens) and speculative passes with random commits must be bit-identical to sequential
///      decode; also times n-token passes.
///  q38_eval <model-mtp.gguf> spec [--prompt code|chat|story|facts|<text>] [--gen N] [--draft K|auto]
///         [--batched-prefill [--chunk N]] [--draft-rows R] [--mmproj F --image F]
///      MTP speculative greedy decoding; the output must equal plain greedy decoding token for token. --image puts
///      the image before the prompt text (batched prefill).
///  q38_eval <model-mtp.gguf> specset [--prompts-file F] [--nprompts N] [--gen N] [--draft auto|K]
///      Speculative throughput aggregated over many prompts in one engine load (the metric for comparisons).
///  q38_eval <model-mtp.gguf> draftstats [--prompts-file F] [--nprompts N] [--gen N] [--draft-max K]
///      Greedy generation with K MTP drafts made at every position: acceptance per draft depth and the tokens per
///      speculative cycle a draft limit of 1..K would give (sizes the benefit of longer drafts).
///  q38_eval <model.gguf> vision --mmproj F --image F [--ref ref.bin] [--image-max-tokens N] [--min-cos C]
///                       [--min-mean-cos C]
///      Vision encoder: preprocess and encode an image on the GPU; with --ref (tools/ref_mtmd_embd), compare the
///      embeddings with llama.cpp's per token (cosine similarity, relative error).
///  q38_eval <model-mtp.gguf> longctx --depth N --ctx C [--gen N] [--nprompts N] [--prompts-file F] [--kv T]
///      Long-context coding benchmark: a fixed snapshot of C++ sources (data/longctx_code.txt) fills the prompt to N
///      tokens, then each task (data/longctx_tasks.txt) continues from an engine snapshot of that context, as the
///      next turn of a coding agent does: prefill speed over the context, greedy speculative decode speed and draft
///      acceptance at that depth, plain decode speed at the same depth.
///  q38_eval <model-mtp.gguf> needle --depth N
///      Long-context retrieval: three codes hidden at 10/50/90% of an N-token prompt must be recalled.
///  q38_eval <model-mtp.gguf> snaptest
///      save_snapshot / restore_snapshot: after a disturbing speculative generation, restoring and decoding a suffix
///      must give bit-identical logits and the same greedy continuation as never having generated.
///  q38_eval <model-mtp.gguf> resizetest [--depth N]
///      Growing the context mid-sequence (Engine::resize) must be bit-identical to not resizing.
///
///  common: --ctx N  --kv f16|q8_0|q4_0  --threads N  --cap-mib X (GPU weight budget cap)  --reserve-mib X
///          --stream-mib X  --plan-k K (weight plan for K drafts; default 2 in spec/specset/needle, else 0)
///          --no-overlap (draft after the verify pass instead of during its LM head)
///          --no-mtp  --mtp-q6k (keep the MTP block in Q6_K)  --no-r4 / --no-r2 (CPU gather kernels for IQ3 / IQ2)

#include <algorithm>
#include <map>
#include <chrono>
#include <vector>
#include <cstdio>
#include <cstring>
#include <string>
#include "runtime/engine.h"
#include "runtime/vision.h"
#include "runtime/tokenizer.h"
#include "../tests/golden/golden_io.h"

using namespace q38rt;

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Decode an image file (formats of decode_image).
static bool load_image_file(const std::string& path, Image& img, std::string& err) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<uint8_t> bytes;
    uint8_t buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) bytes.insert(bytes.end(), buf, buf + n);
    fclose(f);
    return decode_image(bytes.data(), bytes.size(), img, err);
}

// User prompts, one per line; escapes: \n = newline, \\ = backslash (multi-line prompts on one line).
static std::vector<std::string> load_prompts(const std::string& path, int n) {
    std::vector<std::string> users;
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "cannot open prompts file %s\n", path.c_str()); return users; }
    char line[16384];
    while (fgets(line, sizeof line, f) && (int)users.size() < n) {
        std::string u(line);
        while (!u.empty() && (u.back() == '\n' || u.back() == '\r')) u.pop_back();
        std::string v;
        for (size_t i = 0; i < u.size(); i++) {
            if (u[i] == '\\' && i + 1 < u.size() && (u[i + 1] == 'n' || u[i + 1] == '\\')) {
                v += u[i + 1] == 'n' ? '\n' : '\\';
                i++;
            } else {
                v += u[i];
            }
        }
        if (!v.empty()) users.push_back(v);
    }
    fclose(f);
    return users;
}

struct Summary { int match = 0; size_t rows = 0; double kl_mean = 0; };
// Greedy speculative cycles with overlapped drafting: the next cycle's drafts are made during the verify pass's LM
// head from the GPU rows' argmax (Engine::decode_batch guess) and used when that guess was right.
struct SpecLoop {
    Engine& eng;
    bool adaptive;
    int fixed_k;
    DraftController dc;
    int k_next = 0;
    double t_mtp = 0, t_verify = 0;
    // one cycle from `next`: batch = [next, drafts], returns accepted drafts; next becomes the token after them
    long k_hist[Engine::MAX_BATCH] = {};   // cycles per draft count
    int step(int32_t& next, int32_t* batch, int& k) {
        k = k_next > 0 ? k_next : adaptive ? dc.choose() : fixed_k;
        k_hist[k]++;
        batch[0] = next;
        const double c0 = now_s();
        if (!eng.take_drafts(next, k, batch + 1)) eng.mtp_draft(next, k, batch + 1);
        const double c1 = now_s();
        k_next = 0;
        const Engine::SpecGuessFn guess = [&](const DecodeResult* part, int) {
            Engine::SpecGuess g;
            int j = 0;
            while (j < k && batch[1 + j] == part[j].topk_ids[0]) j++;
            g.accepted = j;
            g.next = part[j].topk_ids[0];
            g.k = k_next = adaptive ? dc.choose() : fixed_k;
            return g;
        };
        DecodeResult out[Engine::MAX_BATCH];
        eng.decode_batch(batch, k + 1, 1, false, out, true, &guess);
        t_mtp += c1 - c0;
        t_verify += now_s() - c1;
        int j = 0;
        while (j < k && batch[1 + j] == out[j].topk_ids[0]) j++;
        eng.commit(j + 1);
        if (adaptive) dc.update(k, j, now_s() - c0);
        next = out[j].topk_ids[0];
        return j;
    }
};

static Summary report(const char* label, const golden::File& ref, const golden::File& test) {
    double kl_sum = 0, kl_max = 0, cos_min = 1;
    int match = 0;
    for (size_t r = 0; r < ref.rows.size(); r++) {
        golden::RowMetrics m = golden::compare(ref.rows[r].data(), test.rows[r].data(), ref.n_vocab);
        match += m.top1_match;
        kl_sum += m.kl;
        kl_max = std::max(kl_max, m.kl);
        cos_min = std::min(cos_min, m.cos);
        if (!m.top1_match)
            printf("  [%s] pos %d: top1 ref %d (p=%.3f) vs engine %d, KL %.4f\n", label, ref.row_pos[r], m.top1_ref,
                   m.p_ref_top1, m.top1_test, m.kl);
    }
    printf("%-26s rows %2zu  top1 %2d/%zu  KL mean %.5f max %.5f  cos min %.6f\n", label, ref.rows.size(), match,
           ref.rows.size(), kl_sum / ref.rows.size(), kl_max, cos_min);
    return Summary{match, ref.rows.size(), kl_sum / ref.rows.size()};
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> golden <golden.bin> [--ref ref.bin] | bench [--depth N] [--gen N] [opts]\n", argv[0]);
        return 1;
    }
    EngineConfig cfg;
    cfg.model_path = argv[1];
    const std::string mode = argv[2];
    std::string golden_path, ref_path, save_path;
    int depth = 0, gen = 32, min_top1 = -1, synth_depth = 0, chunk = 0;
    bool batched = false;
    double max_kl = -1, min_cos = 0.99, min_mean_cos = 0;
    int draft_k = 1, plan_k = -1, nprompts = 12, draft_max = 7, image_max_tokens = 1024;
    std::string mmproj_path, image_path;
    std::string prompt_name = "code", prompts_file;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--ref") ref_path = argv[++i];
        else if (a == "--save") save_path = argv[++i];
        else if (a == "--depth") depth = atoi(argv[++i]);
        else if (a == "--synthetic-depth") synth_depth = atoi(argv[++i]);
        else if (a == "--batched-prefill") batched = true;
        else if (a == "--chunk") chunk = atoi(argv[++i]);
        else if (a == "--min-top1") min_top1 = atoi(argv[++i]);
        else if (a == "--max-kl") max_kl = atof(argv[++i]);
        else if (a == "--min-cos") min_cos = atof(argv[++i]);
        else if (a == "--min-mean-cos") min_mean_cos = atof(argv[++i]);
        else if (a == "--gen") gen = atoi(argv[++i]);
        else if (a == "--draft") draft_k = std::string(argv[i + 1]) == "auto" ? (++i, 0) : atoi(argv[++i]);
        else if (a == "--plan-k") plan_k = atoi(argv[++i]);
        else if (a == "--draft-max") draft_max = atoi(argv[++i]);
        else if (a == "--mmproj") mmproj_path = argv[++i];
        else if (a == "--image") image_path = argv[++i];
        else if (a == "--image-max-tokens") image_max_tokens = atoi(argv[++i]);
        else if (a == "--no-overlap") cfg.mtp_overlap = false;
        else if (a == "--prompt") prompt_name = argv[++i];
        else if (a == "--prompts-file") prompts_file = argv[++i];
        else if (a == "--nprompts") nprompts = atoi(argv[++i]);
        else if (a == "--draft-rows") cfg.draft_rows = atoi(argv[++i]);
        else if (a == "--no-mtp") cfg.mtp = false;
        else if (a == "--mtp-q6k") cfg.mtp_q4k = false;
        else if (a == "--no-r4") cfg.cpu_r4 = false;
        else if (a == "--no-r2") cfg.cpu_r2 = false;
        else if (a == "--ctx") cfg.max_context = atoi(argv[++i]);
        else if (a == "--threads") cfg.cpu_threads = atoi(argv[++i]);
        else if (a == "--cap-mib") cfg.gpu_weight_cap_mib = atof(argv[++i]);
        else if (a == "--reserve-mib") cfg.vram_reserve_mib = atof(argv[++i]);
        else if (a == "--stream-mib") cfg.stream_mib = atof(argv[++i]);
        else if (a == "--kv") {
            const std::string k = argv[++i];
            cfg.kv_type = k == "f16" ? cuda::KVType::F16 : k == "q4_0" ? cuda::KVType::Q4_0 : cuda::KVType::Q8_0;
        } else if (mode == "golden" && golden_path.empty()) golden_path = a;
    }
    // weight plan for speculative cycles only in the modes that decode speculatively (others measure plain decode)
    if (plan_k >= 0) cfg.spec_k = plan_k;
    else if (mode != "spec" && mode != "specset" && mode != "needle") cfg.spec_k = 0;
    Engine eng;
    if (!eng.init(cfg)) return 1;

    if (mode == "golden") {
        golden::File g;
        if (!golden::read(golden_path, g)) { fprintf(stderr, "cannot read %s\n", golden_path.c_str()); return 1; }
        golden::File ours;
        ours.n_prompt = g.n_prompt; ours.n_gen = g.n_gen; ours.n_vocab = g.n_vocab; ours.tokens = g.tokens;
        DecodeResult res;
        size_t row = 0;
        if (batched) {
            // prompt rows via batched prefill, generated rows via decode (teacher forced)
            std::vector<int32_t> prompt(g.tokens.begin(), g.tokens.begin() + g.n_prompt);
            std::vector<int> want_rows;
            for (size_t r = 0; r < g.rows.size(); r++) if (g.row_pos[r] < (int)g.n_prompt) want_rows.push_back(g.row_pos[r]);
            std::vector<std::vector<float>> rows_out;
            const double tp = now_s();
            eng.prefill_batched(prompt, nullptr, 1, want_rows, &rows_out, chunk);
            printf("batched prefill of %u tokens: %.2f s (incl. %zu logit rows)\n", g.n_prompt, now_s() - tp, want_rows.size());
            for (size_t r = 0; r < want_rows.size(); r++) {
                ours.row_pos.push_back(want_rows[r]);
                ours.rows.push_back(rows_out[r]);
            }
            row = want_rows.size();
            for (size_t i = g.n_prompt; i < g.tokens.size() && row < g.rows.size(); i++) {
                const bool want = (int)i == g.row_pos[row];
                eng.decode(g.tokens[i], want ? 1 : 0, want, res);
                if (want) { ours.row_pos.push_back((int)i); ours.rows.push_back(res.logits); row++; }
            }
        }
        for (size_t i = batched ? g.tokens.size() : 0; i < g.tokens.size() && row < g.rows.size(); i++) {
            const bool want = (int)i == g.row_pos[row];
            eng.decode(g.tokens[i], want ? 1 : 0, want, res);
            if (want) {
                ours.row_pos.push_back((int)i);
                ours.rows.push_back(res.logits);
                row++;
            }
        }
        // replay in top-k-only mode: the GPU-share top-k + CPU per-thread top-k merge must reproduce the
        // top-20 ranking of the full logits exactly (ties excluded). Only meaningful when the rows came from decode.
        if (!batched) {
            eng.reset();
            int bad = 0, checked = 0;
            size_t r2 = 0;
            for (size_t i = 0; i < g.tokens.size() && r2 < ours.rows.size(); i++) {
                const bool want = (int)i == ours.row_pos[r2];
                eng.decode(g.tokens[i], want ? 20 : 0, false, res);
                if (!want) continue;
                const std::vector<float>& lg = ours.rows[r2++];
                std::vector<int> order(lg.size());
                for (size_t j = 0; j < order.size(); j++) order[j] = (int)j;
                std::partial_sort(order.begin(), order.begin() + 21, order.end(), [&](int a, int b) { return lg[a] > lg[b]; });
                for (int j = 0; j < 20; j++) {
                    if (lg[order[j]] == lg[order[j + 1]]) break;   // tie: order not unique beyond this point
                    checked++;
                    bad += res.topk_ids[j] != order[j];
                }
            }
            printf("top-k path vs full logits: %d/%d ranks mismatched\n", bad, checked);
            if (bad) { printf("GOLDEN CHECK FAILED (top-k path)\n"); return 4; }
        }
        Summary sm = report("llama.cpp golden vs engine", g, ours);
        if (!ref_path.empty()) {
            golden::File r;
            if (golden::read(ref_path, r)) sm = report("FP32 reference vs engine", r, ours);
        }
        if (!save_path.empty()) golden::write(save_path, ours);
        // thresholds apply to the last comparison (the FP32 reference when given)
        if ((min_top1 >= 0 && sm.match < min_top1) || (max_kl >= 0 && sm.kl_mean > max_kl)) {
            printf("GOLDEN CHECK FAILED (top1 %d < %d or KL %.5f > %.5f)\n", sm.match, min_top1, sm.kl_mean, max_kl);
            return 3;
        }
        return 0;
    }

    if (mode == "bench") {
        DecodeResult res;
        const double t0 = now_s();
        for (int i = 0; i < depth; i++) eng.decode(1000 + (i * 7919) % 50000, 0, false, res);
        const double t1 = now_s();
        if (synth_depth > 0) eng.skip_to(synth_depth);
        // warm-up step (first launches, page faults) excluded from timing
        int tok = 9707;   // arbitrary start token; greedy continuation
        eng.decode(tok, 1, false, res);
        tok = res.topk_ids[0];
        eng.stats() = DecodeStats();
        std::vector<double> per_tok;
        const double t1b = now_s();
        for (int i = 0; i < gen; i++) {
            const double a = now_s();
            eng.decode(tok, 1, false, res);
            per_tok.push_back(now_s() - a);
            tok = res.topk_ids[0];
        }
        const double t2 = now_s();
        std::vector<double> sorted = per_tok;
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted[sorted.size() / 2];
        const DecodeStats& s = eng.stats();
        printf("depth %d (+synthetic %d) fill %.1f s (%.2f tok/s) | decode %d tokens: %.2f tok/s, median %.1f ms/token "
               "(cpu work %.1f ms, wait gpu %.1f ms, lm head %.1f ms per token)\n",
               depth, synth_depth, t1 - t0, depth / std::max(t1 - t0, 1e-9), gen, gen / (t2 - t1b), median * 1e3,
               s.t_cpu_work / gen * 1e3, s.t_wait_gpu / gen * 1e3, s.t_lm / gen * 1e3);
        printf("RESULT decode_tok_s=%.4f median_ms=%.3f prefill_tok_s=%.4f vram_mib=%.1f host_weight_mib=%.1f "
               "est_ms=%.2f cpu_ms=%.3f wait_ms=%.3f lm_ms=%.3f\n",
               gen / (t2 - t1b), median * 1e3, depth > 0 ? depth / (t1 - t0) : 0.0, eng.vram_used_bytes() / 1048576.0,
               eng.host_weight_bytes() / 1048576.0, eng.plan().est_total_s * 1e3, s.t_cpu_work / gen * 1e3,
               s.t_wait_gpu / gen * 1e3, s.t_lm / gen * 1e3);
        eng.print_wait_profile();
        return 0;
    }
    if (mode == "specset") {
        // Speculative throughput over many prompts in one engine load (adaptive drafts, batched prefill, stops at EOS):
        // weight-plan changes also change near-tie numerics, hence the text and the acceptance of a single prompt,
        // so plan-changing experiments are compared on this aggregate.
        if (!eng.has_mtp()) { fprintf(stderr, "specset needs the MTP GGUF\n"); return 1; }
        Tokenizer tk;
        if (!tk.load(eng.gguf())) return 1;
        const std::vector<std::string> users =
            load_prompts(prompts_file.empty() ? Q38_DATA_DIR "/corpus_prompts.txt" : prompts_file, nprompts);
        if (users.empty()) return 1;
        long tokens = 0, cycles = 0, drafted = 0, accepted = 0;
        double t_dec = 0, t_mtp = 0, t_verify = 0;
        // one draft controller for all prompts, as the server keeps one per session
        SpecLoop sl{eng, draft_k == 0, std::max(1, std::min(draft_k, Engine::MAX_BATCH - 1))};
        for (const std::string& u : users) {
            const std::vector<int32_t> prompt =
                tk.encode("<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            eng.reset();
            DecodeResult first;
            eng.prefill_batched(prompt, &first, 1, {}, nullptr, chunk);
            int32_t next = first.topk_ids[0];
            sl.k_next = 0;
            const double mtp0 = sl.t_mtp, ver0 = sl.t_verify;
            int n = 0;
            std::vector<int32_t> produced;
            const double a = now_s();
            while (n < gen && next != 248046 && next != 248044) {
                int32_t batch[Engine::MAX_BATCH];
                int k;
                const int j = sl.step(next, batch, k);
                produced.insert(produced.end(), batch, batch + j + 1);
                n += j + 1;
                drafted += k;
                accepted += j;
                cycles++;
            }
            t_mtp += sl.t_mtp - mtp0;
            t_verify += sl.t_verify - ver0;
            t_dec += now_s() - a;
            if (const char* dump = getenv("Q38_DUMP_TOKENS")) {   // prompt and generated tokens, one prompt per line
                if (FILE* f = fopen(dump, "a")) {
                    for (int32_t t : prompt) fprintf(f, "%d ", t);
                    fprintf(f, "|");
                    for (int32_t t : produced) fprintf(f, " %d", t);
                    fprintf(f, "\n");
                    fclose(f);
                }
            }
            tokens += n;
        }
        printf("specset: %zu prompts, %ld tokens in %ld cycles (%.2f tokens/cycle, acceptance %.1f%%), %.2f tok/s; "
               "per cycle MTP %.1f ms, verify %.1f ms\n", users.size(), tokens, cycles, (double)tokens / cycles,
               100.0 * accepted / std::max(drafted, 1L), tokens / t_dec, t_mtp / cycles * 1e3, t_verify / cycles * 1e3);
        eng.print_wait_profile();   // with Q38_PROFILE_WAITS=1 (per position: divide by tokens per cycle for per pass)
        printf("RESULT specset_tok_s=%.4f tokens_per_cycle=%.4f acceptance=%.4f mtp_ms=%.2f verify_ms=%.2f\n",
               tokens / t_dec, (double)tokens / cycles, (double)accepted / std::max(drafted, 1L), t_mtp / cycles * 1e3,
               t_verify / cycles * 1e3);
        printf("overlapped drafting: %ld drafts made during verify passes, %ld discarded (guess wrong)\n",
               (long)eng.stats().predrafts, (long)eng.stats().predraft_misses);
        printf("draft count per cycle:");
        for (int k = 1; k < Engine::MAX_BATCH; k++) printf(" k=%d %.0f%%", k, 100.0 * sl.k_hist[k] / std::max(cycles, 1L));
        printf("\n");
        return 0;
    }
    if (mode == "vision") {
        VisionModel vm;
        if (mmproj_path.empty() || image_path.empty()) { fprintf(stderr, "vision needs --mmproj and --image\n"); return 2; }
        if (!vm.load(mmproj_path, 8, image_max_tokens)) return 1;
        Image img;
        std::string err;
        if (!load_image_file(image_path, img, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const double t0 = now_s();
        const ImagePatches ip = vm.preprocess(img);
        const double t1 = now_s();
        std::vector<float> e;
        if (!eng.encode_image(vm, ip, e, err)) { fprintf(stderr, "encode_image: %s\n", err.c_str()); return 1; }
        const double t2 = now_s();
        printf("image %dx%d -> %dx%d pixels, %d x %d merged tokens: preprocess %.0f ms, encode %.0f ms\n", img.width,
               img.height, ip.width, ip.height, ip.nx, ip.ny, (t1 - t0) * 1e3, (t2 - t1) * 1e3);
        if (!ref_path.empty()) {
            FILE* f = fopen(ref_path.c_str(), "rb");
            char magic[4];
            int hdr[3];
            if (!f || fread(magic, 1, 4, f) != 4 || memcmp(magic, "Q38V", 4) || fread(hdr, 4, 3, f) != 3) {
                fprintf(stderr, "cannot read %s\n", ref_path.c_str());
                return 1;
            }
            if (hdr[0] != ip.nx || hdr[1] != ip.ny || hdr[2] != VisionModel::OUT) {
                printf("reference grid %dx%d (dim %d) differs from ours %dx%d\n", hdr[0], hdr[1], hdr[2], ip.nx, ip.ny);
                return 4;
            }
            std::vector<float> r((size_t)hdr[0] * hdr[1] * hdr[2]);
            if (fread(r.data(), 4, r.size(), f) != r.size()) { fprintf(stderr, "short reference file\n"); return 1; }
            fclose(f);
            double cos_min = 1, cos_sum = 0, rel_max = 0;
            const int T = ip.n_tokens(), Dm = VisionModel::OUT;
            std::vector<std::pair<double, int>> worst;
            for (int t = 0; t < T; t++) {
                double ab = 0, aa = 0, bb = 0, dd = 0;
                for (int i = 0; i < Dm; i++) {
                    const double x = e[(size_t)t * Dm + i], y = r[(size_t)t * Dm + i];
                    ab += x * y; aa += x * x; bb += y * y; dd += (x - y) * (x - y);
                }
                const double c = ab / std::sqrt(aa * bb + 1e-30);
                cos_min = std::min(cos_min, c);
                cos_sum += c;
                worst.push_back({c, t});
                rel_max = std::max(rel_max, std::sqrt(dd / (bb + 1e-30)));
            }
            std::sort(worst.begin(), worst.end());
            printf("lowest cosines (token at column, row):");
            for (size_t i = 0; i < std::min<size_t>(5, worst.size()); i++)
                printf(" %.4f (%d, %d)", worst[i].first, worst[i].second % ip.nx, worst[i].second / ip.nx);
            printf("\nvs llama.cpp mtmd: cosine min %.6f mean %.6f, relative error max %.4f\n", cos_min, cos_sum / T, rel_max);
            return cos_min >= min_cos && cos_sum / T >= min_mean_cos ? 0 : 4;
        }
        return 0;
    }
    if (mode == "draftstats") {
        if (!eng.has_mtp()) { fprintf(stderr, "draftstats needs the MTP GGUF\n"); return 1; }
        Tokenizer tk;
        if (!tk.load(eng.gguf())) return 1;
        const std::vector<std::string> users =
            load_prompts(prompts_file.empty() ? Q38_DATA_DIR "/corpus_prompts.txt" : prompts_file, nprompts);
        if (users.empty()) return 1;
        const int KM = std::max(1, std::min(draft_max, 16));
        std::vector<long> reach(KM + 1, 0);           // positions whose drafts match at least d tokens
        std::vector<long> cyc(KM + 1, 0), tok(KM + 1, 0);
        long positions = 0;
        for (const std::string& u : users) {
            const std::vector<int32_t> prompt =
                tk.encode("<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            eng.reset();
            DecodeResult r;
            eng.prefill_batched(prompt, &r, 1, {}, nullptr, chunk);
            std::vector<int32_t> seq;                   // greedy tokens
            std::vector<std::vector<int32_t>> drafts;  // drafts[i]: KM MTP drafts for seq[i+1..]
            int32_t next = r.topk_ids[0];
            while ((int)seq.size() < gen && next != 248046 && next != 248044) {
                std::vector<int32_t> d(KM);
                eng.mtp_draft(next, KM, d.data());
                drafts.push_back(d);
                seq.push_back(next);
                eng.decode(next, 1, false, r);
                next = r.topk_ids[0];
            }
            seq.push_back(next);
            const int n = (int)drafts.size();
            std::vector<int> a(n);   // accepted prefix length at each position
            for (int i = 0; i < n; i++) {
                int m = 0;
                while (m < KM && i + 1 + m < (int)seq.size() && drafts[i][m] == seq[i + 1 + m]) m++;
                a[i] = m;
                for (int dd = 1; dd <= m; dd++) reach[dd]++;
            }
            positions += n;
            for (int k = 1; k <= KM; k++)   // speculative cycles with drafts of length k
                for (int i = 0; i < n;) {
                    const int adv = std::min(a[i], k) + 1;
                    cyc[k]++;
                    tok[k] += std::min(adv, n - i);
                    i += adv;
                }
        }
        printf("draftstats: %zu prompts, %ld positions, %d drafts per position\n", users.size(), positions, KM);
        printf("depth  P(match | previous matched)  P(match)   |  draft limit k: tokens/cycle\n");
        for (int dd = 1; dd <= KM; dd++)
            printf("%5d  %27.3f  %9.3f   |  k=%d: %.3f\n", dd, (double)reach[dd] / std::max(dd > 1 ? reach[dd - 1] : positions, 1L),
                   (double)reach[dd] / std::max(positions, 1L), dd, (double)tok[dd] / std::max(cyc[dd], 1L));
        return 0;
    }
    if (mode == "spec") {
        if (!eng.has_mtp()) { fprintf(stderr, "spec mode needs the MTP GGUF\n"); return 1; }
        Tokenizer tk;
        if (!tk.load(eng.gguf())) return 1;
        const std::map<std::string, std::string> prompts = {
            {"code", "Write a Python function that parses a CSV file and returns the average of each numeric column. Explain briefly."},
            {"chat", "What are three practical tips for sleeping better? Keep it short."},
            {"story", "Write a short story (about 150 words) about a lighthouse keeper who finds a message in a bottle."},
            {"facts", "Explain how photosynthesis works to a high school student."}};
        const std::string user = prompts.count(prompt_name) ? prompts.at(prompt_name) : prompt_name;   // or literal text
        const std::string text = "<|im_start|>user\n" + user +
                                 "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        std::vector<int32_t> prompt = tk.encode(text);
        // --image: the image before the question (M-RoPE positions, MTP on image embeddings); batched prefill only
        std::vector<Engine::PrefillImage> images;
        std::vector<float> img_emb;
        if (!image_path.empty()) {
            VisionModel vm;
            Image img;
            std::string err;
            if (mmproj_path.empty() || !vm.load(mmproj_path, 8, image_max_tokens)) { fprintf(stderr, "--image needs --mmproj\n"); return 1; }
            if (!load_image_file(image_path, img, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
            const ImagePatches ip = vm.preprocess(img);
            if (!eng.encode_image(vm, ip, img_emb, err)) { fprintf(stderr, "encode_image: %s\n", err.c_str()); return 1; }
            prompt = tk.encode("<|im_start|>user\n<|vision_start|>");
            images.push_back({(int)prompt.size(), ip.nx, ip.ny, img_emb.data()});
            prompt.insert(prompt.end(), ip.n_tokens(), tk.encode("<|image_pad|>")[0]);
            const std::vector<int32_t> tail = tk.encode("<|vision_end|>" + user + text.substr(text.find("<|im_end|>")));
            prompt.insert(prompt.end(), tail.begin(), tail.end());
            batched = true;
            printf("image %dx%d: %d x %d tokens, prompt %zu tokens\n", img.width, img.height, ip.nx, ip.ny, prompt.size());
        }
        const bool adaptive = draft_k == 0;   // --draft auto
        int k = std::max(1, std::min(draft_k, Engine::MAX_BATCH - 1));
        int k_hist[4] = {0, 0, 0, 0};
        // prompt in chunks of MAX_BATCH (keeps the MTP catch-up within its ring); returns the first generated token
        auto run_prompt = [&](bool mtp) {
            eng.reset();
            DecodeResult out[Engine::MAX_BATCH];
            int32_t next = -1;
            if (batched) {   // batched prefill (fills the MTP KV cache too when the MTP block is loaded)
                eng.prefill_batched(prompt, &out[0], 1, {}, nullptr, chunk, {}, images);
                return out[0].topk_ids[0];
            }
            for (size_t i = 0; i < prompt.size(); i += Engine::MAX_BATCH) {
                const int n = (int)std::min<size_t>(Engine::MAX_BATCH, prompt.size() - i);
                eng.decode_batch(prompt.data() + i, n, 1, false, out);
                if (mtp) eng.mtp_catchup();
                next = out[n - 1].topk_ids[0];
            }
            return next;
        };
        // plain greedy reference
        std::vector<int32_t> ref;
        {
            int32_t next = run_prompt(false);
            DecodeResult r;
            const double a = now_s();
            for (int i = 0; i < gen; i++) {
                ref.push_back(next);
                eng.decode(next, 1, false, r);
                next = r.topk_ids[0];
            }
            printf("plain greedy: %d tokens, %.2f tok/s\n", gen, gen / (now_s() - a));
        }
        // speculative: draft k with MTP, verify k+1 in one pass, keep the agreeing prefix + the target's next token
        std::vector<int32_t> outt;
        int drafted = 0, accepted = 0, cycles = 0;
        double t_mtp = 0, t_verify = 0;
        {
            int32_t next = run_prompt(true);
            const double a = now_s();
            eng.stats() = DecodeStats();
            SpecLoop sl{eng, adaptive, k};
            while ((int)outt.size() < gen) {
                int32_t batch[Engine::MAX_BATCH];
                const int j = sl.step(next, batch, k);
                k_hist[k]++;
                for (int t = 0; t <= j; t++) outt.push_back(batch[t]);
                drafted += k;
                accepted += j;
                cycles++;
            }
            const double dt = now_s() - a;
            t_mtp = sl.t_mtp;
            t_verify = sl.t_verify;
            outt.resize(gen);
            if (adaptive) printf("adaptive k: cycles with k=1/2/3: %d/%d/%d\n", k_hist[1], k_hist[2], k_hist[3]);
            printf("speculative (k=%s): %d tokens in %d cycles, %.2f tokens/cycle, acceptance %.1f%% (%d/%d), %.2f tok/s\n",
                   adaptive ? "auto" : std::to_string(k).c_str(),
                   gen, cycles, (double)gen / cycles, 100.0 * accepted / std::max(drafted, 1), accepted, drafted, gen / dt);
            printf("per cycle: MTP drafting %.1f ms, verify pass %.1f ms\n", t_mtp / cycles * 1e3, t_verify / cycles * 1e3);
            printf("RESULT spec_tok_s=%.4f tokens_per_cycle=%.4f acceptance=%.4f mtp_ms=%.2f verify_ms=%.2f\n", gen / dt,
                   (double)gen / cycles, (double)accepted / std::max(drafted, 1), t_mtp / cycles * 1e3,
                   t_verify / cycles * 1e3);
        }
        int diff = -1;
        for (int i = 0; i < gen; i++) if (outt[i] != ref[i]) { diff = i; break; }
        if (diff >= 0) { printf("SPECULATIVE OUTPUT DIFFERS from plain greedy at token %d\n", diff); return 6; }
        printf("speculative output identical to plain greedy (%d tokens)\n", gen);
        printf("--- %s\n", tk.decode(outt).c_str());
        return 0;
    }
    if (mode == "snaptest") {
        Tokenizer tk;
        if (!tk.load(eng.gguf())) return 1;
        std::string text;
        for (int i = 0; i < 12; i++)
            text += "Paragraph " + std::to_string(i) + ": the hybrid engine splits each matrix between the GPU and the CPU. ";
        const std::vector<int32_t> A = tk.encode("<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n");
        const std::vector<int32_t> S = tk.encode("A different continuation that the snapshot must handle exactly.");
        double save_ms = 0, restore_ms = 0;
        auto run = [&](bool disturb, std::vector<float>& logits, std::vector<int32_t>& cont) {
            eng.reset();
            DecodeResult r;
            eng.prefill_batched(A, &r, 1);
            if (disturb) {   // speculative generation moves KV, DeltaNet, MTP state and rings past the snapshot
                if (!eng.save_snapshot()) return false;   // first call allocates the pinned buffer
                const auto s0 = std::chrono::steady_clock::now();
                if (!eng.save_snapshot()) return false;
                save_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
                int32_t next = r.topk_ids[0];
                DraftController dc;
                for (int c = 0; c < 12; c++) {
                    const int k = eng.has_mtp() ? dc.choose() : 0;
                    int32_t batch[Engine::MAX_BATCH] = {next};
                    if (k > 0) eng.mtp_draft(next, k, batch + 1);
                    DecodeResult out[Engine::MAX_BATCH];
                    eng.decode_batch(batch, k + 1, 1, false, out, true);
                    int j = 0;
                    while (j < k && batch[1 + j] == out[j].topk_ids[0]) j++;
                    eng.commit(j + 1);
                    next = out[j].topk_ids[0];
                }
                const auto r0 = std::chrono::steady_clock::now();
                if (!eng.restore_snapshot()) return false;
                restore_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count();
            }
            for (size_t i = 0; i + 1 < S.size(); i++) eng.decode(S[i], 1, false, r);
            eng.decode(S.back(), 1, true, r);
            logits = r.logits;
            int32_t tok = r.topk_ids[0];
            cont.clear();
            for (int i = 0; i < 16; i++) { cont.push_back(tok); eng.decode(tok, 1, false, r); tok = r.topk_ids[0]; }
            return true;
        };
        std::vector<float> l0, l1;
        std::vector<int32_t> c0, c1;
        if (!run(false, l0, c0) || !run(true, l1, c1)) { printf("snapshot save/restore failed\n"); return 3; }
        const bool same_logits = l0.size() == l1.size() && memcmp(l0.data(), l1.data(), l0.size() * 4) == 0;
        printf("snapshot at %zu tokens, suffix %zu: logits %s, continuation %s (save %.1f ms, restore %.1f ms)\n", A.size(),
               S.size(), same_logits ? "bit-identical" : "DIFFER", c0 == c1 ? "identical" : "DIFFERS", save_ms, restore_ms);
        if (!same_logits && l0.size() == l1.size()) {
            size_t nd = 0;
            double mx = 0;
            for (size_t i = 0; i < l0.size(); i++)
                if (l0[i] != l1[i]) { nd++; mx = std::max(mx, (double)std::fabs(l0[i] - l1[i])); }
            printf("  %zu of %zu logits differ, max |difference| %.3g\n", nd, l0.size(), mx);
        }
        return same_logits && c0 == c1 ? 0 : 4;
    }
    if (mode == "resizetest") {
        // Same weight plan before/after (use --cap-mib below both budgets): decoding with a resize in the middle
        // (speculative passes and the MTP state across it) must be bit-identical to never resizing.
        const int L = depth > 0 ? depth : 48;
        std::vector<int32_t> toks(L);
        for (int i = 0; i < L; i++) toks[i] = 1000 + (i * 7919) % 50000;
        const int cap0 = cfg.max_context;
        auto run = [&](bool do_resize, std::vector<DecodeResult>& out) {
            if (eng.max_context() != cap0) { fprintf(stderr, "engine must start at the initial capacity\n"); exit(1); }
            eng.reset();
            out.assign(L, DecodeResult());
            for (int i = 0; i < L;) {
                if (do_resize && i >= L / 2 && eng.max_context() == cap0) {
                    if (!eng.resize(cap0 * 2)) exit(1);
                }
                const int n = std::min(3, L - i);
                DecodeResult o[Engine::MAX_BATCH];
                eng.decode_batch(toks.data() + i, n, 8, true, o, true);
                const int m = 1 + (i % n);   // deterministic partial commits
                for (int t = 0; t < m; t++) out[i + t] = o[t];
                eng.commit(m);
                if (eng.has_mtp()) { int32_t d[2]; eng.mtp_draft(toks[std::min(i + m, L - 1)], 1, d); }
                i += m;
            }
        };
        std::vector<DecodeResult> a, b;
        run(false, a);
        // second run starts again from the small capacity: re-init by resizing a fresh engine is not possible,
        // so compare the resized run against the first (identical plans thanks to the budget cap)
        run(true, b);
        int bad = 0;
        for (int i = 0; i < L; i++)
            if (a[i].logits != b[i].logits) { if (bad < 3) printf("  position %d differs\n", i); bad++; }
        printf("resize %d -> %d mid-sequence: %d/%d positions differ\n", cap0, cap0 * 2, bad, L);
        return bad ? 7 : 0;
    }
    if (mode == "batchtest") {
        // Reference: sequential decode of L tokens (top-8 and full logits at every position). Then the same tokens
        // through decode_batch in chunks of 2, 3, 4, 1, 2, ... — every position must match bit for bit.
        const int L = depth > 0 ? depth : 40;
        // the timing loop below reads up to 7 * MAX_BATCH tokens: size the token list for both uses
        std::vector<int32_t> toks(std::max(L, 7 * Engine::MAX_BATCH));
        for (size_t i = 0; i < toks.size(); i++) toks[i] = 1000 + ((int)i * 7919) % 50000;
        std::vector<DecodeResult> ref(L);
        eng.reset();
        for (int i = 0; i < L; i++) eng.decode(toks[i], 8, true, ref[i]);
        int bad = 0, checked = 0;
        for (int want_logits = 0; want_logits < 2; want_logits++) {
            eng.reset();
            const int pattern[] = {2, 3, 4, 1, 7, 5, 6};   // every pass width up to Engine::MAX_BATCH
            for (int i = 0, k = 0; i < L; k++) {
                const int n = std::min(pattern[k % 7], L - i);
                DecodeResult out[Engine::MAX_BATCH];
                eng.decode_batch(toks.data() + i, n, 8, want_logits != 0, out);
                for (int t = 0; t < n; t++) {
                    const DecodeResult& r = ref[i + t];
                    bool ok = out[t].topk_ids == r.topk_ids &&
                              !memcmp(out[t].topk_vals.data(), r.topk_vals.data(), r.topk_vals.size() * 4);
                    if (want_logits) ok = ok && out[t].logits.size() == r.logits.size() &&
                                          !memcmp(out[t].logits.data(), r.logits.data(), r.logits.size() * 4);
                    checked++;
                    if (!ok) {
                        if (bad < 5) printf("  mismatch at position %d (chunk of %d, logits %d): top1 %d %.6f vs %d %.6f\n", i + t,
                                            n, want_logits, out[t].topk_ids.empty() ? -1 : out[t].topk_ids[0],
                                            out[t].topk_vals.empty() ? 0.f : out[t].topk_vals[0], r.topk_ids[0], r.topk_vals[0]);
                        bad++;
                    }
                }
                i += n;
            }
        }
        printf("decode_batch vs sequential decode: %d/%d positions differ\n", bad, checked);
        // speculative passes with random commit counts (lazy DeltaNet commit), mixed with plain passes; every
        // committed position must still match the sequential reference bit for bit
        {
            int sbad = 0, schecked = 0;
            uint32_t rs = 12345;
            auto rnd = [&](int n) { rs = rs * 1664525u + 1013904223u; return (int)((rs >> 8) % (uint32_t)n); };
            eng.reset();
            int pos = 0, passes = 0;
            while (pos < L) {
                const int n = std::min(1 + rnd(Engine::MAX_BATCH), L - pos);
                const bool spec = rnd(4) != 0;
                DecodeResult out[Engine::MAX_BATCH];
                eng.decode_batch(toks.data() + pos, n, 8, true, out, spec);
                const int m = spec ? 1 + rnd(n) : n;
                for (int t = 0; t < m; t++) {
                    const DecodeResult& r = ref[pos + t];
                    const bool ok = out[t].topk_ids == r.topk_ids &&
                                    !memcmp(out[t].logits.data(), r.logits.data(), r.logits.size() * 4);
                    schecked++;
                    if (!ok) {
                        if (sbad < 5) printf("  speculative mismatch at position %d (pass of %d, keep %d)\n", pos + t, n, m);
                        sbad++;
                    }
                }
                if (spec) eng.commit(m);
                if (eng.position() != pos + m) { printf("  position bookkeeping error\n"); sbad++; break; }
                pos += m;
                passes++;
            }
            printf("speculative passes (%d) with random commits vs sequential decode: %d/%d positions differ\n", passes,
                   sbad, schecked);
            bad += sbad;
        }
        // timing of n-token passes at the current depth (no correctness meaning)
        for (int n = 1; n <= Engine::MAX_BATCH; n++) {
            eng.reset();
            DecodeResult out[Engine::MAX_BATCH];
            eng.decode_batch(toks.data(), n, 1, false, out);   // warm-up
            const int reps = 6;
            const double a = now_s();
            for (int r = 0; r < reps; r++) eng.decode_batch(toks.data() + n * (r + 1), n, 1, false, out);
            const double ms = (now_s() - a) / reps * 1e3;
            printf("  %d-token pass: %.1f ms (%.2f ms/token)\n", n, ms, ms / n);
        }
        return bad ? 5 : 0;
    }
    if (mode == "longctx") {
        if (!eng.has_mtp()) { fprintf(stderr, "longctx needs the MTP GGUF\n"); return 1; }
        if (depth <= 0) { fprintf(stderr, "longctx needs --depth N\n"); return 2; }
        Tokenizer tk;
        if (!tk.load(eng.gguf())) return 1;
        std::string code;
        {
            const std::string path = Q38_DATA_DIR "/longctx_code.txt";
            FILE* f = fopen(path.c_str(), "rb");
            if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return 1; }
            char buf[65536];
            size_t n;
            while ((n = fread(buf, 1, sizeof buf, f)) > 0) code.append(buf, n);
            fclose(f);
        }
        const std::vector<std::string> tasks =
            load_prompts(prompts_file.empty() ? Q38_DATA_DIR "/longctx_tasks.txt" : prompts_file, nprompts);
        if (tasks.empty()) return 1;
        std::vector<int32_t> ctx = tk.encode("<|im_start|>user\nHere is the source code of a C++/CUDA project.\n\n" + code);
        if ((int)ctx.size() < depth) {
            fprintf(stderr, "the code snapshot has %zu tokens, fewer than --depth %d\n", ctx.size(), depth);
            return 1;
        }
        printf("code snapshot: %zu tokens, cut to %d\n", ctx.size(), depth);
        ctx.resize(depth);   // the tasks follow the cut
        std::vector<std::vector<int32_t>> suffix;
        size_t longest = 0;
        for (const std::string& t : tasks) {
            suffix.push_back(tk.encode("\n\n(end of the code)\n\n" + t + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
            longest = std::max(longest, suffix.back().size());
        }
        if (depth + (int)longest + gen + Engine::MAX_BATCH + 1 > eng.max_context()) {
            fprintf(stderr, "--ctx %d is too small for --depth %d, a task of %zu tokens and --gen %d\n", eng.max_context(),
                    depth, longest, gen);
            return 2;
        }
        DecodeResult r;
        const double t0 = now_s();
        eng.prefill_batched(ctx, nullptr, 1, {}, nullptr, chunk);
        const double t_prefill = now_s() - t0;
        printf("context: %d tokens in %.1f s (%.1f tok/s)\n", depth, t_prefill, depth / t_prefill);
        fflush(stdout);
        eng.save_snapshot();
        long n_out = 0, drafted = 0, accepted = 0, cycles = 0;
        double t_dec = 0;
        for (size_t ti = 0; ti < tasks.size(); ti++) {
            eng.restore_snapshot();
            eng.prefill_batched(suffix[ti], &r, 1, {}, nullptr, chunk);
            int32_t next = r.topk_ids[0];
            SpecLoop sl{eng, true, 0};
            std::vector<int32_t> out;
            int dr = 0, ac = 0, cy = 0;
            bool eos = false;
            const double a = now_s();
            while (!eos && (int)out.size() < gen && next != 248046 && next != 248044) {
                int32_t batch[Engine::MAX_BATCH];
                int k;
                const int j = sl.step(next, batch, k);
                for (int t = 0; t <= j && (int)out.size() < gen; t++) {
                    if (t > 0 && (batch[t] == 248046 || batch[t] == 248044)) { eos = true; break; }
                    out.push_back(batch[t]);
                }
                dr += k;
                ac += j;
                cy++;
            }
            const double dt = now_s() - a;
            std::string text;
            for (size_t i = 0; i < out.size() && text.size() < 100; i++) text += tk.token_to_piece(out[i]);
            for (char& ch : text) if (ch == '\n') ch = ' ';
            printf("task %zu: %zu tokens in %.1f s = %.2f tok/s, acceptance %.2f, %.2f tokens per pass | %s...\n", ti + 1,
                   out.size(), dt, out.size() / dt, dr ? (double)ac / dr : 0.0, cy ? (double)out.size() / cy : 0.0,
                   text.c_str());
            fflush(stdout);
            n_out += (long)out.size();
            drafted += dr;
            accepted += ac;
            cycles += cy;
            t_dec += dt;
        }
        // plain decode at the same depth (last: the MTP block falls behind and is not used again)
        eng.restore_snapshot();
        eng.prefill_batched(suffix[0], &r, 1, {}, nullptr, chunk);
        int32_t tok = r.topk_ids[0];
        const int n_plain = std::min(gen, 48);
        const double a = now_s();
        for (int i = 0; i < n_plain; i++) { eng.decode(tok, 1, false, r); tok = r.topk_ids[0]; }
        const double plain = n_plain / (now_s() - a);
        printf("plain decode at depth %d: %.2f tok/s\n", depth, plain);
        printf("RESULT longctx_depth=%d prefill_tok_s=%.3f spec_tok_s=%.4f acceptance=%.4f tokens_per_pass=%.4f "
               "plain_tok_s=%.4f\n", depth, depth / t_prefill, n_out / std::max(t_dec, 1e-9),
               drafted ? (double)accepted / drafted : 0.0, cycles ? (double)n_out / cycles : 0.0, plain);
        return 0;
    }
    if (mode == "needle") {
        // Long-context retrieval: deterministic filler text with three secret codes at ~10/50/90% depth,
        // grown with our tokenizer to `depth` tokens, chat-formatted (non-thinking), then greedy answer.
        Tokenizer tk;
        if (!tk.load(eng.gguf())) return 1;
        const char* adj[] = {"quiet", "ancient", "bright", "distant", "curious", "gentle", "massive", "silver", "hidden",
                             "narrow", "sudden", "golden", "frozen", "busy", "patient", "restless"};
        const char* noun[] = {"river", "library", "merchant", "garden", "lantern", "harbor", "scholar", "orchard", "bridge",
                              "village", "machine", "forest", "teacher", "market", "tower", "valley"};
        const char* verb[] = {"watched", "crossed", "described", "repaired", "followed", "painted", "measured", "visited",
                              "remembered", "carried", "studied", "guarded"};
        const char* place[] = {"old mill", "northern road", "city wall", "quiet bay", "stone square", "eastern hills",
                               "train station", "public archive"};
        uint64_t rs = 12345;
        auto rnd = [&](int n) { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return (int)((rs >> 33) % n); };
        auto sentence = [&]() {
            std::string x = "The ";
            x += adj[rnd(16)]; x += " "; x += noun[rnd(16)]; x += " "; x += verb[rnd(12)]; x += " the ";
            x += adj[rnd(16)]; x += " "; x += noun[rnd(16)]; x += " near the "; x += place[rnd(8)]; x += ".";
            return x;
        };
        const char* colors[3] = {"red", "green", "blue"};
        const char* codes[3] = {"4817", "2953", "7106"};
        const std::string head = "<|im_start|>user\nRead the following document carefully.\n\n";
        const std::string tail =
            "\n\nAccording to the document, what are the secret codes for the red door, the green door and the blue "
            "door? Answer with the three codes only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        // build sentences until the token budget is reached (approximate, then exact via tokenizer)
        std::vector<std::string> sents;
        const int target = depth > 0 ? depth : 100000;
        int est = 0;
        while (est < target) { sents.push_back(sentence()); est += 10; }   // ~12.7 tokens per sentence: over-provision
        auto build = [&](size_t n) {
            std::string doc;
            const size_t pos[3] = {n / 10, n / 2, n * 9 / 10};
            for (size_t i = 0; i < n; i++) {
                for (int k = 0; k < 3; k++)
                    if (i == pos[k]) doc += std::string(" Important: the secret code for the ") + colors[k] + " door is " +
                                           codes[k] + ".";
                doc += (i % 7 == 0 ? "\n" : " ");
                doc += sents[i];
            }
            return head + doc + tail;
        };
        size_t lo = 1, hi = sents.size();
        while (lo < hi) {   // largest sentence count whose prompt fits the target
            const size_t mid = (lo + hi + 1) / 2;
            if ((int)tk.encode(build(mid)).size() <= target) lo = mid; else hi = mid - 1;
        }
        const std::vector<int32_t> prompt = tk.encode(build(lo));
        printf("needle prompt: %zu tokens (%zu sentences), KV %s, ctx %d\n", prompt.size(), lo,
               cfg.kv_type == cuda::KVType::Q4_0 ? "q4_0" : cfg.kv_type == cuda::KVType::Q8_0 ? "q8_0" : "f16", cfg.max_context);
        fflush(stdout);
        DecodeResult res;
        const double t0 = now_s();
        eng.prefill_batched(prompt, &res, 1, {}, nullptr, chunk);
        const double t1 = now_s();
        std::vector<int32_t> out;
        int tok = res.topk_ids[0];
        eng.stats() = DecodeStats();
        const double t2 = now_s();
        int steps = 0;
        if (eng.has_mtp()) {   // speculative greedy with adaptive draft length
            DraftController dc;
            int cycles = 0, acc = 0, drafted = 0;
            bool done = false;
            while (!done && (int)out.size() < gen) {
                const int k = dc.choose();
                int32_t batch[Engine::MAX_BATCH];
                batch[0] = tok;
                const double c0 = now_s();
                eng.mtp_draft(tok, k, batch + 1);
                DecodeResult o[Engine::MAX_BATCH];
                eng.decode_batch(batch, k + 1, 1, false, o, true);
                int j = 0;
                while (j < k && batch[1 + j] == o[j].topk_ids[0]) j++;
                eng.commit(j + 1);
                dc.update(k, j, now_s() - c0);
                for (int t = 0; t <= j && !done; t++) {
                    out.push_back(batch[t]);
                    done = batch[t] == tk.eos() || batch[t] == 248046;
                }
                tok = o[j].topk_ids[0];
                cycles++; acc += j; drafted += k;
            }
            steps = (int)out.size();
            printf("speculative: %d tokens in %d cycles, acceptance %.0f%%\n", steps, cycles, 100.0 * acc / std::max(drafted, 1));
        } else {
            for (; steps < gen; steps++) {
                out.push_back(tok);
                if (tok == tk.eos() || tok == 248046) break;
                eng.decode(tok, 1, false, res);
                tok = res.topk_ids[0];
            }
        }
        const double t3 = now_s();
        const std::string answer = tk.decode(out);
        int found = 0;
        for (int k = 0; k < 3; k++) found += answer.find(codes[k]) != std::string::npos;
        size_t free_b, total_b;
        cudaMemGetInfo(&free_b, &total_b);
        long rss_kb = 0;
        if (FILE* f = fopen("/proc/self/status", "r")) {
            char line[256];
            while (fgets(line, sizeof line, f)) if (!strncmp(line, "VmRSS:", 6)) rss_kb = atol(line + 6);
            fclose(f);
        }
        printf("prefill %zu tokens: %.1f s (%.1f tok/s) | decode %d tokens at depth %zu: %.2f tok/s\n", prompt.size(),
               t1 - t0, prompt.size() / (t1 - t0), steps, prompt.size(), steps / std::max(t3 - t2, 1e-9));
        printf("answer: %s\n", answer.c_str());
        printf("needles found: %d/3 (red@10%%=%s green@50%%=%s blue@90%%=%s) | VRAM free %.0f MiB of %.0f | RSS %.1f GiB\n",
               found, answer.find(codes[0]) != std::string::npos ? "yes" : "NO",
               answer.find(codes[1]) != std::string::npos ? "yes" : "NO",
               answer.find(codes[2]) != std::string::npos ? "yes" : "NO", free_b / 1048576.0, total_b / 1048576.0,
               rss_kb / 1048576.0);
        printf("RESULT needle_found=%d prefill_tok_s=%.3f decode_tok_s=%.4f\n", found, prompt.size() / (t1 - t0),
               steps / std::max(t3 - t2, 1e-9));
        return found == 3 ? 0 : 5;
    }
    if (mode == "pbench") {
        // batched prefill of `depth` pseudo-random tokens, then `gen` real decode steps at that depth
        std::vector<int32_t> toks(depth);
        for (int i = 0; i < depth; i++) toks[i] = 1000 + (int)(((uint64_t)i * 2654435761u) % 150000);
        DecodeResult res;
        const double t0 = now_s();
        eng.prefill_batched(toks, &res, 1, {}, nullptr, chunk);
        const double t1 = now_s();
        const Engine::PrefillStats& ps = eng.prefill_stats();
        int tok = res.topk_ids.empty() ? 9707 : res.topk_ids[0];
        eng.stats() = DecodeStats();
        const double t2 = now_s();
        for (int i = 0; i < gen; i++) { eng.decode(tok, 1, false, res); tok = res.topk_ids[0]; }
        const double t3 = now_s();
        printf("prefill %d tokens: %.1f s = %.1f tok/s (streamed %.2f GB, %.2f GB/s) | decode %d @ depth %d: %.2f tok/s\n",
               depth, t1 - t0, depth / (t1 - t0), ps.bytes_streamed / 1e9, ps.bytes_streamed / 1e9 / (t1 - t0), gen, depth,
               gen / (t3 - t2));
        const DecodeStats& ds = eng.stats();
        printf("decode breakdown per token: cpu work %.1f ms, wait gpu %.1f ms, lm head %.1f ms\n", ds.t_cpu_work / gen * 1e3,
               ds.t_wait_gpu / gen * 1e3, ds.t_lm / gen * 1e3);
        printf("RESULT prefill_tok_s=%.3f decode_tok_s=%.4f\n", depth / (t1 - t0), gen / (t3 - t2));
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
