/// @file ref_compare.cpp
/// Run the FP32 reference model on a golden prompt and compare its logits with a golden file
/// (llama.cpp). Prompt rows come from one batched forward (prefill path); generated rows come
/// from single-token steps with the golden's own tokens (teacher forcing, decode path).
/// Usage: ref_compare <model.gguf> <golden.bin> [--save ours.bin] [--threads N] [--golden2 other.bin]

#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include "reference/ref_model.h"
#include "../tests/golden/golden_io.h"

using namespace q38rt;

static void report(const char* label, const golden::File& ref, const golden::File& test) {
    double kl_sum = 0, kl_max = 0, cos_min = 1, abs_max = 0;
    int match = 0;
    for (size_t r = 0; r < ref.rows.size(); r++) {
        golden::RowMetrics m = golden::compare(ref.rows[r].data(), test.rows[r].data(), ref.n_vocab);
        match += m.top1_match;
        kl_sum += m.kl;
        kl_max = std::max(kl_max, m.kl);
        cos_min = std::min(cos_min, m.cos);
        abs_max = std::max(abs_max, m.max_abs);
        if (!m.top1_match)
            printf("  [%s] pos %d: top1 ref %d (p=%.3f) vs test %d, KL %.4f\n", label, ref.row_pos[r], m.top1_ref,
                   m.p_ref_top1, m.top1_test, m.kl);
    }
    printf("%-28s rows %2zu  top1 %2d/%zu  KL mean %.5f max %.5f  cos min %.6f  max|dlogit| %.3f\n", label,
           ref.rows.size(), match, ref.rows.size(), kl_sum / ref.rows.size(), kl_max, cos_min, abs_max);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <golden.bin> [--save ours.bin] [--threads N] [--golden2 b.bin]\n", argv[0]);
        return 1;
    }
    std::string save, golden2;
    int threads = 8;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--save")) save = argv[++i];
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--golden2")) golden2 = argv[++i];
    }
    golden::File g;
    if (!golden::read(argv[2], g)) { fprintf(stderr, "cannot read golden %s\n", argv[2]); return 1; }
    GGUFParser p;
    if (!p.parse(argv[1])) return 1;
    RefModel m(p);
    m.set_threads(threads);

    golden::File ours;
    ours.n_prompt = g.n_prompt; ours.n_gen = g.n_gen; ours.n_vocab = g.n_vocab; ours.tokens = g.tokens;
    // prompt rows
    std::vector<int> prompt_rows;
    for (size_t r = 0; r < g.rows.size(); r++)
        if (g.row_pos[r] < (int)g.n_prompt) prompt_rows.push_back(g.row_pos[r]);
    std::vector<float> logits;
    auto t0 = std::chrono::steady_clock::now();
    m.forward(g.tokens.data(), (int)g.n_prompt, prompt_rows, logits);
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "reference prefill of %u tokens: %.1f s\n", g.n_prompt, std::chrono::duration<double>(t1 - t0).count());
    for (size_t i = 0; i < prompt_rows.size(); i++) {
        ours.row_pos.push_back(prompt_rows[i]);
        ours.rows.emplace_back(logits.begin() + i * g.n_vocab, logits.begin() + (i + 1) * g.n_vocab);
    }
    // decode steps (teacher forced): stored rows at positions n_prompt .. n_prompt+n_gen-1
    for (size_t r = 0; r < g.rows.size(); r++) {
        const int pos = g.row_pos[r];
        if (pos < (int)g.n_prompt) continue;
        if (pos != m.pos()) { fprintf(stderr, "unexpected row position %d (model at %d)\n", pos, m.pos()); return 1; }
        m.forward(&g.tokens[pos], 1, {0}, logits);
        ours.row_pos.push_back(pos);
        ours.rows.push_back(logits);
    }
    auto t2 = std::chrono::steady_clock::now();
    fprintf(stderr, "reference decode of %u tokens: %.1f s\n", g.n_gen, std::chrono::duration<double>(t2 - t1).count());
    if (!save.empty()) golden::write(save, ours);

    report("llama.cpp(golden) vs ref", g, ours);
    if (!golden2.empty()) {
        golden::File g2;
        if (golden::read(golden2, g2) && g2.rows.size() == g.rows.size()) {
            report("golden2 vs ref", g2, ours);
            report("golden vs golden2", g, g2);
        }
    }
    return 0;
}
