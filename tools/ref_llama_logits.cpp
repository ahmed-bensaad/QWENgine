/// @file ref_llama_logits.cpp
/// Golden-reference generator using the trusted llama.cpp implementation (libllama).
///
/// Usage: ref_llama_logits <model.gguf> <prompt.txt> <out.bin> [--ngl N] [--gen N] [--kv f32|f16|q8_0|q4_0]
///                         [--threads N] [--last N] [--tokens-only]
///
/// Tokenizes the prompt (no BOS: add_bos=false for qwen35; special tokens parsed), evaluates it in one
/// batch, then greedily decodes --gen tokens one at a time. Output file (little endian):
///   char magic[4] = "Q38G"; uint32 version = 1; uint32 n_prompt; uint32 n_gen; uint32 n_vocab;
///   uint32 n_logit_rows; int32 tokens[n_prompt + n_gen];
///   n_logit_rows x { int32 position; float logits[n_vocab]; }
/// Logit rows: the last --last prompt positions (default 8) and every generated step. The logits at
/// position p predict token p+1.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "llama.h"

static std::string read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt.txt> <out.bin> [--ngl N] [--gen N] [--kv f32|f16|q8_0|q4_0] "
                        "[--threads N] [--last N] [--tokens-only]\n", argv[0]);
        return 1;
    }
    int ngl = 0, n_gen = 8, threads = 8, n_last = 8;
    bool tokens_only = false;
    ggml_type kv = GGML_TYPE_F32;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--ngl")) ngl = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen")) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--last")) n_last = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens-only")) tokens_only = true;
        else if (!strcmp(argv[i], "--kv")) {
            std::string k = argv[++i];
            kv = k == "f16" ? GGML_TYPE_F16 : k == "q8_0" ? GGML_TYPE_Q8_0 : k == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F32;
        }
    }
    const std::string prompt = read_file(argv[2]);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.data(), (int32_t)prompt.size(), toks.data(), (int32_t)toks.size(),
                           /*add_special=*/false, /*parse_special=*/true);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);
    fprintf(stderr, "prompt tokens: %d\n", n);
    FILE* fo = fopen(argv[3], "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    if (tokens_only) {
        uint32_t hdr[5] = {1, (uint32_t)n, 0, (uint32_t)n_vocab, 0};
        fwrite("Q38G", 1, 4, fo);
        fwrite(hdr, 4, 5, fo);
        fwrite(toks.data(), 4, n, fo);
        fclose(fo);
        llama_model_free(model);
        return 0;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)(n + n_gen + 16);
    cp.n_batch = (uint32_t)std::max(n, 1);
    cp.n_ubatch = (uint32_t)std::max(n, 1);
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.type_k = kv;
    cp.type_v = kv;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (kv == GGML_TYPE_Q8_0 || kv == GGML_TYPE_Q4_0) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.no_perf = true;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }

    std::vector<std::pair<int32_t, std::vector<float>>> rows;
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i >= n - n_last;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    for (int i = std::max(0, n - n_last); i < n; i++) {
        const float* lg = llama_get_logits_ith(ctx, i);
        rows.emplace_back(i, std::vector<float>(lg, lg + n_vocab));
    }
    std::vector<llama_token> all = toks;
    llama_token next = 0;
    {
        const std::vector<float>& lg = rows.back().second;
        next = (llama_token)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    }
    llama_batch one = llama_batch_init(1, 0, 1);
    for (int g = 0; g < n_gen; g++) {
        all.push_back(next);
        one.token[0] = next;
        one.pos[0] = (llama_pos)(all.size() - 1);
        one.n_seq_id[0] = 1;
        one.seq_id[0][0] = 0;
        one.logits[0] = true;
        one.n_tokens = 1;
        if (llama_decode(ctx, one) != 0) { fprintf(stderr, "decode step failed\n"); return 1; }
        const float* lg = llama_get_logits_ith(ctx, 0);
        rows.emplace_back((int32_t)(all.size() - 1), std::vector<float>(lg, lg + n_vocab));
        next = (llama_token)(std::max_element(lg, lg + n_vocab) - lg);
    }

    uint32_t hdr[5] = {1, (uint32_t)n, (uint32_t)n_gen, (uint32_t)n_vocab, (uint32_t)rows.size()};
    fwrite("Q38G", 1, 4, fo);
    fwrite(hdr, 4, 5, fo);
    fwrite(all.data(), 4, all.size(), fo);
    for (auto& r : rows) {
        fwrite(&r.first, 4, 1, fo);
        fwrite(r.second.data(), 4, n_vocab, fo);
    }
    fclose(fo);

    std::string text;
    for (size_t i = toks.size(); i < all.size(); i++) {
        char piece[256];
        int len = llama_token_to_piece(vocab, all[i], piece, sizeof piece, 0, true);
        if (len > 0) text.append(piece, len);
    }
    fprintf(stderr, "greedy continuation: %s\n", text.c_str());
    llama_batch_free(batch);
    llama_batch_free(one);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
