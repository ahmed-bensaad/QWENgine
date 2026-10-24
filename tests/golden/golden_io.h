#pragma once
/// @file golden_io.h
/// Reader/writer and comparison metrics for golden logit files (format: tools/ref_llama_logits.cpp).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace golden {

struct File {
    uint32_t n_prompt = 0, n_gen = 0, n_vocab = 0;
    std::vector<int32_t> tokens;                 // n_prompt + n_gen
    std::vector<int32_t> row_pos;                // position whose logits are stored
    std::vector<std::vector<float>> rows;        // logits per stored position
};

inline bool read(const std::string& path, File& g) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    uint32_t hdr[5];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "Q38G", 4) || fread(hdr, 4, 5, f) != 5 || hdr[0] != 1) {
        fclose(f);
        return false;
    }
    g.n_prompt = hdr[1]; g.n_gen = hdr[2]; g.n_vocab = hdr[3];
    const uint32_t nrows = hdr[4];
    g.tokens.resize(g.n_prompt + g.n_gen);
    bool ok = fread(g.tokens.data(), 4, g.tokens.size(), f) == g.tokens.size();
    g.row_pos.resize(nrows);
    g.rows.assign(nrows, std::vector<float>(g.n_vocab));
    for (uint32_t r = 0; ok && r < nrows; r++) {
        ok = fread(&g.row_pos[r], 4, 1, f) == 1 && fread(g.rows[r].data(), 4, g.n_vocab, f) == g.n_vocab;
    }
    fclose(f);
    return ok;
}

inline bool write(const std::string& path, const File& g) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    uint32_t hdr[5] = {1, g.n_prompt, g.n_gen, g.n_vocab, (uint32_t)g.rows.size()};
    fwrite("Q38G", 1, 4, f);
    fwrite(hdr, 4, 5, f);
    fwrite(g.tokens.data(), 4, g.tokens.size(), f);
    for (size_t r = 0; r < g.rows.size(); r++) {
        fwrite(&g.row_pos[r], 4, 1, f);
        fwrite(g.rows[r].data(), 4, g.n_vocab, f);
    }
    fclose(f);
    return true;
}

struct RowMetrics {
    bool top1_match = false;
    double kl = 0;          // KL(P_ref || P_test), nats
    double cos = 0;         // cosine similarity of logit vectors
    double max_abs = 0;     // max |logit diff|
    int top1_ref = -1, top1_test = -1;
    double p_ref_top1 = 0;  // reference probability of its own top-1
};

inline RowMetrics compare(const float* ref, const float* test, size_t n) {
    RowMetrics m;
    m.top1_ref = (int)(std::max_element(ref, ref + n) - ref);
    m.top1_test = (int)(std::max_element(test, test + n) - test);
    m.top1_match = m.top1_ref == m.top1_test;
    const double mr = ref[m.top1_ref], mt = test[m.top1_test];
    double zr = 0, zt = 0, dot = 0, nr = 0, nt = 0;
    for (size_t i = 0; i < n; i++) {
        zr += std::exp(ref[i] - mr);
        zt += std::exp(test[i] - mt);
        dot += (double)ref[i] * test[i];
        nr += (double)ref[i] * ref[i];
        nt += (double)test[i] * test[i];
        m.max_abs = std::max(m.max_abs, (double)std::fabs(ref[i] - test[i]));
    }
    const double lzr = mr + std::log(zr), lzt = mt + std::log(zt);
    double kl = 0;
    for (size_t i = 0; i < n; i++) {
        const double lpr = ref[i] - lzr, lpt = test[i] - lzt;
        const double pr = std::exp(lpr);
        if (pr > 0) kl += pr * (lpr - lpt);
    }
    m.kl = kl;
    m.cos = dot / std::sqrt(nr * nt);
    m.p_ref_top1 = std::exp(ref[m.top1_ref] - lzr);
    return m;
}

} // namespace golden
