/// @file test_quant_ref.cpp
/// Bit-exact check of our reference (de)quantizers against libggml (the trusted
/// implementation, loaded from the local llama.cpp build). Uses real rows from
/// every quantized tensor of the target GGUF, plus random/edge-case inputs for
/// the activation and KV quantizers.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>
#include "quant/quant_formats.h"
#include "runtime/gguf_parser.h"

extern "C" {
// libggml-base.so (signatures from ggml/src/ggml-quants.h; block pointers as void*)
void dequantize_row_q2_K(const void* x, float* y, int64_t k);
void dequantize_row_q4_K(const void* x, float* y, int64_t k);
void dequantize_row_q6_K(const void* x, float* y, int64_t k);
void dequantize_row_iq2_xxs(const void* x, float* y, int64_t k);
void dequantize_row_iq2_xs(const void* x, float* y, int64_t k);
void dequantize_row_iq2_s(const void* x, float* y, int64_t k);
void dequantize_row_iq3_xxs(const void* x, float* y, int64_t k);
void dequantize_row_iq3_s(const void* x, float* y, int64_t k);
void dequantize_row_iq1_s(const void* x, float* y, int64_t k);
void dequantize_row_iq1_m(const void* x, float* y, int64_t k);
void dequantize_row_iq4_xs(const void* x, float* y, int64_t k);
void dequantize_row_q8_0(const void* x, float* y, int64_t k);
void dequantize_row_q4_0(const void* x, float* y, int64_t k);
void quantize_row_q8_K_ref(const float* x, void* y, int64_t k);
void quantize_row_q8_0_ref(const float* x, void* y, int64_t k);
void quantize_row_q4_K_ref(const float* x, void* y, int64_t k);
void quantize_row_q4_0_ref(const float* x, void* y, int64_t k);
}

using namespace q38rt;
typedef void (*ggml_dq_fn)(const void*, float*, int64_t);

static ggml_dq_fn ggml_dequant(QuantFormat t) {
    switch (t) {
        case QuantFormat::Q2_K: return dequantize_row_q2_K;
        case QuantFormat::Q4_K: return dequantize_row_q4_K;
        case QuantFormat::Q6_K: return dequantize_row_q6_K;
        case QuantFormat::IQ2_XXS: return dequantize_row_iq2_xxs;
        case QuantFormat::IQ2_XS: return dequantize_row_iq2_xs;
        case QuantFormat::IQ2_S: return dequantize_row_iq2_s;
        case QuantFormat::IQ3_XXS: return dequantize_row_iq3_xxs;
        case QuantFormat::IQ3_S: return dequantize_row_iq3_s;
        case QuantFormat::IQ1_S: return dequantize_row_iq1_s;
        case QuantFormat::IQ1_M: return dequantize_row_iq1_m;
        case QuantFormat::IQ4_XS: return dequantize_row_iq4_xs;
        case QuantFormat::Q8_0: return dequantize_row_q8_0;
        case QuantFormat::Q4_0: return dequantize_row_q4_0;
        default: return nullptr;
    }
}

static int g_fail = 0;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 2; }
    GGUFParser p;
    if (!p.parse(argv[1])) return 1;

    // 1) Weight dequantization on real data: rows 0, 1, middle, last of every quantized tensor.
    std::map<std::string, std::pair<long, long>> per_type;  // rows checked, mismatching rows
    std::vector<float> a, b;
    for (const TensorInfo& t : p.get_tensors()) {
        ggml_dq_fn ref = ggml_dequant(t.quant);
        if (!ref) continue;
        const int64_t ne0 = (int64_t)t.shape.dims[0];
        const int64_t nrows = (int64_t)(t.shape.size() / t.shape.dims[0]);
        const size_t row_bytes = quant_row_bytes(t.quant, ne0);
        a.resize(ne0);
        b.resize(ne0);
        const int64_t rows[] = {0, 1, nrows / 2, nrows - 1};
        for (int64_t r : rows) {
            const uint8_t* src = p.tensor_data(t) + (size_t)r * row_bytes;
            std::fill(a.begin(), a.end(), NAN);
            std::fill(b.begin(), b.end(), -1.0f);
            if (!dequantize_row_ref(t.quant, src, a.data(), ne0)) { g_fail++; continue; }
            ref(src, b.data(), ne0);
            auto& e = per_type[quant_name(t.quant)];
            e.first++;
            if (memcmp(a.data(), b.data(), ne0 * sizeof(float)) != 0) {
                e.second++;
                if (e.second <= 3) {
                    for (int64_t i = 0; i < ne0; i++)
                        if (memcmp(&a[i], &b[i], 4)) {
                            fprintf(stderr, "MISMATCH %s row %lld elem %lld: ours %.9g ggml %.9g\n", t.name.c_str(),
                                    (long long)r, (long long)i, a[i], b[i]);
                            break;
                        }
                }
            }
        }
    }
    for (const auto& kv : per_type) {
        printf("dequant %-8s rows %5ld  mismatches %ld\n", kv.first.c_str(), kv.second.first, kv.second.second);
        if (kv.second.second) g_fail++;
    }
    // trunk: 10 quantized formats; the MTP variant adds Q6_K (its nextn block)
    const size_t want_types = p.get_config().num_nextn_layers > 0 ? 11 : 10;
    if (per_type.size() != want_types) {
        fprintf(stderr, "expected %zu quantized weight types, got %zu\n", want_types, per_type.size());
        g_fail++;
    }

    // 2) Activation / KV quantizers: random rows with varied scale, plus edge cases.
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int64_t K = 17408;
    std::vector<float> x(K);
    std::vector<uint8_t> qa(K * 2), qb(K * 2);
    int quant_mismatch[3] = {0, 0, 0};
    int q4k_mismatch = 0;
    for (int trial = 0; trial < 200; trial++) {
        const float scale = std::pow(10.f, (float)(trial % 9) - 4.f);
        for (int64_t i = 0; i < K; i++) x[i] = nd(rng) * scale * (1 + (i % 7));
        if (trial == 1) std::fill(x.begin(), x.end(), 0.f);
        if (trial == 2) for (int64_t i = 0; i < K; i++) x[i] = (i % 256 == 3) ? -5.f : 0.f;
        if (trial == 3) for (int64_t i = 0; i < K; i++) x[i] = 1e-30f * (float)(i % 3);
        std::fill(qa.begin(), qa.end(), 0xAB); std::fill(qb.begin(), qb.end(), 0xAB);
        quantize_row_q8_K_ref(x.data(), (block_q8_K*)qa.data(), K);
        ::quantize_row_q8_K_ref(x.data(), qb.data(), K);
        // ggml leaves bsums untouched for all-zero blocks; compare d and qs always, bsums for nonzero blocks
        for (int64_t blk = 0; blk < K / QK_K; blk++) {
            const block_q8_K* ba = (const block_q8_K*)qa.data() + blk;
            const block_q8_K* bb = (const block_q8_K*)qb.data() + blk;
            if (memcmp(&ba->d, &bb->d, 4) || memcmp(ba->qs, bb->qs, QK_K) ||
                (ba->d != 0 && memcmp(ba->bsums, bb->bsums, sizeof(ba->bsums)))) { quant_mismatch[0]++; break; }
        }
        quantize_row_q8_0_ref(x.data(), (block_q8_0*)qa.data(), K);
        ::quantize_row_q8_0_ref(x.data(), qb.data(), K);
        if (memcmp(qa.data(), qb.data(), K / 32 * sizeof(block_q8_0))) quant_mismatch[1]++;
        quantize_row_q4_0_ref(x.data(), (block_q4_0*)qa.data(), K);
        ::quantize_row_q4_0_ref(x.data(), qb.data(), K);
        if (memcmp(qa.data(), qb.data(), K / 32 * sizeof(block_q4_0))) quant_mismatch[2]++;
        if (trial % 4 == 0) {   // Q4_K (MTP requantization), slower: every 4th trial
            std::vector<uint8_t> ka(K / QK_K * sizeof(block_q4_K), 0xAB), kb(ka.size(), 0xAB);
            quantize_row_q4_K_ref(x.data(), (block_q4_K*)ka.data(), K);
            ::quantize_row_q4_K_ref(x.data(), kb.data(), K);
            if (memcmp(ka.data(), kb.data(), ka.size())) q4k_mismatch++;
        }
        // and our KV dequantizers vs ggml's
        for (QuantFormat kt : {QuantFormat::Q8_0, QuantFormat::Q4_0}) {
            const void* q = qb.data();
            if (kt == QuantFormat::Q8_0) ::quantize_row_q8_0_ref(x.data(), qb.data(), K);
            a.assign(K, 0.f); b.assign(K, 1.f);
            dequantize_row_ref(kt, q, a.data(), K);
            ggml_dequant(kt)(q, b.data(), K);
            if (memcmp(a.data(), b.data(), K * 4)) quant_mismatch[kt == QuantFormat::Q8_0 ? 1 : 2]++;
        }
    }
    printf("quantize q8_K mismatching trials %d / 200\n", quant_mismatch[0]);
    printf("quantize q8_0 mismatching trials %d / 200\n", quant_mismatch[1]);
    printf("quantize q4_0 mismatching trials %d / 200\n", quant_mismatch[2]);
    printf("quantize q4_K mismatching trials %d / 50\n", q4k_mismatch);
    g_fail += quant_mismatch[0] + quant_mismatch[1] + quant_mismatch[2] + q4k_mismatch;

    // 3) fp16 conversion: exhaustive round trip vs reference semantics
    int fp16_bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        const float f = fp16_to_fp32((fp16_t)h);
        if (std::isnan(f)) continue;
        if (fp32_to_fp16(f) != (fp16_t)h) fp16_bad++;
    }
    printf("fp16 round-trip failures: %d\n", fp16_bad);
    g_fail += fp16_bad;

    if (g_fail) { fprintf(stderr, "test_quant_ref: %d failures\n", g_fail); return 1; }
    printf("test_quant_ref: all bit-exact\n");
    return 0;
}
