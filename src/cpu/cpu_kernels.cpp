/// @file cpu_kernels.cpp
/// AVX2 dot products for the 10 weight formats of the target GGUF against Q8_K activations.
///
/// Common structure per 32-weight sub-block: build 32 weight bytes (grid formats: vpgather from
/// the L1-resident grid — this Coffee Lake CPU has no gather microcode mitigation; LUT formats:
/// pshufb), apply the weight signs to the activations (_mm256_sign_epi8), maddubs (u8 x s8 ->
/// s16 pairs) and madd with the sub-block scale (-> s32). Integer arithmetic is exact; the
/// per-super-block float scale is applied once. IQ1 formats fold their +-1/8 delta into the
/// weights as 8*(g + delta) in {+-1, +-7, +-9}, which keeps everything in int8.
/// Bit layouts: ggml-common.h (MIT, The ggml authors).

#include "cpu/cpu_kernels.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include "quant/iq_tables.h"

namespace q38rt {
namespace cpu {

namespace {

inline float f16(fp16_t h) { return _cvtsh_ss(h); }

inline float hsum8(__m256 v) {
    // fold 8 floats to 4, then 2, then 1
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline int hsum8i(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

inline uint32_t ld32(const void* p) { uint32_t v; memcpy(&v, p, 4); return v; }
inline uint64_t ld64(const void* p) { uint64_t v; memcpy(&v, p, 8); return v; }

// 32 sign bits (bit w -> weight w) -> -1/+1 bytes for _mm256_sign_epi8.
inline __m256i signs_to_vec(uint32_t bits) {
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                          2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bitm = _mm256_set1_epi64x(0x8040201008040201LL);
    __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32((int)bits), shuf);   // byte k of the result = sign byte k / 8
    v = _mm256_cmpeq_epi8(_mm256_and_si256(v, bitm), bitm);   // 0xFF where the weight's sign bit is set
    return _mm256_or_si256(v, _mm256_set1_epi8(1));   // -1 (0xFF) or +1 per byte
}

// ksigns: 7 stored sign bits per 8 weights, the 8th = parity. Table: 7-bit group -> 8 sign bytes (+1 / -1) for
// _mm256_sign_epi8 (4 loads instead of pdep + parity + signs_to_vec, ~15 operations).
struct KSignsTable {
    uint64_t v[128];
    KSignsTable() {
        for (int g = 0; g < 128; g++) {
            const int bits = g | (__builtin_parity(g) << 7);   // 8th sign bit restores even parity
            uint64_t e = 0;
            for (int i = 0; i < 8; i++) e |= (uint64_t)((bits >> i) & 1 ? 0xFF : 0x01) << (8 * i);
            v[g] = e;
        }
    }
};
const KSignsTable kKSigns;

// Four 7-bit ksigns groups (low 28 bits, group g = weights 8g..8g+7) -> sign vector.
inline __m256i ksigns_vec(uint32_t v28) {
    const uint64_t* t = kKSigns.v;
    return _mm256_set_epi64x((long long)t[(v28 >> 21) & 127], (long long)t[(v28 >> 14) & 127],
                             (long long)t[(v28 >> 7) & 127], (long long)t[v28 & 127]);
}

// Per-sub-block scale bytes (low nibble -> first 16 weights, high nibble -> last 16) of 8 sub-blocks -> words
// 2*nibble+1 with the low nibbles in the low lane and the high nibbles in the high lane (for bcast_word_ctrl(s)).
inline __m256i lohi_nibble_scales_odd(uint64_t v) {
    const __m128i b = _mm_cvtsi64_si128((long long)v);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i lh = _mm_unpacklo_epi64(_mm_and_si128(b, m4), _mm_and_si128(_mm_srli_epi16(b, 4), m4));
    return _mm256_or_si256(_mm256_slli_epi16(_mm256_cvtepu8_epi16(lh), 1), _mm256_set1_epi16(1));
}

// Broadcast 16-bit word k of each 128-bit lane (pshufb control).
inline __m256i bcast_word_ctrl(int k) { return _mm256_set1_epi16((short)((2 * k + 1) << 8 | (2 * k))); }

// 8 scale nibbles (low nibble first) of 4 bytes -> words 2*ls+1, in both lanes (for bcast_word_ctrl(s)).
inline __m256i nibble_scales_odd(uint32_t v) {
    const __m128i b = _mm_cvtsi32_si128((int)v);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i ls = _mm_unpacklo_epi8(_mm_and_si128(b, m4), _mm_and_si128(_mm_srli_epi16(b, 4), m4));
    const __m128i w = _mm_or_si128(_mm_slli_epi16(_mm_cvtepu8_epi16(ls), 1), _mm_set1_epi16(1));
    return _mm256_broadcastsi128_si256(w);
}

// R4 re-encoding of IQ3_S / IQ3_XXS: per 256 weights fp16 d (copied), scale nibble per 32-weight sub-block (IQ3_S:
// its 4-bit scale, IQ3_XXS: aux >> 28; both used as 2*ls+1), and per weight a 4-bit code: level index (0-7) of the
// grid byte, +8 if the weight is negative. Sub-block byte j holds weight j (low nibble) and weight j+16 (high).
struct block_r4 { fp16_t d; uint8_t scales[4]; uint8_t qs[128]; };
static_assert(sizeof(block_r4) == 134, "r4");
// R2 re-encoding of IQ2_XXS / IQ2_XS / IQ2_S (grid bytes in {8, 25, 43}): per 256 weights fp16 d (copied), a scale
// nibble per 16 weights (sub-block byte s: low nibble = weights 0-15, high = 16-31; IQ2_XXS repeats its one scale),
// 2-bit magnitude codes in the Q2_K layout (32-byte chunk n, byte j, bits 2k: sub-block 4n+k, weight j) and the 256
// sign bits (sub-block s: 32 bits, bit w = weight w).
struct block_r2 { fp16_t d; uint8_t scales[8]; uint8_t mags[64]; uint8_t signs[32]; };
static_assert(sizeof(block_r2) == 106, "r2");
alignas(16) const uint8_t kR2Lut[16] = {8, 25, 43, 0, 8, 25, 43, 0, 8, 25, 43, 0, 8, 25, 43, 0};
alignas(16) const int8_t kR4Lut[2][16] = {
    {1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15},            // IQ3_S grid levels
    {4, 12, 20, 28, 36, 44, 52, 62, -4, -12, -20, -28, -36, -44, -52, -62}};   // IQ3_XXS grid levels

// ---------------------------------------------------------------------------
// Kernels are templated on NB = number of activation columns (tokens) sharing one pass over the weights:
// each sub-block is decoded once and dotted with NB activations (x + b * xs, b < NB). The per-column
// arithmetic is identical for every NB, so results are bit-identical to the single-column kernel.

#define Q38_COLS_INIT                                   \
    __m256 acc[NB];                                     \
    for (int b = 0; b < NB; b++) acc[b] = _mm256_setzero_ps()
#define Q38_SUMI_INIT                                   \
    __m256i sumi[NB];                                   \
    for (int b = 0; b < NB; b++) sumi[b] = _mm256_setzero_si256()
#define Q38_Q8(b, off) _mm256_loadu_si256((const __m256i*)(x[(b) * xs + i].qs + (off)))
// Materializes an accumulator in a register at this point. Without it GCC reassociates the per-column additions of
// the fully unrolled sub-block loop into a tree, keeping all partial products live: ~13 vector spills and reloads per
// super-block at 3 columns, and the float accumulators kept in memory. Integer sums are exact in any order and the
// float operations are unchanged, so results are bit-identical.
#define Q38_KEEP(v) __asm__ volatile("" : "+x"(v))

template <int NB>
void dot_iq3_s(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_iq3_s* w = (const block_iq3_s*)vw;
    const int nb = K / QK_K;
    const __m256i hshift = _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1);
    const __m256i m256 = _mm256_set1_epi32(256);
    Q38_COLS_INIT;
    for (int i = 0; i < nb; i++) {
        const uint8_t* qs = w[i].qs;
        Q38_SUMI_INIT;
#pragma GCC unroll 8
        for (int s = 0; s < 8; s++) {
            __m256i idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qs + 8 * s)));
            // 9th index bit of each of the 8 groups from qh
            idx = _mm256_or_si256(idx, _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(w[i].qh[s]), hshift), m256));
            // 8 gathers of 4 grid bytes = the 32 weight magnitudes
            const __m256i g = _mm256_i32gather_epi32((const int*)iq3s_grid, idx, 4);
            // signs go onto the activations (sign_epi8): maddubs needs unsigned weights
            const __m256i sv = signs_to_vec(ld32(w[i].signs + 4 * s));
            const int ls = (w[i].scales[s / 2] >> (4 * (s & 1))) & 0xF;
            // odd scale 1..31, applied in the int16 -> int32 madd
            const __m256i sc = _mm256_set1_epi16((short)(2 * ls + 1));
            for (int b = 0; b < NB; b++)
                // u8 weight x s8 signed activation -> s16 pairs, x scale -> s32: exact integer sum per column
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(g, _mm256_sign_epi8(Q38_Q8(b, 32 * s), sv)), sc)); Q38_KEEP(sumi[b]); }
        }
        const float dw = f16(w[i].d);
        for (int b = 0; b < NB; b++)
            // float scale once per 256 weights
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[b * xs + i].d), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]);
}

template <int NB>
void dot_iq3_xxs(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_iq3_xxs* w = (const block_iq3_xxs*)vw;
    const int nb = K / QK_K;
    Q38_COLS_INIT;
    for (int i = 0; i < nb; i++) {
        const uint8_t* qs = w[i].qs;
        Q38_SUMI_INIT;
#pragma GCC unroll 8
        for (int s = 0; s < 8; s++) {
            const __m256i idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qs + 8 * s)));
            const __m256i g = _mm256_i32gather_epi32((const int*)iq3xxs_grid, idx, 4);
            // sign indices and the 4-bit scale of the sub-block after the 64 index bytes
            const uint32_t aux = ld32(qs + QK_K / 4 + 4 * s);
            const __m256i sv = ksigns_vec(aux);
            const __m256i sc = _mm256_set1_epi16((short)(2 * (aux >> 28) + 1));
            for (int b = 0; b < NB; b++)
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(g, _mm256_sign_epi8(Q38_Q8(b, 32 * s), sv)), sc)); Q38_KEEP(sumi[b]); }
        }
        const float dw = f16(w[i].d);
        for (int b = 0; b < NB; b++)
            // IQ3_XXS's /2 and the (2s+1)/2 scale form: x 1/4
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[b * xs + i].d * 0.25f), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]);
}

template <int NB>
void dot_iq2_xxs(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_iq2_xxs* w = (const block_iq2_xxs*)vw;
    const int nb = K / QK_K;
    Q38_COLS_INIT;
    for (int i = 0; i < nb; i++) {
        const uint8_t* qs = (const uint8_t*)w[i].qs;
        Q38_SUMI_INIT;
#pragma GCC unroll 8
        for (int s = 0; s < 8; s++) {
            const uint32_t a0 = ld32(qs + 8 * s), a1 = ld32(qs + 8 * s + 4);
            const __m128i idx = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)a0));
            // 4 gathers of 8 grid bytes
            const __m256i g = _mm256_i32gather_epi64((const long long*)iq2xxs_grid, idx, 8);
            const __m256i sv = ksigns_vec(a1);
            const __m256i sc = _mm256_set1_epi16((short)(2 * (a1 >> 28) + 1));
            for (int b = 0; b < NB; b++)
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(g, _mm256_sign_epi8(Q38_Q8(b, 32 * s), sv)), sc)); Q38_KEEP(sumi[b]); }
        }
        const float dw = f16(w[i].d);
        for (int b = 0; b < NB; b++)
            // IQ2 scales (2s+1)/8
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[b * xs + i].d * 0.125f), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]);
}

template <int NB>
void dot_iq2_xs(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_iq2_xs* w = (const block_iq2_xs*)vw;
    const int nb = K / QK_K;
    const __m128i m511 = _mm_set1_epi32(511);
    Q38_COLS_INIT;
    for (int i = 0; i < nb; i++) {
        const uint8_t* qs = (const uint8_t*)w[i].qs;
        const __m256i sc16 = lohi_nibble_scales_odd(ld64(w[i].scales));
        Q38_SUMI_INIT;
#pragma GCC unroll 8
        for (int s = 0; s < 8; s++) {
            const uint64_t q64 = ld64(qs + 8 * s);
            // four 16-bit codes: 9-bit grid index ...
            const __m128i idx = _mm_and_si128(_mm_cvtepu16_epi32(_mm_cvtsi64_si128((long long)q64)), m511);
            const __m256i g = _mm256_i32gather_epi64((const long long*)iq2xs_grid, idx, 8);
            // ... and their 7-bit sign indices, packed together by pext
            const __m256i sv = ksigns_vec((uint32_t)_pext_u64(q64, 0xFE00FE00FE00FE00ULL));
            // scale of weights 0-15 in the low lane, 16-31 in the high lane
            const __m256i sc = _mm256_shuffle_epi8(sc16, bcast_word_ctrl(s));
            for (int b = 0; b < NB; b++)
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(g, _mm256_sign_epi8(Q38_Q8(b, 32 * s), sv)), sc)); Q38_KEEP(sumi[b]); }
        }
        const float dw = f16(w[i].d);
        for (int b = 0; b < NB; b++)
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[b * xs + i].d * 0.125f), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]);
}

template <int NB>
void dot_iq2_s(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_iq2_s* w = (const block_iq2_s*)vw;
    const int nb = K / QK_K;
    const __m128i hshift = _mm_setr_epi32(8, 6, 4, 2);
    const __m128i m300 = _mm_set1_epi32(0x300);
    Q38_COLS_INIT;
    for (int i = 0; i < nb; i++) {
        const uint8_t* qs = w[i].qs;
        const __m256i sc16 = lohi_nibble_scales_odd(ld64(w[i].scales));
        Q38_SUMI_INIT;
#pragma GCC unroll 8
        for (int s = 0; s < 8; s++) {
            __m128i idx = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)ld32(qs + 4 * s)));
            // 2 high index bits per group from qh: 10-bit indices
            idx = _mm_or_si128(idx, _mm_and_si128(_mm_sllv_epi32(_mm_set1_epi32(w[i].qh[s]), hshift), m300));
            const __m256i g = _mm256_i32gather_epi64((const long long*)iq2s_grid, idx, 8);
            const __m256i sv = signs_to_vec(ld32(qs + QK_K / 8 + 4 * s));
            const __m256i sc = _mm256_shuffle_epi8(sc16, bcast_word_ctrl(s));
            for (int b = 0; b < NB; b++)
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(g, _mm256_sign_epi8(Q38_Q8(b, 32 * s), sv)), sc)); Q38_KEEP(sumi[b]); }
        }
        const float dw = f16(w[i].d);
        for (int b = 0; b < NB; b++)
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[b * xs + i].d * 0.125f), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]);
}

template <int NB>
void dot_iq1_s(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_iq1_s* w = (const block_iq1_s*)vw;
    const int nb = K / QK_K;
    const __m128i shifts = _mm_setr_epi32(0, 3, 6, 9);
    const __m128i m7 = _mm_set1_epi32(7);
    const __m256i eight = _mm256_set1_epi8(8);
    Q38_COLS_INIT;
    for (int i = 0; i < nb; i++) {
        const uint8_t* qs = w[i].qs;
        Q38_SUMI_INIT;
#pragma GCC unroll 8
        for (int s = 0; s < 8; s++) {
            const int qh = w[i].qh[s];
            __m128i idx = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)ld32(qs + 4 * s)));
            // 11-bit indices: 3 bits per group from qh
            idx = _mm_or_si128(idx, _mm_slli_epi32(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(qh), shifts), m7), 8));
            const __m256i g = _mm256_i32gather_epi64((const long long*)iq1s_grid, idx, 8);
            // 8 * (grid + delta) with grid in {-1,0,1}, delta = +-1/8: values in {+-1, +-7, +-9}
            const __m256i wv = _mm256_add_epi8(_mm256_sign_epi8(eight, g), _mm256_set1_epi8((qh & 0x8000) ? -1 : 1));
            // |w| as the unsigned operand, the sign moves to the activations
            const __m256i aw = _mm256_sign_epi8(wv, wv);
            const __m256i sc = _mm256_set1_epi16((short)(2 * ((qh >> 12) & 7) + 1));
            for (int b = 0; b < NB; b++)
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(Q38_Q8(b, 32 * s), wv)), sc)); Q38_KEEP(sumi[b]); }
        }
        const float dw = f16(w[i].d);
        for (int b = 0; b < NB; b++)
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[b * xs + i].d * 0.125f), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]);
}

template <int NB>
void dot_iq1_m(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_iq1_m* w = (const block_iq1_m*)vw;
    const int nb = K / QK_K;
    const __m128i m7 = _mm_set1_epi32(7);
    const __m256i eight = _mm256_set1_epi8(8);
    Q38_COLS_INIT;
    for (int i = 0; i < nb; i++) {
        uint16_t scw[4];
        memcpy(scw, w[i].scales, 8);
        // FP16 super-scale scattered over the top nibbles of the 4 scale words
        const uint16_t su = (scw[0] >> 12) | ((scw[1] >> 8) & 0x00f0) | ((scw[2] >> 4) & 0x0f00) | (scw[3] & 0xf000);
        const uint8_t* qs = w[i].qs;
        const uint8_t* qh = w[i].qh;
        Q38_SUMI_INIT;
#pragma GCC unroll 8
        for (int s = 0; s < 8; s++) {
            const int h0 = qh[2 * s], h1 = qh[2 * s + 1];
            const __m128i hv = _mm_setr_epi32(h0, h0 >> 4, h1, h1 >> 4);
            const __m128i idx = _mm_or_si128(_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)ld32(qs + 4 * s))),
                                             _mm_slli_epi32(_mm_and_si128(hv, m7), 8));
            const __m256i g = _mm256_i32gather_epi64((const long long*)iq1s_grid, idx, 8);
            const long long dm = -1LL, dp = 0x0101010101010101LL;
            // per group of 8 weights: delta sign from its qh nibble
            const __m256i dv = _mm256_setr_epi64x((h0 & 0x08) ? dm : dp, (h0 & 0x80) ? dm : dp,
                                                  (h1 & 0x08) ? dm : dp, (h1 & 0x80) ? dm : dp);
            const __m256i wv = _mm256_add_epi8(_mm256_sign_epi8(eight, g), dv);
            const __m256i aw = _mm256_sign_epi8(wv, wv);
            const int t = scw[s / 2] >> (6 * (s % 2));
            const __m256i sc = _mm256_set_m128i(_mm_set1_epi16((short)(2 * ((t >> 3) & 7) + 1)),
                                                _mm_set1_epi16((short)(2 * (t & 7) + 1)));
            for (int b = 0; b < NB; b++)
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(Q38_Q8(b, 32 * s), wv)), sc)); Q38_KEEP(sumi[b]); }
        }
        const float dw = f16(su);
        for (int b = 0; b < NB; b++)
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[b * xs + i].d * 0.125f), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]);
}

template <int NB>
void dot_iq4_xs(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_iq4_xs* w = (const block_iq4_xs*)vw;
    const int nb = K / QK_K;
    const __m256i table = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_iq4nl));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    Q38_COLS_INIT;
    for (int i = 0; i < nb; i++) {
        const uint8_t* qs = w[i].qs;
        // 6-bit scales: low nibble from scales_l (2 per byte), 2 high bits from scales_h -> words ls - 32
        const __m128i sl = _mm_cvtsi32_si128((int)ld32(w[i].scales_l));
        const __m128i lo = _mm_unpacklo_epi8(_mm_and_si128(sl, m4), _mm_and_si128(_mm_srli_epi16(sl, 4), m4));
        // spread the 2 high bits of each 6-bit scale to bits 4-5 of its byte
        const __m128i hi = _mm_cvtsi64_si128((long long)(_pdep_u64(w[i].scales_h, 0x0303030303030303ULL) << 4));
        const __m256i sc16 = _mm256_broadcastsi128_si256(
            _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_or_si128(lo, hi)), _mm_set1_epi16(32)));
        Q38_SUMI_INIT;
#pragma GCC unroll 8
        for (int s = 0; s < 8; s++) {
            const __m128i q4 = _mm_loadu_si128((const __m128i*)(qs + 16 * s));
            // low nibbles = weights 0-15, high nibbles = 16-31
            const __m256i nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q4, 4), m4), _mm_and_si128(q4, m4));
            const __m256i v = _mm256_shuffle_epi8(table, nib);   // no zero entries in kvalues_iq4nl
            const __m256i av = _mm256_sign_epi8(v, v);
            const __m256i sc = _mm256_shuffle_epi8(sc16, bcast_word_ctrl(s));
            for (int b = 0; b < NB; b++)
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(av, _mm256_sign_epi8(Q38_Q8(b, 32 * s), v)), sc)); Q38_KEEP(sumi[b]); }
        }
        const float dw = f16(w[i].d);
        for (int b = 0; b < NB; b++)
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[b * xs + i].d), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]);
}

template <int NB>
void dot_q4_K(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_q4_K* w = (const block_q4_K*)vw;
    const int nb = K / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    Q38_COLS_INIT;
    float acc_m[NB];
    for (int b = 0; b < NB; b++) acc_m[b] = 0.0f;
    for (int i = 0; i < nb; i++) {
        // 12 bytes of 6-bit scales/mins -> sc[0..7] (bytes 0-7), mn[0..7] (bytes 8-15) (ggml's kmask unpacking)
        uint32_t u[4];
        memcpy(u, w[i].scales, 12);
        u[3] = ((u[2] >> 4) & 0x0f0f0f0fu) | (((u[1] >> 6) & 0x03030303u) << 4);   // mins 4-7: low 4 bits + 2 high bits
        const uint32_t ua = u[1] & 0x3f3f3f3fu;
        u[1] = (u[2] & 0x0f0f0f0fu) | (((u[0] >> 6) & 0x03030303u) << 4);   // scales 4-7
        u[2] = ua;
        u[0] &= 0x3f3f3f3fu;
        const __m128i scmn = _mm_loadu_si128((const __m128i*)u);
        const __m256i sc16 = _mm256_broadcastsi128_si256(_mm_cvtepu8_epi16(scmn));            // sc0..7 per lane
        const __m128i mnb = _mm_srli_si128(scmn, 8);
        const __m256i mn16 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mnb, mnb));                // mn0,mn0,mn1,mn1..
        Q38_SUMI_INIT;
#pragma GCC unroll 4
        for (int p = 0; p < 4; p++) {
            const __m256i q4 = _mm256_loadu_si256((const __m256i*)(w[i].qs + 32 * p));
            const __m256i lo = _mm256_and_si256(q4, m4);   // 64 weights per 32 bytes: low nibbles (sub-block 2p) ...
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q4, 4), m4);   // ... and high nibbles (sub-block 2p+1)
            const __m256i scl = _mm256_shuffle_epi8(sc16, bcast_word_ctrl(2 * p));
            const __m256i sch = _mm256_shuffle_epi8(sc16, bcast_word_ctrl(2 * p + 1));
            for (int b = 0; b < NB; b++) {
                const __m256i pl = _mm256_maddubs_epi16(lo, Q38_Q8(b, 64 * p));
                const __m256i ph = _mm256_maddubs_epi16(hi, Q38_Q8(b, 64 * p + 32));
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(pl, scl)); Q38_KEEP(sumi[b]); }
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(ph, sch)); Q38_KEEP(sumi[b]); }
            }
        }
        const float dw = f16(w[i].d), dmin = f16(w[i].dmin);
        for (int b = 0; b < NB; b++) {
            const block_q8_K& xb = x[b * xs + i];
            // sum_j mn[j] * (bsums[2j] + bsums[2j+1]), exact in int32
            const int msum = hsum8i(_mm256_madd_epi16(mn16, _mm256_loadu_si256((const __m256i*)xb.bsums)));
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb.d), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
            acc_m[b] += dmin * xb.d * (float)msum;   // the min term: dmin * sum(m_j * activation sum of sub-block j)
        }
    }
    // weight = d*sc*q - dmin*m: scale term minus min term
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]) - acc_m[b];
}

template <int NB>
void dot_q2_K(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_q2_K* w = (const block_q2_K*)vw;
    const int nb = K / QK_K;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i evod = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    Q38_COLS_INIT;
    float acc_m[NB];
    for (int b = 0; b < NB; b++) acc_m[b] = 0.0f;
    for (int i = 0; i < nb; i++) {
        // 16 scale bytes: low nibble = scale of 16-weight group j, high nibble = min. Scales as words with the even
        // groups in the low lane and the odd groups in the high lane: sub-block sb (groups 2sb | 2sb+1) = word sb.
        const __m128i scb = _mm_loadu_si128((const __m128i*)w[i].scales);
        const __m256i sc16 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(_mm_and_si128(scb, m4), evod));
        const __m256i mn16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(scb, 4), m4));
        Q38_SUMI_INIT;
#pragma GCC unroll 2
        for (int n = 0; n < 2; n++) {
            const __m256i q2b = _mm256_loadu_si256((const __m256i*)(w[i].qs + 32 * n));
#pragma GCC unroll 4
            for (int j = 0; j < 4; j++) {
                const int sb = 4 * n + j;
                // 2-bit values of sub-block 4n+j: same bytes, shifted by 2j
                const __m256i q2 = _mm256_and_si256(_mm256_srli_epi16(q2b, 2 * j), m3);
                const __m256i sc = _mm256_shuffle_epi8(sc16, bcast_word_ctrl(sb));
                for (int b = 0; b < NB; b++)
                    // unsigned 0..3 weights need no sign trick
                    { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(q2, Q38_Q8(b, 32 * sb)), sc)); Q38_KEEP(sumi[b]); }
            }
        }
        const float dw = f16(w[i].d), dmin = f16(w[i].dmin);
        for (int b = 0; b < NB; b++) {
            const block_q8_K& xb = x[b * xs + i];
            const int msum = hsum8i(_mm256_madd_epi16(mn16, _mm256_loadu_si256((const __m256i*)xb.bsums)));   // exact
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb.d), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
            acc_m[b] += dmin * xb.d * (float)msum;
        }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]) - acc_m[b];
}

// Q6_K: weight = d * scale[16-group] * (q - 32), q = 4 low bits (ql) | 2 high bits (qh). maddubs on the unsigned
// q (0..63), the -32 offset subtracted per 16-group as 32 * scale * bsum (exact integers).
template <int NB>
void dot_q6_K(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_q6_K* w = (const block_q6_K*)vw;
    const int nb = K / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(3);
    Q38_COLS_INIT;
    float acc_m[NB];
    for (int b = 0; b < NB; b++) acc_m[b] = 0.0f;
    for (int i = 0; i < nb; i++) {
        const int8_t* sc = w[i].scales;
        Q38_SUMI_INIT;
#pragma GCC unroll 8
        for (int s = 0; s < 8; s++) {
            const int half = s >> 2, quarter = s & 3;
            const __m256i ql = _mm256_loadu_si256((const __m256i*)(w[i].ql + 64 * half + 32 * (quarter & 1)));
            const __m256i qh = _mm256_loadu_si256((const __m256i*)(w[i].qh + 32 * half));
            const __m256i lo = _mm256_and_si256(_mm256_srli_epi16(ql, 4 * (quarter >> 1)), m4);
            // 2 high bits to bits 4-5: q = 0..63
            const __m256i hi = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * quarter), m3), 4);
            const __m256i q = _mm256_or_si256(lo, hi);
            const int g = 8 * half + 2 * quarter;
            const __m256i scv = _mm256_set_m128i(_mm_set1_epi16(sc[g + 1]), _mm_set1_epi16(sc[g]));
            for (int b = 0; b < NB; b++)
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(q, Q38_Q8(b, 32 * s)), scv)); Q38_KEEP(sumi[b]); }
        }
        const float dw = f16(w[i].d);
        for (int b = 0; b < NB; b++) {
            const block_q8_K& xb = x[b * xs + i];
            int moff = 0;
            // the -32 offset: 32 * sum(scale_j * activation sum of group j)
            for (int j = 0; j < 16; j++) moff += sc[j] * xb.bsums[j];
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb.d), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
            acc_m[b] += dw * xb.d * (float)(32 * moff);
        }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]) - acc_m[b];
}

// R4: v = signed weight from the 16-entry table; maddubs(|v|, sign(x, v)) = maddubs(grid, sign(x, signs)) of the source
// kernels (no zero levels), same scales and float steps -> bit-identical.
template <int NB, int F>   // F: 0 = from IQ3_S, 1 = from IQ3_XXS (its d carries a factor 1/4)
void dot_r4(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_r4* w = (const block_r4*)vw;
    const int nb = K / QK_K;
    const __m256i table = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*)kR4Lut[F]));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    Q38_COLS_INIT;
    for (int i = 0; i < nb; i++) {
        const uint8_t* qs = w[i].qs;
        const __m256i sc16 = nibble_scales_odd(ld32(w[i].scales));
        Q38_SUMI_INIT;
#pragma GCC unroll 8
        for (int s = 0; s < 8; s++) {
            const __m128i q4 = _mm_loadu_si128((const __m128i*)(qs + 16 * s));
            const __m256i nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q4, 4), m4), _mm_and_si128(q4, m4));
            const __m256i v = _mm256_shuffle_epi8(table, nib);   // 4-bit code -> signed grid level (sign in bit 3)
            const __m256i av = _mm256_sign_epi8(v, v);
            const __m256i sc = _mm256_shuffle_epi8(sc16, bcast_word_ctrl(s));
            for (int b = 0; b < NB; b++)
                { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(av, _mm256_sign_epi8(Q38_Q8(b, 32 * s), v)), sc)); Q38_KEEP(sumi[b]); }
        }
        const float dw = f16(w[i].d);
        for (int b = 0; b < NB; b++) {
            // same float factors as the source format's kernel
            const float f = F == 0 ? dw * x[b * xs + i].d : dw * x[b * xs + i].d * 0.25f;
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(f), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
        }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]);
}

// R2: grid magnitudes from the 2-bit codes by pshufb, signs from the stored bits; the rest as the IQ2 kernels.
template <int NB>
void dot_r2(int K, const void* vw, const block_q8_K* x, size_t xs, float* y) {
    const block_r2* w = (const block_r2*)vw;
    const int nb = K / QK_K;
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*)kR2Lut));
    const __m256i m3 = _mm256_set1_epi8(3);
    Q38_COLS_INIT;
    for (int i = 0; i < nb; i++) {
        const __m256i sc16 = lohi_nibble_scales_odd(ld64(w[i].scales));
        Q38_SUMI_INIT;
#pragma GCC unroll 2
        for (int n = 0; n < 2; n++) {
            const __m256i q2b = _mm256_loadu_si256((const __m256i*)(w[i].mags + 32 * n));
#pragma GCC unroll 4
            for (int k = 0; k < 4; k++) {
                const int s = 4 * n + k;
                // 2-bit code -> grid byte 8 / 25 / 43, no gather
                const __m256i g = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(q2b, 2 * k), m3));
                const __m256i sv = signs_to_vec(ld32(w[i].signs + 4 * s));
                const __m256i sc = _mm256_shuffle_epi8(sc16, bcast_word_ctrl(s));
                for (int b = 0; b < NB; b++)
                    { sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(_mm256_maddubs_epi16(g, _mm256_sign_epi8(Q38_Q8(b, 32 * s), sv)), sc)); Q38_KEEP(sumi[b]); }
            }
        }
        const float dw = f16(w[i].d);
        for (int b = 0; b < NB; b++)
            { acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[b * xs + i].d * 0.125f), _mm256_cvtepi32_ps(sumi[b]), acc[b]); Q38_KEEP(acc[b]); }
    }
    for (int b = 0; b < NB; b++) y[b] = hsum8(acc[b]);
}

#undef Q38_COLS_INIT
#undef Q38_SUMI_INIT
#undef Q38_Q8
#undef Q38_KEEP

template <void (*F)(int, const void*, const block_q8_K*, size_t, float*)>
float one_col(int K, const void* w, const block_q8_K* x) {
    float y;
    F(K, w, x, 0, &y);
    return y;
}

template <int NB>
vec_dot_n_fn pick_n(QuantFormat t) {
    switch (t) {
        case QuantFormat::IQ3_S: return dot_iq3_s<NB>;
        case QuantFormat::IQ3_XXS: return dot_iq3_xxs<NB>;
        case QuantFormat::IQ2_XXS: return dot_iq2_xxs<NB>;
        case QuantFormat::IQ2_XS: return dot_iq2_xs<NB>;
        case QuantFormat::IQ2_S: return dot_iq2_s<NB>;
        case QuantFormat::IQ1_S: return dot_iq1_s<NB>;
        case QuantFormat::IQ1_M: return dot_iq1_m<NB>;
        case QuantFormat::IQ4_XS: return dot_iq4_xs<NB>;
        case QuantFormat::Q4_K: return dot_q4_K<NB>;
        case QuantFormat::Q2_K: return dot_q2_K<NB>;
        case QuantFormat::Q6_K: return dot_q6_K<NB>;
        default: return nullptr;
    }
}

} // namespace

vec_dot_fn get_vec_dot(QuantFormat t) {
    switch (t) {
        case QuantFormat::IQ3_S: return one_col<dot_iq3_s<1>>;
        case QuantFormat::IQ3_XXS: return one_col<dot_iq3_xxs<1>>;
        case QuantFormat::IQ2_XXS: return one_col<dot_iq2_xxs<1>>;
        case QuantFormat::IQ2_XS: return one_col<dot_iq2_xs<1>>;
        case QuantFormat::IQ2_S: return one_col<dot_iq2_s<1>>;
        case QuantFormat::IQ1_S: return one_col<dot_iq1_s<1>>;
        case QuantFormat::IQ1_M: return one_col<dot_iq1_m<1>>;
        case QuantFormat::IQ4_XS: return one_col<dot_iq4_xs<1>>;
        case QuantFormat::Q4_K: return one_col<dot_q4_K<1>>;
        case QuantFormat::Q2_K: return one_col<dot_q2_K<1>>;
        case QuantFormat::Q6_K: return one_col<dot_q6_K<1>>;
        default: return nullptr;
    }
}

vec_dot_n_fn get_vec_dot_n(QuantFormat t, int nb) {
    switch (nb) {
        case 1: return pick_n<1>(t);
        case 2: return pick_n<2>(t);
        case 3: return pick_n<3>(t);
        case 4: return pick_n<4>(t);
        case 5: return pick_n<5>(t);
        case 6: return pick_n<6>(t);
        case 7: return pick_n<7>(t);
        case 8: return pick_n<8>(t);
        default: return nullptr;
    }
}

namespace {
bool repack_r4(QuantFormat t, const void* src, int K, void* dst) {
    const int F = t == QuantFormat::IQ3_S ? 0 : 1;
    block_r4* o = (block_r4*)dst;
    for (int i = 0; i < K / QK_K; i++) {
        uint8_t g[QK_K], neg[QK_K], ls[8];
        if (F == 0) {
            const block_iq3_s& b = ((const block_iq3_s*)src)[i];
            o[i].d = b.d;
            for (int s = 0; s < 8; s++) {
                ls[s] = (b.scales[s / 2] >> (4 * (s & 1))) & 0xF;
                for (int j = 0; j < 8; j++) {   // as the kernel: index bit 8 = bit j of qh[s]
                    // the same 9-bit index the IQ3_S kernel builds
                    const int idx = b.qs[8 * s + j] | ((b.qh[s] << (8 - j)) & 256);
                    memcpy(g + 32 * s + 4 * j, iq3s_grid + idx, 4);
                }
                const uint32_t sg = ld32(b.signs + 4 * s);
                for (int k = 0; k < 32; k++) neg[32 * s + k] = (sg >> k) & 1;
            }
        } else {
            const block_iq3_xxs& b = ((const block_iq3_xxs*)src)[i];
            o[i].d = b.d;
            for (int s = 0; s < 8; s++) {
                const uint32_t aux = ld32(b.qs + QK_K / 4 + 4 * s);
                ls[s] = (uint8_t)(aux >> 28);
                for (int j = 0; j < 8; j++) memcpy(g + 32 * s + 4 * j, iq3xxs_grid + b.qs[8 * s + j], 4);
                for (int grp = 0; grp < 4; grp++) {   // 7 sign bits + parity per 8 weights
                    const int g7 = (aux >> (7 * grp)) & 127, bits = g7 | (__builtin_parity(g7) << 7);
                    for (int k = 0; k < 8; k++) neg[32 * s + 8 * grp + k] = (bits >> k) & 1;
                }
            }
        }
        for (int q = 0; q < 4; q++) o[i].scales[q] = (uint8_t)(ls[2 * q] | (ls[2 * q + 1] << 4));
        uint8_t code[QK_K];
        for (int k = 0; k < QK_K; k++) {
            int lvl = -1;
            // grid byte -> level index; IQ3 grids use only 8 distinct bytes
            for (int l = 0; l < 8; l++) if (kR4Lut[F][l] == g[k]) lvl = l;
            if (lvl < 0) return false;   // not representable: the caller keeps the original rows
            code[k] = (uint8_t)(lvl + 8 * neg[k]);
        }
        for (int s = 0; s < 8; s++)
            // weights j and j+16 share a byte (the kernel's nibble order)
            for (int j = 0; j < 16; j++) o[i].qs[16 * s + j] = (uint8_t)(code[32 * s + j] | (code[32 * s + j + 16] << 4));
    }
    return true;
}


// IQ2 rows -> R2 (see block_r2); false if a grid byte is not 8, 25 or 43.
bool repack_r2(QuantFormat t, const void* src, int K, void* dst) {
    block_r2* o = (block_r2*)dst;
    for (int i = 0; i < K / QK_K; i++) {
        uint8_t g[QK_K], neg[QK_K], sc[8];
        if (t == QuantFormat::IQ2_XXS) {
            const block_iq2_xxs& b = ((const block_iq2_xxs*)src)[i];
            o[i].d = b.d;
            const uint8_t* qs = (const uint8_t*)b.qs;
            for (int s = 0; s < 8; s++) {
                const uint32_t a0 = ld32(qs + 8 * s), a1 = ld32(qs + 8 * s + 4);
                for (int j = 0; j < 4; j++) memcpy(g + 32 * s + 8 * j, iq2xxs_grid + ((a0 >> (8 * j)) & 255), 8);
                for (int grp = 0; grp < 4; grp++) {   // 7 sign bits + parity per 8 weights
                    const int g7 = (a1 >> (7 * grp)) & 127, bits = g7 | (__builtin_parity(g7) << 7);
                    for (int k = 0; k < 8; k++) neg[32 * s + 8 * grp + k] = (bits >> k) & 1;
                }
                sc[s] = (uint8_t)((a1 >> 28) | ((a1 >> 28) << 4));
            }
        } else if (t == QuantFormat::IQ2_XS) {
            const block_iq2_xs& b = ((const block_iq2_xs*)src)[i];
            o[i].d = b.d;
            for (int s = 0; s < 8; s++) {
                for (int j = 0; j < 4; j++) {
                    const uint16_t q = b.qs[4 * s + j];
                    memcpy(g + 32 * s + 8 * j, iq2xs_grid + (q & 511), 8);
                    const int g7 = q >> 9, bits = g7 | (__builtin_parity(g7) << 7);
                    for (int k = 0; k < 8; k++) neg[32 * s + 8 * j + k] = (bits >> k) & 1;
                }
                sc[s] = b.scales[s];
            }
        } else if (t == QuantFormat::IQ2_S) {
            const block_iq2_s& b = ((const block_iq2_s*)src)[i];
            o[i].d = b.d;
            for (int s = 0; s < 8; s++) {
                for (int j = 0; j < 4; j++) {   // as the kernel: index bits 8-9 from qh[s] bits 2j, 2j+1
                    const int idx = b.qs[4 * s + j] | ((b.qh[s] << (8 - 2 * j)) & 0x300);
                    memcpy(g + 32 * s + 8 * j, iq2s_grid + idx, 8);
                }
                const uint32_t sg = ld32(b.qs + QK_K / 8 + 4 * s);
                for (int k = 0; k < 32; k++) neg[32 * s + k] = (sg >> k) & 1;
                sc[s] = b.scales[s];
            }
        } else {
            return false;
        }
        memcpy(o[i].scales, sc, 8);
        memset(o[i].mags, 0, sizeof o[i].mags);
        memset(o[i].signs, 0, sizeof o[i].signs);
        for (int s = 0; s < 8; s++)
            for (int w = 0; w < 32; w++) {
                // IQ2 grids use only the bytes 8, 25 and 43
                const int v = g[32 * s + w], code = v == 8 ? 0 : v == 25 ? 1 : v == 43 ? 2 : -1;
                if (code < 0) return false;
                // Q2_K-style layout: 4 sub-blocks per byte
                o[i].mags[32 * (s / 4) + w] |= (uint8_t)(code << (2 * (s % 4)));
                o[i].signs[4 * s + w / 8] |= (uint8_t)(neg[32 * s + w] << (w % 8));
            }
    }
    return true;
}

bool is_r4(QuantFormat t) { return t == QuantFormat::IQ3_S || t == QuantFormat::IQ3_XXS; }
bool is_r2(QuantFormat t) { return t == QuantFormat::IQ2_XXS || t == QuantFormat::IQ2_XS || t == QuantFormat::IQ2_S; }
} // namespace

bool re_supported(QuantFormat t) { return is_r4(t) || is_r2(t); }

size_t re_row_bytes(QuantFormat t, int K) {
    return (size_t)(K / QK_K) * (is_r4(t) ? sizeof(block_r4) : is_r2(t) ? sizeof(block_r2) : 0);
}

bool re_repack_row(QuantFormat t, const void* src, int K, void* dst) {
    return is_r4(t) ? repack_r4(t, src, K, dst) : is_r2(t) ? repack_r2(t, src, K, dst) : false;
}

vec_dot_fn get_vec_dot_re(QuantFormat t) {
    if (t == QuantFormat::IQ3_S) return one_col<dot_r4<1, 0>>;
    if (t == QuantFormat::IQ3_XXS) return one_col<dot_r4<1, 1>>;
    if (is_r2(t)) return one_col<dot_r2<1>>;
    return nullptr;
}

vec_dot_n_fn get_vec_dot_re_n(QuantFormat t, int nb) {
    if (!re_supported(t) || nb < 1 || nb > 8) return nullptr;
    static const vec_dot_n_fn r4[2][8] = {{dot_r4<1, 0>, dot_r4<2, 0>, dot_r4<3, 0>, dot_r4<4, 0>, dot_r4<5, 0>, dot_r4<6, 0>, dot_r4<7, 0>, dot_r4<8, 0>},
                                          {dot_r4<1, 1>, dot_r4<2, 1>, dot_r4<3, 1>, dot_r4<4, 1>, dot_r4<5, 1>, dot_r4<6, 1>, dot_r4<7, 1>, dot_r4<8, 1>}};
    static const vec_dot_n_fn r2[8] = {dot_r2<1>, dot_r2<2>, dot_r2<3>, dot_r2<4>, dot_r2<5>, dot_r2<6>, dot_r2<7>, dot_r2<8>};
    if (is_r2(t)) return r2[nb - 1];
    return r4[t == QuantFormat::IQ3_S ? 0 : 1][nb - 1];
}

void quantize_q8_K(const float* x, block_q8_K* y, int K) {
    const int nb = K / QK_K;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (int i = 0; i < nb; i++, x += QK_K) {
        __m256 vmax = _mm256_setzero_ps();
        // max |x| of the 256 values
        for (int j = 0; j < QK_K; j += 8) vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(x + j)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4);
        if (amax == 0.0f) {
            memset(&y[i], 0, sizeof(block_q8_K));
            continue;
        }
        float max = 0.0f;   // first element with |x| == amax (signed), as the reference does
        for (int j = 0; j < QK_K; j++) {
            if (fabsf(x[j]) == amax) { max = x[j]; break; }
        }
        const float iscale = -127.f / max;   // the extreme value maps to -127 (ggml's quantize_row_q8_K)
        const __m256 vs = _mm256_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 32) {
            // round half to even, as nearest_int
            __m256i a = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i b = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j + 8)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i c = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j + 16)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i d = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j + 24)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            const __m256i lim = _mm256_set1_epi32(127);
            a = _mm256_min_epi32(a, lim); b = _mm256_min_epi32(b, lim); c = _mm256_min_epi32(c, lim); d = _mm256_min_epi32(d, lim);
            // pack 32-bit -> 16-bit -> 8-bit; packs work per 128-bit lane, fix order with a permute
            const __m256i ab = _mm256_packs_epi32(a, b), cd = _mm256_packs_epi32(c, d);
            __m256i q = _mm256_packs_epi16(ab, cd);
            q = _mm256_permutevar8x32_epi32(q, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256((__m256i*)(y[i].qs + j), q);
        }
        for (int j = 0; j < QK_K / 16; j++) {
            const __m128i v = _mm_loadu_si128((const __m128i*)(y[i].qs + 16 * j));
            const __m256i w16 = _mm256_cvtepi8_epi16(v);
            // sums of 16 quants: used by the min / offset terms of K-quant kernels
            y[i].bsums[j] = (int16_t)hsum8i(_mm256_madd_epi16(w16, _mm256_set1_epi16(1)));
        }
        y[i].d = 1.0f / iscale;
    }
}

} // namespace cpu
} // namespace q38rt
