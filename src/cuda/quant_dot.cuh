#pragma once
/// @file quant_dot.cuh
/// Device-side dot products between one 32-weight sub-block of a ggml weight
/// super-block and 32 int8-quantized activations (per-32 scale d8 and float sum s8).
///
/// Bit layouts follow ggml (ggml-common.h); the integer kernels follow ggml-cuda's
/// vecdotq.cuh (MIT, The ggml authors) but apply the per-block scales in float, so the
/// result equals  sum_j w_j * (d8 * q8_j)  up to float rounding, where w_j is exactly the
/// ggml dequantized weight. (ggml-cuda truncates some of these scales with integer divides.)
///
/// Super-blocks are only 2-byte aligned in the GGUF layout (e.g. 110-byte IQ3_S), so
/// multi-byte loads go through 16-bit halves.

#include <cstdint>
#include <cuda_fp16.h>
#include "quant/quant_formats.h"
#include "cuda/iq_tables_dev.cuh"

namespace q38rt {
namespace cuda {

__device__ __forceinline__ int ld_b2(const void* p, int i32) {   // 32-bit load from 2-byte aligned data
    const uint16_t* x16 = (const uint16_t*)p;
    return (int)((uint32_t)x16[2 * i32] | ((uint32_t)x16[2 * i32 + 1] << 16));
}
__device__ __forceinline__ int ld_b4(const void* p, int i32) { return ((const int*)p)[i32]; }

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// Conditionally negate the 4 int8 grid values packed in g: sign bit k of s4 applies to byte k.
// Spread the 4 bits to the byte LSBs with one multiply (shifted copies are bit-disjoint, no carries),
// then negate as (g ^ 0xFF) + 1 per selected byte. Exact because i-quant grid values are never 0,
// so (0xFF ^ g) + 1 = 256 - g never carries into the next byte.
__device__ __forceinline__ int neg4(int g, uint32_t s4) {
    const uint32_t m = ((s4 & 0xFu) * 0x00204081u) & 0x01010101u;
    return (int)(((uint32_t)g ^ (m * 0xFFu)) + m);
}
// 7-bit ksigns index -> 8 sign bits (8th bit = parity), as ksigns_iq2xs[] in ggml.
__device__ __forceinline__ uint32_t ksigns8(uint32_t v) {
    v &= 0x7Fu;
    return v | ((uint32_t)(__popc(v) & 1) << 7);
}

// 16-entry int8 table lookup for 8 4-bit indices (ggml-cuda get_int_from_table_16).
__device__ __forceinline__ int2 table16(int q4, const int8_t* table) {
    const uint32_t* t32 = (const uint32_t*)table;
    uint32_t tmp[2];
    // bit 3 of each nibble selects the upper 8 table entries
    const uint32_t sel = 0x32103210u | (((uint32_t)q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        // byte-select from entries 0-7 with the nibbles' low 3 bits
        const uint32_t lo = __byte_perm(t32[0], t32[1], (uint32_t)q4 >> sh);
        const uint32_t hi = __byte_perm(t32[2], t32[3], (uint32_t)q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2((int)__byte_perm(tmp[0], tmp[1], 0x6420), (int)__byte_perm(tmp[0], tmp[1], 0x7531));
}

/// Lookup table a format needs during the dot product (staged in shared memory by the GEMV kernels).
template <QuantFormat T> struct QTable { static constexpr int BYTES = 0; __device__ static const void* src() { return nullptr; } };
template <> struct QTable<QuantFormat::IQ2_XXS> { static constexpr int BYTES = 256 * 8; __device__ static const void* src() { return iq2xxs_grid; } };
template <> struct QTable<QuantFormat::IQ2_XS>  { static constexpr int BYTES = 512 * 8; __device__ static const void* src() { return iq2xs_grid; } };
template <> struct QTable<QuantFormat::IQ2_S>   { static constexpr int BYTES = 1024 * 8; __device__ static const void* src() { return iq2s_grid; } };
template <> struct QTable<QuantFormat::IQ3_XXS> { static constexpr int BYTES = 256 * 4; __device__ static const void* src() { return iq3xxs_grid; } };
template <> struct QTable<QuantFormat::IQ3_S>   { static constexpr int BYTES = 512 * 4; __device__ static const void* src() { return iq3s_grid; } };
// IQ1_S and IQ4_XS measured faster with the table read through L1 (global) than staged in shared memory.
template <> struct QTable<QuantFormat::IQ1_S>   { static constexpr int BYTES = 0; __device__ static const void* src() { return nullptr; } };
template <> struct QTable<QuantFormat::IQ1_M>   { static constexpr int BYTES = 2048 * 4; __device__ static const void* src() { return iq1s_grid_gpu; } };
template <> struct QTable<QuantFormat::IQ4_XS>  { static constexpr int BYTES = 0; __device__ static const void* src() { return nullptr; } };

/// Per-type traits. dotn<NB>(blk, sub, q8, d8, s8, tab, acc): tab -> the format's QTable (shared memory);
/// blk -> super-block, sub in [0,8) selects weights [32*sub, 32*sub+32); for each of NB activation columns b:
/// q8[b] = 8 ints (32 int8 activations, in registers), d8[b] = activation scale, s8[b] = sum of the
/// (unquantized) activations of the sub-block (only formats with offsets use it); acc[b] += dot.
/// The weights are decoded once; the per-column integer and float operations are identical for every NB and
/// use explicitly rounded float intrinsics (no FMA contraction that could differ between unrolled columns),
/// so each column's result is bit-identical to the NB = 1 result.
template <QuantFormat T> struct QDot;

template <> struct QDot<QuantFormat::IQ2_XXS> {
    static constexpr int BYTES = 66;
    /// Prefill dequantization: the weights of sub-block `sub` are a[j/8] * q_j + o[j/8], q_j the 32 int8 values in w
    /// (natural order; the same decoding as dotn). a * q and o are exact in float, so one rounding to FP16 gives the
    /// correctly rounded ggml weight.
    __device__ __forceinline__ static void decode(const uint8_t* blk, int sub, const void* tab, int* w, float* a, float* o) {
        const block_iq2_xxs* b = (const block_iq2_xxs*)blk;
        const int q2 = ld_b2(b->qs, 2 * sub);   // 4 grid indices of 8 bits, one per group of 8 weights
        const uint8_t* idx = (const uint8_t*)&q2;
        // 4 x 7-bit sign indices (bits 0-27) and the 4-bit scale (bits 28-31)
        const uint32_t aux = (uint32_t)ld_b2(b->qs, 2 * sub + 1);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*)tab)[idx[l]];   // 8 grid magnitudes (bytes) of group l
            const uint32_t sg = ksigns8(aux >> (7 * l));   // 7 stored sign bits + even parity = 8 signs
            w[2 * l + 0] = neg4((int)g.x, sg);
            w[2 * l + 1] = neg4((int)g.y, sg >> 4);
        }
        const float db = h2f(b->d) * (0.5f + (float)(aux >> 28)) * 0.25f;   // scale (0.5 + s) * d / 4, as in ggml
        for (int g = 0; g < 4; ++g) { a[g] = db; o[g] = 0.0f; }
    }
    __device__ __forceinline__ static float dot(const uint8_t* blk, int sub, const int* q8, float d8, float, const void* tab) {
        const block_iq2_xxs* b = (const block_iq2_xxs*)blk;
        const int q2 = ld_b2(b->qs, 2 * sub);
        const uint8_t* idx = (const uint8_t*)&q2;
        const uint32_t aux = (uint32_t)ld_b2(b->qs, 2 * sub + 1);
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*)tab)[idx[l]];
            const uint32_t sg = ksigns8(aux >> (7 * l));
            sumi = __dp4a(neg4((int)g.x, sg), q8[2 * l + 0], sumi);
            sumi = __dp4a(neg4((int)g.y, sg >> 4), q8[2 * l + 1], sumi);
        }
        const float db = h2f(b->d) * (0.5f + (float)(aux >> 28)) * 0.25f;
        return db * d8 * (float)sumi;
    }
    template <int NB>
    __device__ __forceinline__ static void dotn(const uint8_t* blk, int sub, const int (*q8)[8], const float* d8, const float*,
                                                const void* tab, float* acc) {
        const block_iq2_xxs* b = (const block_iq2_xxs*)blk;
        const int q2 = ld_b2(b->qs, 2 * sub);
        const uint8_t* idx = (const uint8_t*)&q2;
        const uint32_t aux = (uint32_t)ld_b2(b->qs, 2 * sub + 1);
        int w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*)tab)[idx[l]];
            const uint32_t sg = ksigns8(aux >> (7 * l));
            w[2 * l + 0] = neg4((int)g.x, sg);
            w[2 * l + 1] = neg4((int)g.y, sg >> 4);
        }
        const float db = h2f(b->d) * (0.5f + (float)(aux >> 28)) * 0.25f;
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            int sumi = 0;
#pragma unroll
            // dp4a: 4 int8 products per instruction, 32 per sub-block
            for (int j = 0; j < 8; ++j) sumi = __dp4a(w[j], q8[c][j], sumi);
            // explicitly rounded: identical for any column count
            acc[c] = __fadd_rn(acc[c], __fmul_rn(__fmul_rn(db, d8[c]), (float)sumi));
        }
    }
};

template <> struct QDot<QuantFormat::IQ2_XS> {
    static constexpr int BYTES = 74;
    __device__ __forceinline__ static void decode(const uint8_t* blk, int sub, const void* tab, int* w, float* a, float* o) {
        const block_iq2_xs* b = (const block_iq2_xs*)blk;
        const int2 qp = make_int2(ld_b2(b->qs, 2 * sub), ld_b2(b->qs, 2 * sub + 1));
        const uint16_t* q2 = (const uint16_t*)&qp;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*)tab)[q2[l] & 0x1FF];   // 9-bit grid index; the top 7 bits index the signs
            const uint32_t sg = ksigns8(q2[l] >> 9);
            w[2 * l + 0] = neg4((int)g.x, sg);
            w[2 * l + 1] = neg4((int)g.y, sg >> 4);
        }
        const uint8_t sc = b->scales[sub];
        const float d = h2f(b->d) * 0.25f;
        a[0] = a[1] = d * (0.5f + (float)(sc & 0xF));   // two 4-bit scales per sub-block: weights 0-15 and 16-31
        a[2] = a[3] = d * (0.5f + (float)(sc >> 4));
        for (int g = 0; g < 4; ++g) o[g] = 0.0f;
    }
    __device__ __forceinline__ static float dot(const uint8_t* blk, int sub, const int* q8, float d8, float, const void* tab) {
        const block_iq2_xs* b = (const block_iq2_xs*)blk;
        const int2 qp = make_int2(ld_b2(b->qs, 2 * sub), ld_b2(b->qs, 2 * sub + 1));
        const uint16_t* q2 = (const uint16_t*)&qp;
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*)tab)[q2[l] & 0x1FF];
            const uint32_t sg = ksigns8(q2[l] >> 9);
            int acc = l < 2 ? s0 : s1;
            acc = __dp4a(neg4((int)g.x, sg), q8[2 * l + 0], acc);
            acc = __dp4a(neg4((int)g.y, sg >> 4), q8[2 * l + 1], acc);
            if (l < 2) s0 = acc; else s1 = acc;
        }
        const uint8_t sc = b->scales[sub];
        const float d = h2f(b->d) * 0.25f;
        return d * d8 * ((0.5f + (float)(sc & 0xF)) * (float)s0 + (0.5f + (float)(sc >> 4)) * (float)s1);
    }
    template <int NB>
    __device__ __forceinline__ static void dotn(const uint8_t* blk, int sub, const int (*q8)[8], const float* d8, const float*,
                                                const void* tab, float* acc) {
        const block_iq2_xs* b = (const block_iq2_xs*)blk;
        const int2 qp = make_int2(ld_b2(b->qs, 2 * sub), ld_b2(b->qs, 2 * sub + 1));
        const uint16_t* q2 = (const uint16_t*)&qp;
        int w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*)tab)[q2[l] & 0x1FF];
            const uint32_t sg = ksigns8(q2[l] >> 9);
            w[2 * l + 0] = neg4((int)g.x, sg);
            w[2 * l + 1] = neg4((int)g.y, sg >> 4);
        }
        const uint8_t sc = b->scales[sub];
        const float d = h2f(b->d) * 0.25f;
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) s0 = __dp4a(w[j], q8[c][j], s0);
#pragma unroll
            for (int j = 4; j < 8; ++j) s1 = __dp4a(w[j], q8[c][j], s1);
            acc[c] = __fadd_rn(acc[c], __fmul_rn(__fmul_rn(d, d8[c]), __fadd_rn(__fmul_rn(0.5f + (float)(sc & 0xF), (float)s0),
                                                                            __fmul_rn(0.5f + (float)(sc >> 4), (float)s1))));
        }
    }
};

template <> struct QDot<QuantFormat::IQ2_S> {
    static constexpr int BYTES = 82;
    __device__ __forceinline__ static void decode(const uint8_t* blk, int sub, const void* tab, int* w, float* a, float* o) {
        const block_iq2_s* b = (const block_iq2_s*)blk;
        const int qsp = ld_b2(b->qs, sub);
        const uint8_t* qs = (const uint8_t*)&qsp;
        const int qh = b->qh[sub];   // 2 high bits of each of the 4 grid indices (10-bit indices)
        const int sgp = ld_b2(b->qs, QK_K / 32 + sub);   // explicit sign bytes follow the index bytes
        const uint8_t* sg = (const uint8_t*)&sgp;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            // index = low byte | its 2 high bits from qh
            const int* g = (const int*)((const uint64_t*)tab + (qs[l] | ((qh << (8 - 2 * l)) & 0x300)));
            w[2 * l + 0] = neg4(g[0], sg[l]);
            w[2 * l + 1] = neg4(g[1], (uint32_t)sg[l] >> 4);
        }
        const uint8_t sc = b->scales[sub];
        const float d = h2f(b->d) * 0.25f;
        a[0] = a[1] = d * (0.5f + (float)(sc & 0xF));
        a[2] = a[3] = d * (0.5f + (float)(sc >> 4));
        for (int g = 0; g < 4; ++g) o[g] = 0.0f;
    }
    __device__ __forceinline__ static float dot(const uint8_t* blk, int sub, const int* q8, float d8, float, const void* tab) {
        const block_iq2_s* b = (const block_iq2_s*)blk;
        const int qsp = ld_b2(b->qs, sub);
        const uint8_t* qs = (const uint8_t*)&qsp;
        const int qh = b->qh[sub];
        const int sgp = ld_b2(b->qs, QK_K / 32 + sub);
        const uint8_t* sg = (const uint8_t*)&sgp;
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int* g = (const int*)((const uint64_t*)tab + (qs[l] | ((qh << (8 - 2 * l)) & 0x300)));
            const int gl = neg4(g[0], sg[l]);
            const int gh = neg4(g[1], (uint32_t)sg[l] >> 4);
            int acc = l < 2 ? s0 : s1;
            acc = __dp4a(gl, q8[2 * l + 0], acc);
            acc = __dp4a(gh, q8[2 * l + 1], acc);
            if (l < 2) s0 = acc; else s1 = acc;
        }
        const uint8_t sc = b->scales[sub];
        const float d = h2f(b->d) * 0.25f;
        return d * d8 * ((0.5f + (float)(sc & 0xF)) * (float)s0 + (0.5f + (float)(sc >> 4)) * (float)s1);
    }
    template <int NB>
    __device__ __forceinline__ static void dotn(const uint8_t* blk, int sub, const int (*q8)[8], const float* d8, const float*,
                                                const void* tab, float* acc) {
        const block_iq2_s* b = (const block_iq2_s*)blk;
        const int qsp = ld_b2(b->qs, sub);
        const uint8_t* qs = (const uint8_t*)&qsp;
        const int qh = b->qh[sub];
        const int sgp = ld_b2(b->qs, QK_K / 32 + sub);
        const uint8_t* sg = (const uint8_t*)&sgp;
        int w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int* g = (const int*)((const uint64_t*)tab + (qs[l] | ((qh << (8 - 2 * l)) & 0x300)));
            w[2 * l + 0] = neg4(g[0], sg[l]);
            w[2 * l + 1] = neg4(g[1], (uint32_t)sg[l] >> 4);
        }
        const uint8_t sc = b->scales[sub];
        const float d = h2f(b->d) * 0.25f;
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) s0 = __dp4a(w[j], q8[c][j], s0);
#pragma unroll
            for (int j = 4; j < 8; ++j) s1 = __dp4a(w[j], q8[c][j], s1);
            acc[c] = __fadd_rn(acc[c], __fmul_rn(__fmul_rn(d, d8[c]), __fadd_rn(__fmul_rn(0.5f + (float)(sc & 0xF), (float)s0),
                                                                            __fmul_rn(0.5f + (float)(sc >> 4), (float)s1))));
        }
    }
};

template <> struct QDot<QuantFormat::IQ3_XXS> {
    static constexpr int BYTES = 98;
    __device__ __forceinline__ static void decode(const uint8_t* blk, int sub, const void* tab, int* w, float* a, float* o) {
        const block_iq3_xxs* b = (const block_iq3_xxs*)blk;
        const int2 qp = make_int2(ld_b2(b->qs, 2 * sub), ld_b2(b->qs, 2 * sub + 1));
        const uint8_t* q3 = (const uint8_t*)&qp;
        // signs and scale after the 64 index bytes, packed as in IQ2_XXS
        const uint32_t aux = (uint32_t)ld_b2(b->qs, QK_K / 16 + sub);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int g0 = (int)((const uint32_t*)tab)[q3[2 * l + 0]];   // 4 magnitudes per 8-bit index (256-entry grid)
            const int g1 = (int)((const uint32_t*)tab)[q3[2 * l + 1]];
            const uint32_t sg = ksigns8(aux >> (7 * l));
            w[2 * l + 0] = neg4(g0, sg);
            w[2 * l + 1] = neg4(g1, sg >> 4);
        }
        const float db = h2f(b->d) * (0.5f + (float)(aux >> 28)) * 0.5f;   // scale (0.5 + s) * d / 2
        for (int g = 0; g < 4; ++g) { a[g] = db; o[g] = 0.0f; }
    }
    __device__ __forceinline__ static float dot(const uint8_t* blk, int sub, const int* q8, float d8, float, const void* tab) {
        const block_iq3_xxs* b = (const block_iq3_xxs*)blk;
        const int2 qp = make_int2(ld_b2(b->qs, 2 * sub), ld_b2(b->qs, 2 * sub + 1));
        const uint8_t* q3 = (const uint8_t*)&qp;
        const uint32_t aux = (uint32_t)ld_b2(b->qs, QK_K / 16 + sub);
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int g0 = (int)((const uint32_t*)tab)[q3[2 * l + 0]];
            const int g1 = (int)((const uint32_t*)tab)[q3[2 * l + 1]];
            const uint32_t sg = ksigns8(aux >> (7 * l));
            sumi = __dp4a(neg4(g0, sg), q8[2 * l + 0], sumi);
            sumi = __dp4a(neg4(g1, sg >> 4), q8[2 * l + 1], sumi);
        }
        const float db = h2f(b->d) * (0.5f + (float)(aux >> 28)) * 0.5f;
        return db * d8 * (float)sumi;
    }
    template <int NB>
    __device__ __forceinline__ static void dotn(const uint8_t* blk, int sub, const int (*q8)[8], const float* d8, const float*,
                                                const void* tab, float* acc) {
        const block_iq3_xxs* b = (const block_iq3_xxs*)blk;
        const int2 qp = make_int2(ld_b2(b->qs, 2 * sub), ld_b2(b->qs, 2 * sub + 1));
        const uint8_t* q3 = (const uint8_t*)&qp;
        const uint32_t aux = (uint32_t)ld_b2(b->qs, QK_K / 16 + sub);
        int w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int g0 = (int)((const uint32_t*)tab)[q3[2 * l + 0]];
            const int g1 = (int)((const uint32_t*)tab)[q3[2 * l + 1]];
            const uint32_t sg = ksigns8(aux >> (7 * l));
            w[2 * l + 0] = neg4(g0, sg);
            w[2 * l + 1] = neg4(g1, sg >> 4);
        }
        const float db = h2f(b->d) * (0.5f + (float)(aux >> 28)) * 0.5f;
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            int sumi = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) sumi = __dp4a(w[j], q8[c][j], sumi);
            acc[c] = __fadd_rn(acc[c], __fmul_rn(__fmul_rn(db, d8[c]), (float)sumi));
        }
    }
};

template <> struct QDot<QuantFormat::IQ3_S> {
    static constexpr int BYTES = 110;
    __device__ __forceinline__ static void decode(const uint8_t* blk, int sub, const void* tab, int* w, float* a, float* o) {
        const block_iq3_s* b = (const block_iq3_s*)blk;
        const int2 qp = make_int2(ld_b2(b->qs, 2 * sub), ld_b2(b->qs, 2 * sub + 1));
        const uint8_t* qs = (const uint8_t*)&qp;
        const int qh = b->qh[sub];   // 9th bit of each of the 8 grid indices
        const int sgp = ld_b2(b->signs, sub);
        const uint8_t* sg = (const uint8_t*)&sgp;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            // 9-bit index: byte | its bit from qh
            const int g0 = (int)((const uint32_t*)tab)[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 0x100)];
            const int g1 = (int)((const uint32_t*)tab)[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 0x100)];
            w[2 * l + 0] = neg4(g0, sg[l]);
            w[2 * l + 1] = neg4(g1, (uint32_t)sg[l] >> 4);
        }
        const int ls = (b->scales[sub / 2] >> (4 * (sub & 1))) & 0xF;   // 4-bit scale, two sub-blocks per byte
        const float dl = h2f(b->d) * (float)(1 + 2 * ls);   // odd multipliers 1..31
        for (int g = 0; g < 4; ++g) { a[g] = dl; o[g] = 0.0f; }
    }
    __device__ __forceinline__ static float dot(const uint8_t* blk, int sub, const int* q8, float d8, float, const void* tab) {
        const block_iq3_s* b = (const block_iq3_s*)blk;
        const int2 qp = make_int2(ld_b2(b->qs, 2 * sub), ld_b2(b->qs, 2 * sub + 1));
        const uint8_t* qs = (const uint8_t*)&qp;
        const int qh = b->qh[sub];
        const int sgp = ld_b2(b->signs, sub);
        const uint8_t* sg = (const uint8_t*)&sgp;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int g0 = (int)((const uint32_t*)tab)[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 0x100)];
            const int g1 = (int)((const uint32_t*)tab)[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 0x100)];
            sumi = __dp4a(neg4(g0, sg[l]), q8[2 * l + 0], sumi);
            sumi = __dp4a(neg4(g1, (uint32_t)sg[l] >> 4), q8[2 * l + 1], sumi);
        }
        const int ls = (b->scales[sub / 2] >> (4 * (sub & 1))) & 0xF;
        return h2f(b->d) * (float)(1 + 2 * ls) * d8 * (float)sumi;
    }
    template <int NB>
    __device__ __forceinline__ static void dotn(const uint8_t* blk, int sub, const int (*q8)[8], const float* d8, const float*,
                                                const void* tab, float* acc) {
        const block_iq3_s* b = (const block_iq3_s*)blk;
        const int2 qp = make_int2(ld_b2(b->qs, 2 * sub), ld_b2(b->qs, 2 * sub + 1));
        const uint8_t* qs = (const uint8_t*)&qp;
        const int qh = b->qh[sub];
        const int sgp = ld_b2(b->signs, sub);
        const uint8_t* sg = (const uint8_t*)&sgp;
        int w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int g0 = (int)((const uint32_t*)tab)[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 0x100)];
            const int g1 = (int)((const uint32_t*)tab)[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 0x100)];
            w[2 * l + 0] = neg4(g0, sg[l]);
            w[2 * l + 1] = neg4(g1, (uint32_t)sg[l] >> 4);
        }
        const int ls = (b->scales[sub / 2] >> (4 * (sub & 1))) & 0xF;
        const float dl = h2f(b->d) * (float)(1 + 2 * ls);
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            int sumi = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) sumi = __dp4a(w[j], q8[c][j], sumi);
            acc[c] = __fadd_rn(acc[c], __fmul_rn(__fmul_rn(dl, d8[c]), (float)sumi));
        }
    }
};

template <> struct QDot<QuantFormat::IQ1_S> {
    static constexpr int BYTES = 50;
    // q_j = grid + 1 in {0, 1, 2}: weight = dl * (grid + delta) = dl * q_j + dl * (delta - 1)
    __device__ __forceinline__ static void decode(const uint8_t* blk, int sub, const void* tab, int* w, float* a, float* o) {
        const block_iq1_s* b = (const block_iq1_s*)blk;
        const int qsp = ld_b2(b->qs, sub);
        const uint8_t* qs = (const uint8_t*)&qsp;
        const int qh = b->qh[sub];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            // 11-bit index (byte | 3 bits of qh): 8 values in {0,1,2} as nibbles
            const int g = (int)iq1s_grid_gpu[qs[l] | (((qh >> (3 * l)) & 7) << 8)];
            w[2 * l + 0] = (g >> 0) & 0x0F0F0F0F;   // even nibbles: 4 values, one per int8 lane
            w[2 * l + 1] = (g >> 4) & 0x0F0F0F0F;
        }
        const float dl = h2f(b->d) * (float)(2 * ((qh >> 12) & 7) + 1);   // 3-bit odd scale in qh bits 12-14
        const float delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;   // bit 15: sign of the sub-block's shift
        for (int g = 0; g < 4; ++g) { a[g] = dl; o[g] = dl * (delta - 1.0f); }
        (void)tab;
    }
    __device__ __forceinline__ static float dot(const uint8_t* blk, int sub, const int* q8, float d8, float s8, const void* tab) {
        const block_iq1_s* b = (const block_iq1_s*)blk;
        const int qsp = ld_b2(b->qs, sub);
        const uint8_t* qs = (const uint8_t*)&qsp;
        const int qh = b->qh[sub];
        int sumi = 0;   // sum of (grid + 1) * q8, grid in {-1, 0, 1}
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int g = (int)iq1s_grid_gpu[qs[l] | (((qh >> (3 * l)) & 7) << 8)];
            sumi = __dp4a((g >> 0) & 0x0F0F0F0F, q8[2 * l + 0], sumi);
            sumi = __dp4a((g >> 4) & 0x0F0F0F0F, q8[2 * l + 1], sumi);
        }
        const float dl = h2f(b->d) * (float)(2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
        // sum_j dl*(g_j + delta)*x_j = dl*(d8*sumi + (delta - 1)*sum_j x_j)
        return dl * (d8 * (float)sumi + (delta - 1.0f) * s8);
    }
    template <int NB>
    __device__ __forceinline__ static void dotn(const uint8_t* blk, int sub, const int (*q8)[8], const float* d8, const float* s8,
                                                const void*, float* acc) {
        const block_iq1_s* b = (const block_iq1_s*)blk;
        const int qsp = ld_b2(b->qs, sub);
        const uint8_t* qs = (const uint8_t*)&qsp;
        const int qh = b->qh[sub];
        int w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int g = (int)iq1s_grid_gpu[qs[l] | (((qh >> (3 * l)) & 7) << 8)];
            w[2 * l + 0] = (g >> 0) & 0x0F0F0F0F;
            w[2 * l + 1] = (g >> 4) & 0x0F0F0F0F;
        }
        const float dl = h2f(b->d) * (float)(2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            int sumi = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) sumi = __dp4a(w[j], q8[c][j], sumi);
            acc[c] = __fadd_rn(acc[c], __fmul_rn(dl, __fadd_rn(__fmul_rn(d8[c], (float)sumi), __fmul_rn(delta - 1.0f, s8[c]))));
        }
    }
};

template <> struct QDot<QuantFormat::IQ1_M> {
    static constexpr int BYTES = 56;
    __device__ __forceinline__ static void decode(const uint8_t* blk, int sub, const void* tab, int* w, float* a, float* o) {
        const block_iq1_m* b = (const block_iq1_m*)blk;
        const int qsp = ld_b4(b->qs, sub);
        const uint8_t* qs = (const uint8_t*)&qsp;
        float dm1[4];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int qhl = b->qh[2 * sub + l / 2] >> (4 * (l % 2));
            const int g = (int)((const uint32_t*)tab)[qs[l] | ((qhl & 7) << 8)];
            w[2 * l + 0] = (g >> 0) & 0x0F0F0F0F;
            w[2 * l + 1] = (g >> 4) & 0x0F0F0F0F;
            // per group of 8: shift sign in bit 3 of its qh nibble (-1: q = grid + 1)
            dm1[l] = ((qhl & 0x08) ? -IQ1M_DELTA : IQ1M_DELTA) - 1.0f;
        }
        uint16_t sc[4];
        memcpy(sc, b->scales, 8);
        // the FP16 super-scale is scattered over the top nibbles of the 4 scale words
        const uint16_t su = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const float d = h2f(su);
        const int tmp = sc[sub / 2] >> (6 * (sub % 2));   // two 3-bit scales per sub-block (one per 16 weights)
        const float sc0 = (float)(2 * ((tmp >> 0) & 7) + 1);
        const float sc1 = (float)(2 * ((tmp >> 3) & 7) + 1);
        for (int g = 0; g < 4; ++g) { a[g] = d * (g < 2 ? sc0 : sc1); o[g] = a[g] * dm1[g]; }
    }
    __device__ __forceinline__ static float dot(const uint8_t* blk, int sub, const int* q8, float d8, float, const void* tab) {
        const block_iq1_m* b = (const block_iq1_m*)blk;
        const int qsp = ld_b4(b->qs, sub);   // qs at offset 0, block size 56 -> 4-byte aligned when base is
        const uint8_t* qs = (const uint8_t*)&qsp;
        int sumi[2] = {0, 0};
        int sumy[2] = {0, 0};
        float dsum[2] = {0.f, 0.f};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int qhl = b->qh[2 * sub + l / 2] >> (4 * (l % 2));
            const int g = (int)((const uint32_t*)tab)[qs[l] | ((qhl & 7) << 8)];
            const int u0 = q8[2 * l + 0], u1 = q8[2 * l + 1];
            sumi[l / 2] = __dp4a((g >> 0) & 0x0F0F0F0F, u0, sumi[l / 2]);
            sumi[l / 2] = __dp4a((g >> 4) & 0x0F0F0F0F, u1, sumi[l / 2]);
            int sy = __dp4a(u0, 0x01010101, 0);
            sy = __dp4a(u1, 0x01010101, sy);
            sumy[l / 2] += sy;
            // per-8 delta: value = g + delta; with (g+1) packed: (g+1) + (delta - 1)
            const float delta = (qhl & 0x08) ? -IQ1M_DELTA : IQ1M_DELTA;
            dsum[l / 2] += (delta - 1.0f) * (float)sy;
        }
        uint16_t sc[4];
        memcpy(sc, b->scales, 8);
        const uint16_t su = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const float d = h2f(su);
        const int tmp = sc[sub / 2] >> (6 * (sub % 2));
        const float sc0 = (float)(2 * ((tmp >> 0) & 7) + 1);
        const float sc1 = (float)(2 * ((tmp >> 3) & 7) + 1);
        (void)sumy;
        return d * d8 * (sc0 * ((float)sumi[0] + dsum[0]) + sc1 * ((float)sumi[1] + dsum[1]));
    }
    template <int NB>
    __device__ __forceinline__ static void dotn(const uint8_t* blk, int sub, const int (*q8)[8], const float* d8, const float*,
                                                const void* tab, float* acc) {
        const block_iq1_m* b = (const block_iq1_m*)blk;
        const int qsp = ld_b4(b->qs, sub);
        const uint8_t* qs = (const uint8_t*)&qsp;
        int w[8];
        float dm1[4];   // (delta - 1) per 8 weights
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int qhl = b->qh[2 * sub + l / 2] >> (4 * (l % 2));
            const int g = (int)((const uint32_t*)tab)[qs[l] | ((qhl & 7) << 8)];
            w[2 * l + 0] = (g >> 0) & 0x0F0F0F0F;
            w[2 * l + 1] = (g >> 4) & 0x0F0F0F0F;
            dm1[l] = ((qhl & 0x08) ? -IQ1M_DELTA : IQ1M_DELTA) - 1.0f;
        }
        uint16_t sc[4];
        memcpy(sc, b->scales, 8);
        const uint16_t su = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const float d = h2f(su);
        const int tmp = sc[sub / 2] >> (6 * (sub % 2));
        const float sc0 = (float)(2 * ((tmp >> 0) & 7) + 1);
        const float sc1 = (float)(2 * ((tmp >> 3) & 7) + 1);
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            int sumi[2] = {0, 0};
            float dsum[2] = {0.f, 0.f};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int u0 = q8[c][2 * l + 0], u1 = q8[c][2 * l + 1];
                sumi[l / 2] = __dp4a(w[2 * l + 0], u0, sumi[l / 2]);
                sumi[l / 2] = __dp4a(w[2 * l + 1], u1, sumi[l / 2]);
                int sy = __dp4a(u0, 0x01010101, 0);
                sy = __dp4a(u1, 0x01010101, sy);
                dsum[l / 2] = __fadd_rn(dsum[l / 2], __fmul_rn(dm1[l], (float)sy));
            }
            acc[c] = __fadd_rn(acc[c], __fmul_rn(__fmul_rn(d, d8[c]), __fadd_rn(__fmul_rn(sc0, __fadd_rn((float)sumi[0], dsum[0])),
                                                                            __fmul_rn(sc1, __fadd_rn((float)sumi[1], dsum[1])))));
        }
    }
};

template <> struct QDot<QuantFormat::IQ4_XS> {
    static constexpr int BYTES = 136;
    __device__ __forceinline__ static void decode(const uint8_t* blk, int sub, const void* tab, int* w, float* a, float* o) {
        const block_iq4_xs* b = (const block_iq4_xs*)blk;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            // non-linear 4-bit codebook for 8 nibbles
            const int2 v = table16(ld_b4(b->qs, 4 * sub + j), kvalues_iq4nl);
            w[j] = v.x;       // low nibbles: weights 4j..4j+3
            w[j + 4] = v.y;   // high nibbles: weights 16+4j..
        }
        // 6-bit scale: 4 low bits + 2 high bits
        const int ls = ((b->scales_l[sub / 2] >> (4 * (sub & 1))) & 0xF) | (((b->scales_h >> (2 * sub)) & 3) << 4);
        const float dl = h2f(b->d) * (float)(ls - 32);   // signed: -32..31
        for (int g = 0; g < 4; ++g) { a[g] = dl; o[g] = 0.0f; }
        (void)tab;
    }
    __device__ __forceinline__ static float dot(const uint8_t* blk, int sub, const int* q8, float d8, float, const void* tab) {
        const block_iq4_xs* b = (const block_iq4_xs*)blk;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q4 = ld_b4(b->qs, 4 * sub + j);   // qs at offset 8, block 136 -> 4-byte aligned
            const int2 v = table16(q4, kvalues_iq4nl);
            sumi = __dp4a(v.x, q8[j + 0], sumi);
            sumi = __dp4a(v.y, q8[j + 4], sumi);
        }
        const int ls = ((b->scales_l[sub / 2] >> (4 * (sub & 1))) & 0xF) | (((b->scales_h >> (2 * sub)) & 3) << 4);
        return h2f(b->d) * (float)(ls - 32) * d8 * (float)sumi;
    }
    template <int NB>
    __device__ __forceinline__ static void dotn(const uint8_t* blk, int sub, const int (*q8)[8], const float* d8, const float*,
                                                const void*, float* acc) {
        const block_iq4_xs* b = (const block_iq4_xs*)blk;
        int2 v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = table16(ld_b4(b->qs, 4 * sub + j), kvalues_iq4nl);
        const int ls = ((b->scales_l[sub / 2] >> (4 * (sub & 1))) & 0xF) | (((b->scales_h >> (2 * sub)) & 3) << 4);
        const float dl = h2f(b->d) * (float)(ls - 32);
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            int sumi = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sumi = __dp4a(v[j].x, q8[c][j + 0], sumi);
                sumi = __dp4a(v[j].y, q8[c][j + 4], sumi);
            }
            acc[c] = __fadd_rn(acc[c], __fmul_rn(__fmul_rn(dl, d8[c]), (float)sumi));
        }
    }
};

template <> struct QDot<QuantFormat::Q2_K> {
    static constexpr int BYTES = 84;
    __device__ __forceinline__ static void decode(const uint8_t* blk, int sub, const void* tab, int* w, float* a, float* o) {
        const block_q2_K* b = (const block_q2_K*)blk;
        const int shift = 2 * (sub & 3);   // each qs byte holds one 2-bit weight of 4 different sub-blocks
        const uint8_t* qs = b->qs + 32 * (sub >> 2);
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = (ld_b4(qs, j) >> shift) & 0x03030303;   // 2-bit values 0..3
        // per 16 weights: 4-bit scale (low nibble), 4-bit min (high)
        const uint8_t sc0 = b->scales[2 * sub], sc1 = b->scales[2 * sub + 1];
        const float d = h2f(b->d), dmin = h2f(b->dmin);
        a[0] = a[1] = d * (float)(sc0 & 0xF);
        a[2] = a[3] = d * (float)(sc1 & 0xF);
        o[0] = o[1] = -(dmin * (float)(sc0 >> 4));   // weight = d * sc * q - dmin * m
        o[2] = o[3] = -(dmin * (float)(sc1 >> 4));
        (void)tab;
    }
    __device__ __forceinline__ static float dot(const uint8_t* blk, int sub, const int* q8, float d8, float, const void* tab) {
        const block_q2_K* b = (const block_q2_K*)blk;
        // weights 32*sub..: qs bytes [32*(sub/4), +32), shift 2*(sub%4); scales 2*sub, 2*sub+1 (16 each)
        const int shift = 2 * (sub & 3);
        const uint8_t* qs = b->qs + 32 * (sub >> 2);
        int sd[2] = {0, 0}, sm[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int v = (ld_b4(qs, j) >> shift) & 0x03030303;   // qs at offset 16 in an 84-byte block
            sd[j / 4] = __dp4a(v, q8[j], sd[j / 4]);
            sm[j / 4] = __dp4a(0x01010101, q8[j], sm[j / 4]);
        }
        const uint8_t sc0 = b->scales[2 * sub], sc1 = b->scales[2 * sub + 1];
        const float d = h2f(b->d), dmin = h2f(b->dmin);
        return d8 * (d * ((float)(sc0 & 0xF) * (float)sd[0] + (float)(sc1 & 0xF) * (float)sd[1]) -
                     dmin * ((float)(sc0 >> 4) * (float)sm[0] + (float)(sc1 >> 4) * (float)sm[1]));
    }
    template <int NB>
    __device__ __forceinline__ static void dotn(const uint8_t* blk, int sub, const int (*q8)[8], const float* d8, const float*,
                                                const void*, float* acc) {
        const block_q2_K* b = (const block_q2_K*)blk;
        const int shift = 2 * (sub & 3);
        const uint8_t* qs = b->qs + 32 * (sub >> 2);
        int w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = (ld_b4(qs, j) >> shift) & 0x03030303;
        const uint8_t sc0 = b->scales[2 * sub], sc1 = b->scales[2 * sub + 1];
        const float d = h2f(b->d), dmin = h2f(b->dmin);
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            int sd[2] = {0, 0}, sm[2] = {0, 0};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                sd[j / 4] = __dp4a(w[j], q8[c][j], sd[j / 4]);
                sm[j / 4] = __dp4a(0x01010101, q8[c][j], sm[j / 4]);
            }
            const float pd = __fadd_rn(__fmul_rn((float)(sc0 & 0xF), (float)sd[0]), __fmul_rn((float)(sc1 & 0xF), (float)sd[1]));
            const float pm = __fadd_rn(__fmul_rn((float)(sc0 >> 4), (float)sm[0]), __fmul_rn((float)(sc1 >> 4), (float)sm[1]));
            acc[c] = __fadd_rn(acc[c], __fmul_rn(d8[c], __fsub_rn(__fmul_rn(d, pd), __fmul_rn(dmin, pm))));
        }
    }
};

template <> struct QDot<QuantFormat::Q4_K> {
    static constexpr int BYTES = 144;
    __device__ __forceinline__ static void decode(const uint8_t* blk, int sub, const void* tab, int* w, float* a, float* o) {
        const block_q4_K* b = (const block_q4_K*)blk;
        const int p = sub >> 1, hi = sub & 1;   // sub-blocks 2p and 2p+1 share 32 bytes: low and high nibbles
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = (ld_b4(b->qs, 8 * p + j) >> (4 * hi)) & 0x0F0F0F0F;
        uint8_t sc, m;
        const uint8_t* s = b->scales;
        // 6-bit scales and mins packed in 12 bytes (ggml get_scale_min_k4)
        if (sub < 4) { sc = s[sub] & 63; m = s[sub + 4] & 63; }
        else { sc = (s[sub + 4] & 0xF) | ((s[sub - 4] >> 6) << 4); m = (s[sub + 4] >> 4) | ((s[sub] >> 6) << 4); }
        const float dsc = h2f(b->d) * (float)sc, dm = h2f(b->dmin) * (float)m;
        for (int g = 0; g < 4; ++g) { a[g] = dsc; o[g] = -dm; }
        (void)tab;
    }
    __device__ __forceinline__ static float dot(const uint8_t* blk, int sub, const int* q8, float d8, float s8, const void* tab) {
        const block_q4_K* b = (const block_q4_K*)blk;
        // sub-block pair p = sub/2 uses qs[32p .. 32p+32): low nibbles -> sub 2p, high -> 2p+1
        const int p = sub >> 1, hi = sub & 1;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int v = (ld_b4(b->qs, 8 * p + j) >> (4 * hi)) & 0x0F0F0F0F;
            sumi = __dp4a(v, q8[j], sumi);
        }
        uint8_t sc, m;
        const uint8_t* s = b->scales;
        if (sub < 4) { sc = s[sub] & 63; m = s[sub + 4] & 63; }
        else { sc = (s[sub + 4] & 0xF) | ((s[sub - 4] >> 6) << 4); m = (s[sub + 4] >> 4) | ((s[sub] >> 6) << 4); }
        return h2f(b->d) * (float)sc * d8 * (float)sumi - h2f(b->dmin) * (float)m * s8;
    }
    template <int NB>
    __device__ __forceinline__ static void dotn(const uint8_t* blk, int sub, const int (*q8)[8], const float* d8, const float* s8,
                                                const void*, float* acc) {
        const block_q4_K* b = (const block_q4_K*)blk;
        const int p = sub >> 1, hi = sub & 1;
        int w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = (ld_b4(b->qs, 8 * p + j) >> (4 * hi)) & 0x0F0F0F0F;
        uint8_t sc, m;
        const uint8_t* s = b->scales;
        if (sub < 4) { sc = s[sub] & 63; m = s[sub + 4] & 63; }
        else { sc = (s[sub + 4] & 0xF) | ((s[sub - 4] >> 6) << 4); m = (s[sub + 4] >> 4) | ((s[sub] >> 6) << 4); }
        const float dsc = h2f(b->d) * (float)sc, dm = h2f(b->dmin) * (float)m;
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            int sumi = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) sumi = __dp4a(w[j], q8[c][j], sumi);
            acc[c] = __fadd_rn(acc[c], __fsub_rn(__fmul_rn(__fmul_rn(dsc, d8[c]), (float)sumi), __fmul_rn(dm, s8[c])));
        }
    }
};

// Q6_K (MTP block): weight = d * scale[16-group] * (q - 32), q = 4 low bits (ql) | 2 high bits (qh); 210-byte blocks,
// 2-byte aligned. Sub-block sub: half = sub/4 (128 weights), quarter = sub%4 selects the ql half/nibble and qh bit pair.
template <> struct QDot<QuantFormat::Q6_K> {
    static constexpr int BYTES = 210;
    __device__ __forceinline__ static void decode(const uint8_t* blk, int sub, const void* tab, int* w, float* a, float* o) {
        const block_q6_K* b = (const block_q6_K*)blk;
        const int half = sub >> 2, quarter = sub & 3;   // two halves of 128 weights, each in 4 bit planes
        const uint8_t* ql = b->ql + 64 * half + 32 * (quarter & 1);
        const uint8_t* qh = b->qh + 32 * half;
        const int ns = 4 * (quarter >> 1), hs = 2 * quarter;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int lo = (ld_b2(ql, j) >> ns) & 0x0F0F0F0F;   // 4 low bits
            const int hi = ((ld_b2(qh, j) >> hs) & 0x03030303) << 4;   // 2 high bits
            w[j] = (int)__vsub4((unsigned)(lo | hi), 0x20202020u);   // 6-bit value - 32 = signed -32..31, per byte
        }
        const float d = h2f(b->d);
        a[0] = a[1] = d * (float)b->scales[8 * half + 2 * quarter];   // int8 scale per 16 weights
        a[2] = a[3] = d * (float)b->scales[8 * half + 2 * quarter + 1];
        for (int g = 0; g < 4; ++g) o[g] = 0.0f;
        (void)tab;
    }
    template <int NB>
    __device__ __forceinline__ static void dotn(const uint8_t* blk, int sub, const int (*q8)[8], const float* d8, const float*,
                                                const void*, float* acc) {
        const block_q6_K* b = (const block_q6_K*)blk;
        const int half = sub >> 2, quarter = sub & 3;
        const uint8_t* ql = b->ql + 64 * half + 32 * (quarter & 1);
        const uint8_t* qh = b->qh + 32 * half;
        const int ns = 4 * (quarter >> 1), hs = 2 * quarter;
        int w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int lo = (ld_b2(ql, j) >> ns) & 0x0F0F0F0F;
            const int hi = ((ld_b2(qh, j) >> hs) & 0x03030303) << 4;
            w[j] = (int)__vsub4((unsigned)(lo | hi), 0x20202020u);   // q - 32 per byte, in [-32, 31]
        }
        const float sc0 = (float)b->scales[8 * half + 2 * quarter], sc1 = (float)b->scales[8 * half + 2 * quarter + 1];
        const float d = h2f(b->d);
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) s0 = __dp4a(w[j], q8[c][j], s0);
#pragma unroll
            for (int j = 4; j < 8; ++j) s1 = __dp4a(w[j], q8[c][j], s1);
            acc[c] = __fadd_rn(acc[c], __fmul_rn(__fmul_rn(d, d8[c]), __fadd_rn(__fmul_rn(sc0, (float)s0), __fmul_rn(sc1, (float)s1))));
        }
    }
};

} // namespace cuda
} // namespace q38rt
