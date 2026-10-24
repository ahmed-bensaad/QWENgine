#pragma once
/// @file quant_formats.h
/// Bit-exact block layouts of the ggml quantization formats used by
/// Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf (weights) and by our KV cache / activations.
/// Layouts follow ggml/src/ggml-common.h (MIT License, The ggml authors).
/// All structs are plain bytes (no padding): they alias the GGUF mapping directly.

#include <cstdint>
#include <cstring>
#include "q38rt/q38rt_types.h"

#if defined(__CUDACC__)
#define Q38_HD __host__ __device__
#else
#define Q38_HD
#endif

namespace q38rt {

static constexpr int QK_K  = 256;   // super-block size of K/IQ formats
static constexpr int QK8_0 = 32;
static constexpr int QK4_0 = 32;
static constexpr int QK8_1 = 32;

using fp16_t = uint16_t;   // raw IEEE binary16 bits
using bf16_t = uint16_t;   // raw bfloat16 bits

// Exact IEEE half -> float (handles subnormals, inf, nan).
Q38_HD inline float fp16_to_fp32(fp16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal: normalize
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// float -> IEEE half, round to nearest even (matches F16C / ggml's FP32_TO_FP16).
Q38_HD inline fp16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t absx = x & 0x7FFFFFFFu;
    if (absx >= 0x7F800000u) {                     // inf / nan
        return (fp16_t)(sign | 0x7C00u | (absx > 0x7F800000u ? 0x200u : 0));
    }
    if (absx >= 0x477FF000u) return (fp16_t)(sign | 0x7C00u);   // overflow -> inf
    if (absx < 0x38800000u) {                      // result is subnormal or zero
        if (absx < 0x33000000u) return (fp16_t)sign;
        const uint32_t e = absx >> 23;
        uint32_t m = (absx & 0x7FFFFFu) | 0x800000u;
        const uint32_t shift = 126 - e;            // 14 + (113 - e)
        const uint32_t half_m = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1);
        const uint32_t halfway = 1u << (shift - 1);
        uint32_t r = half_m + ((rem > halfway || (rem == halfway && (half_m & 1))) ? 1 : 0);
        return (fp16_t)(sign | r);
    }
    uint32_t r = ((absx - 0x38000000u) >> 13);     // rebias exponent 127 -> 15
    const uint32_t rem = absx & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) r++;
    return (fp16_t)(sign | r);
}

Q38_HD inline float bf16_to_fp32(bf16_t b) {
    const uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

#pragma pack(push, 1)
struct block_q4_0 { fp16_t d; uint8_t qs[QK4_0 / 2]; };
struct block_q8_0 { fp16_t d; int8_t qs[QK8_0]; };
struct block_q8_1 { fp16_t d; fp16_t s; int8_t qs[QK8_1]; };          // s = d * sum(qs)

struct block_q2_K { uint8_t scales[QK_K / 16]; uint8_t qs[QK_K / 4]; fp16_t d; fp16_t dmin; };
struct block_q4_K { fp16_t d; fp16_t dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; };
struct block_q6_K { uint8_t ql[QK_K / 2]; uint8_t qh[QK_K / 4]; int8_t scales[QK_K / 16]; fp16_t d; };
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; };

struct block_iq2_xxs { fp16_t d; uint16_t qs[QK_K / 8]; };
struct block_iq2_xs  { fp16_t d; uint16_t qs[QK_K / 8]; uint8_t scales[QK_K / 32]; };
struct block_iq2_s   { fp16_t d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t scales[QK_K / 32]; };
struct block_iq3_xxs { fp16_t d; uint8_t qs[3 * QK_K / 8]; };
struct block_iq3_s   { fp16_t d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t signs[QK_K / 8]; uint8_t scales[QK_K / 64]; };
struct block_iq1_s   { fp16_t d; uint8_t qs[QK_K / 8]; uint16_t qh[QK_K / 32]; };
struct block_iq1_m   { uint8_t qs[QK_K / 8]; uint8_t qh[QK_K / 16]; uint8_t scales[QK_K / 32]; };
struct block_iq4_xs  { fp16_t d; uint16_t scales_h; uint8_t scales_l[QK_K / 64]; uint8_t qs[QK_K / 2]; };
#pragma pack(pop)

static_assert(sizeof(block_q4_0) == 18, "q4_0");
static_assert(sizeof(block_q8_0) == 34, "q8_0");
static_assert(sizeof(block_q8_1) == 36, "q8_1");
static_assert(sizeof(block_q2_K) == 84, "q2_K");
static_assert(sizeof(block_q4_K) == 144, "q4_K");
static_assert(sizeof(block_q6_K) == 210, "q6_K");
static_assert(sizeof(block_q8_K) == 292, "q8_K");
static_assert(sizeof(block_iq2_xxs) == 66, "iq2_xxs");
static_assert(sizeof(block_iq2_xs) == 74, "iq2_xs");
static_assert(sizeof(block_iq2_s) == 82, "iq2_s");
static_assert(sizeof(block_iq3_xxs) == 98, "iq3_xxs");
static_assert(sizeof(block_iq3_s) == 110, "iq3_s");
static_assert(sizeof(block_iq1_s) == 50, "iq1_s");
static_assert(sizeof(block_iq1_m) == 56, "iq1_m");
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs");

// ---------------------------------------------------------------------------
// Reference (scalar, exact) routines — src/quant/quant_ref.cpp
// ---------------------------------------------------------------------------

/// Dequantize `n` elements (multiple of the block size) of storage type `t` to float.
/// Supports every weight type in the target GGUF plus F32/F16/BF16/Q8_0/Q4_0.
/// Returns false for unsupported types.
bool dequantize_row_ref(QuantFormat t, const void* src, float* dst, int64_t n);

/// ggml-identical activation / KV quantizers.
void quantize_row_q8_K_ref(const float* x, block_q8_K* y, int64_t n);
void quantize_row_q8_0_ref(const float* x, block_q8_0* y, int64_t n);
/// ggml's reference Q4_K quantizer (bit-exact port; used to requantize the MTP block).
void quantize_row_q4_K_ref(const float* x, block_q4_K* y, int64_t n);
void quantize_row_q4_0_ref(const float* x, block_q4_0* y, int64_t n);

} // namespace q38rt
