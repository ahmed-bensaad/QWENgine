#pragma once
/// @file cpu_kernels.h
/// AVX2 quantized dot products (weights in ggml formats x activations in Q8_K) for the
/// CPU-resident share of the decode GEMVs.

#include <cstdint>
#include "q38rt/q38rt_types.h"
#include "quant/quant_formats.h"

namespace q38rt {
namespace cpu {

/// dot(K, w_row, x) = sum_k dequant(w)[k] * (x.d * x.qs[k]) over K elements (multiple of 256).
using vec_dot_fn = float (*)(int K, const void* w_row, const block_q8_K* x);

/// Optimized kernel for a weight format, nullptr if unsupported.
vec_dot_fn get_vec_dot(QuantFormat t);

/// Multi-column variant for nb = 1..4 activation columns sharing one pass over the weight row:
/// y[b] = dot(K, w_row, x + b * x_stride) (x_stride in blocks). Bit-identical per column to vec_dot.
using vec_dot_n_fn = void (*)(int K, const void* w_row, const block_q8_K* x, size_t x_stride, float* y);
vec_dot_n_fn get_vec_dot_n(QuantFormat t, int nb);

/// CPU-only exact re-encodings of grid formats, decoded without gathers; the kernels form the same integer products
/// as the source kernels, so results are bit-identical (faster with 2-4 columns):
///   R4 (IQ3_S, IQ3_XXS): 134 B per 256 weights = fp16 d, 8 scale nibbles, 4-bit signed level codes.
///   R2 (IQ2_XXS, IQ2_XS, IQ2_S): 106 B = fp16 d, 16 scale nibbles, 2-bit magnitude codes, sign bits.
bool re_supported(QuantFormat t);
size_t re_row_bytes(QuantFormat t, int K);
/// Re-encode one row; false if a value lies outside the format's level set (never for valid data).
bool re_repack_row(QuantFormat t, const void* src, int K, void* dst);
vec_dot_fn get_vec_dot_re(QuantFormat t);
vec_dot_n_fn get_vec_dot_re_n(QuantFormat t, int nb);

/// AVX2 Q8_K activation quantization (same math as quantize_row_q8_K_ref).
void quantize_q8_K(const float* x, block_q8_K* y, int K);

} // namespace cpu
} // namespace q38rt
