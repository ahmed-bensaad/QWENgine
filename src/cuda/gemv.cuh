#pragma once
/// @file gemv.cuh
/// Host-side API of the decode GEMV kernels (src/cuda/gemv.cu).

#include <cstdint>
#include <cuda_runtime.h>
#include "q38rt/q38rt_types.h"

namespace q38rt {
namespace cuda {

/// Activation vector quantized for the dp4a GEMV: int8 values, per-32 scale d and
/// per-32 s = d * sum(q) (the sum of the dequantized activations of the block).
struct ActQ8 {
    int8_t* q = nullptr;   // [K]
    float* d = nullptr;    // [K/32]
    float* s = nullptr;    // [K/32]
};

/// Quantize n_rows activation vectors of length K (row-major, stride K) into ActQ8 (same layout).
void quantize_act_q8(const float* x, int K, int n_rows, ActQ8 out, cudaStream_t st);

/// y[r] = W[r, :] . act for r in [0, N), W row-major with `row_bytes` bytes per row in ggml
/// layout `type` (K elements per row). Returns false if `type` has no GEMV kernel.
bool gemv_q8(QuantFormat type, const void* W, size_t row_bytes, int K, int N, const ActQ8& act, float* y,
             cudaStream_t st, bool accumulate = false);

/// y[r] = W[r, :] . x for BF16 weights (small projections: ssm_alpha / ssm_beta).
void gemv_bf16(const uint16_t* W, int K, int N, const float* x, float* y, cudaStream_t st);

/// Multi-column GEMV: y[c*y_ld + r] (+)= W[r,:] . x_c for c < nb (1..4), x_c = activation c at act.q + c*act_ld
/// (d/s at + c*act_ld/32). Each column's result is bit-identical to gemv_q8 on that column alone.
bool gemv_q8n(QuantFormat type, const void* W, size_t row_bytes, int K, int N, const ActQ8& act, int nb, int act_ld,
              float* y, int y_ld, cudaStream_t st, bool accumulate = false);

/// Supported weight formats of gemv_q8.
bool gemv_q8_supported(QuantFormat type);

} // namespace cuda
} // namespace q38rt
