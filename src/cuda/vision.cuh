#pragma once
/// @file vision.cuh
/// Kernels of the vision encoder (runtime/vision.h): the element-wise and normalization steps between its cuBLAS GEMMs.
/// Activations are FP32 [rows][features], GEMM inputs FP16.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace q38rt {
namespace cuda {

/// Y[r][e] += bias[e]
void vis_bias(float* Y, const float* bias, int rows, int n, cudaStream_t st);
/// X[r][e] += Y[r][e] + bias[e]
void vis_residual(float* X, const float* Y, const float* bias, int rows, int n, cudaStream_t st);
/// LayerNorm of each row of n <= 1152 * 4 values: out16 = (x - mean) / sqrt(var + eps) * w + b, in FP16
void vis_layernorm(const float* X, const float* w, const float* b, int rows, int n, float eps, __half* out16, cudaStream_t st);
/// out16 = gelu(U + bias), tanh approximation, FP16
void vis_bias_gelu(const float* U, const float* bias, int rows, int n, __half* out16, cudaStream_t st);
/// FP32 -> FP16
void vis_to_f16(const float* x, size_t n, __half* out, cudaStream_t st);
/// Rows t0 .. t0+C-1 of the QKV projection ([C][3 * 1152] FP32, before its bias) -> Q and K with the 2D rotary
/// positions of their patches ([16][N][72] FP16) and V transposed ([16][72][N] FP16). Patch t is in merge order (the
/// 4 patches of each 2x2 group consecutive); nx = merged tokens per row. Rotation: head-dim pairs (j, j + 36), angle
/// row * 10000^(-j/18) for j < 18, column * 10000^(-(j-18)/18) for 18 <= j < 36 (ggml rope_multi, vision mode).
void vis_qkv_rope(const float* qkv, const float* bias, int t0, int C, int N, int nx, __half* Q, __half* K, __half* Vt,
                  cudaStream_t st);
/// P = softmax(scale * S) per row ([rows][cols] FP32 -> FP16)
void vis_softmax(const float* S, int rows, int cols, float scale, __half* P, cudaStream_t st);

} // namespace cuda
} // namespace q38rt
