#pragma once
/// @file prefill.cuh
/// Batched (prefill) kernels: B tokens at once. GEMMs go through cuBLAS on FP16 weights dequantized per row block;
/// attention is a tensor-core flash attention over the quantized KV cache (producer warps fill a double-buffered
/// K/V tile); everything else is token-parallel except the Gated DeltaNet recurrence (sequential over tokens,
/// state held in registers).

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cuda/ops.cuh"
#include "q38rt/q38rt_types.h"

namespace q38rt {
namespace cuda {

/// Dequantize `rows` rows (K elements each, `row_bytes` apart) of a ggml-format matrix to FP16 [rows][K].
/// BF16 is supported as well (row_bytes = 2K).
bool dequant_rows_f16(QuantFormat type, const void* W, size_t row_bytes, int K, int rows, __half* out, cudaStream_t st);

/// Y[b][n] (+)= sum_k X[b][k] * W[n][k] for an FP16 weight block W [N][K] and FP16 activations X [B][ldx];
/// Y fp32 with row stride ldy. accumulate: beta = 1 instead of 0.
void gemm_f16(void* cublas, const __half* W, int N, int K, const __half* X, int ldx, int B, float* Y, int ldy,
              bool accumulate, cudaStream_t st);
void* cublas_create();
/// Bind the handle to a stream and to a caller-owned workspace (no allocations by cuBLAS afterwards).
bool cublas_bind(void* cublas, cudaStream_t st, void* workspace, size_t bytes);
void cublas_destroy(void* h);

/// out16[b][i] (and/or out32) = rmsnorm(X[b]) * w, for B rows of n. X optionally += A (+ C) first (written back).
void rmsnorm_rows(float* X, const float* A, const float* C, const float* w, int B, int n, float eps, __half* out16,
                  float* out32, cudaStream_t st);
/// silu(G) * U for B rows of n (row stride ld floats), written in place as FP16: row b of the result is the first
/// 2*n bytes of G's row b (H16 row stride 2*ld halves).
void swiglu_rows_inplace(float* G, const float* U, int B, int n, int ld, cudaStream_t st);

/// Gated DeltaNet over B tokens (positions p0..p0+B-1). qkv [B][10240], z [B][6144], beta/alpha raw [B][48].
/// Updates conv/ssm state (conv state parity convention of deltanet_step: the final state lands in buffer
/// (p0+B)&1). work: >= dn_work_bytes(B). core_out [B][6144] fp32 scratch, written after qkv was last read (may
/// alias qkv). Output o16 [B][6144] (gated-normed) in FP16 (must not overlap core_out).
size_t dn_work_bytes(int B);
void deltanet_rows(const DeltaNetParams& p, const float* qkv, const float* z, const float* beta_raw,
                   const float* alpha_raw, int B, int cin, float eps, void* work, float* core_out, __half* o16,
                   cudaStream_t st);

/// RoPE table of B consecutive positions p0..: rope[t][i] = cos, rope[t][32 + i] = sin of (p0 + t) * theta^(-i/32)
/// (angle in double, reduced mod 2 pi; computed once per prefill chunk for all attention layers).
void rope_table(int p0, int B, float theta, float* rope, cudaStream_t st);
/// RoPE table of B tokens with 3D positions pos [B][3] (t, h, w; device), Qwen3-VL's interleaved M-RoPE: frequency i
/// turns with pos[i % 3] (the model's sections 11 / 11 / 10, checked by the GGUF parser). Equals rope_table for text
/// tokens (t = h = w).
void rope_table_pos(const int* pos, int B, float theta, float* rope, cudaStream_t st);
/// Attention prep for B tokens: q/k RMSNorm + RoPE (rope: the chunk's rope_table / rope_table_pos), K/V appended to
/// the cache at positions p0+t. qg [B][24*512] (Q|gate per head), k/v [B][1024]. Writes q [B][24][256] fp32 and gate [B][24][256].
void attn_prep_rows(const float* qg, const float* k, const float* v, const float* q_norm, const float* k_norm, int B,
                    int p0, const float* rope, float eps, KVType kvt, KVLayer& kv, float* q, float* gate, cudaStream_t st);
/// Causal attention for B queries at positions p0..p0+B-1 over the cache [0, p0+B); output * sigmoid(gate)
/// written as FP16 [B][24*256].
void attn_prefill(const float* q, const float* gate, const KVLayer& kv, KVType kvt, int B, int p0, __half* out16,
                  cudaStream_t st);   // tensor cores (mma.sync)
void attn_prefill_simple(const float* q, const float* gate, const KVLayer& kv, KVType kvt, int B, int p0,
                         __half* out16, cudaStream_t st);   // CUDA-core reference version

} // namespace cuda
} // namespace q38rt
