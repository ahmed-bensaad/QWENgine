#pragma once
/// @file ops.cuh
/// Non-GEMV decode kernels for passes of 1..4 tokens: residual + RMSNorm (with Q8 activation outputs for the
/// GPU and CPU GEMVs), Gated DeltaNet recurrence with lazy commit, gated attention over the quantized KV cache
/// (tensor-core decode attention), SwiGLU, top-k over the LM-head logits, copies between device and mapped host
/// memory. Single-token wrappers (attn_decode, deltanet_step, ...) call the multi-token kernels.

#include <cstdint>
#include <cuda_runtime.h>
#include "cuda/gemv.cuh"

namespace q38rt {
namespace cuda {

/// Kernel-based copy of n floats (device <-> mapped pinned host memory). Small activation transfers use this
/// instead of cudaMemcpyAsync so they never queue behind bulk weight DMA on the copy engines. Writes to host
/// memory are fenced (system scope) before the kernel ends.
void copy_floats(float* dst, const float* src, int n, cudaStream_t st);
/// Same for `rows` rows of n floats: dst[r*dst_ld + i] = src[r*src_ld + i].
void copy_rows(float* dst, int dst_ld, const float* src, int src_ld, int n, int rows, cudaStream_t st);


/// A row-wise source of n floats: element i of row r comes from dev[r*n + i] for i < split, else from (mapped
/// host) host[r*host_ld + i - split]. dev == host == nullptr: absent.
struct RowSrc {
    const float* dev = nullptr;
    const float* host = nullptr;
    int split = 0, host_ld = 0;
    static RowSrc device(const float* p, int n) { RowSrc s; s.dev = p; s.split = n; return s; }
    static RowSrc split_at(const float* dev, int split, const float* host, int host_ld) {
        RowSrc s; s.dev = dev; s.split = split; s.host = host; s.host_ld = host_ld; return s;
    }
};

/// x += a (+ b if non-null); then xn = rmsnorm(x) * w with optional fp32/ActQ8 outputs.
/// If w == nullptr only the residual add is performed.
void add_rmsnorm(float* x, const float* a, const float* b, const float* w, int n, float eps, float* xn,
                 const ActQ8* q, cudaStream_t st);
/// add_rmsnorm on `rows` contiguous rows of n (x, a, b, xn and the ActQ8 rows all have stride n); per-row math
/// identical to add_rmsnorm.
void add_rmsnorm_rows(float* x, const float* a, const float* b, const float* w, int n, int rows, float eps, float* xn,
                      const ActQ8* q, cudaStream_t st);
/// Same with split device/host sources and an optional second output to (mapped) host memory.
/// xq_host (optional, mapped host memory): the normalized rows also as Q8_K blocks for the CPU kernels (bit-identical
/// to cpu::quantize_q8_K of the fp32 rows; 292 B per 256 elements instead of 1 KB of fp32 over PCIe).
void add_rmsnorm_rows2(float* x, const RowSrc& a, const RowSrc& b, const float* w, int n, int rows, float eps, float* xn,
                       float* xn_host, const ActQ8* q, cudaStream_t st, void* xq_host = nullptr);

/// h[i] = silu(g[i]) * u[i] for i < n; optionally quantized.
void swiglu(const float* g, const float* u, int n, float* h, const ActQ8* q, cudaStream_t st);
/// swiglu on the first n elements of `rows` rows with stride ld (g, u, h and the ActQ8 rows).
void swiglu_rows(const float* g, const float* u, int n, int ld, int rows, float* h, const ActQ8* q, cudaStream_t st);

/// Parameters of one Gated DeltaNet layer (device pointers).
struct DeltaNetParams {
    const float* conv_w;     // [10240][4]
    const float* dt_bias;    // [48]
    const float* ssm_a;      // [48]  (= -exp(A_log))
    const float* ssm_norm;   // [128]
    float* conv_state;       // [2][3][10240] ping-pong (index = parity)
    float* ssm_state;        // [48][128 col j][128 row i] fp32
};

/// One token of a Gated DeltaNet layer. qkv [10240], z [6144], beta_raw/alpha_raw [48] are the
/// projection outputs. Writes o [6144] (gated-normed core output) fp32 and optionally ActQ8.
/// The conv state is read from buffer `parity` and written to buffer parity^1.
void deltanet_step(const DeltaNetParams& p, const float* qkv, const float* z, const float* beta_raw,
                   const float* alpha_raw, int parity, float eps, float* o, const ActQ8* oq, cudaStream_t st);

/// Token inputs of deltanet_multi. Pending tokens (already output by an earlier pass, applied to the state
/// now): sqkv [n][10240], sbeta/salpha [n][64]. New tokens: qkv [n][10240], z [n][6144], beta/alpha [n][64].
/// oqkv/obeta/oalpha (optional) receive the new tokens' inputs (save buffer for a later commit; must not
/// alias the pending inputs).
struct DNTokens {
    const float* sqkv = nullptr;
    const float* sbeta = nullptr;
    const float* salpha = nullptr;
    const float* qkv = nullptr;
    const float* z = nullptr;
    const float* beta = nullptr;
    const float* alpha = nullptr;
    float* oqkv = nullptr;
    float* obeta = nullptr;
    float* oalpha = nullptr;
    // new tokens' qkv / z channels >= *_split come from (mapped host) *_host [n][*_ld] (the CPU share)
    const float* qkv_host = nullptr;
    int qkv_split = 1 << 30, qkv_ld = 0;
    const float* z_host = nullptr;
    int z_split = 1 << 30, z_ld = 0;
    float* o_host = nullptr;   // optional copy of the outputs in (mapped host) memory, rows of 6144
};

/// Lazy-commit multi-token step: applies n_commit pending tokens then n_new new tokens (outputs o [n_new][6144]
/// and ActQ8 rows); the conv window (to buffer cin^1) and the state (in place) are written back after
/// `write_after` tokens (0 = leave both unchanged). Bit-identical to the same tokens applied one per launch.
void deltanet_multi(const DeltaNetParams& p, int cin, const DNTokens& in, int n_commit, int n_new, int write_after,
                    float eps, float* o, const ActQ8* oq, cudaStream_t st);

enum class KVType : int { F16 = 0, Q8_0 = 1, Q4_0 = 2 };
/// Split KV layout: per 256-element head row, the values are contiguous and 16-byte aligned
/// (F16: 512 B, Q8_0: 256 int8, Q4_0: 128 B, element 2i in the low nibble of byte i) and the
/// 8 per-32 fp16 scales (Q8_0/Q4_0, ggml semantics) live in a separate array.
size_t kv_value_bytes(KVType t);
size_t kv_scale_bytes(KVType t);

/// Per-layer KV cache (device). values: [kv_head][pos][value_bytes]; scales: [kv_head][pos][8] fp16.
struct KVLayer {
    uint8_t* k = nullptr;
    uint8_t* v = nullptr;
    uint16_t* ks = nullptr;
    uint16_t* vs = nullptr;
    int capacity = 0;   // positions
};

/// q/k RMSNorm per head, RoPE (NeoX on first 64 dims, theta base), K/V append at `pos`.
/// qg: [24][512] (Q | gate per head), k/v: [4][256]. Writes q_out [24][256] fp32 and gate [24][256].
void attn_prep(const float* qg, const float* k, const float* v, const float* q_norm, const float* k_norm,
               int pos, float theta, float eps, KVType kvt, KVLayer& kv, float* q_out, float* gate, cudaStream_t st);
/// attn_prep for n consecutive tokens (cache positions pos0.., rotary positions rope_pos0..: they differ after an image,
/// see Engine::PrefillImage), qg / k / v rows of 2*24*256 / 4*256 / 4*256 from split device/host sources; q_out / gate
/// rows of 24*256.
void attn_prep_multi(const RowSrc& qg, const RowSrc& k, const RowSrc& v, const float* q_norm, const float* k_norm, int pos0,
                     int rope_pos0, int n, float theta, float eps, KVType kvt, KVLayer& kv, float* q_out, float* gate,
                     cudaStream_t st);

/// Attention over positions [0, n_pos) for 24 query heads / 4 KV heads, output * sigmoid(gate).
/// work: scratch of attn_work_bytes(n_pos). Writes out [24*256] fp32 and optionally ActQ8.
/// Tensor-core kernel (fp16 Q/K/V/P, fp32 accumulation). attn_work_bytes covers both kernels (up to 4 queries).
size_t attn_work_bytes(int n_pos);
/// Longest context the decode attention kernels support (positions).
int attn_max_positions();
void attn_decode(const float* q, const float* gate, const KVLayer& kv, KVType kvt, int n_pos, void* work,
                 float* out, const ActQ8* oq, cudaStream_t st);
/// nq <= 4 consecutive queries at positions pos0 .. pos0+nq-1 (query t sees keys [0, pos0+t]); q/gate/out rows of
/// 24*256, ActQ8 rows of 6144. Each query's result is bit-identical to attn_decode on it alone.
void attn_decode_multi(const float* q, const float* gate, const KVLayer& kv, KVType kvt, int pos0, int nq, void* work,
                       float* out, const ActQ8* oq, cudaStream_t st, float* out_host = nullptr);
/// FP32 reference kernel (CUDA cores), kept for tests.
void attn_decode_simple(const float* q, const float* gate, const KVLayer& kv, KVType kvt, int n_pos, void* work,
                        float* out, const ActQ8* oq, cudaStream_t st);

/// Top-k over logits[0..n): writes k (value, index) pairs sorted descending (k <= 64).
void topk(const float* logits, int n, int k, int index_offset, float* vals, int* idx, void* work, cudaStream_t st);
size_t topk_work_bytes(int n);

} // namespace cuda
} // namespace q38rt
