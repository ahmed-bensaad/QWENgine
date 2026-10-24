/// @file ops.cu
/// Non-GEMV decode kernels. Each mirrors one step of llama.cpp's qwen35 graph (validated by the
/// FP32 reference model, src/reference/ref_model.cpp) and is unit-tested against it.

#include "cuda/ops.cuh"
#include "cuda/common.cuh"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include "common/utils.h"
#include "quant/quant_formats.h"

namespace q38rt {
namespace cuda {

namespace {

using namespace dev;

// ---------------------------------------------------------------------------
// RMSNorm (+ residual) — one CTA per row (blockIdx.y), rows contiguous with stride n
// ---------------------------------------------------------------------------

__device__ __forceinline__ float src_at(const RowSrc& s, int r, int n, int i) {
    // columns below the split were computed on the GPU, the rest by the CPU into mapped host memory
    return i < s.split ? s.dev[(size_t)r * n + i] : s.host[(size_t)r * s.host_ld + (i - s.split)];
}

// x += a (+ b) per row (blockIdx.y), then xn = rmsnorm(x) * w to the device and/or mapped host memory and ActQ8.
// a / b elements may come from device memory (head) and mapped host memory (tail, the CPU share).
// Q8_K block (quant_formats.h): float d, int8 qs[256], int16 bsums[16] = 292 bytes
constexpr int Q8K_BYTES = 292, Q8K_MAXN = 8192;
__device__ __forceinline__ int warp_min_i(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// One 256-element block per warp (lane: elements 8 lane .. 8 lane + 7), exactly quantize_row_q8_K_ref: first element
// of maximal |v| (signed) -> iscale = -127 / max, q = min(127, round-half-even(iscale * v)), bsums of 16, d = 1 / iscale.
__device__ __forceinline__ void q8k_block_store(const float* vb, uint8_t* blk) {
    const int lane = threadIdx.x & 31;
    float v8[8];
    float am = 0.0f;
#pragma unroll
    // this lane's 8 elements and their largest magnitude
    for (int e = 0; e < 8; e++) { v8[e] = vb[8 * lane + e]; am = fmaxf(am, fabsf(v8[e])); }
    am = warp_max(am);
    uint32_t w0 = 0, w1 = 0;
    int s8 = 0;
    float d = 0.0f;
    if (am != 0.0f) {
        int idx = 256;
#pragma unroll
        // first element reaching the maximum, as the reference picks it
        for (int e = 7; e >= 0; e--) if (fabsf(v8[e]) == am) idx = 8 * lane + e;
        idx = warp_min_i(idx);
        // signed: the extreme value maps to -127 exactly (ggml's convention)
        const float iscale = __fdiv_rn(-127.0f, vb[idx]);
#pragma unroll
        for (int e = 0; e < 8; e++) {
            // round half to even like nearest_int(); +128 clamps to 127
            const int qv = min(127, __float2int_rn(__fmul_rn(iscale, v8[e])));
            s8 += qv;
            if (e < 4) w0 |= (uint32_t)(qv & 0xFF) << (8 * e); else w1 |= (uint32_t)(qv & 0xFF) << (8 * (e - 4));
        }
        d = __fdiv_rn(1.0f, iscale);
    }
    ((uint32_t*)(blk + 4))[2 * lane] = w0;       // blk is 4-byte aligned (292 = 4 * 73)
    ((uint32_t*)(blk + 4))[2 * lane + 1] = w1;
    const int s16 = s8 + __shfl_xor_sync(0xffffffffu, s8, 1);   // bsums are per 16 elements = 2 lanes
    if ((lane & 1) == 0) ((int16_t*)(blk + 260))[lane >> 1] = (int16_t)s16;
    if (lane == 0) *(float*)blk = d;
}

__global__ void k_add_rmsnorm(float* __restrict__ x, RowSrc a, RowSrc b, const float* __restrict__ w, int n, float eps,
                              float* __restrict__ xn, float* __restrict__ xn_host, ActQ8 q, int do_q,
                              uint8_t* __restrict__ xq_host) {
    __shared__ float red[32];
    __shared__ float s_scale;
    __shared__ float sv[Q8K_MAXN];   // normalized row for the Q8_K blocks
    const int r = blockIdx.y;
    {
        const size_t ro = (size_t)r * n;
        x += ro;
        if (xn) xn += ro;
        if (xn_host) xn_host += ro;
        q.q += ro; q.d += ro / 32; q.s += ro / 32;
    }
    const bool has_a = a.dev || a.host, has_b = b.dev || b.host;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (has_a) v += src_at(a, r, n, i);   // residual add: x += layer output (GPU head + CPU tail)
        if (has_b) v += src_at(b, r, n, i);
        if (has_a) x[i] = v;   // the residual stream is updated in place
        ss += v * v;
    }
    if (!w) return;   // residual-only call: no norm
    const float tot = block_sum(ss, red);
    if (threadIdx.x == 0) s_scale = rsqrtf(tot / n + eps);   // 1 / RMS of the row
    __syncthreads();
    const float scale = s_scale;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int blk = warp; blk < n / 32; blk += nw) {
        const int i = blk * 32 + lane;
        const float v = x[i] * scale * w[i];
        if (xn) xn[i] = v;
        if (xn_host) xn_host[i] = v;
        if (xq_host) sv[i] = v;
        // int8 per 32 activations for the GPU GEMVs, produced while the values are in registers
        if (do_q) quant_block_q8(v, blk, q);
    }
    if (xq_host) {
        __syncthreads();
        // Q8_K blocks for the CPU kernels, written straight to mapped host memory
        for (int qb = warp; qb < n / 256; qb += nw) q8k_block_store(sv + qb * 256, xq_host + ((size_t)r * (n / 256) + qb) * Q8K_BYTES);
    }
    // make the host-visible writes visible before the event the CPU waits on
    if (xn_host || xq_host) __threadfence_system();
}

// rows (blockIdx.y) with stride ld; n valid elements per row
__global__ void k_swiglu(const float* __restrict__ g, const float* __restrict__ u, int n, int ld, float* __restrict__ h,
                         ActQ8 q, int do_q) {
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk >= n / 32) return;
    {
        const size_t ro = (size_t)blockIdx.y * ld;
        g += ro; u += ro;
        if (h) h += ro;
        q.q += ro; q.d += ro / 32; q.s += ro / 32;
    }
    const int i = blk * 32 + lane;
    const float v = siluf_(g[i]) * u[i];   // SwiGLU: silu(gate) * up
    if (h) h[i] = v;
    if (do_q) quant_block_q8(v, blk, q);
}

// ---------------------------------------------------------------------------
// Gated DeltaNet step: one CTA per V head (48), 512 threads
// ---------------------------------------------------------------------------

constexpr int DN_CH = 10240, DN_QK = 2048, DN_HD = 128, DN_KH = 16, DN_VH = 48;
constexpr int DN_MAXN = 7;   // new tokens per call (Engine::MAX_BATCH)

// Multi-token step with lazy commit. Token sequence = n_commit pending tokens (inputs from a save buffer: their
// outputs were produced by an earlier pass) followed by n_new new tokens (outputs written). The conv window and
// the state live in registers; the state and conv window are written back after `write_after` tokens
// (0 = unchanged): conv to buffer cin^1, state in place. New tokens' inputs are copied to the out-save buffer.
// The per-token loop body is the same code for every token, so results do not depend on how a token sequence
// is split into launches.
__global__ void __launch_bounds__(512) k_deltanet(DeltaNetParams p, int cin, DNTokens in, int n_commit, int n_new,
                                                  int write_after, float eps, float* __restrict__ o, ActQ8 oq,
                                                  int do_q) {
    const int h = blockIdx.x, kh = h % DN_KH, t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5;
    __shared__ float sq[DN_HD], sk[DN_HD], sv[DN_HD], so[DN_HD];
    __shared__ float red[32];
    __shared__ float s_misc[4];   // q scale, k scale, beta, decay
    __shared__ float s_xc[DN_MAXN][3 * DN_HD], s_z[DN_MAXN][DN_HD];   // new tokens' conv inputs / gate inputs
    const float* cs_in = p.conv_state + (size_t)cin * 3 * DN_CH;
    float* cs_out = p.conv_state + (size_t)(cin ^ 1) * 3 * DN_CH;
    // conv window of this thread's channel (q/k channels are shared by heads kh, kh+16, kh+32: owners write)
    int c = 0;
    float x0 = 0.f, x1 = 0.f, x2 = 0.f;
    float4 w = make_float4(0.f, 0.f, 0.f, 0.f);
    if (t < 3 * DN_HD) {
        // channel of this thread: q (by key head), k (by key head) or v (by value head)
        c = t < DN_HD ? kh * DN_HD + t : t < 2 * DN_HD ? DN_QK + kh * DN_HD + (t - DN_HD) : 2 * DN_QK + h * DN_HD + (t - 2 * DN_HD);
        x0 = cs_in[c]; x1 = cs_in[DN_CH + c]; x2 = cs_in[2 * DN_CH + c];
        w = *(const float4*)(p.conv_w + (size_t)c * 4);   // the 4 conv taps of the channel
    }
    // q/k channels are shared by 3 value heads: only the first one writes them back
    const bool owner = t < 3 * DN_HD && (t >= 2 * DN_HD || h < DN_KH);
    // Prefetch the new tokens' conv and gate inputs: the CPU share comes from mapped host memory, and reading it inside
    // the sequential token loop cost one PCIe round trip per token and stage.
    if (t < 3 * DN_HD) {
#pragma unroll
        for (int ni = 0; ni < DN_MAXN; ni++)
            if (ni < n_new) s_xc[ni][t] = c >= in.qkv_split ? in.qkv_host[(size_t)ni * in.qkv_ld + (c - in.qkv_split)]
                                                             : in.qkv[(size_t)ni * DN_CH + c];
    }
    if (warp < 4) {
        const int zi = h * DN_HD + warp * 32 + lane;
#pragma unroll
        for (int ni = 0; ni < DN_MAXN; ni++)
            if (ni < n_new) s_z[ni][warp * 32 + lane] = zi >= in.z_split ? in.z_host[(size_t)ni * in.z_ld + (zi - in.z_split)]
                                                                          : in.z[(size_t)ni * (DN_VH * DN_HD) + zi];
    }
    __syncthreads();
    // state: 8 columns per warp, column j contiguous over i
    float s[8][4];
    float* Sh = p.ssm_state + (size_t)h * DN_HD * DN_HD;
#pragma unroll
    for (int cc = 0; cc < 8; cc++) {
        const float* col = Sh + (size_t)(warp * 8 + cc) * DN_HD;
#pragma unroll
        // the 128x128 state stays in registers: 8 columns x 4 rows per lane
        for (int r = 0; r < 4; r++) s[cc][r] = col[lane + 32 * r];
    }
    const float oscale = rsqrtf((float)DN_HD);
    const int ntot = n_commit + n_new;
#pragma unroll 1
    for (int i = 0; i < ntot; i++) {
        const bool is_new = i >= n_commit;
        const int ni = i - n_commit;
        // pending tokens replay their saved inputs (lazy commit)
        const float* qkv = is_new ? in.qkv + (size_t)ni * DN_CH : in.sqkv + (size_t)i * DN_CH;
        const float* beta_raw = is_new ? in.beta + ni * 64 : in.sbeta + i * 64;
        const float* alpha_raw = is_new ? in.alpha + ni * 64 : in.salpha + i * 64;
        // 1) causal conv (width 4) + SiLU on this head's q/k/v channels
        if (t < 3 * DN_HD) {
            const float xc = is_new ? s_xc[ni][t] : qkv[c];
            // causal depthwise conv over the last 4 inputs, then SiLU
            const float val = siluf_(w.x * x0 + w.y * x1 + w.z * x2 + w.w * xc);
            if (t < DN_HD) sq[t] = val; else if (t < 2 * DN_HD) sk[t - DN_HD] = val; else sv[t - 2 * DN_HD] = val;
            x0 = x1; x1 = x2; x2 = xc;   // slide the window
            if (owner) {
                if (is_new && in.oqkv) in.oqkv[(size_t)ni * DN_CH + c] = xc;
                // conv window at the commit point, to the other buffer
                if (i == write_after - 1) { cs_out[c] = x0; cs_out[DN_CH + c] = x1; cs_out[2 * DN_CH + c] = x2; }
            }
        }
        if (is_new && in.obeta && t == 0) { in.obeta[ni * 64 + h] = beta_raw[h]; in.oalpha[ni * 64 + h] = alpha_raw[h]; }
        __syncthreads();
        // 2) L2 norms (warp 0: q, warp 1: k), gating scalars (warp 2)
        if (warp < 2) {
            const float* src = warp == 0 ? sq : sk;
            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; r++) { const float v = src[lane + 32 * r]; sum += v * v; }
            sum = warp_sum(sum);
            if (lane == 0) s_misc[warp] = 1.0f / fmaxf(sqrtf(sum), eps);   // L2 normalization of q and k
        } else if (warp == 2 && lane == 0) {
            const float a = alpha_raw[h] + p.dt_bias[h];
            const float sp = a > 20.0f ? a : logf(1.0f + expf(a));   // softplus (linear above 20, as in ggml)
            s_misc[2] = 1.0f / (1.0f + expf(-beta_raw[h]));   // beta = sigmoid: the write strength
            s_misc[3] = expf(sp * p.ssm_a[h]);   // decay = exp(softplus(alpha + dt_bias) * A), A < 0
        }
        __syncthreads();
        const float qs = s_misc[0], ks = s_misc[1], beta = s_misc[2], decay = s_misc[3];
        float kr[4], qr[4];
#pragma unroll
        for (int r = 0; r < 4; r++) { kr[r] = sk[lane + 32 * r] * ks; qr[r] = sq[lane + 32 * r] * qs; }
        // 3) recurrence
#pragma unroll
        for (int cc = 0; cc < 8; cc++) {
            const int j = warp * 8 + cc;
            float kv = 0.0f;
#pragma unroll
            // (S^T k)_j: what the state currently predicts for value j
            for (int r = 0; r < 4; r++) kv += s[cc][r] * kr[r];
            kv = warp_sum(kv);
            const float delta = (sv[j] - decay * kv) * beta;   // delta rule: correct towards v by the prediction error
            float oj = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const float nv = decay * s[cc][r] + kr[r] * delta;   // S = decay * S + k (v - decay S^T k)^T * beta
                s[cc][r] = nv;
                oj += nv * qr[r];   // output o_j = (S^T q)_j with the updated state
            }
            if (is_new) {
                oj = warp_sum(oj);
                if (lane == 0) so[j] = oj * oscale;   // scaled by 1 / sqrt(head dim)
            }
        }
        if (i == write_after - 1) {   // state at the commit point back to global memory
#pragma unroll
            for (int cc = 0; cc < 8; cc++) {
                float* col = Sh + (size_t)(warp * 8 + cc) * DN_HD;
#pragma unroll
                for (int r = 0; r < 4; r++) col[lane + 32 * r] = s[cc][r];
            }
        }
        // 4) new tokens: gated RMSNorm over the head: rmsnorm(o) * ssm_norm * silu(z)
        if (is_new) {
            __syncthreads();
            const float ss = t < DN_HD ? so[t] * so[t] : 0.0f;
            const float tot = block_sum(ss, red);
            const float rs = rsqrtf(tot / DN_HD + eps);
            if (warp < 4) {
                const int j = warp * 32 + lane;
                const float zv = s_z[ni][j];
                const float v = so[j] * rs * p.ssm_norm[j] * siluf_(zv);   // gated RMSNorm: norm(o) * weight * silu(z)
                o[(size_t)ni * (DN_VH * DN_HD) + h * DN_HD + j] = v;
                if (in.o_host) in.o_host[(size_t)ni * (DN_VH * DN_HD) + h * DN_HD + j] = v;
                if (do_q) {
                    const ActQ8 qr8{oq.q + (size_t)ni * (DN_VH * DN_HD), oq.d + (size_t)ni * (DN_VH * DN_HD / 32),
                                    oq.s + (size_t)ni * (DN_VH * DN_HD / 32)};
                    quant_block_q8(v, h * 4 + warp, qr8);
                }
            }
        }
        __syncthreads();
    }
    if (in.o_host) __threadfence_system();
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

constexpr int AH = 24, AKV = 4, AD = 256, AG = AH / AKV, NROT = 64;

// token r = blockIdx.y at cache position pos0 + r, rotary position rpos0 + r; qg / k / v rows from split device/host
// sources
__global__ void __launch_bounds__(256) k_attn_prep(RowSrc qg_s, RowSrc k_s, RowSrc v_s, const float* __restrict__ qn_w,
                                                   const float* __restrict__ kn_w, int pos0, int rpos0, float theta,
                                                   float eps, KVType kvt, KVLayer kv, size_t vb,
                                                   float* __restrict__ q_out, float* __restrict__ gate) {
    const int hb = blockIdx.x, t = threadIdx.x, r = blockIdx.y;
    const int pos = pos0 + r;
    q_out += (size_t)r * AH * AD;
    gate += (size_t)r * AH * AD;
    __shared__ float sx[AD];
    __shared__ float red[32];
    __shared__ float s_cos[NROT / 2], s_sin[NROT / 2];
    if (t < NROT / 2) {
        // angle in double: theta^(-2i/64) * pos, reduced mod 2pi, then float sincos
        const double inv = pow((double)theta, -2.0 * t / NROT);
        // reduce the angle before converting to float: positions reach 262144, too large for float sincos
        double ang = fmod((double)(rpos0 + r) * inv, 6.283185307179586476925286766559);
        double sn, cs;
        sincos(ang, &sn, &cs);
        s_cos[t] = (float)cs;
        s_sin[t] = (float)sn;
    }
    const bool is_q = hb < AH;
    const int head = is_q ? hb : hb - AH;
    // one CTA per head: 24 query heads then 4 KV heads; q and gate are interleaved per head
    float x = is_q ? src_at(qg_s, r, 2 * AH * AD, head * 2 * AD + t) : src_at(k_s, r, AKV * AD, head * AD + t);
    // the output gate (sigmoid later) is the second half of each q head
    if (is_q) gate[head * AD + t] = src_at(qg_s, r, 2 * AH * AD, head * 2 * AD + AD + t);
    const float tot = block_sum(x * x, red);
    x = x * rsqrtf(tot / AD + eps) * (is_q ? qn_w[t] : kn_w[t]);   // per-head RMSNorm of q and k
    sx[t] = x;
    __syncthreads();
    if (t < NROT / 2) {
        const float x0 = sx[t], x1 = sx[t + NROT / 2];
        sx[t] = x0 * s_cos[t] - x1 * s_sin[t];   // NeoX RoPE on the first 64 dims: pairs (i, i + 32)
        sx[t + NROT / 2] = x0 * s_sin[t] + x1 * s_cos[t];
    }
    __syncthreads();
    x = sx[t];
    if (is_q) {
        q_out[head * AD + t] = x;
    } else {
        // KV layout [head][position]: one head's keys are contiguous
        const size_t row = (size_t)head * kv.capacity + pos;
        // quantize to the KV type (Q8_0 / Q4_0 blocks of 32 with an FP16 scale) and store
        kv_store_row(kv.k + row * vb, kv.ks + row * 8, kvt, x, t);
        kv_store_row(kv.v + row * vb, kv.vs + row * 8, kvt, src_at(v_s, r, AKV * AD, head * AD + t), t);
    }
}

constexpr int ATT_CHUNK = 256;   // positions per CTA (= threads: phase 1 maps one position per thread)

// Partial attention for one KV head and its 6 query heads over the chunks s, s+nsplit, s+2*nsplit, ...
// (grid = nsplit x 4 KV heads, one wave). Per chunk of 256 positions:
//  phase 1: thread = position, full 256-dim dots with the 6 q vectors (q in shared memory, broadcast reads);
//  online softmax update per head (running max/sum shared by the CTA, one exp per score);
//  phase 2: warp w = dims [32w, 32w+32), lane = dim, accumulate P*V in registers across chunks.
// No shuffles in the inner loops. work layout per (split, kv head): [AG] m, [AG] l, [AG][AD] acc.
__global__ void __launch_bounds__(ATT_CHUNK) k_attn_chunk(const float* __restrict__ q, KVLayer kv, KVType kvt,
                                                          size_t vb, int n_pos, int nsplit, float* __restrict__ work) {
    const int split = blockIdx.x, g = blockIdx.y, t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5;
    __shared__ __align__(16) float sq[AG][AD];          // 6 KB
    __shared__ __align__(16) float sp[ATT_CHUNK][8];    // scores -> probabilities, [pos][head], 8 KB
    __shared__ float s_m[AG], s_l[AG], s_corr[AG];
    const float scale = rsqrtf((float)AD);
    // the 6 query heads sharing this KV head, pre-scaled by 1/sqrt(256)
    for (int i = t; i < AG * AD; i += ATT_CHUNK) sq[i / AD][i % AD] = q[(g * AG) * AD + i] * scale;
    if (t < AG) { s_m[t] = -FLT_MAX; s_l[t] = 0.0f; }
    float acc[AG];
#pragma unroll
    for (int h = 0; h < AG; h++) acc[h] = 0.0f;
    const int dim = warp * 32 + lane;
    const int n_chunks = (n_pos + ATT_CHUNK - 1) / ATT_CHUNK;
    __syncthreads();
    // this CTA's chunks: interleaved so all splits finish together
    for (int chunk = split; chunk < n_chunks; chunk += nsplit) {
        const int p0 = chunk * ATT_CHUNK;
        const int np = min(ATT_CHUNK, n_pos - p0);
        const size_t row0 = (size_t)g * kv.capacity + p0;
        // ---- phase 1: scores for position t
        {
            float s[AG];
#pragma unroll
            for (int h = 0; h < AG; h++) s[h] = 0.0f;
            if (t < np) {
                const uint8_t* krow = kv.k + (row0 + t) * vb;
                uint4 scl = make_uint4(0, 0, 0, 0);
                if (kvt != KVType::F16) scl = *(const uint4*)(kv.ks + (row0 + t) * 8);
                const uint16_t* sc16 = (const uint16_t*)&scl;
#pragma unroll 2
                for (int b = 0; b < AD / 32; b++) {
                    float part[AG];
#pragma unroll
                    for (int h = 0; h < AG; h++) part[h] = 0.0f;
                    float kf[4][8];
#pragma unroll
                    // dequantize 32 key values (one quant block)
                    for (int jj = 0; jj < 4; jj++) kv_vals8(krow, kvt, b * 4 + jj, kf[jj]);
#pragma unroll
                    for (int jj = 0; jj < 4; jj++) {
                        const int d0 = b * 32 + jj * 8;
#pragma unroll
                        for (int h = 0; h < AG; h++) {
                            const float4 qa = *(const float4*)&sq[h][d0];
                            const float4 qb = *(const float4*)&sq[h][d0 + 4];
                            part[h] += qa.x * kf[jj][0] + qa.y * kf[jj][1] + qa.z * kf[jj][2] + qa.w * kf[jj][3] +
                                       qb.x * kf[jj][4] + qb.y * kf[jj][5] + qb.z * kf[jj][6] + qb.w * kf[jj][7];
                        }
                    }
                    // block scale applied once per 32 dims, not per value
                    const float dk = kvt == KVType::F16 ? 1.0f : __half2float(__ushort_as_half(sc16[b]));
#pragma unroll
                    for (int h = 0; h < AG; h++) s[h] += dk * part[h];
                }
            }
#pragma unroll
            for (int h = 0; h < AG; h++) sp[t][h] = t < np ? s[h] : -FLT_MAX;   // positions past the end score -inf
        }
        __syncthreads();
        // ---- online softmax: warp h handles head h
        if (warp < AG) {
            float mx = -FLT_MAX;
            for (int p = lane; p < ATT_CHUNK; p += 32) mx = fmaxf(mx, sp[p][warp]);
            mx = warp_max(mx);
            const float m_old = s_m[warp];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
            for (int p = lane; p < ATT_CHUNK; p += 32) {
                // probabilities relative to the running max (online softmax)
                const float e = p < np ? __expf(sp[p][warp] - m_new) : 0.0f;
                sp[p][warp] = e;
                sum += e;
            }
            sum = warp_sum(sum);
            if (lane == 0) {
                const float corr = __expf(m_old - m_new);   // rescale factor for what was accumulated with the old max
                s_corr[warp] = corr;
                s_l[warp] = s_l[warp] * corr + sum;
                s_m[warp] = m_new;
            }
        }
        __syncthreads();
        // ---- phase 2: P*V for dim `dim`
#pragma unroll
        for (int h = 0; h < AG; h++) acc[h] *= s_corr[h];   // earlier chunks' P*V rescaled to the new max
        const uint8_t* vbase = kv.v + row0 * vb;
        const uint16_t* vsbase = kv.vs + row0 * 8;
#pragma unroll 8
        for (int p = 0; p < np; p++) {
            float val;
            const uint8_t* vrow = vbase + (size_t)p * vb;
            if (kvt == KVType::F16) {
                val = __half2float(((const __half*)vrow)[dim]);
            } else {
                // warp w covers dims 32w..32w+31 = one V quant block
                const float dv = __half2float(__ushort_as_half(vsbase[(size_t)p * 8 + warp]));
                if (kvt == KVType::Q8_0) val = dv * (float)((const int8_t*)vrow)[dim];
                else val = dv * (float)((int)((vrow[dim >> 1] >> (4 * (dim & 1))) & 0xF) - 8);   // Q4_0: nibble - 8
            }
            const float4 pa = *(const float4*)&sp[p][0];
            const float2 pb = *(const float2*)&sp[p][4];
            acc[0] += pa.x * val; acc[1] += pa.y * val; acc[2] += pa.z * val;
            acc[3] += pa.w * val; acc[4] += pb.x * val; acc[5] += pb.y * val;
        }
        __syncthreads();   // sp reused by the next chunk
    }
    float* wout = work + ((size_t)split * AKV + g) * (AG * 2 + AG * AD);
#pragma unroll
    for (int h = 0; h < AG; h++) wout[AG * 2 + h * AD + dim] = acc[h];
    // per split: max, sum and unnormalized output, merged by k_attn_combine
    if (t < AG) { wout[t] = s_m[t]; wout[AG + t] = s_l[t]; }
}

__global__ void __launch_bounds__(AD) k_attn_combine(const float* __restrict__ work, int n_chunks,
                                                     const float* __restrict__ gate, float* __restrict__ out, ActQ8 oq,
                                                     int do_q) {
    const int h = blockIdx.x, g = h / AG, hh = h % AG, d = threadIdx.x;
    const size_t stride = (size_t)AKV * (AG * 2 + AG * AD);
    const float* base = work + (size_t)g * (AG * 2 + AG * AD);
    float M = -FLT_MAX;
#pragma unroll 8
    for (int c = 0; c < n_chunks; c++) M = fmaxf(M, base[c * stride + hh]);   // global max over the splits
    float L = 0.0f, A = 0.0f;
#pragma unroll 8
    for (int c = 0; c < n_chunks; c++) {
        const float* w = base + c * stride;
        // each split's partial sums rescaled to the global max; empty splits add nothing
        const float f = w[AG + hh] > 0.0f ? __expf(w[hh] - M) : 0.0f;
        L += w[AG + hh] * f;
        A += w[AG * 2 + hh * AD + d] * f;
    }
    const float v = (A / L) * sigmoidf_(gate[h * AD + d]);   // softmax-weighted V, times the sigmoid output gate
    out[h * AD + d] = v;
    if (do_q) quant_block_q8(v, h * (AD / 32) + (d >> 5), oq);
}

// ---------------------------------------------------------------------------
// Tensor-core decode attention (mma.sync m16n8k8 f16 -> f32), NQ <= 4 consecutive queries per call.
// Rows of the mma = (query, head) pairs: one 16-row tile holds the 6 heads of a KV group for 2 queries
// (rows 12-15 padding); a second tile serves queries 2-3. A warp owns one fixed 256-key split, one row tile and
// all 256 output dims; K/V fragments are loaded straight from the KV cache (no shared memory): the head
// dimension is permuted (S: logical k-step ks / lane tg <-> dim 64*tg + 2*ks + j, O: n-tile nd / lane gr <->
// dim 32*gr + nd) so every thread reads contiguous runs (one quant block of V per key). Split boundaries are
// fixed multiples of 256 keys and fully masked tiles contribute exact zeros, so a query's result does not depend
// on how many other queries share the call (decode == decode_batch, bit for bit).
// ---------------------------------------------------------------------------
constexpr int ATT_SPLIT = 256;   // keys per split
constexpr int ATT_MAXQ = 8;   // queries per call (>= Engine::MAX_BATCH, even: 2 per 16-row MMA tile)
constexpr int ATT_WPB = 4;       // warps (splits) per CTA
constexpr int ATT_REC = 2 + AD;  // partial record per (split, kv head, query, head): m, l, O[256]

__device__ __forceinline__ void mma1688(float* c, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ uint32_t h2pack(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t*)&h;
}

// FP16 conversion by magic numbers: the half 0x6400 | x encodes 1024 + x (x < 1024), so subtracting the bias in
// half2 recovers x exactly and one HMUL2 by the fp16 scale rounds once — the same value as fp32 d*x -> fp16.
__device__ __forceinline__ __half2 h2u(uint32_t u) { return *(const __half2*)&u; }
__device__ __forceinline__ uint32_t u2h(__half2 h) { return *(const uint32_t*)&h; }

// Q8_0: 4 int8 (word w) -> half2 {x0,x1}*d, {x2,x3}*d
__device__ __forceinline__ void q8x4_h2(uint32_t w, __half2 d, uint32_t& lo, uint32_t& hi) {
    const uint32_t u = w ^ 0x80808080u;                         // biased bytes: x + 128
    const __half2 bias = __float2half2_rn(1152.0f);             // 1024 + 128
    lo = u2h(__hmul2(__hsub2(h2u(__byte_perm(u, 0x64646464u, 0x4140)), bias), d));
    hi = u2h(__hmul2(__hsub2(h2u(__byte_perm(u, 0x64646464u, 0x4342)), bias), d));
}
// Q4_0: 4 bytes (8 nibbles, element 2i low nibble of byte i) -> 4 half2 {(lo_i - 8)*d, (hi_i - 8)*d}
__device__ __forceinline__ void q4x8_h2(uint32_t w, __half2 d, uint32_t* out) {
    const __half2 bias = __float2half2_rn(1032.0f);             // 1024 + 8
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint32_t t = w >> (8 * k);
        const uint32_t h = (t & 0xFu) | ((t << 12) & 0xF0000u) | 0x64006400u;
        out[k] = u2h(__hmul2(__hsub2(h2u(h), bias), d));
    }
}

// 8 dims [64*tg + 8*c, +8) of one KV row (c = 0..7) as 4 half2 (dims 2i, 2i+1 in word i)
template <KVType kvt>
__device__ __forceinline__ void kv_k8(const uint8_t* vals, const uint16_t* scales, int tg, int c, uint32_t* w) {
    if constexpr (kvt == KVType::F16) {
        const uint4 u = *(const uint4*)(vals + 128 * tg + 16 * c);
        w[0] = u.x; w[1] = u.y; w[2] = u.z; w[3] = u.w;
    } else {
        const __half2 d = __half2half2(__ushort_as_half(scales[2 * tg + (c >> 2)]));
        if constexpr (kvt == KVType::Q8_0) {
            const uint2 u = *(const uint2*)(vals + 64 * tg + 8 * c);
            q8x4_h2(u.x, d, w[0], w[1]);
            q8x4_h2(u.y, d, w[2], w[3]);
        } else {
            q4x8_h2(*(const uint32_t*)(vals + 32 * tg + 4 * c), d, w);
        }
    }
}

// V B-fragments: dims [32*gr, 32*gr+32) of rows key0 and key1 -> 32 words {V0[d], V1[d]} (d = 32*gr + nd)
template <KVType kvt>
__device__ __forceinline__ void kv_v32(const uint8_t* r0, const uint8_t* r1, __half2 dd, int gr, uint32_t* w) {
    if constexpr (kvt == KVType::F16) {
        const uint4* a = (const uint4*)(r0 + 64 * gr);
        const uint4* b = (const uint4*)(r1 + 64 * gr);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint4 x = a[i], y = b[i];
            const uint32_t xs[4] = {x.x, x.y, x.z, x.w}, ys[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
            for (int k = 0; k < 4; k++) {
                w[8 * i + 2 * k] = __byte_perm(xs[k], ys[k], 0x5410);       // {x lo half, y lo half}
                w[8 * i + 2 * k + 1] = __byte_perm(xs[k], ys[k], 0x7632);   // {x hi half, y hi half}
            }
        }
    } else if constexpr (kvt == KVType::Q8_0) {
        const uint4* a = (const uint4*)(r0 + 32 * gr);
        const uint4* b = (const uint4*)(r1 + 32 * gr);
        const __half2 bias = __float2half2_rn(1152.0f);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint4 x = a[i], y = b[i];
            const uint32_t xs[4] = {x.x ^ 0x80808080u, x.y ^ 0x80808080u, x.z ^ 0x80808080u, x.w ^ 0x80808080u};
            const uint32_t ys[4] = {y.x ^ 0x80808080u, y.y ^ 0x80808080u, y.z ^ 0x80808080u, y.w ^ 0x80808080u};
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t lo = __byte_perm(xs[k], ys[k], 0x5140), hi = __byte_perm(xs[k], ys[k], 0x7362);   // [x0 y0 x1 y1], [x2 y2 x3 y3]
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const uint32_t pr = e == 0 ? lo : hi;
                    w[16 * i + 4 * k + 2 * e] = u2h(__hmul2(__hsub2(h2u(__byte_perm(pr, 0x64646464u, 0x4140)), bias), dd));
                    w[16 * i + 4 * k + 2 * e + 1] = u2h(__hmul2(__hsub2(h2u(__byte_perm(pr, 0x64646464u, 0x4342)), bias), dd));
                }
            }
        }
    } else {
        const uint4 x = *(const uint4*)(r0 + 16 * gr), y = *(const uint4*)(r1 + 16 * gr);
        const uint32_t xs[4] = {x.x, x.y, x.z, x.w}, ys[4] = {y.x, y.y, y.z, y.w};
        const __half2 bias = __float2half2_rn(1032.0f);
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int k = 0; k < 4; k++) {   // byte k of word i: dims 8i + 2k (low nibble), 8i + 2k + 1 (high nibble)
                const uint32_t bx = (xs[i] >> (8 * k)) & 0xFFu, by = (ys[i] >> (8 * k)) & 0xFFu;
                const uint32_t l = (bx & 0xFu) | ((by & 0xFu) << 16) | 0x64006400u;
                const uint32_t h = (bx >> 4) | ((by >> 4) << 16) | 0x64006400u;
                w[8 * i + 2 * k] = u2h(__hmul2(__hsub2(h2u(l), bias), dd));
                w[8 * i + 2 * k + 1] = u2h(__hmul2(__hsub2(h2u(h), bias), dd));
            }
        }
    }
}

template <KVType kvt>
__global__ void __launch_bounds__(ATT_WPB * 32) k_attn_dec_mma(const float* __restrict__ q, KVLayer kv, size_t vb,
                                                              int pos0, int nq, int nsplit, float* __restrict__ work) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int split = blockIdx.x * ATT_WPB + warp;   // each warp owns one 256-key split
    if (split >= nsplit) return;
    const int g = blockIdx.y, rt = blockIdx.z;
    const int gr = lane >> 2, tg = lane & 3;   // mma fragment coordinates: row group (0-7) and thread in group (0-3)
    // this lane's two rows: r = gr and gr + 8 of tile rt -> (query, head)
    int qt[2], hd[2], qp[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = gr + 8 * i;
        qt[i] = 2 * rt + r / AG;   // rows 0-5: query 2rt, heads 0-5; rows 6-11: query 2rt+1
        hd[i] = r % AG;
        const bool valid = r < 2 * AG && qt[i] < nq;
        qp[i] = valid ? pos0 + qt[i] : -1;   // last visible key (-1: padding row, sees nothing)
    }
    // Q A-fragments, permuted dims (k-step ks, lane tg <-> dims 64*tg + 2*ks + {0,1}), pre-scaled by 1/16
    uint32_t qa[AD / 8][2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const float* qrow = q + ((size_t)(qp[i] >= 0 ? qt[i] : 0) * AH + g * AG + hd[i]) * AD + 64 * tg;
#pragma unroll
        for (int ks = 0; ks < AD / 8; ks++) {
            const float2 x = qp[i] >= 0 ? *(const float2*)(qrow + 2 * ks) : make_float2(0.f, 0.f);
            // 1/16 = 1/sqrt(256): the softmax scale, folded into Q (exact in FP16)
            qa[ks][i] = h2pack(x.x * 0.0625f, x.y * 0.0625f);
        }
    }
    // O C-fragments: n-tile nd, column n <-> dim 32*n + nd
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;
    const int kmax = pos0 + nq - 1;                  // last key any query of the call may see
    // fixed split boundaries: results do not depend on how many queries share the call
    const int k_begin = split * ATT_SPLIT, k_end = min(k_begin + ATT_SPLIT, kmax + 1);
    const size_t rowbase = (size_t)g * kv.capacity;
    for (int kb = k_begin; kb < k_end; kb += 32) {
        // ---- S = Q K^T over 4 n-tiles of 8 keys
        float sc[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.0f;
            const int key = kb + 8 * nt + gr;
            const bool live = key <= kmax;   // keys past the window read row 0 and are zeroed below
            const size_t row = rowbase + (live ? key : 0);
            const uint8_t* kr = kv.k + row * vb;
            const uint16_t* ksc = kv.ks + row * 8;
#pragma unroll
            for (int c = 0; c < 8; c++) {
                uint32_t kf[4];
                if (live) kv_k8<kvt>(kr, ksc, tg, c, kf);
                else kf[0] = kf[1] = kf[2] = kf[3] = 0u;
#pragma unroll
                // 16 rows x 8 keys x 8 dims per instruction, FP32 accumulation
                for (int j = 0; j < 4; j++) mma1688(sc[nt], qa[4 * c + j][0], qa[4 * c + j][1], kf[j]);
            }
        }
        // ---- causal mask + online softmax (a row lives in the 4 lanes of a quad)
        float mx0 = -FLT_MAX, mx1 = -FLT_MAX;
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const int kj = kb + 8 * nt + 2 * tg;
            if (kj > qp[0]) sc[nt][0] = -FLT_MAX;   // causal mask: query at position p sees keys 0..p
            if (kj + 1 > qp[0]) sc[nt][1] = -FLT_MAX;
            if (kj > qp[1]) sc[nt][2] = -FLT_MAX;
            if (kj + 1 > qp[1]) sc[nt][3] = -FLT_MAX;
            mx0 = fmaxf(mx0, fmaxf(sc[nt][0], sc[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(sc[nt][2], sc[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));   // row max over the 4 lanes of the quad
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        // rescale of the running sums; a row that has seen nothing stays at 1
        const float c0 = mn0 == -FLT_MAX ? 1.0f : __expf(m0 - mn0);
        const float c1 = mn1 == -FLT_MAX ? 1.0f : __expf(m1 - mn1);
        uint32_t pa[4][2];
        float rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            // masked scores contribute an exact 0
            const float e0 = sc[nt][0] == -FLT_MAX ? 0.0f : __expf(sc[nt][0] - mn0);
            const float e1 = sc[nt][1] == -FLT_MAX ? 0.0f : __expf(sc[nt][1] - mn0);
            const float e2 = sc[nt][2] == -FLT_MAX ? 0.0f : __expf(sc[nt][2] - mn1);
            const float e3 = sc[nt][3] == -FLT_MAX ? 0.0f : __expf(sc[nt][3] - mn1);
            rs0 += e0 + e1;
            rs1 += e2 + e3;
            pa[nt][0] = h2pack(e0, e1);   // probabilities become the FP16 A operand of P V
            pa[nt][1] = h2pack(e2, e3);
        }
        l0 = l0 * c0 + rs0;
        l1 = l1 * c1 + rs1;
        m0 = mn0;
        m1 = mn1;
#pragma unroll
        // rescale the accumulated output to the new max
        for (int i = 0; i < 32; i++) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
        // ---- O += P V : k-step kk = keys 8kk + 2tg + {0,1}
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const int key0 = kb + 8 * kk + 2 * tg;
            uint32_t vf[32];
            if (key0 + 1 <= kmax) {
                const size_t r0 = rowbase + key0;
                __half2 dd = __float2half2_rn(1.0f);
                // the two keys' V scales for this lane's 32-dim block
                if (kvt != KVType::F16) dd = __halves2half2(__ushort_as_half(kv.vs[r0 * 8 + (gr >> 0)]), __ushort_as_half(kv.vs[(r0 + 1) * 8 + gr]));
                kv_v32<kvt>(kv.v + r0 * vb, kv.v + (r0 + 1) * vb, dd, gr, vf);
            } else if (key0 <= kmax) {   // second key beyond the window: zero row (its P is 0 anyway)
                const size_t r0 = rowbase + key0;
                __half2 dd = __float2half2_rn(1.0f);
                if (kvt != KVType::F16) dd = __halves2half2(__ushort_as_half(kv.vs[r0 * 8 + gr]), __float2half(0.0f));
                kv_v32<kvt>(kv.v + r0 * vb, kv.v + r0 * vb, dd, gr, vf);
                if (kvt == KVType::F16) for (int i = 0; i < 32; i++) vf[i] &= 0x0000FFFFu;
            } else {
#pragma unroll
                for (int i = 0; i < 32; i++) vf[i] = 0u;
            }
#pragma unroll
            // O[16 x 256] += P[16 x 8] V[8 x 256], 32 n-tiles
            for (int nd = 0; nd < 32; nd++) mma1688(o[nd], pa[kk][0], pa[kk][1], vf[nd]);
        }
    }
    // ---- partial record: l summed over the quad; O at its permuted dims (C column 2tg+j -> dim 32*(2tg+j) + nd)
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);   // each lane summed 1/4 of the row: reduce over the quad
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float ms[2] = {m0, m1}, ls[2] = {l0, l1};
#pragma unroll
    for (int i = 0; i < 2; i++) {
        if (qp[i] < 0) continue;   // padding rows write nothing
        float* rec = work + ((((size_t)split * AKV + g) * ATT_MAXQ + qt[i]) * AG + hd[i]) * ATT_REC;
        if (tg == 0) { rec[0] = ms[i]; rec[1] = ls[i]; }
#pragma unroll
        for (int nd = 0; nd < 32; nd++) {
            rec[2 + 32 * (2 * tg) + nd] = o[nd][2 * i];
            rec[2 + 32 * (2 * tg + 1) + nd] = o[nd][2 * i + 1];
        }
    }
}

// one CTA per (query, head): combine the splits in a fixed order, normalize, gate, quantize. Split records are
// gathered into shared memory in parallel; the weights are formed sequentially (fixed order, splits beyond a
// query's window add exact zeros), each dim's weighted sum is an unrolled loop of independent loads.
constexpr int ATT_MAXSPLIT = 1024;   // 262144 positions: the model's training context (attn_max_positions)
__global__ void __launch_bounds__(AD) k_attn_dec_combine(const float* __restrict__ work, int nsplit,
                                                         const float* __restrict__ gate, float* __restrict__ out,
                                                         float* __restrict__ out_host, ActQ8 oq, int do_q) {
    __shared__ float sm_m[ATT_MAXSPLIT], sm_l[ATT_MAXSPLIT], sm_f[ATT_MAXSPLIT];
    __shared__ float s_L;
    const int t = blockIdx.x / AH, h = blockIdx.x % AH, g = h / AG, hh = h % AG, d = threadIdx.x;
    const size_t sstride = (size_t)AKV * ATT_MAXQ * AG * ATT_REC;
    const float* base = work + (((size_t)g * ATT_MAXQ + t) * AG + hh) * ATT_REC;
    for (int sp = d; sp < nsplit; sp += AD) {
        sm_m[sp] = base[sp * sstride];
        sm_l[sp] = base[sp * sstride + 1];
    }
    __syncthreads();
    if (d == 0) {
        float M = -FLT_MAX;
        // global max over splits that saw keys
        for (int sp = 0; sp < nsplit; sp++) if (sm_l[sp] > 0.0f) M = fmaxf(M, sm_m[sp]);
        float L = 0.0f;
        for (int sp = 0; sp < nsplit; sp++) {
            const float f = sm_l[sp] > 0.0f ? __expf(sm_m[sp] - M) : 0.0f;   // weight of each split's partial result
            sm_f[sp] = f;
            L += sm_l[sp] * f;
        }
        s_L = L;
    }
    __syncthreads();
    float A = 0.0f;
#pragma unroll 8
    // fixed split order: bit-identical regardless of the batch
    for (int sp = 0; sp < nsplit; sp++) A += base[sp * sstride + 2 + d] * sm_f[sp];
    const size_t o = ((size_t)t * AH + h) * AD + d;
    const float v = (A / s_L) * sigmoidf_(gate[o]);   // normalize and apply the sigmoid output gate
    out[o] = v;
    if (out_host) { out_host[o] = v; __threadfence_system(); }
    if (do_q) {
        const ActQ8 qr{oq.q + (size_t)t * AH * AD, oq.d + (size_t)t * AH * AD / 32, oq.s + (size_t)t * AH * AD / 32};
        quant_block_q8(v, h * (AD / 32) + (d >> 5), qr);
    }
}

// ---------------------------------------------------------------------------
// Top-k (k <= 32): stage 1 per warp over 1024-element chunks, stage 2 one CTA
// ---------------------------------------------------------------------------

constexpr int TK = 32;

// Select the top-k of the values held by a warp (each lane owns `cnt` (v,i) pairs in registers).
template <int CNT>
__device__ void warp_topk(float (&v)[CNT], int (&ix)[CNT], int k, float* ov, int* oi) {
    const int lane = threadIdx.x & 31;
    for (int r = 0; r < k; r++) {
        float bv = -FLT_MAX;
        int bslot = 0;
#pragma unroll
        for (int j = 0; j < CNT; j++) if (v[j] > bv) { bv = v[j]; bslot = j; }   // best candidate of this lane
        float wv = bv;
        int wl = lane;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ovv = __shfl_xor_sync(0xffffffffu, wv, o);
            const int ol = __shfl_xor_sync(0xffffffffu, wl, o);
            // warp argmax; ties go to the lower lane (deterministic)
            if (ovv > wv || (ovv == wv && ol < wl)) { wv = ovv; wl = ol; }
        }
        int widx = 0;
#pragma unroll
        for (int j = 0; j < CNT; j++) if (j == bslot) widx = ix[j];
        widx = __shfl_sync(0xffffffffu, widx, wl);   // the winner's token id, from the lane that holds it
        if (lane == wl) {
#pragma unroll
            // remove the winner and select again: k rounds
            for (int j = 0; j < CNT; j++) if (j == bslot) v[j] = -FLT_MAX;
        }
        if (lane == 0) { ov[r] = wv; oi[r] = widx; }
    }
}

__global__ void k_topk_stage1(const float* __restrict__ x, int n, int k, int offset, float* __restrict__ cv,
                              int* __restrict__ ci) {
    const int lane = threadIdx.x & 31;
    const int wchunk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int base = wchunk * 1024;
    if (base >= n) return;
    float v[32];
    int ix[32];
#pragma unroll
    for (int j = 0; j < 32; j++) {
        const int i = base + j * 32 + lane;
        v[j] = i < n ? x[i] : -FLT_MAX;
        ix[j] = i + offset;
    }
    (void)k;
    // stage 1: top 32 of each 1024 logits, one warp per chunk
    warp_topk<32>(v, ix, TK, cv + (size_t)wchunk * TK, ci + (size_t)wchunk * TK);
}

__global__ void k_topk_stage2(const float* __restrict__ cv, const int* __restrict__ ci, int ncand, int k,
                              float* __restrict__ ov, int* __restrict__ oi) {
    // one warp: repeated selection over all candidates held 32 per lane per pass
    __shared__ float sv[TK];
    __shared__ int si[TK];
    const int lane = threadIdx.x;
    // simple approach: each lane scans a strided subset keeping a sorted top-k list in registers (k <= 32)
    float lv[TK];
    int li[TK];
#pragma unroll
    for (int r = 0; r < TK; r++) { lv[r] = -FLT_MAX; li[r] = 0x7fffffff; }
    for (int c = lane; c < ncand; c += 32) {
        float val = cv[c];
        int id = ci[c];
        // below this lane's 32nd best: skip the insertion
        if (!(val > lv[TK - 1] || (val == lv[TK - 1] && id < li[TK - 1]))) continue;
#pragma unroll
        for (int r = 0; r < TK; r++) {
            if (val > lv[r] || (val == lv[r] && id < li[r])) {
                const float tv = lv[r]; const int ti = li[r];
                lv[r] = val; li[r] = id;
                val = tv; id = ti;
            }
        }
    }
    warp_topk<TK>(lv, li, k, sv, si);   // stage 2: merge the lanes' sorted lists
    __syncwarp();
    if (lane < k) { ov[lane] = sv[lane]; oi[lane] = si[lane]; }
}

// element-wise: offsets into the activation buffers are arbitrary (no vector alignment guarantee)
__global__ void k_copy_floats(float* __restrict__ dst, int dst_ld, const float* __restrict__ src, int src_ld, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(size_t)blockIdx.y * dst_ld + i] = src[(size_t)blockIdx.y * src_ld + i];
    __threadfence_system();   // the destination may be mapped host memory read by the CPU
}

} // namespace

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

void copy_floats(float* dst, const float* src, int n, cudaStream_t st) { copy_rows(dst, n, src, n, n, 1, st); }

void copy_rows(float* dst, int dst_ld, const float* src, int src_ld, int n, int rows, cudaStream_t st) {
    if (n <= 0 || rows <= 0) return;
    const int threads = 256;
    k_copy_floats<<<dim3((n + threads - 1) / threads, rows), threads, 0, st>>>(dst, dst_ld, src, src_ld, n);
}


void add_rmsnorm(float* x, const float* a, const float* b, const float* w, int n, float eps, float* xn,
                 const ActQ8* q, cudaStream_t st) {
    add_rmsnorm_rows(x, a, b, w, n, 1, eps, xn, q, st);
}

void add_rmsnorm_rows(float* x, const float* a, const float* b, const float* w, int n, int rows, float eps, float* xn,
                      const ActQ8* q, cudaStream_t st) {
    add_rmsnorm_rows2(x, RowSrc::device(a, n), RowSrc::device(b, n), w, n, rows, eps, xn, nullptr, q, st);
}

void add_rmsnorm_rows2(float* x, const RowSrc& a, const RowSrc& b, const float* w, int n, int rows, float eps, float* xn,
                       float* xn_host, const ActQ8* q, cudaStream_t st, void* xq_host) {
    if (xq_host && (n % 256 || n > Q8K_MAXN)) { fprintf(stderr, "add_rmsnorm_rows2: Q8_K output needs n %% 256 == 0, n <= %d\n", Q8K_MAXN); abort(); }
    k_add_rmsnorm<<<dim3(1, rows), 1024, 0, st>>>(x, a, b, w, n, eps, xn, xn_host, q ? *q : ActQ8{}, q != nullptr,
                                                  (uint8_t*)xq_host);
}

void swiglu(const float* g, const float* u, int n, float* h, const ActQ8* q, cudaStream_t st) {
    swiglu_rows(g, u, n, n, 1, h, q, st);
}

void swiglu_rows(const float* g, const float* u, int n, int ld, int rows, float* h, const ActQ8* q, cudaStream_t st) {
    const int blocks = n / 32, wpb = 8;
    k_swiglu<<<dim3((blocks + wpb - 1) / wpb, rows), wpb * 32, 0, st>>>(g, u, n, ld, h, q ? *q : ActQ8{}, q != nullptr);
}

void deltanet_step(const DeltaNetParams& p, const float* qkv, const float* z, const float* beta_raw,
                   const float* alpha_raw, int parity, float eps, float* o, const ActQ8* oq, cudaStream_t st) {
    DNTokens in;
    in.qkv = qkv; in.z = z; in.beta = beta_raw; in.alpha = alpha_raw;
    deltanet_multi(p, parity, in, 0, 1, 1, eps, o, oq, st);
}

void deltanet_multi(const DeltaNetParams& p, int cin, const DNTokens& in, int n_commit, int n_new, int write_after,
                    float eps, float* o, const ActQ8* oq, cudaStream_t st) {
    if (n_commit + n_new <= 0) return;
    if (n_new > DN_MAXN) { fprintf(stderr, "deltanet_multi: %d new tokens > %d\n", n_new, DN_MAXN); abort(); }
    k_deltanet<<<DN_VH, 512, 0, st>>>(p, cin, in, n_commit, n_new, write_after, eps, o, oq ? *oq : ActQ8{},
                                      oq != nullptr);
}

size_t kv_value_bytes(KVType t) {
    switch (t) {
        case KVType::F16: return AD * 2;
        case KVType::Q8_0: return AD;
        case KVType::Q4_0: return AD / 2;
    }
    return 0;
}
size_t kv_scale_bytes(KVType t) { return t == KVType::F16 ? 0 : AD / 32 * 2; }

void attn_prep(const float* qg, const float* k, const float* v, const float* q_norm, const float* k_norm, int pos,
               float theta, float eps, KVType kvt, KVLayer& kv, float* q_out, float* gate, cudaStream_t st) {
    attn_prep_multi(RowSrc::device(qg, 2 * AH * AD), RowSrc::device(k, AKV * AD), RowSrc::device(v, AKV * AD), q_norm, k_norm,
                    pos, pos, 1, theta, eps, kvt, kv, q_out, gate, st);
}

void attn_prep_multi(const RowSrc& qg, const RowSrc& k, const RowSrc& v, const float* q_norm, const float* k_norm, int pos0,
                     int rope_pos0, int n, float theta, float eps, KVType kvt, KVLayer& kv, float* q_out, float* gate,
                     cudaStream_t st) {
    k_attn_prep<<<dim3(AH + AKV, n), AD, 0, st>>>(qg, k, v, q_norm, k_norm, pos0, rope_pos0, theta, eps, kvt, kv,
                                                  kv_value_bytes(kvt), q_out, gate);
}

constexpr int ATT_MAX_SPLIT = 40;   // one wave of 160 CTAs (4 KV heads) on 40 SMs
int attn_splits(int n_pos) { return std::min(ATT_MAX_SPLIT, (n_pos + ATT_CHUNK - 1) / ATT_CHUNK); }

int attn_max_positions() { return ATT_MAXSPLIT * ATT_SPLIT; }

size_t attn_work_bytes(int n_pos) {
    const size_t simple = (size_t)attn_splits(n_pos) * AKV * (AG * 2 + AG * AD) * sizeof(float);
    const size_t mma = (size_t)((n_pos + ATT_MAXQ + ATT_SPLIT - 1) / ATT_SPLIT) * AKV * ATT_MAXQ * AG * ATT_REC * sizeof(float);
    return std::max(simple, mma);
}

void attn_decode_multi(const float* q, const float* gate, const KVLayer& kv, KVType kvt, int pos0, int nq, void* work,
                       float* out, const ActQ8* oq, cudaStream_t st, float* out_host) {
    if (nq < 1 || nq > ATT_MAXQ) { fprintf(stderr, "attn_decode_multi: %d queries > %d\n", nq, ATT_MAXQ); std::abort(); }
    // one warp per 256 keys up to the last query's position
    const int nsplit = (pos0 + nq + ATT_SPLIT - 1) / ATT_SPLIT;
    if (nsplit > ATT_MAXSPLIT) {   // skipping the kernel would leave stale attention outputs: garbage text, no error
        fprintf(stderr, "attn_decode_multi: context beyond %d positions\n", ATT_MAXSPLIT * ATT_SPLIT);
        std::abort();
    }
    const int row_tiles = (nq + 1) / 2;   // 2 queries per 16-row MMA tile
    const dim3 grid((nsplit + ATT_WPB - 1) / ATT_WPB, AKV, row_tiles);
    const size_t vb = kv_value_bytes(kvt);
    switch (kvt) {
        case KVType::F16: k_attn_dec_mma<KVType::F16><<<grid, ATT_WPB * 32, 0, st>>>(q, kv, vb, pos0, nq, nsplit, (float*)work); break;
        case KVType::Q8_0: k_attn_dec_mma<KVType::Q8_0><<<grid, ATT_WPB * 32, 0, st>>>(q, kv, vb, pos0, nq, nsplit, (float*)work); break;
        case KVType::Q4_0: k_attn_dec_mma<KVType::Q4_0><<<grid, ATT_WPB * 32, 0, st>>>(q, kv, vb, pos0, nq, nsplit, (float*)work); break;
    }
    k_attn_dec_combine<<<nq * AH, AD, 0, st>>>((const float*)work, nsplit, gate, out, out_host, oq ? *oq : ActQ8{}, oq != nullptr);
}

void attn_decode(const float* q, const float* gate, const KVLayer& kv, KVType kvt, int n_pos, void* work, float* out,
                 const ActQ8* oq, cudaStream_t st) {
    attn_decode_multi(q, gate, kv, kvt, n_pos - 1, 1, work, out, oq, st);
}

void attn_decode_simple(const float* q, const float* gate, const KVLayer& kv, KVType kvt, int n_pos, void* work, float* out,
                        const ActQ8* oq, cudaStream_t st) {
    const int nsplit = attn_splits(n_pos);
    k_attn_chunk<<<dim3(nsplit, AKV), ATT_CHUNK, 0, st>>>(q, kv, kvt, kv_value_bytes(kvt), n_pos, nsplit, (float*)work);
    k_attn_combine<<<AH, AD, 0, st>>>((const float*)work, nsplit, gate, out, oq ? *oq : ActQ8{}, oq != nullptr);
}

size_t topk_work_bytes(int n) {
    const int nchunks = (n + 1023) / 1024;   // stage 1 keeps 32 candidates per 1024 logits
    return (size_t)nchunks * TK * (sizeof(float) + sizeof(int));
}

void topk(const float* logits, int n, int k, int index_offset, float* vals, int* idx, void* work, cudaStream_t st) {
    const int nchunks = (n + 1023) / 1024;
    float* cv = (float*)work;
    int* ci = (int*)(cv + (size_t)nchunks * TK);
    const int wpb = 4;
    k_topk_stage1<<<(nchunks + wpb - 1) / wpb, wpb * 32, 0, st>>>(logits, n, k, index_offset, cv, ci);
    k_topk_stage2<<<1, 32, 0, st>>>(cv, ci, nchunks * TK, k, vals, idx);
}

} // namespace cuda
} // namespace q38rt
