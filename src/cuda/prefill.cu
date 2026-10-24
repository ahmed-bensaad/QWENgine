/// @file prefill.cu
/// Batched prefill kernels (see prefill.cuh).

#include "cuda/prefill.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include "common/utils.h"
#include "cuda/common.cuh"
#include "cuda/iq_tables_dev.cuh"
#include "cuda/quant_dot.cuh"
#include "quant/quant_formats.h"

namespace q38rt {
namespace cuda {

namespace {

using namespace dev;

// ---------------------------------------------------------------------------
// Dequantize weights to FP16 (prefill GEMM operand)
// ---------------------------------------------------------------------------
// One thread per 32-weight sub-block, decoded to int8 by the GEMV decoders (QDot<T>::decode, lookup tables staged in
// shared memory), then weight = a * q + o in float: a * q and o are exact, so the FP16 value is the ggml weight rounded
// once. The kernel is bound by the FP16 writes: a warp's 32 sub-blocks are 2 KB of contiguous output, staged in shared
// memory and stored as 4 fully coalesced 512-byte rows (357 GB/s measured, vs 233 GB/s with each thread storing its
// own 64 bytes). Warps loop over groups of 32 sub-blocks, so a block stages its table once.
constexpr int DQ_WARPS = 8, DQ_STRIDE = 20;   // staging row per thread: 16 words + 4 padding (conflict-free uint4 stores)
template <QuantFormat T>
__global__ void __launch_bounds__(DQ_WARPS * 32) k_dequant_f16(const uint8_t* __restrict__ W, size_t row_bytes, int K,
                                                               int rows, __half* __restrict__ out) {
    constexpr int TB = QTable<T>::BYTES;
    __shared__ __align__(16) uint32_t tab[TB > 0 ? TB / 4 : 1];
    __shared__ __align__(16) uint32_t stage[DQ_WARPS][32 * DQ_STRIDE];
    if constexpr (TB > 0) {
        const uint32_t* src = (const uint32_t*)QTable<T>::src();
        for (int i = threadIdx.x; i < TB / 4; i += blockDim.x) tab[i] = src[i];
        __syncthreads();
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nsub = K / 32;
    const long long n = (long long)rows * nsub;
    uint32_t* st = stage[warp];
    // each warp takes 32 consecutive sub-blocks per step (grid-stride)
    for (long long g0 = ((long long)blockIdx.x * DQ_WARPS + warp) * 32; g0 < n; g0 += (long long)gridDim.x * DQ_WARPS * 32) {
        const long long gid = g0 + lane;
        if (gid < n) {
            const int r = (int)(gid / nsub), sb = (int)(gid % nsub);   // row and 32-weight sub-block of this thread
            int w[8];
            float a[4], o[4];
            // int8 values w and per-8 scale a / offset o, as the GEMV kernels decode them
            QDot<T>::decode(W + (size_t)r * row_bytes + (size_t)(sb / 8) * QDot<T>::BYTES, sb % 8, tab, w, a, o);
            uint32_t h[16];
#pragma unroll
            for (int k = 0; k < 8; ++k)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    // int8 -> float without I2F (quarter rate on sm_75): 1.5 * 2^23 + v is exact
                    const int v0 = (int)(int8_t)(w[k] >> (16 * e)), v1 = (int)(int8_t)(w[k] >> (16 * e + 8));
                    const float f0 = __int_as_float(0x4B400000 + v0) - 12582912.0f;
                    const float f1 = __int_as_float(0x4B400000 + v1) - 12582912.0f;
                    // plain product without an offset: keeps the reference's sign of zero (Q6_K: q = 0, negative scale)
                    const float ak = a[k >> 1], ok = o[k >> 1];
                    const __half2 hv = __floats2half2_rn(ok != 0.0f ? fmaf(f0, ak, ok) : f0 * ak,
                                                         ok != 0.0f ? fmaf(f1, ak, ok) : f1 * ak);
                    h[2 * k + e] = *(const uint32_t*)&hv;
                }
#pragma unroll
            for (int j = 0; j < 4; j++)
                // this thread's 32 halves (64 bytes) to the warp's staging rows
                *(uint4*)&st[lane * DQ_STRIDE + 4 * j] = make_uint4(h[4 * j], h[4 * j + 1], h[4 * j + 2], h[4 * j + 3]);
        }
        __syncwarp();
        // the warp's sub-blocks g0 .. g0+31 are contiguous in `out` (rows of K halves, densely packed)
        const int nchunk = (int)(min(32LL, n - g0) * 4);   // 16-byte chunks
        uint4* dst = (uint4*)(out + (size_t)g0 * 32);   // 32 sub-blocks x 32 halves = 2 KB of contiguous output
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int c = 32 * j + lane;
            // lane c stores 16-byte chunk c: 512 contiguous bytes per instruction
            if (c < nchunk) dst[c] = *(const uint4*)&st[(c >> 2) * DQ_STRIDE + 4 * (c & 3)];
        }
        __syncwarp();
    }
}

__global__ void k_bf16_to_f16(const uint16_t* __restrict__ W, long long n, __half* __restrict__ out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    // BF16 -> FP32 (exact) -> FP16 (rounded once)
    if (i < n) out[i] = __float2half_rn(__uint_as_float((uint32_t)W[i] << 16));
}

template <QuantFormat T>
void launch_dq(const void* W, size_t rb, int K, int rows, __half* out, cudaStream_t st) {
    const long long n = (long long)rows * (K / 32);
    // at most 1280 blocks (32 per SM); warps loop over the rest
    const unsigned blocks = (unsigned)std::min<long long>((n + DQ_WARPS * 32 - 1) / (DQ_WARPS * 32), 1280);
    k_dequant_f16<T><<<blocks, DQ_WARPS * 32, 0, st>>>((const uint8_t*)W, rb, K, rows, out);
}

// ---------------------------------------------------------------------------
// Row-batched elementwise
// ---------------------------------------------------------------------------
__global__ void k_rmsnorm_rows(float* __restrict__ X, const float* __restrict__ A, const float* __restrict__ C,
                               const float* __restrict__ w, int n, float eps, __half* __restrict__ o16,
                               float* __restrict__ o32) {
    __shared__ float red[32];
    float* x = X + (size_t)blockIdx.x * n;
    const float* a = A ? A + (size_t)blockIdx.x * n : nullptr;
    const float* c = C ? C + (size_t)blockIdx.x * n : nullptr;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (a) v += a[i];
        if (c) v += c[i];
        if (a || c) x[i] = v;   // residual adds update the stream in place
        ss += v * v;
    }
    const float tot = block_sum(ss, red);
    const float scale = rsqrtf(tot / n + eps);   // 1 / RMS of the row
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i] * scale * w[i];
        if (o16) o16[(size_t)blockIdx.x * n + i] = __float2half_rn(v);   // FP16 copy: the next GEMM's input
        if (o32) o32[(size_t)blockIdx.x * n + i] = v;
    }
}

// In place: H16 row b overwrites the first half of G's row b (row stride unchanged: ld floats = 2*ld halves). One
// block per row walks it in tiles of 4 * blockDim elements (float4 loads, 8-byte stores); tile k writes halves over G
// elements [i0/2, (i0+T)/2), all read by tiles <= k (the barrier orders this tile's reads before its writes; later
// tiles read only elements >= i0+T). n and ld are multiples of 4.
__global__ void k_swiglu_rows_inplace(float* G, const float* __restrict__ U, int n, int ld) {
    float* g = G + (size_t)blockIdx.x * ld;
    const float* u = U + (size_t)blockIdx.x * ld;
    __half* h = (__half*)g;
    for (int i0 = 0; i0 < n; i0 += 4 * blockDim.x) {
        const int i = i0 + 4 * threadIdx.x;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (i < n) {
            const float4 x = *(const float4*)(g + i), y = *(const float4*)(u + i);
            // silu(gate) * up for 4 elements
            v = make_float4(x.x / (1.0f + __expf(-x.x)) * y.x, x.y / (1.0f + __expf(-x.y)) * y.y,
                            x.z / (1.0f + __expf(-x.z)) * y.z, x.w / (1.0f + __expf(-x.w)) * y.w);
        }
        __syncthreads();   // all reads of this tile before the half writes that overlap it
        if (i < n) {
            const __half2 lo = __floats2half2_rn(v.x, v.y), hi = __floats2half2_rn(v.z, v.w);
            *(uint2*)(h + i) = make_uint2(*(const uint32_t*)&lo, *(const uint32_t*)&hi);
        }
    }
}


// ---------------------------------------------------------------------------
// Gated DeltaNet over B tokens
// ---------------------------------------------------------------------------
constexpr int CH = 10240, QK = 2048, VD = 6144, HD = 128, VH = 48, KH = 16;

struct DNWork {
    float* conv;    // [B][10240] after SiLU
    float* qn;      // [B][2048]
    float* kn;      // [B][2048]
    float* beta;    // [B][48]
    float* decay;   // [B][48]
    float* o;       // [B][6144] raw core output (scaled)
};
DNWork dn_split(void* work, int B) {
    float* f = (float*)work;
    DNWork w;
    w.conv = f; f += (size_t)B * CH;
    w.qn = f; f += (size_t)B * QK;
    w.kn = f; f += (size_t)B * QK;
    w.beta = f; f += (size_t)B * VH;
    w.decay = f;
    w.o = nullptr;   // caller-provided
    return w;
}

// out[t][c] = silu(sum_j w[c][j] * in(t-3+j)), in(i) = qkv[i][c] (i >= 0) or old state row 3+i
__global__ void k_dn_conv_rows(const float* __restrict__ qkv, const float* __restrict__ cs_old,
                               const float* __restrict__ conv_w, int B, float* __restrict__ out) {
    const int t = blockIdx.y;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < CH; c += gridDim.x * blockDim.x) {
        const float4 w = *(const float4*)(conv_w + (size_t)c * 4);
        float in[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = t - 3 + j;
            // the first 3 tokens reach back into the previous chunk's conv window
            in[j] = i >= 0 ? qkv[(size_t)i * CH + c] : cs_old[(size_t)(3 + i) * CH + c];
        }
        const float a = w.x * in[0] + w.y * in[1] + w.z * in[2] + w.w * in[3];
        out[(size_t)t * CH + c] = a / (1.0f + __expf(-a));   // SiLU
    }
}

// new state rows k = 0..2 = in(B-3+k); old and new may alias (all old values read first)
__global__ void k_dn_conv_state(const float* __restrict__ qkv, const float* cs_old, int B, float* cs_new) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CH) return;
    const float o0 = cs_old[c], o1 = cs_old[CH + c], o2 = cs_old[2 * CH + c];
    const float old[3] = {o0, o1, o2};
    float nv[3];
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int i = B - 3 + k;
        // the last 3 inputs become the window of the next chunk (or decode)
        nv[k] = i >= 0 ? qkv[(size_t)i * CH + c] : old[3 + i];
    }
#pragma unroll
    for (int k = 0; k < 3; k++) cs_new[(size_t)k * CH + c] = nv[k];
}

// per token: L2-normalized q/k (16 heads each), beta and decay (48 heads). 256 threads, warp w: 4 normalizations
__global__ void k_dn_prep_rows(DNWork w, const float* __restrict__ beta_raw, const float* __restrict__ alpha_raw,
                               const float* __restrict__ dt_bias, const float* __restrict__ ssm_a, float eps) {
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int jn = warp; jn < 2 * KH; jn += 8) {
        const bool is_k = jn >= KH;
        const int h = jn % KH;
        const float* src = w.conv + (size_t)t * CH + (is_k ? QK : 0) + h * HD;
        float v[4], s = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; r++) { v[r] = src[lane + 32 * r]; s += v[r] * v[r]; }
        s = warp_sum(s);
        const float inv = 1.0f / fmaxf(sqrtf(s), eps);   // L2 normalization of q and k per head
        float* dst = (is_k ? w.kn : w.qn) + (size_t)t * QK + h * HD;
#pragma unroll
        for (int r = 0; r < 4; r++) dst[lane + 32 * r] = v[r] * inv;
    }
    if (threadIdx.x < VH) {
        const int h = threadIdx.x;
        const float a = alpha_raw[(size_t)t * VH + h] + dt_bias[h];
        const float sp = a > 20.0f ? a : logf(1.0f + expf(a));   // softplus
        w.beta[(size_t)t * VH + h] = 1.0f / (1.0f + expf(-beta_raw[(size_t)t * VH + h]));
        w.decay[(size_t)t * VH + h] = expf(sp * ssm_a[h]);   // per-token decay of the state, A < 0
    }
}

// recurrence: grid (48 heads, 4 column groups of 32), 128 threads. Warp w owns columns 8w..8w+7 of its group; lane
// (c = lane / 4, g = lane % 4) holds rows 32g..32g+31 of column c, so the two 128-row dot products per column and
// token reduce over 4 lanes (2 shuffles, all 8 columns at once) instead of a 32-lane warp_sum per column (the
// shuffle throughput was the limit: ~10x fewer shuffles for the same FMAs). 192 blocks in one wave: 4-5 per SM (96
// blocks of 64 columns left the SMs at 2 or 3, the kernel waiting on the latter). The inputs
// of RT tokens are staged in shared memory (k / q rows padded by 4 floats per 32: conflict-free 16-byte loads); the
// warps walk the staged tokens without block barriers. Different summation order than the decode kernel (prefill is
// checked against the FP32 reference, not bit-for-bit).
constexpr int RT = 8, RCOL = 32, KPAD = HD + 16;   // padded k / q row: element i at i + 4 * (i / 32)
__global__ void __launch_bounds__(RCOL / 8 * 32, 6) k_dn_recur_rows(DNWork w, float* __restrict__ state, int B) {
    const int h = blockIdx.x, cg = blockIdx.y, kh = h % KH;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cl = lane >> 2, g = lane & 3;
    const int col = cg * RCOL + warp * 8 + cl;
    __shared__ __align__(16) float sk[RT][KPAD], sq[RT][KPAD];
    __shared__ float sv[RT][RCOL], so[RT][RCOL], sbeta[RT], sdecay[RT];
    float* Sh = state + (size_t)h * HD * HD;
    float s[32];
    {
        const float4* src = (const float4*)(Sh + (size_t)col * HD + 32 * g);
#pragma unroll
        // this lane's 32 rows of its state column, in registers for the whole chunk
        for (int i = 0; i < 8; i++) { const float4 v = src[i]; s[4 * i] = v.x; s[4 * i + 1] = v.y; s[4 * i + 2] = v.z; s[4 * i + 3] = v.w; }
    }
    const float oscale = rsqrtf((float)HD);
    for (int t0 = 0; t0 < B; t0 += RT) {   // tokens in groups of RT staged in shared memory
        const int nt = min(RT, B - t0);
        for (int i = threadIdx.x; i < nt * HD; i += blockDim.x) {
            // padded index: the 4 lanes of a column read different banks
            const int tt = i / HD, e = i % HD, pe = e + 4 * (e / 32);
            sk[tt][pe] = w.kn[(size_t)(t0 + tt) * QK + kh * HD + e];
            sq[tt][pe] = w.qn[(size_t)(t0 + tt) * QK + kh * HD + e];
        }
        for (int i = threadIdx.x; i < nt * RCOL; i += blockDim.x) {
            const int tt = i / RCOL, e = i % RCOL;
            sv[tt][e] = w.conv[(size_t)(t0 + tt) * CH + 2 * QK + h * HD + cg * RCOL + e];
        }
        if (threadIdx.x < nt) {
            sbeta[threadIdx.x] = w.beta[(size_t)(t0 + threadIdx.x) * VH + h];
            sdecay[threadIdx.x] = w.decay[(size_t)(t0 + threadIdx.x) * VH + h];
        }
        __syncthreads();
        for (int tt = 0; tt < nt; tt++) {
            const float4* kp = (const float4*)&sk[tt][36 * g];   // rows 32g.. at padded offset 36g
            const float4* qp = (const float4*)&sq[tt][36 * g];
            float ka = 0.0f, kb = 0.0f, kc = 0.0f, kd = 0.0f;   // 4 short FMA chains
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float4 k4 = kp[i];
                ka += s[4 * i] * k4.x; kb += s[4 * i + 1] * k4.y; kc += s[4 * i + 2] * k4.z; kd += s[4 * i + 3] * k4.w;
            }
            float kv = (ka + kb) + (kc + kd);   // partial (S^T k) over this lane's 32 rows
            kv += __shfl_xor_sync(0xffffffffu, kv, 1);   // sum over the 4 lanes of the column (128 rows)
            kv += __shfl_xor_sync(0xffffffffu, kv, 2);
            const float beta = sbeta[tt], decay = sdecay[tt];
            const float delta = (sv[tt][warp * 8 + cl] - decay * kv) * beta;   // delta rule, as in the decode kernel
            float oa = 0.0f, ob = 0.0f, oc = 0.0f, od = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float4 k4 = kp[i], q4 = qp[i];
                // state update fused with the output dot product
                s[4 * i] = decay * s[4 * i] + k4.x * delta;         oa += s[4 * i] * q4.x;
                s[4 * i + 1] = decay * s[4 * i + 1] + k4.y * delta; ob += s[4 * i + 1] * q4.y;
                s[4 * i + 2] = decay * s[4 * i + 2] + k4.z * delta; oc += s[4 * i + 2] * q4.z;
                s[4 * i + 3] = decay * s[4 * i + 3] + k4.w * delta; od += s[4 * i + 3] * q4.w;
            }
            float o = (oa + ob) + (oc + od);
            o += __shfl_xor_sync(0xffffffffu, o, 1);
            o += __shfl_xor_sync(0xffffffffu, o, 2);
            if (g == 0) so[tt][warp * 8 + cl] = o * oscale;   // one lane per column writes the output
        }
        __syncthreads();
        for (int i = threadIdx.x; i < nt * RCOL; i += blockDim.x) {
            const int tt = i / RCOL, e = i % RCOL;
            w.o[(size_t)(t0 + tt) * VD + h * HD + cg * RCOL + e] = so[tt][e];
        }
    }
    {
        float4* dst = (float4*)(Sh + (size_t)col * HD + 32 * g);
#pragma unroll
        // state back to global memory once per chunk
        for (int i = 0; i < 8; i++) dst[i] = make_float4(s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]);
    }
}

// gated RMSNorm: grid (B, 48), 128 threads
__global__ void k_dn_gnorm_rows(const float* __restrict__ o, const float* __restrict__ z, const float* __restrict__ nw,
                                float eps, __half* __restrict__ o16) {
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    const size_t idx = (size_t)t * VD + h * HD + j;
    const float v = o[idx];
    const float tot = block_sum(v * v, red);
    const float rs = rsqrtf(tot / HD + eps);
    const float zz = z[idx];
    // gated RMSNorm: norm(o) * weight * silu(z), FP16 for the output GEMM
    o16[idx] = __float2half_rn(v * rs * nw[j] * (zz / (1.0f + __expf(-zz))));
}

// ---------------------------------------------------------------------------
// Attention prep / prefill attention
// ---------------------------------------------------------------------------
constexpr int AH = 24, AKV = 4, AD = 256, AG = 6, NROT = 64;

__global__ void __launch_bounds__(256) k_attn_prep_rows(const float* __restrict__ qg, const float* __restrict__ k,
                                                        const float* __restrict__ v, const float* __restrict__ qn_w,
                                                        const float* __restrict__ kn_w, int p0,
                                                        const float* __restrict__ rope, float eps, KVType kvt, KVLayer kv,
                                                        size_t vb, float* __restrict__ q_out, float* __restrict__ gate) {
    const int t = blockIdx.y, hb = blockIdx.x, tid = threadIdx.x;
    const int pos = p0 + t;
    __shared__ float sx[AD];
    __shared__ float red[32];
    __shared__ float s_cos[NROT / 2], s_sin[NROT / 2];
    if (tid < NROT / 2) {
        s_cos[tid] = rope[(size_t)t * NROT + tid];   // precomputed per chunk (rope_table / rope_table_pos)
        s_sin[tid] = rope[(size_t)t * NROT + NROT / 2 + tid];
    }
    const bool is_q = hb < AH;
    const int head = is_q ? hb : hb - AH;
    const float* qrow = qg + (size_t)t * AH * 2 * AD;
    float x = is_q ? qrow[head * 2 * AD + tid] : k[(size_t)t * AKV * AD + head * AD + tid];
    // output gate = second half of each q head
    if (is_q) gate[((size_t)t * AH + head) * AD + tid] = qrow[head * 2 * AD + AD + tid];
    const float tot = block_sum(x * x, red);
    x = x * rsqrtf(tot / AD + eps) * (is_q ? qn_w[tid] : kn_w[tid]);
    sx[tid] = x;
    __syncthreads();
    if (tid < NROT / 2) {
        const float x0 = sx[tid], x1 = sx[tid + NROT / 2];
        sx[tid] = x0 * s_cos[tid] - x1 * s_sin[tid];
        sx[tid + NROT / 2] = x0 * s_sin[tid] + x1 * s_cos[tid];
    }
    __syncthreads();
    x = sx[tid];
    if (is_q) {
        q_out[((size_t)t * AH + head) * AD + tid] = x;
    } else {
        const size_t row = (size_t)head * kv.capacity + pos;
        kv_store_row(kv.k + row * vb, kv.ks + row * 8, kvt, x, tid);
        kv_store_row(kv.v + row * vb, kv.vs + row * 8, kvt, v[(size_t)t * AKV * AD + head * AD + tid], tid);
    }
}

// One block per token, one thread per rotary frequency. Double precision (1/32 rate on sm_75) runs here once per
// chunk instead of in each of the 28 head blocks of every attention layer.
__global__ void k_rope_table(int p0, float theta, float* __restrict__ rope) {
    const int t = blockIdx.x, i = threadIdx.x;
    const double inv = pow((double)theta, -2.0 * i / NROT);
    // angle in double, reduced mod 2 pi, then rounded to float once
    const double ang = fmod((double)(p0 + t) * inv, 6.283185307179586476925286766559);
    double sn, cs;
    sincos(ang, &sn, &cs);
    rope[(size_t)t * NROT + i] = (float)cs;
    rope[(size_t)t * NROT + NROT / 2 + i] = (float)sn;
}

// k_rope_table with per-token (t, h, w) positions: frequency i turns with component i % 3 (interleaved M-RoPE)
__global__ void k_rope_table_pos(const int* __restrict__ pos, float theta, float* __restrict__ rope) {
    const int t = blockIdx.x, i = threadIdx.x;
    const double inv = pow((double)theta, -2.0 * i / NROT);
    // frequency i turns with t, h or w by i % 3; text tokens have t = h = w
    const double ang = fmod((double)pos[t * 3 + i % 3] * inv, 6.283185307179586476925286766559);
    double sn, cs;
    sincos(ang, &sn, &cs);
    rope[(size_t)t * NROT + i] = (float)cs;
    rope[(size_t)t * NROT + NROT / 2 + i] = (float)sn;
}

// One CTA per (query token, KV head): same two-phase structure as decode, all chunks sequentially
// (B provides the parallelism), causal limit n_pos = p0 + t + 1, normalized output * sigmoid(gate) in FP16.
constexpr int PCH = 256;
__global__ void __launch_bounds__(PCH) k_attn_prefill(const float* __restrict__ q, const float* __restrict__ gate,
                                                      KVLayer kv, KVType kvt, size_t vb, int p0,
                                                      __half* __restrict__ out16) {
    const int g = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int n_pos = p0 + t + 1;   // causal: token t sees the cache up to its own position
    __shared__ __align__(16) float sq[AG][AD];
    __shared__ __align__(16) float sp[PCH][8];
    __shared__ float s_m[AG], s_l[AG], s_corr[AG];
    const float scale = rsqrtf((float)AD);
    for (int i = tid; i < AG * AD; i += PCH) sq[i / AD][i % AD] = q[((size_t)t * AH + g * AG) * AD + i] * scale;
    if (tid < AG) { s_m[tid] = -FLT_MAX; s_l[tid] = 0.0f; }
    float acc[AG];
#pragma unroll
    for (int h = 0; h < AG; h++) acc[h] = 0.0f;
    const int dim = warp * 32 + lane;
    __syncthreads();
    for (int p0c = 0; p0c < n_pos; p0c += PCH) {   // all chunks sequentially: the B query tokens are the parallelism
        const int np = min(PCH, n_pos - p0c);
        const size_t row0 = (size_t)g * kv.capacity + p0c;
        {
            float s[AG];
#pragma unroll
            for (int h = 0; h < AG; h++) s[h] = 0.0f;
            if (tid < np) {
                const uint8_t* krow = kv.k + (row0 + tid) * vb;
                uint4 scl = make_uint4(0, 0, 0, 0);
                if (kvt != KVType::F16) scl = *(const uint4*)(kv.ks + (row0 + tid) * 8);
                const uint16_t* sc16 = (const uint16_t*)&scl;
#pragma unroll 2
                for (int b = 0; b < AD / 32; b++) {
                    float part[AG];
#pragma unroll
                    for (int h = 0; h < AG; h++) part[h] = 0.0f;
                    float kf[4][8];
#pragma unroll
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
                    const float dk = kvt == KVType::F16 ? 1.0f : __half2float(__ushort_as_half(sc16[b]));
#pragma unroll
                    for (int h = 0; h < AG; h++) s[h] += dk * part[h];
                }
            }
#pragma unroll
            for (int h = 0; h < AG; h++) sp[tid][h] = tid < np ? s[h] : -FLT_MAX;
        }
        __syncthreads();
        if (warp < AG) {
            float mx = -FLT_MAX;
            for (int p = lane; p < PCH; p += 32) mx = fmaxf(mx, sp[p][warp]);
            mx = warp_max(mx);
            const float m_old = s_m[warp];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
            for (int p = lane; p < PCH; p += 32) {
                const float e = p < np ? __expf(sp[p][warp] - m_new) : 0.0f;
                sp[p][warp] = e;
                sum += e;
            }
            sum = warp_sum(sum);
            if (lane == 0) {
                const float corr = __expf(m_old - m_new);
                s_corr[warp] = corr;
                s_l[warp] = s_l[warp] * corr + sum;
                s_m[warp] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < AG; h++) acc[h] *= s_corr[h];
        const uint8_t* vbase = kv.v + row0 * vb;
        const uint16_t* vsbase = kv.vs + row0 * 8;
#pragma unroll 8
        for (int p = 0; p < np; p++) {
            float val;
            const uint8_t* vrow = vbase + (size_t)p * vb;
            if (kvt == KVType::F16) {
                val = __half2float(((const __half*)vrow)[dim]);
            } else {
                const float dv = __half2float(__ushort_as_half(vsbase[(size_t)p * 8 + warp]));
                if (kvt == KVType::Q8_0) val = dv * (float)((const int8_t*)vrow)[dim];
                else val = dv * (float)((int)((vrow[dim >> 1] >> (4 * (dim & 1))) & 0xF) - 8);
            }
            const float4 pa = *(const float4*)&sp[p][0];
            const float2 pb = *(const float2*)&sp[p][4];
            acc[0] += pa.x * val; acc[1] += pa.y * val; acc[2] += pa.z * val;
            acc[3] += pa.w * val; acc[4] += pb.x * val; acc[5] += pb.y * val;
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = 0; h < AG; h++) {
        const size_t idx = ((size_t)t * AH + g * AG + h) * AD + dim;
        const float gv = gate[idx];
        // normalize, sigmoid gate, FP16 for the output GEMM
        out16[idx] = __float2half_rn(acc[h] / s_l[h] * (1.0f / (1.0f + __expf(-gv))));
    }
}


// ---------------------------------------------------------------------------
// Tensor-core causal attention for prefill (mma.sync m16n8k8 f16 -> f32, ldmatrix), FlashAttention-2 style.
// CTA = (16 query positions, KV head g); warp h (0..5) owns query head g*6+h for those 16 queries.
// Q (pre-scaled by 1/16) lives in registers as A fragments, O (16 x 256) as 32 C fragments.
// Per 16-key tile: K and V dequantized to FP16 in shared memory (rows padded to 264 halves:
// conflict-free ldmatrix), S = Q K^T, online softmax in registers, O += P V with P reused as A fragments.
// ---------------------------------------------------------------------------
constexpr int TQ = 16, TK = 16, KROW = AD + 8, NPROD = 2;   // NPROD producer warps (see k_attn_prefill_mma)

__device__ __forceinline__ void mma16816h(float* c, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t*)&h;
}

// Dequantize 8 elements [8j, 8j+8) of a KV row (split layout) into 8 halves. FP16 by magic numbers: the half
// 0x6400|x is 1024+x, so subtracting the bias recovers x exactly and one HMUL2 by the fp16 scale rounds once
// (same value as fp32 d*x -> fp16).
template <KVType kvt>
__device__ __forceinline__ uint4 kv_row8_h(const uint8_t* vals, const uint16_t* scales, int j) {
    if constexpr (kvt == KVType::F16) {
        return ((const uint4*)vals)[j];
    } else {
        const __half2 d = __half2half2(__ushort_as_half(scales[j >> 2]));
        uint32_t w[4];
        if constexpr (kvt == KVType::Q8_0) {
            const uint2 u = ((const uint2*)vals)[j];
            const __half2 bias = __float2half2_rn(1152.0f);
            const uint32_t a = u.x ^ 0x80808080u, b = u.y ^ 0x80808080u;
            const uint32_t h[4] = {__byte_perm(a, 0x64646464u, 0x4140), __byte_perm(a, 0x64646464u, 0x4342),
                                   __byte_perm(b, 0x64646464u, 0x4140), __byte_perm(b, 0x64646464u, 0x4342)};
#pragma unroll
            for (int k = 0; k < 4; k++) { const __half2 r = __hmul2(__hsub2(*(const __half2*)&h[k], bias), d); w[k] = *(const uint32_t*)&r; }
        } else {
            const uint32_t u = ((const uint32_t*)vals)[j];
            const __half2 bias = __float2half2_rn(1032.0f);
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t t = u >> (8 * k);
                const uint32_t h = (t & 0xFu) | ((t << 12) & 0xF0000u) | 0x64006400u;
                const __half2 r = __hmul2(__hsub2(*(const __half2*)&h, bias), d);
                w[k] = *(const uint32_t*)&r;
            }
        }
        return make_uint4(w[0], w[1], w[2], w[3]);
    }
}

// Warp-specialized: warps 0..AG-1 compute (one head each, 16 query rows), NPROD producer warps load and dequantize
// the next 16-key K/V tile into the other half of a double buffer meanwhile (sm_75 has no cp.async; with one buffer the
// load phase was ~55% of the time). One block barrier per tile: the buffer being written is never the one being read.
// Split form of kv_row8_h for batched loads: raw words of chunk j (+ its scale), then the conversion.
template <KVType kvt> struct KVRaw { using T = uint32_t; };
template <> struct KVRaw<KVType::F16> { using T = uint4; };
template <> struct KVRaw<KVType::Q8_0> { using T = uint2; };
template <KVType kvt>
__device__ __forceinline__ typename KVRaw<kvt>::T kv_raw8(const uint8_t* vals, int j) {
    return ((const typename KVRaw<kvt>::T*)vals)[j];
}
template <KVType kvt>
__device__ __forceinline__ uint4 kv_conv8(typename KVRaw<kvt>::T u, uint16_t sbits) {
    if constexpr (kvt == KVType::F16) {
        return u;
    } else {
        const __half2 d = __half2half2(__ushort_as_half(sbits));
        uint32_t w[4];
        if constexpr (kvt == KVType::Q8_0) {
            const __half2 bias = __float2half2_rn(1152.0f);
            const uint32_t a = u.x ^ 0x80808080u, b = u.y ^ 0x80808080u;
            const uint32_t h[4] = {__byte_perm(a, 0x64646464u, 0x4140), __byte_perm(a, 0x64646464u, 0x4342),
                                   __byte_perm(b, 0x64646464u, 0x4140), __byte_perm(b, 0x64646464u, 0x4342)};
#pragma unroll
            for (int k = 0; k < 4; k++) { const __half2 r = __hmul2(__hsub2(*(const __half2*)&h[k], bias), d); w[k] = *(const uint32_t*)&r; }
        } else {
            const __half2 bias = __float2half2_rn(1032.0f);
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t t = u >> (8 * k);
                const uint32_t h = (t & 0xFu) | ((t << 12) & 0xF0000u) | 0x64006400u;
                const __half2 r = __hmul2(__hsub2(*(const __half2*)&h, bias), d);
                w[k] = *(const uint32_t*)&r;
            }
        }
        return make_uint4(w[0], w[1], w[2], w[3]);
    }
}

template <KVType kvt>
__global__ void __launch_bounds__((AG + NPROD) * 32, 1) k_attn_prefill_mma(const float* __restrict__ q, const float* __restrict__ gate,
                                                                KVLayer kv, size_t vb, int B, int p0,
                                                                __half* __restrict__ out16) {
    const int qt = blockIdx.x, g = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gr = lane >> 2, tg = lane & 3;          // mma groupID / threadID_in_group
    const bool producer = warp >= AG;
    const int head = g * AG + (producer ? 0 : warp);
    const int t0 = qt * TQ;
    __shared__ __align__(16) __half Ks[2][TK][KROW];
    __shared__ __align__(16) __half Vs[2][TK][KROW];
    // keys beyond the tile's last query are never needed
    const int kmax = p0 + min(t0 + TQ - 1, B - 1);    // last key any row of this tile may see
    const int n_tiles = kmax / TK + 1;
    const size_t rowbase = (size_t)g * kv.capacity;
    // producer: K and V rows [k0, k0 + TK) -> buffer b (rows past kmax zero: they are masked anyway). All loads of a
    // thread are issued before the first conversion (one memory latency per tile, not one per chunk). Thread p
    // handles chunk j = p % 32 of rows p / 32 + 2 i (i < 8) of K, then of V.
    constexpr int PER = 2 * TK * (AD / 8) / (NPROD * 32);
    static_assert(PER * NPROD * 32 == 2 * TK * (AD / 8) && (NPROD * 32) % (AD / 8) == 0, "producer layout");
    auto produce = [&](int k0, int b) {
        const int p = threadIdx.x - AG * 32, j = p % (AD / 8), rr = p / (AD / 8);
        typename KVRaw<kvt>::T raw[PER];
        uint16_t sb[PER];
#pragma unroll
        for (int i = 0; i < PER; i++) {
            // first half of the loads: K rows, second half: V rows
            const int isv = i >= PER / 2, r = rr + (NPROD * 32 / (AD / 8)) * (i % (PER / 2)), kpos = k0 + r;
            raw[i] = {};
            sb[i] = 0;
            if (kpos <= kmax) {
                const size_t row = rowbase + kpos;
                // issue every load before converting any (one memory latency per tile)
                raw[i] = kv_raw8<kvt>((isv ? kv.v : kv.k) + row * vb, j);
                if constexpr (kvt != KVType::F16) sb[i] = (isv ? kv.vs : kv.ks)[row * 8 + (j >> 2)];
            }
        }
#pragma unroll
        for (int i = 0; i < PER; i++) {
            const int isv = i >= PER / 2, r = rr + (NPROD * 32 / (AD / 8)) * (i % (PER / 2));
            // dequantized to FP16 in shared memory
            *(uint4*)(isv ? &Vs[b][r][8 * j] : &Ks[b][r][8 * j]) = kv_conv8<kvt>(raw[i], sb[i]);
        }
    };
    if (producer) produce(0, 0);   // the first tile before the loop; later tiles one step ahead

    // Q fragments (rows t0+gr and t0+gr+8 of this head), pre-scaled by 1/sqrt(256) = 1/16 (exact in fp16)
    uint32_t qa[AD / 8][2];
    float o[AD / 8][4];
    if (!producer) {
        // rows past the chunk end reuse the last row (their output is not stored)
        const int r0 = min(t0 + gr, B - 1), r1 = min(t0 + gr + 8, B - 1);
        const float* q0 = q + ((size_t)r0 * AH + head) * AD;
        const float* q1 = q + ((size_t)r1 * AH + head) * AD;
#pragma unroll
        for (int ks = 0; ks < AD / 8; ks++) {
            const float2 x0 = *(const float2*)(q0 + 8 * ks + 2 * tg);
            const float2 x1 = *(const float2*)(q1 + 8 * ks + 2 * tg);
            qa[ks][0] = pack_h2(x0.x * 0.0625f, x0.y * 0.0625f);
            qa[ks][1] = pack_h2(x1.x * 0.0625f, x1.y * 0.0625f);
        }
#pragma unroll
        for (int i = 0; i < AD / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    }
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;
    const int qp0 = p0 + t0 + gr, qp1 = qp0 + 8;      // query positions of this lane's two rows
    __syncthreads();

    for (int kt = 0; kt < n_tiles; kt++) {
        const int k0 = kt * TK, b = kt & 1;   // double buffer: compute reads b while the producers fill b ^ 1
        if (producer) {
            if (kt + 1 < n_tiles) produce(k0 + TK, b ^ 1);
        } else {
            // S = Q K^T : 2 n-tiles of 8 keys
            float sc[2][4];
#pragma unroll
            for (int nt = 0; nt < 2; nt++) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < AD / 8; ks += 4) {
#pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    uint32_t bf[4];
                    // ldmatrix: K fragments for 4 k-steps in one instruction
                    ldsm_x4(bf, &Ks[b][8 * nt + (lane & 7)][8 * (ks + (lane >> 3))]);
#pragma unroll
                    for (int i = 0; i < 4; i++) mma16816h(sc[nt], qa[ks + i][0], qa[ks + i][1], bf[i]);
                }
            }
            // causal mask + online softmax (rows gr and gr+8; a row lives in the 4 lanes of a quad)
            float mx0 = -FLT_MAX, mx1 = -FLT_MAX;
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                const int kj = k0 + 8 * nt + 2 * tg;
                if (kj > qp0) sc[nt][0] = -FLT_MAX;   // causal mask
                if (kj + 1 > qp0) sc[nt][1] = -FLT_MAX;
                if (kj > qp1) sc[nt][2] = -FLT_MAX;
                if (kj + 1 > qp1) sc[nt][3] = -FLT_MAX;
                mx0 = fmaxf(mx0, fmaxf(sc[nt][0], sc[nt][1]));
                mx1 = fmaxf(mx1, fmaxf(sc[nt][2], sc[nt][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
            // rows with no valid key yet keep m = -FLT_MAX: guard exp(-inf - -inf)
            const float c0 = mn0 == -FLT_MAX ? 1.0f : __expf(m0 - mn0);
            const float c1 = mn1 == -FLT_MAX ? 1.0f : __expf(m1 - mn1);
            uint32_t pa[2][2];
            float rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                const float e0 = sc[nt][0] == -FLT_MAX ? 0.0f : __expf(sc[nt][0] - mn0);
                const float e1 = sc[nt][1] == -FLT_MAX ? 0.0f : __expf(sc[nt][1] - mn0);
                const float e2 = sc[nt][2] == -FLT_MAX ? 0.0f : __expf(sc[nt][2] - mn1);
                const float e3 = sc[nt][3] == -FLT_MAX ? 0.0f : __expf(sc[nt][3] - mn1);
                rs0 += e0 + e1;
                rs1 += e2 + e3;
                pa[nt][0] = pack_h2(e0, e1);
                pa[nt][1] = pack_h2(e2, e3);
            }
            l0 = l0 * c0 + rs0;
            l1 = l1 * c1 + rs1;
            m0 = mn0;
            m1 = mn1;
            // rescale only if some row's max moved in this tile (c == 1 exactly otherwise)
            if (__any_sync(0xffffffffu, c0 != 1.0f || c1 != 1.0f)) {
#pragma unroll
                for (int i = 0; i < AD / 8; i++) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
            }
            // O += P V : k-steps = the 2 key groups of 8, n-tiles = 32 groups of 8 dims
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
#pragma unroll
                for (int dt = 0; dt < AD / 8; dt += 4) {
                    uint32_t bf[4];
                    // transposed load: V as the B operand
                    ldsm_x4_t(bf, &Vs[b][8 * kk + (lane & 7)][8 * (dt + (lane >> 3))]);
#pragma unroll
                    // P (FP16 probabilities) reused directly as the A fragments
                    for (int i = 0; i < 4; i++) mma16816h(o[dt + i], pa[kk][0], pa[kk][1], bf[i]);
                }
            }
        }
        __syncthreads();   // the only block barrier per tile: swaps the buffers
    }
    if (producer) return;
    // finalize: row sums over the quad, normalize, gate, store fp16
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float il0 = 1.0f / l0, il1 = 1.0f / l1;   // softmax normalization deferred to the end (FlashAttention-2)
    const int r0 = t0 + gr, r1 = t0 + gr + 8;
#pragma unroll
    for (int dt = 0; dt < AD / 8; dt++) {
        const int d = 8 * dt + 2 * tg;
        if (r0 < B) {
            const size_t idx = ((size_t)r0 * AH + head) * AD + d;
            const float2 gv = *(const float2*)(gate + idx);
            *(__half2*)(out16 + idx) = __floats2half2_rn(o[dt][0] * il0 / (1.0f + __expf(-gv.x)),
                                                         o[dt][1] * il0 / (1.0f + __expf(-gv.y)));
        }
        if (r1 < B) {
            const size_t idx = ((size_t)r1 * AH + head) * AD + d;
            const float2 gv = *(const float2*)(gate + idx);
            *(__half2*)(out16 + idx) = __floats2half2_rn(o[dt][2] * il1 / (1.0f + __expf(-gv.x)),
                                                         o[dt][3] * il1 / (1.0f + __expf(-gv.y)));
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

bool dequant_rows_f16(QuantFormat t, const void* W, size_t rb, int K, int rows, __half* out, cudaStream_t st) {
    switch (t) {
        case QuantFormat::IQ2_XXS: launch_dq<QuantFormat::IQ2_XXS>(W, rb, K, rows, out, st); return true;
        case QuantFormat::IQ2_XS: launch_dq<QuantFormat::IQ2_XS>(W, rb, K, rows, out, st); return true;
        case QuantFormat::IQ2_S: launch_dq<QuantFormat::IQ2_S>(W, rb, K, rows, out, st); return true;
        case QuantFormat::IQ3_XXS: launch_dq<QuantFormat::IQ3_XXS>(W, rb, K, rows, out, st); return true;
        case QuantFormat::IQ3_S: launch_dq<QuantFormat::IQ3_S>(W, rb, K, rows, out, st); return true;
        case QuantFormat::IQ1_S: launch_dq<QuantFormat::IQ1_S>(W, rb, K, rows, out, st); return true;
        case QuantFormat::IQ1_M: launch_dq<QuantFormat::IQ1_M>(W, rb, K, rows, out, st); return true;
        case QuantFormat::IQ4_XS: launch_dq<QuantFormat::IQ4_XS>(W, rb, K, rows, out, st); return true;
        case QuantFormat::Q2_K: launch_dq<QuantFormat::Q2_K>(W, rb, K, rows, out, st); return true;
        case QuantFormat::Q4_K: launch_dq<QuantFormat::Q4_K>(W, rb, K, rows, out, st); return true;
        case QuantFormat::Q6_K: launch_dq<QuantFormat::Q6_K>(W, rb, K, rows, out, st); return true;
        case QuantFormat::BF16: {
            const long long n = (long long)rows * K;
            k_bf16_to_f16<<<(unsigned)((n + 255) / 256), 256, 0, st>>>((const uint16_t*)W, n, out);
            return true;
        }
        default: return false;
    }
}

void* cublas_create() {
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { LOG_ERROR("cublasCreate failed"); return nullptr; }
    cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);   // FP16 tensor cores
    return h;
}

void cublas_destroy(void* h) { cublasDestroy((cublasHandle_t)h); }

bool cublas_bind(void* h, cudaStream_t st, void* workspace, size_t bytes) {
    // cublasSetStream resets the workspace to cuBLAS's own pool: set the stream first, then our buffer
    return cublasSetStream((cublasHandle_t)h, st) == CUBLAS_STATUS_SUCCESS &&
           cublasSetWorkspace((cublasHandle_t)h, workspace, bytes) == CUBLAS_STATUS_SUCCESS;
}

void gemm_f16(void* handle, const __half* W, int N, int K, const __half* X, int ldx, int B, float* Y, int ldy,
              bool accumulate, cudaStream_t st) {
    cublasHandle_t h = (cublasHandle_t)handle;   // bound to st by cublas_bind
    (void)st;
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    // column-major view: Y^T [N x B] = W^T(view of row-major W [N][K] as col-major [K][N]) * X^T [K x B]
    const cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, B, K, &alpha, W, CUDA_R_16F, K, X, CUDA_R_16F, ldx,
                                          &beta, Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) {   // a skipped GEMM would silently corrupt every later result
        LOG_ERROR("cublasGemmEx failed (%d) N=%d B=%d K=%d", (int)s, N, B, K);
        exit(1);
    }
}

void rmsnorm_rows(float* X, const float* A, const float* C, const float* w, int B, int n, float eps, __half* out16,
                  float* out32, cudaStream_t st) {
    k_rmsnorm_rows<<<B, 1024, 0, st>>>(X, A, C, w, n, eps, out16, out32);
}


void swiglu_rows_inplace(float* G, const float* U, int B, int n, int ld, cudaStream_t st) {
    k_swiglu_rows_inplace<<<B, 512, 0, st>>>(G, U, n, ld);
}


size_t dn_work_bytes(int B) { return (size_t)B * (CH + 2 * QK + 2 * VH) * sizeof(float); }

void deltanet_rows(const DeltaNetParams& p, const float* qkv, const float* z, const float* beta_raw,
                   const float* alpha_raw, int B, int cin, float eps, void* work, float* core_out, __half* o16,
                   cudaStream_t st) {
    DNWork w = dn_split(work, B);
    w.o = core_out;
    const float* cs_old = p.conv_state + (size_t)cin * 3 * CH;
    float* cs_new = p.conv_state + (size_t)(cin ^ 1) * 3 * CH;
    // 1) causal conv + SiLU of all B tokens
    k_dn_conv_rows<<<dim3((CH + 255) / 256, B), 256, 0, st>>>(qkv, cs_old, p.conv_w, B, w.conv);
    // 2) conv window for whatever follows (the other buffer)
    k_dn_conv_state<<<(CH + 255) / 256, 256, 0, st>>>(qkv, cs_old, B, cs_new);
    // 3) per token: normalized q/k, beta, decay
    k_dn_prep_rows<<<B, 256, 0, st>>>(w, beta_raw, alpha_raw, p.dt_bias, p.ssm_a, eps);
    // 4) the sequential recurrence (the only part that cannot be token-parallel)
    k_dn_recur_rows<<<dim3(VH, HD / RCOL), RCOL / 8 * 32, 0, st>>>(w, p.ssm_state, B);
    // 5) gated norm, FP16 input of the output projection
    k_dn_gnorm_rows<<<dim3(B, VH), HD, 0, st>>>(w.o, z, p.ssm_norm, eps, o16);
}

void rope_table(int p0, int B, float theta, float* rope, cudaStream_t st) {
    k_rope_table<<<B, NROT / 2, 0, st>>>(p0, theta, rope);
}

void rope_table_pos(const int* pos, int B, float theta, float* rope, cudaStream_t st) {
    k_rope_table_pos<<<B, NROT / 2, 0, st>>>(pos, theta, rope);
}

void attn_prep_rows(const float* qg, const float* k, const float* v, const float* q_norm, const float* k_norm, int B,
                    int p0, const float* rope, float eps, KVType kvt, KVLayer& kv, float* q, float* gate, cudaStream_t st) {
    k_attn_prep_rows<<<dim3(AH + AKV, B), AD, 0, st>>>(qg, k, v, q_norm, k_norm, p0, rope, eps, kvt, kv,
                                                       kv_value_bytes(kvt), q, gate);
}

void attn_prefill(const float* q, const float* gate, const KVLayer& kv, KVType kvt, int B, int p0, __half* out16,
                  cudaStream_t st) {
    const dim3 grid((B + TQ - 1) / TQ, AKV);   // one CTA per 16 queries and KV head
    const size_t vb = kv_value_bytes(kvt);
    switch (kvt) {
        case KVType::F16: k_attn_prefill_mma<KVType::F16><<<grid, (AG + NPROD) * 32, 0, st>>>(q, gate, kv, vb, B, p0, out16); break;
        case KVType::Q8_0: k_attn_prefill_mma<KVType::Q8_0><<<grid, (AG + NPROD) * 32, 0, st>>>(q, gate, kv, vb, B, p0, out16); break;
        case KVType::Q4_0: k_attn_prefill_mma<KVType::Q4_0><<<grid, (AG + NPROD) * 32, 0, st>>>(q, gate, kv, vb, B, p0, out16); break;
    }
}

void attn_prefill_simple(const float* q, const float* gate, const KVLayer& kv, KVType kvt, int B, int p0, __half* out16,
                         cudaStream_t st) {
    k_attn_prefill<<<dim3(AKV, B), PCH, 0, st>>>(q, gate, kv, kvt, kv_value_bytes(kvt), p0, out16);
}

} // namespace cuda
} // namespace q38rt
