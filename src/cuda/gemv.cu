/// @file gemv.cu
/// Decode-path quantized GEMV for 1-4 token columns: y_c = W x_c with W in a ggml format and each x_c quantized
/// to int8 per 32 elements (dp4a dot products, per-block scales applied in float). The activations are staged once
/// per CTA in shared memory; each warp owns a row at a time and its lanes split K into 32-weight sub-blocks; each
/// weight sub-block is decoded once and dotted with all columns (per-column results bit-identical to 1 column).

#include "cuda/gemv.cuh"
#include "cuda/quant_dot.cuh"
#include "common/utils.h"

namespace q38rt {
namespace cuda {

namespace {

constexpr int GEMV_WARPS = 8;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    // butterfly: every lane ends with the full sum
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// One warp per 32-element block: d = amax/127, q = round(x/d), s = d*sum(q).
__global__ void k_quantize_act_q8(const float* __restrict__ x, int n_blocks, int8_t* __restrict__ q,
                                  float* __restrict__ d, float* __restrict__ s) {
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (blk >= n_blocks) return;
    const float v = x[(size_t)blk * 32 + lane];
    const float amax = warp_max(fabsf(v));   // largest magnitude of the 32 values
    const float dd = amax / 127.0f;
    const int qi = dd > 0.0f ? __float2int_rn(v / dd) : 0;   // round to nearest: |qi| <= 127; an all-zero block stays 0
    q[(size_t)blk * 32 + lane] = (int8_t)qi;
    float sum = warp_sum((float)qi);
    if (lane == 0) {
        d[blk] = dd;
        // d * sum(q): Q2_K / Q4_K-style formats subtract min * this instead of one product per weight
        s[blk] = dd * sum;
    }
}

// y[c*y_ld + row] (+)= W[row,:] . x_c for NB activation columns (x_c = act + c*act_ld). With STAGE the
// activations are copied to shared memory once per CTA, else read through L1 (identical arithmetic).
template <QuantFormat T, int NB, bool STAGE>
__global__ void __launch_bounds__(GEMV_WARPS * 32)
k_gemv_q8(const uint8_t* __restrict__ W, size_t row_bytes, int K, int N, const int8_t* __restrict__ aq,
          const float* __restrict__ ad, const float* __restrict__ as, int act_ld, float* __restrict__ y, int y_ld,
          int accumulate) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int nsb = K / 32;                       // 32-weight sub-blocks per row
    constexpr int TB = QTable<T>::BYTES;          // lookup table (16-byte multiple)
    uint8_t* stab = smem;                         // [TB]
    int8_t* sq = (int8_t*)(smem + TB);            // [NB][K]
    float* sd = (float*)(smem + TB + NB * K);     // [NB][nsb]
    float* ss = sd + NB * nsb;                    // [NB][nsb]
    if constexpr (TB > 0) {
        const int4* src = (const int4*)QTable<T>::src();
        // codebook to shared memory: random-access lookups would serialize in global memory
        for (int i = threadIdx.x; i < TB / 16; i += blockDim.x) ((int4*)stab)[i] = src[i];
    }
    if constexpr (STAGE) {
        for (int c = 0; c < NB; c++) {
            const int4* src = (const int4*)(aq + (size_t)c * act_ld);
            int4* dst = (int4*)(sq + c * K);
            // int8 activations of column c, 16 bytes per thread
            for (int i = threadIdx.x; i < K / 16; i += blockDim.x) dst[i] = src[i];
            for (int i = threadIdx.x; i < nsb; i += blockDim.x) {
                sd[c * nsb + i] = ad[(size_t)c * (act_ld / 32) + i];
                ss[c * nsb + i] = as[(size_t)c * (act_ld / 32) + i];
            }
        }
    }
    __syncthreads();   // tables and activations visible to all warps
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // grid-stride over rows: one warp per row at a time
    for (int row = blockIdx.x * GEMV_WARPS + warp; row < N; row += gridDim.x * GEMV_WARPS) {
        const uint8_t* wr = W + (size_t)row * row_bytes;
        float acc[NB];
#pragma unroll
        for (int c = 0; c < NB; c++) acc[c] = 0.0f;
#pragma unroll 4
        for (int sb = lane; sb < nsb; sb += 32) {   // lanes take interleaved 32-weight sub-blocks of the row
            int q8[NB][8];   // explicit 16-byte loads (c*K / c*act_ld offsets are 16-byte multiples)
            float d8[NB], s8[NB];
#pragma unroll
            for (int c = 0; c < NB; c++) {
                int4 lo, hi;
                if constexpr (STAGE) {
                    const int4* src = (const int4*)(sq + c * K + sb * 32);
                    lo = src[0];
                    hi = src[1];
                    d8[c] = sd[c * nsb + sb];
                    s8[c] = ss[c * nsb + sb];
                } else {
                    const int4* src = (const int4*)(aq + (size_t)c * act_ld + sb * 32);
                    lo = __ldg(src);
                    hi = __ldg(src + 1);
                    d8[c] = __ldg(ad + (size_t)c * (act_ld / 32) + sb);
                    s8[c] = __ldg(as + (size_t)c * (act_ld / 32) + sb);
                }
                q8[c][0] = lo.x; q8[c][1] = lo.y; q8[c][2] = lo.z; q8[c][3] = lo.w;
                q8[c][4] = hi.x; q8[c][5] = hi.y; q8[c][6] = hi.z; q8[c][7] = hi.w;
            }
            // a ggml block holds 8 sub-blocks of 32 (256 weights): block sb/8, sub-block sb%8; decoded once for all NB
            // columns
            QDot<T>::template dotn<NB>(wr + (size_t)(sb >> 3) * QDot<T>::BYTES, sb & 7, q8, d8, s8, stab, acc);
        }
#pragma unroll
        for (int c = 0; c < NB; c++) {
            const float v = warp_sum(acc[c]);   // row total over the 32 lanes
            if (lane == 0) {
                float* yo = y + (size_t)c * y_ld + row;
                *yo = accumulate ? *yo + v : v;   // accumulate: add to a partial sum already in y (split rows)
            }
        }
    }
}

__global__ void k_gemv_bf16(const uint16_t* __restrict__ W, int K, int N, const float* __restrict__ x,
                            float* __restrict__ y) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= N) return;
    const uint16_t* w = W + (size_t)row * K;
    float acc = 0.0f;
    for (int k = lane * 8; k < K; k += 32 * 8) {
        const uint4 p = *(const uint4*)(w + k);
        const uint32_t u[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            // BF16 = the top 16 bits of an FP32: shift the low half up
            acc += __uint_as_float(u[j] << 16) * x[k + 2 * j];
            acc += __uint_as_float(u[j] & 0xFFFF0000u) * x[k + 2 * j + 1];   // high half is already in place
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

int g_num_sms = 0;

constexpr size_t GEMV_SMEM_MAX = 64 * 1024;   // Turing opt-in maximum per CTA

template <QuantFormat T, int NB, bool STAGE>
void launch_gemv_t(const void* W, size_t row_bytes, int K, int N, const ActQ8& act, int act_ld, float* y, int y_ld,
                   cudaStream_t st, bool accumulate) {
    const size_t smem = QTable<T>::BYTES + (STAGE ? (size_t)NB * ((size_t)K + 2 * sizeof(float) * (K / 32)) : 0);
    // one instantiation per <format, columns, staging>: set its shared-memory limit once
    static bool configured = false;
    if (!configured) {
        CHECK_CUDA(cudaFuncSetAttribute(k_gemv_q8<T, NB, STAGE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GEMV_SMEM_MAX));
        configured = true;
    }
    if (!g_num_sms) {
        int dev;
        CHECK_CUDA(cudaGetDevice(&dev));
        CHECK_CUDA(cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev));
    }
    const int max_ctas = g_num_sms * 4;   // 4 CTAs per SM: enough warps to hide memory latency, rows strided beyond that
    const int ctas = std::min(max_ctas, (N + GEMV_WARPS - 1) / GEMV_WARPS);
    k_gemv_q8<T, NB, STAGE><<<ctas, GEMV_WARPS * 32, smem, st>>>((const uint8_t*)W, row_bytes, K, N, act.q, act.d, act.s,
                                                                act_ld, y, y_ld, accumulate ? 1 : 0);
}

template <QuantFormat T, int NB>
void launch_gemv_nb(const void* W, size_t row_bytes, int K, int N, const ActQ8& act, int act_ld, float* y, int y_ld,
                    cudaStream_t st, bool accumulate) {
    // stage the activations in shared memory when they fit (keeps 4 CTAs/SM for small K)
    const size_t need = QTable<T>::BYTES + (size_t)NB * ((size_t)K + 2 * sizeof(float) * (K / 32));
    // above 48 KB of shared memory fewer CTAs fit per SM: read activations through L1 instead
    if (need <= 48 * 1024) launch_gemv_t<T, NB, true>(W, row_bytes, K, N, act, act_ld, y, y_ld, st, accumulate);
    else launch_gemv_t<T, NB, false>(W, row_bytes, K, N, act, act_ld, y, y_ld, st, accumulate);
}

template <QuantFormat T>
void launch_gemv(const void* W, size_t row_bytes, int K, int N, const ActQ8& act, int nb, int act_ld, float* y, int y_ld,
                 cudaStream_t st, bool accumulate) {
    switch (nb) {
        case 1: launch_gemv_nb<T, 1>(W, row_bytes, K, N, act, act_ld, y, y_ld, st, accumulate); break;
        case 2: launch_gemv_nb<T, 2>(W, row_bytes, K, N, act, act_ld, y, y_ld, st, accumulate); break;
        case 3: launch_gemv_nb<T, 3>(W, row_bytes, K, N, act, act_ld, y, y_ld, st, accumulate); break;
        case 4: launch_gemv_nb<T, 4>(W, row_bytes, K, N, act, act_ld, y, y_ld, st, accumulate); break;
        case 5: launch_gemv_nb<T, 5>(W, row_bytes, K, N, act, act_ld, y, y_ld, st, accumulate); break;
        case 6: launch_gemv_nb<T, 6>(W, row_bytes, K, N, act, act_ld, y, y_ld, st, accumulate); break;
        case 7: launch_gemv_nb<T, 7>(W, row_bytes, K, N, act, act_ld, y, y_ld, st, accumulate); break;
        default: break;
    }
}

} // namespace

void quantize_act_q8(const float* x, int K, int n_rows, ActQ8 out, cudaStream_t st) {
    const int n_blocks = K / 32 * n_rows;
    const int warps = 8;
    k_quantize_act_q8<<<(n_blocks + warps - 1) / warps, warps * 32, 0, st>>>(x, n_blocks, out.q, out.d, out.s);
}

bool gemv_q8_supported(QuantFormat t) {
    switch (t) {
        case QuantFormat::IQ2_XXS: case QuantFormat::IQ2_XS: case QuantFormat::IQ2_S: case QuantFormat::IQ3_XXS:
        case QuantFormat::IQ3_S: case QuantFormat::IQ1_S: case QuantFormat::IQ1_M: case QuantFormat::IQ4_XS:
        case QuantFormat::Q2_K: case QuantFormat::Q4_K: case QuantFormat::Q6_K: return true;
        default: return false;
    }
}

bool gemv_q8n(QuantFormat t, const void* W, size_t row_bytes, int K, int N, const ActQ8& act, int nb, int act_ld,
              float* y, int y_ld, cudaStream_t st, bool accumulate) {
    if (nb < 1 || nb > 7) return false;
#define Q38_CASE(F)                                                                                   \
    case QuantFormat::F:                                                                              \
        launch_gemv<QuantFormat::F>(W, row_bytes, K, N, act, nb, act_ld, y, y_ld, st, accumulate);    \
        return true;
    switch (t) {
        Q38_CASE(IQ2_XXS) Q38_CASE(IQ2_XS) Q38_CASE(IQ2_S) Q38_CASE(IQ3_XXS) Q38_CASE(IQ3_S)
        Q38_CASE(IQ1_S) Q38_CASE(IQ1_M) Q38_CASE(IQ4_XS) Q38_CASE(Q2_K) Q38_CASE(Q4_K) Q38_CASE(Q6_K)
        default: return false;
    }
#undef Q38_CASE
}

bool gemv_q8(QuantFormat t, const void* W, size_t row_bytes, int K, int N, const ActQ8& act, float* y, cudaStream_t st,
             bool accumulate) {
    return gemv_q8n(t, W, row_bytes, K, N, act, 1, K, y, N, st, accumulate);
}

void gemv_bf16(const uint16_t* W, int K, int N, const float* x, float* y, cudaStream_t st) {
    const int warps = 8;
    k_gemv_bf16<<<(N + warps - 1) / warps, warps * 32, 0, st>>>(W, K, N, x, y);
}

} // namespace cuda
} // namespace q38rt
