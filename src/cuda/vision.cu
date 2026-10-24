/// @file vision.cu
/// See vision.cuh.

#include "cuda/vision.cuh"
#include <cfloat>
#include <cmath>

namespace q38rt {
namespace cuda {

namespace {

constexpr int VD = 1152, VHEADS = 16, VHD = 72;

__device__ __forceinline__ float vis_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float vis_warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum / max (blockDim multiple of 32, <= 1024); every thread gets the result
__device__ float vis_block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = vis_warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : 0.0f;
    return vis_warp_sum(v);
}
__device__ float vis_block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = vis_warp_max(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : -FLT_MAX;
    return vis_warp_max(v);
}

__global__ void k_vis_bias(float* Y, const float* __restrict__ bias, long long total, int n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x)
        Y[i] += bias[i % n];
}

__global__ void k_vis_residual(float* X, const float* __restrict__ Y, const float* __restrict__ bias, long long total, int n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x)
        X[i] += Y[i] + bias[i % n];
}

// one block per row; two passes (mean, then variance of x - mean) as ggml_norm
__global__ void k_vis_layernorm(const float* __restrict__ X, const float* __restrict__ w, const float* __restrict__ b,
                                int n, float eps, __half* __restrict__ out) {
    __shared__ float red[32];
    const float* x = X + (size_t)blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i];
    const float mean = vis_block_sum(s, red) / n;
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float d = x[i] - mean; v += d * d; }
    const float rs = 1.0f / sqrtf(vis_block_sum(v, red) / n + eps);
    __half* o = out + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = __float2half_rn((x[i] - mean) * rs * w[i] + b[i]);
}

__global__ void k_vis_bias_gelu(const float* __restrict__ U, const float* __restrict__ bias, long long total, int n,
                                __half* __restrict__ out) {
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;   // sqrt(2 / pi)
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const float x = U[i] + bias[i % n];
        out[i] = __float2half_rn(0.5f * x * (1.0f + tanhf(k0 * x * (1.0f + k1 * x * x))));
    }
}

__global__ void k_vis_to_f16(const float* __restrict__ x, long long n, __half* __restrict__ out) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = __float2half_rn(x[i]);
}

// grid (C tokens, 16 heads), 72 threads: thread j < 36 rotates pairs (j, j + 36) of q and k; every thread moves v[j]
__global__ void k_vis_qkv_rope(const float* __restrict__ qkv, const float* __restrict__ bias, int t0, int N, int nx,
                               __half* __restrict__ Q, __half* __restrict__ K, __half* __restrict__ Vt) {
    const int c = blockIdx.x, h = blockIdx.y, j = threadIdx.x, t = t0 + c;
    const float* row = qkv + (size_t)c * 3 * VD;
    const int m = t >> 2, y = 2 * (m / nx) + ((t >> 1) & 1), x = 2 * (m % nx) + (t & 1);
    const size_t qo = ((size_t)h * N + t) * VHD;
    if (j < VHD / 2) {
        const int sec = j < VHD / 4 ? 0 : 1, k = j - sec * (VHD / 4);
        const float theta = (float)(sec == 0 ? y : x) * powf(10000.0f, -2.0f * k / (VHD / 2));
        float sn, cs;
        sincosf(theta, &sn, &cs);
        for (int part = 0; part < 2; part++) {   // q, k
            const int off = part * VD + h * VHD;
            const float a = row[off + j] + bias[off + j], b2 = row[off + j + VHD / 2] + bias[off + j + VHD / 2];
            __half* dst = (part == 0 ? Q : K) + qo;
            dst[j] = __float2half_rn(a * cs - b2 * sn);
            dst[j + VHD / 2] = __float2half_rn(a * sn + b2 * cs);
        }
    }
    const int vo = 2 * VD + h * VHD + j;
    Vt[((size_t)h * VHD + j) * N + t] = __float2half_rn(row[vo] + bias[vo]);
}

// one block per row
__global__ void k_vis_softmax(const float* __restrict__ S, int cols, float scale, __half* __restrict__ P) {
    __shared__ float red[32];
    const float* s = S + (size_t)blockIdx.x * cols;
    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, s[i]);
    m = vis_block_max(m, red) * scale;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sum += expf(s[i] * scale - m);
    const float inv = 1.0f / vis_block_sum(sum, red);
    __half* p = P + (size_t)blockIdx.x * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) p[i] = __float2half_rn(expf(s[i] * scale - m) * inv);
}

unsigned grid_for(long long n) { return (unsigned)std::min<long long>((n + 255) / 256, 40 * 32); }

} // namespace

void vis_bias(float* Y, const float* bias, int rows, int n, cudaStream_t st) {
    const long long total = (long long)rows * n;
    k_vis_bias<<<grid_for(total), 256, 0, st>>>(Y, bias, total, n);
}

void vis_residual(float* X, const float* Y, const float* bias, int rows, int n, cudaStream_t st) {
    const long long total = (long long)rows * n;
    k_vis_residual<<<grid_for(total), 256, 0, st>>>(X, Y, bias, total, n);
}

void vis_layernorm(const float* X, const float* w, const float* b, int rows, int n, float eps, __half* out16, cudaStream_t st) {
    k_vis_layernorm<<<rows, 256, 0, st>>>(X, w, b, n, eps, out16);
}

void vis_bias_gelu(const float* U, const float* bias, int rows, int n, __half* out16, cudaStream_t st) {
    const long long total = (long long)rows * n;
    k_vis_bias_gelu<<<grid_for(total), 256, 0, st>>>(U, bias, total, n, out16);
}

void vis_to_f16(const float* x, size_t n, __half* out, cudaStream_t st) {
    k_vis_to_f16<<<grid_for((long long)n), 256, 0, st>>>(x, (long long)n, out);
}

void vis_qkv_rope(const float* qkv, const float* bias, int t0, int C, int N, int nx, __half* Q, __half* K, __half* Vt,
                  cudaStream_t st) {
    k_vis_qkv_rope<<<dim3(C, VHEADS), VHD, 0, st>>>(qkv, bias, t0, N, nx, Q, K, Vt);
}

void vis_softmax(const float* S, int rows, int cols, float scale, __half* P, cudaStream_t st) {
    k_vis_softmax<<<rows, 256, 0, st>>>(S, cols, scale, P);
}

} // namespace cuda
} // namespace q38rt
