#pragma once
/// @file common.cuh
/// Device helpers shared by the decode (ops.cu) and prefill (prefill.cu) kernels.

#include <cfloat>
#include <cstdint>
#include <cuda_fp16.h>
#include "cuda/gemv.cuh"
#include "cuda/ops.cuh"

namespace q38rt {
namespace cuda {
namespace dev {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.0f + __expf(-x)); }

// Block-wide sum (blockDim.x multiple of 32, <= 1024). `red` = 32 floats of shared memory.
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    float t = lane < nw ? red[lane] : 0.0f;
    t = warp_sum(t);
    return t;
}

// Quantize 32 values held one per lane into ActQ8 block `blk` (warp-collective).
__device__ __forceinline__ void quant_block_q8(float v, int blk, const ActQ8& q) {
    const int lane = threadIdx.x & 31;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.0f;
    const int qi = d > 0.0f ? __float2int_rn(v / d) : 0;
    q.q[blk * 32 + lane] = (int8_t)qi;
    const float s = warp_sum((float)qi);
    if (lane == 0) { q.d[blk] = d; q.s[blk] = d * s; }
}

// Store one 256-element row (256 threads, element t per thread) in the split KV layout.
// NB: no stores through packed-struct members in device code (nvcc 12.4 miscompiled them).
__device__ __forceinline__ void kv_store_row(uint8_t* vals, uint16_t* scales, KVType kvt, float v, int t) {
    const int lane = t & 31, blk = t >> 5;
    if (kvt == KVType::F16) {
        ((__half*)vals)[t] = __float2half_rn(v);
    } else if (kvt == KVType::Q8_0) {
        const float amax = warp_max(fabsf(v));
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        ((int8_t*)vals)[t] = (int8_t)roundf(v * id);
        if (lane == 0) scales[blk] = __half_as_ushort(__float2half_rn(d));
    } else {   // Q4_0, ggml semantics: d = max / -8 with max the signed value of the first largest |x|
        float amax = fabsf(v);
        float mx = v;
        int bi = lane;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float oa = __shfl_xor_sync(0xffffffffu, amax, o), om = __shfl_xor_sync(0xffffffffu, mx, o);
            const int ob = __shfl_xor_sync(0xffffffffu, bi, o);
            if (oa > amax || (oa == amax && ob < bi)) { amax = oa; mx = om; bi = ob; }
        }
        const float d = mx / -8.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const int qv = min(15, (int)(int8_t)(v * id + 8.5f));
        const int nxt = __shfl_down_sync(0xffffffffu, qv, 1);
        if ((t & 1) == 0) vals[t >> 1] = (uint8_t)(qv | (nxt << 4));
        if (lane == 0) scales[blk] = __half_as_ushort(__float2half_rn(d));
    }
}

// Load 8 consecutive key/value elements [8j, 8j+8) of a row as floats (before scale for Q formats).
__device__ __forceinline__ void kv_vals8(const uint8_t* row, KVType kvt, int j, float* out) {
    if (kvt == KVType::F16) {
        const uint4 u = ((const uint4*)row)[j];
        const __half2* hp = (const __half2*)&u;
#pragma unroll
        for (int i = 0; i < 4; i++) { const float2 f = __half22float2(hp[i]); out[2 * i] = f.x; out[2 * i + 1] = f.y; }
    } else if (kvt == KVType::Q8_0) {
        const uint2 u = ((const uint2*)row)[j];
        const int8_t* q = (const int8_t*)&u;
#pragma unroll
        for (int i = 0; i < 8; i++) out[i] = (float)q[i];
    } else {
        const uint32_t u = ((const uint32_t*)row)[j];
#pragma unroll
        for (int i = 0; i < 8; i++) out[i] = (float)((int)((u >> (4 * i)) & 0xF) - 8);
    }
}


} // namespace dev
} // namespace cuda
} // namespace q38rt
