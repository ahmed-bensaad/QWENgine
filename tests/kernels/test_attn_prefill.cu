/// @file test_attn_prefill.cu
/// Tensor-core prefill attention vs the CUDA-core version (itself the decode kernel structure, validated
/// against fp64 math in test_ops) on random data: several chunk positions, partial tiles, all KV types.
/// Also reports throughput at long context.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "cuda/prefill.cuh"
#include "common/utils.h"

using namespace q38rt;

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const int AH = 24, AKV = 4, AD = 256;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd;
    int fail = 0;
    for (cuda::KVType kvt : {cuda::KVType::F16, cuda::KVType::Q8_0, cuda::KVType::Q4_0}) {
        const int cap = 2048;
        cuda::KVLayer kv; kv.capacity = cap;
        const size_t vb = cuda::kv_value_bytes(kvt);
        CHECK_CUDA(cudaMalloc(&kv.k, (size_t)AKV * cap * vb)); CHECK_CUDA(cudaMalloc(&kv.v, (size_t)AKV * cap * vb));
        CHECK_CUDA(cudaMalloc(&kv.ks, (size_t)AKV * cap * 16)); CHECK_CUDA(cudaMalloc(&kv.vs, (size_t)AKV * cap * 16));
        // fill the cache through the real prep path with random k/v for all positions
        const int TOT = 1500;
        {
            float *qg, *k, *v, *q, *gate, *w;
            CHECK_CUDA(cudaMalloc(&qg, (size_t)TOT * AH * 512 * 4)); CHECK_CUDA(cudaMalloc(&k, (size_t)TOT * 1024 * 4));
            CHECK_CUDA(cudaMalloc(&v, (size_t)TOT * 1024 * 4)); CHECK_CUDA(cudaMalloc(&q, (size_t)TOT * 6144 * 4));
            CHECK_CUDA(cudaMalloc(&gate, (size_t)TOT * 6144 * 4)); CHECK_CUDA(cudaMalloc(&w, AD * 4));
            std::vector<float> h((size_t)TOT * AH * 512);
            for (auto& x : h) x = nd(rng);
            CHECK_CUDA(cudaMemcpy(qg, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
            h.resize((size_t)TOT * 1024);
            for (auto& x : h) x = nd(rng);
            CHECK_CUDA(cudaMemcpy(k, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
            for (auto& x : h) x = nd(rng);
            CHECK_CUDA(cudaMemcpy(v, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
            std::vector<float> ones(AD, 1.0f);
            CHECK_CUDA(cudaMemcpy(w, ones.data(), AD * 4, cudaMemcpyHostToDevice));
            float* rope;
            CHECK_CUDA(cudaMalloc(&rope, (size_t)TOT * 64 * 4));
            cuda::rope_table(0, TOT, 1e7f, rope, 0);
            cuda::attn_prep_rows(qg, k, v, w, w, TOT, 0, rope, 1e-6f, kvt, kv, q, gate, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            // test chunks: (p0, B)
            const int cases[][2] = {{0, 37}, {0, 512}, {300, 129}, {1000, 500}, {1480, 20}};
            for (auto& cs : cases) {
                const int p0 = cs[0], B = cs[1];
                __half *o1, *o2;
                CHECK_CUDA(cudaMalloc(&o1, (size_t)B * 6144 * 2)); CHECK_CUDA(cudaMalloc(&o2, (size_t)B * 6144 * 2));
                const float* qc = q + (size_t)p0 * 6144;
                const float* gc = gate + (size_t)p0 * 6144;
                cuda::attn_prefill(qc, gc, kv, kvt, B, p0, o1, 0);
                cuda::attn_prefill_simple(qc, gc, kv, kvt, B, p0, o2, 0);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaGetLastError());
                std::vector<uint16_t> a((size_t)B * 6144), b(a.size());
                CHECK_CUDA(cudaMemcpy(a.data(), o1, a.size() * 2, cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(b.data(), o2, b.size() * 2, cudaMemcpyDeviceToHost));
                double maxd = 0, maxv = 0;
                for (size_t i = 0; i < a.size(); i++) {
                    const float x = __half2float(*(__half*)&a[i]), y = __half2float(*(__half*)&b[i]);
                    maxd = std::max(maxd, (double)std::fabs(x - y));
                    maxv = std::max(maxv, (double)std::fabs(y));
                }
                const double rel = maxd / maxv;
                const bool ok = rel < 5e-3;   // fp16 operands (S and P) vs fp32 reference path
                fail += !ok;
                printf("kv %d p0 %4d B %3d: max rel diff %.2e %s\n", (int)kvt, p0, B, rel, ok ? "ok" : "FAIL");
                cudaFree(o1); cudaFree(o2);
            }
            cudaFree(qg); cudaFree(k); cudaFree(v); cudaFree(q); cudaFree(gate); cudaFree(w);
        }
        cudaFree(kv.k); cudaFree(kv.v); cudaFree(kv.ks); cudaFree(kv.vs);
    }
    if (bench) {
        const int cap = 100000, B = 512;
        for (cuda::KVType kvt : {cuda::KVType::Q8_0, cuda::KVType::Q4_0}) {
            cuda::KVLayer kv; kv.capacity = cap;
            const size_t vb = cuda::kv_value_bytes(kvt);
            CHECK_CUDA(cudaMalloc(&kv.k, (size_t)AKV * cap * vb)); CHECK_CUDA(cudaMalloc(&kv.v, (size_t)AKV * cap * vb));
            CHECK_CUDA(cudaMalloc(&kv.ks, (size_t)AKV * cap * 16)); CHECK_CUDA(cudaMalloc(&kv.vs, (size_t)AKV * cap * 16));
            CHECK_CUDA(cudaMemset(kv.k, 0, (size_t)AKV * cap * vb)); CHECK_CUDA(cudaMemset(kv.v, 0, (size_t)AKV * cap * vb));
            CHECK_CUDA(cudaMemset(kv.ks, 0, (size_t)AKV * cap * 16)); CHECK_CUDA(cudaMemset(kv.vs, 0, (size_t)AKV * cap * 16));
            float *q, *gate; __half* o;
            CHECK_CUDA(cudaMalloc(&q, (size_t)B * 6144 * 4)); CHECK_CUDA(cudaMalloc(&gate, (size_t)B * 6144 * 4));
            CHECK_CUDA(cudaMalloc(&o, (size_t)B * 6144 * 2));
            CHECK_CUDA(cudaMemset(q, 0, (size_t)B * 6144 * 4)); CHECK_CUDA(cudaMemset(gate, 0, (size_t)B * 6144 * 4));
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            for (int p0 : {8192 - B, 32768 - B, cap - B}) {
                cuda::attn_prefill(q, gate, kv, kvt, B, p0, o, 0);
                cudaEventRecord(e0);
                cuda::attn_prefill(q, gate, kv, kvt, B, p0, o, 0);
                cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                const double flop = 2.0 * B * AH * (double)(p0 + B / 2) * AD * 2;
                printf("bench kv %d: chunk of %d at p0 %6d: %.2f ms/layer (%.1f TFLOPS)\n", (int)kvt, B, p0, ms, flop / ms / 1e9);
            }
            cudaFree(kv.k); cudaFree(kv.v); cudaFree(kv.ks); cudaFree(kv.vs); cudaFree(q); cudaFree(gate); cudaFree(o);
        }
    }
    if (fail) { printf("test_attn_prefill: %d failures\n", fail); return 1; }
    printf("test_attn_prefill: tensor-core path matches\n");
    return 0;
}
