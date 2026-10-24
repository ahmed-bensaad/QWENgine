/// @file test_gemv.cu
/// Correctness + bandwidth of the decode GEMV kernels on real tensors of every weight format.
/// Reference: sum_k dequant_ref(W)[r,k] * (d8*q8)[k] in double, i.e. exactly what the kernel computes
/// up to float summation order. Tolerance is relative to sum_k |w||x|.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>
#include "cuda/gemv.cuh"
#include "cuda/prefill.cuh"
#include <cuda_fp16.h>
#include "quant/quant_formats.h"
#include "runtime/gguf_parser.h"
#include "common/utils.h"

using namespace q38rt;

constexpr int NC = 7;   // activation columns tested (Engine::MAX_BATCH)

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [--bench]\n", argv[0]); return 2; }
    const bool bench = argc > 2 && std::string(argv[2]) == "--bench";
    GGUFParser p;
    if (!p.parse(argv[1])) return 1;

    // pick, per (type, K), the largest tensor (for bandwidth) — correctness checked on its first rows. Both K = 5120
    // and K = 17408 (FFN down) are covered so the multi-column kernels run with staged and unstaged activations.
    std::map<std::pair<QuantFormat, int>, const TensorInfo*> pick;
    for (const TensorInfo& t : p.get_tensors()) {
        if (!cuda::gemv_q8_supported(t.quant) || t.name == "output.weight" || t.name == "token_embd.weight") continue;
        const auto key = std::make_pair(t.quant, (int)t.shape.dims[0]);
        auto it = pick.find(key);
        if (it == pick.end() || t.size_bytes > it->second->size_bytes) pick[key] = &t;
    }
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    int failures = 0;
    cudaStream_t st;
    CHECK_CUDA(cudaStreamCreate(&st));
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    printf("%-8s %-28s %6s %6s %10s %10s %8s   %s\n", "type", "tensor", "K", "N", "max_rel", "rel_vs_fp", "GB/s",
           "GB/s with 1/2/3/4 columns");
    for (auto& kv : pick) {
        const TensorInfo& t = *kv.second;
        const int K = (int)t.shape.dims[0], N = (int)t.shape.dims[1];
        const size_t rb = quant_row_bytes(t.quant, K);
        void* dW;
        CHECK_CUDA(cudaMalloc(&dW, t.size_bytes));
        CHECK_CUDA(cudaMemcpy(dW, p.tensor_data(t), t.size_bytes, cudaMemcpyHostToDevice));
        std::vector<float> x(K);
        for (auto& v : x) v = nd(rng);
        float *dx, *dy;
        cuda::ActQ8 act;
        CHECK_CUDA(cudaMalloc(&dx, K * 4));
        CHECK_CUDA(cudaMalloc(&dy, (size_t)N * 4));
        CHECK_CUDA(cudaMalloc(&act.q, (size_t)NC * K));
        CHECK_CUDA(cudaMalloc(&act.d, (size_t)NC * (K / 32) * 4));
        CHECK_CUDA(cudaMalloc(&act.s, (size_t)NC * (K / 32) * 4));
        CHECK_CUDA(cudaMemcpy(dx, x.data(), K * 4, cudaMemcpyHostToDevice));
        cuda::quantize_act_q8(dx, K, 1, act, st);
        cuda::gemv_q8(t.quant, dW, rb, K, N, act, dy, st);
        CHECK_CUDA(cudaStreamSynchronize(st));
        CHECK_CUDA(cudaGetLastError());
        std::vector<int8_t> q(K);
        std::vector<float> d(K / 32), y(N);
        CHECK_CUDA(cudaMemcpy(q.data(), act.q, K, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(d.data(), act.d, K / 32 * 4, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(y.data(), dy, (size_t)N * 4, cudaMemcpyDeviceToHost));
        std::vector<float> xq(K), w(K);
        for (int k = 0; k < K; k++) xq[k] = d[k / 32] * q[k];
        double max_rel = 0, max_rel_fp = 0;
        const int check_rows = std::min(N, 512);
        for (int r = 0; r < check_rows; r++) {
            dequantize_row_ref(t.quant, p.tensor_data(t) + (size_t)r * rb, w.data(), K);
            double ref = 0, ref_fp = 0, mag = 0;
            for (int k = 0; k < K; k++) {
                ref += (double)w[k] * xq[k];
                ref_fp += (double)w[k] * x[k];
                mag += std::fabs((double)w[k] * x[k]);
            }
            max_rel = std::max(max_rel, std::fabs(y[r] - ref) / mag);
            max_rel_fp = std::max(max_rel_fp, std::fabs(y[r] - ref_fp) / mag);
        }
        double gbs = 0;
        if (bench) {
            for (int i = 0; i < 5; i++) cuda::gemv_q8(t.quant, dW, rb, K, N, act, dy, st);
            CHECK_CUDA(cudaEventRecord(e0, st));
            const int reps = 50;
            for (int i = 0; i < reps; i++) cuda::gemv_q8(t.quant, dW, rb, K, N, act, dy, st);
            CHECK_CUDA(cudaEventRecord(e1, st));
            CHECK_CUDA(cudaEventSynchronize(e1));
            float ms;
            CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
            gbs = (double)t.size_bytes * reps / (ms * 1e-3) / 1e9;
        }
        bool ok = max_rel < 2e-6;
        // prefill dequantizer (fp16) must equal the reference dequant rounded to fp16
        {
            const int rows = std::min(N, 8);
            __half* dh;
            CHECK_CUDA(cudaMalloc(&dh, (size_t)rows * K * sizeof(__half)));
            if (!cuda::dequant_rows_f16(t.quant, dW, rb, K, rows, dh, st)) { printf("  no dequant_rows_f16\n"); ok = false; }
            CHECK_CUDA(cudaStreamSynchronize(st));
            std::vector<__half> hh((size_t)rows * K);
            CHECK_CUDA(cudaMemcpy(hh.data(), dh, hh.size() * sizeof(__half), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int r = 0; r < rows; r++) {
                dequantize_row_ref(t.quant, p.tensor_data(t) + (size_t)r * rb, w.data(), K);
                for (int k = 0; k < K; k++) {
                    const __half ref = __float2half_rn(w[k]);
                    bad += memcmp(&ref, &hh[(size_t)r * K + k], sizeof(__half)) != 0;
                }
            }
            if (bad) printf("  dequant_rows_f16: %d values differ from the reference\n", bad);
            ok = ok && bad == 0;
            cudaFree(dh);
        }
        // multi-column: NC activation columns; every column of gemv_q8n(nb) must be bit-identical to gemv_q8 alone
        double gcols[NC + 1] = {};
        {
            float *dx4, *dy1, *dyn;
            CHECK_CUDA(cudaMalloc(&dx4, (size_t)NC * K * 4));
            CHECK_CUDA(cudaMalloc(&dy1, (size_t)NC * N * 4));
            CHECK_CUDA(cudaMalloc(&dyn, (size_t)NC * N * 4));
            std::vector<float> x4((size_t)NC * K);
            for (auto& v : x4) v = nd(rng);
            CHECK_CUDA(cudaMemcpy(dx4, x4.data(), x4.size() * 4, cudaMemcpyHostToDevice));
            cuda::quantize_act_q8(dx4, K, NC, act, st);
            for (int c = 0; c < NC; c++) {
                cuda::ActQ8 ac{act.q + (size_t)c * K, act.d + (size_t)c * (K / 32), act.s + (size_t)c * (K / 32)};
                cuda::gemv_q8(t.quant, dW, rb, K, N, ac, dy1 + (size_t)c * N, st);
            }
            std::vector<float> y1((size_t)NC * N), yn((size_t)NC * N);
            CHECK_CUDA(cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int nbc = 2; nbc <= NC; nbc++) {
                cuda::gemv_q8n(t.quant, dW, rb, K, N, act, nbc, K, dyn, N, st);
                CHECK_CUDA(cudaMemcpy(yn.data(), dyn, (size_t)nbc * N * 4, cudaMemcpyDeviceToHost));
                bad += memcmp(y1.data(), yn.data(), (size_t)nbc * N * 4) != 0;
            }
            if (bad) printf("  multi-column results differ from single-column (%d of %d column counts)\n", bad, NC - 1);
            ok = ok && bad == 0;
            if (bench) {
                for (int nbc = 1; nbc <= NC; nbc++) {
                    for (int i = 0; i < 3; i++) cuda::gemv_q8n(t.quant, dW, rb, K, N, act, nbc, K, dyn, N, st);
                    CHECK_CUDA(cudaEventRecord(e0, st));
                    const int reps = 30;
                    for (int i = 0; i < reps; i++) cuda::gemv_q8n(t.quant, dW, rb, K, N, act, nbc, K, dyn, N, st);
                    CHECK_CUDA(cudaEventRecord(e1, st));
                    CHECK_CUDA(cudaEventSynchronize(e1));
                    float ms;
                    CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
                    gcols[nbc] = (double)t.size_bytes * reps / (ms * 1e-3) / 1e9;
                }
            }
            cudaFree(dx4); cudaFree(dy1); cudaFree(dyn);
        }
        failures += !ok;
        printf("%-8s %-28s %6d %6d %10.2e %10.2e %8.1f", quant_name(t.quant), t.name.c_str(), K, N, max_rel, max_rel_fp, gbs);
        if (bench) for (int c = 1; c <= NC; c++) printf(" %6.1f", gcols[c]);
        printf(" %s\n", ok ? "" : "FAIL");
        cudaFree(dW); cudaFree(dx); cudaFree(dy); cudaFree(act.q); cudaFree(act.d); cudaFree(act.s);
    }
    if (failures) { printf("test_gemv: %d failures\n", failures); return 1; }
    printf("test_gemv: all formats match the reference\n");
    return 0;
}
