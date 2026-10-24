/// @file bench_cpu_ggml.cpp
/// Per-format CPU GEMV throughput of ggml's own vec_dot kernels (libggml-cpu, the trusted
/// reference), on the largest real tensor of each format, with the activation quantized to
/// the kernel's vec_dot_type. Rows are split statically over N threads pinned to physical cores.
/// Usage: bench_cpu_ggml <model.gguf> [threads=8]

#include <pthread.h>
#include <sys/mman.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <thread>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"
#include "runtime/gguf_parser.h"

using namespace q38rt;

static void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [threads]\n", argv[0]); return 1; }
    const int nth = argc > 2 ? atoi(argv[2]) : 8;
    ggml_cpu_init();
    GGUFParser p;
    if (!p.parse(argv[1])) return 1;
    std::map<QuantFormat, const TensorInfo*> pick;
    for (const TensorInfo& t : p.get_tensors()) {
        if (t.shape.ndim != 2 || t.quant == QuantFormat::FP32 || t.quant == QuantFormat::BF16) continue;
        if (t.name == "token_embd.weight") continue;
        auto it = pick.find(t.quant);
        if (it == pick.end() || t.size_bytes > it->second->size_bytes) pick[t.quant] = &t;
    }
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    printf("%-8s %-26s %6s %6s %9s %9s\n", "type", "tensor", "K", "N", "GB/s(1t)", "GB/s(Nt)");
    for (auto& kv : pick) {
        const TensorInfo& t = *kv.second;
        const ggml_type gt = (ggml_type)t.quant;
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu(gt);
        const int K = (int)t.shape.dims[0], N = (int)t.shape.dims[1];
        const size_t rb = quant_row_bytes(t.quant, K);
        // private copy in anonymous (huge-page advised) memory
        uint8_t* W = (uint8_t*)aligned_alloc(2 << 20, (t.size_bytes + (2 << 20) - 1) / (2 << 20) * (2 << 20));
        madvise(W, t.size_bytes, MADV_HUGEPAGE);
        memcpy(W, p.tensor_data(t), t.size_bytes);
        std::vector<float> x(K), y(N);
        for (auto& v : x) v = nd(rng);
        std::vector<uint8_t> xq(ggml_row_size(tr->vec_dot_type, K));
        ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(x.data(), xq.data(), K);
        auto run = [&](int threads, int reps) {
            std::atomic<int> go{0};
            std::vector<std::thread> th;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < threads; i++) {
                th.emplace_back([&, i] {
                    pin(i);
                    const int r0 = (int)((int64_t)N * i / threads), r1 = (int)((int64_t)N * (i + 1) / threads);
                    for (int r = 0; r < reps; r++)
                        for (int row = r0; row < r1; row++)
                            tr->vec_dot(K, &y[row], 0, W + (size_t)row * rb, 0, xq.data(), 0, 1);
                });
            }
            for (auto& h : th) h.join();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return (double)t.size_bytes * reps / s / 1e9;
        };
        run(nth, 1);  // warm
        const double g1 = run(1, 2);
        const double gn = run(nth, 10);
        printf("%-8s %-26s %6d %6d %9.2f %9.2f\n", quant_name(t.quant), t.name.c_str(), K, N, g1, gn);
        free(W);
    }
    return 0;
}
