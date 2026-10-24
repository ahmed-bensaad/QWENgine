/// @file test_cpu_gemv.cpp
/// Correctness and multi-threaded bandwidth of the AVX2 CPU dot products on real tensors.
/// Reference: sum_k dequant_ref(W)[r,k] * (x.d * x.qs[k]) in double (exactly what the kernel computes).
/// Bandwidth: all tensors of a format concatenated (working set >> 16 MB L3), rows split statically.
/// Usage: test_cpu_gemv <model.gguf> [--bench] [--threads N]

#include <sys/mman.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "cpu/cpu_kernels.h"
#include "cpu/thread_pool.h"
#include "quant/quant_formats.h"
#include "runtime/gguf_parser.h"

using namespace q38rt;

struct BenchCtx {
    cpu::vec_dot_fn fn;
    const uint8_t* W;
    size_t rb;
    int K;
    int64_t nrows;
    const block_q8_K* x;
    float* y;
    int reps;
};

static void bench_task(void* p, int tid, int n) {
    BenchCtx& c = *(BenchCtx*)p;
    const int64_t r0 = c.nrows * tid / n, r1 = c.nrows * (tid + 1) / n;
    for (int r = 0; r < c.reps; r++)
        for (int64_t row = r0; row < r1; row++) c.y[row] = c.fn(c.K, c.W + (size_t)row * c.rb, c.x);
}

struct BenchCtxN {
    cpu::vec_dot_n_fn fn;
    const uint8_t* W;
    size_t rb;
    int K;
    int64_t nrows;
    const block_q8_K* x;
    size_t xs;
    float* y;
};

static void bench_task_n(void* p, int tid, int n) {
    BenchCtxN& c = *(BenchCtxN*)p;
    const int64_t r0 = c.nrows * tid / n, r1 = c.nrows * (tid + 1) / n;
    float out[8];
    for (int64_t row = r0; row < r1; row++) {
        c.fn(c.K, c.W + (size_t)row * c.rb, c.x, c.xs, out);
        c.y[row] = out[0];
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [--bench] [--threads N]\n", argv[0]); return 2; }
    bool bench = false;
    int nth = 8;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--threads")) nth = atoi(argv[++i]);
    }
    GGUFParser p;
    if (!p.parse(argv[1])) return 1;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    int failures = 0;

    // Q8_K quantizer must be bit-exact with the reference
    {
        std::vector<float> x(17408);
        std::vector<block_q8_K> a(68), b(68);
        int bad = 0;
        for (int t = 0; t < 50; t++) {
            for (auto& v : x) v = nd(rng) * std::pow(10.f, (float)(t % 7) - 3);
            if (t == 1) std::fill(x.begin(), x.end(), 0.f);
            memset(a.data(), 0, a.size() * sizeof(block_q8_K));
            memset(b.data(), 0, b.size() * sizeof(block_q8_K));
            cpu::quantize_q8_K(x.data(), a.data(), 17408);
            quantize_row_q8_K_ref(x.data(), b.data(), 17408);
            if (memcmp(a.data(), b.data(), a.size() * sizeof(block_q8_K))) bad++;
        }
        printf("quantize_q8_K vs ref: %d/50 mismatching\n", bad);
        failures += bad;
    }

    std::map<QuantFormat, std::vector<const TensorInfo*>> by_type;
    for (const TensorInfo& t : p.get_tensors())
        if (cpu::get_vec_dot(t.quant) && t.name != "token_embd.weight" && t.shape.ndim == 2) by_type[t.quant].push_back(&t);

    cpu::ThreadPool pool(nth);
    printf("%-8s %6s %10s %10s %9s %9s\n", "type", "K", "max_rel", "rel_vs_fp", "GB/s(1t)", "GB/s(Nt)");
    for (auto& kv : by_type) {
        const TensorInfo& t = *kv.second.front();
        const int K = (int)t.shape.dims[0];
        const size_t rb = quant_row_bytes(t.quant, K);
        cpu::vec_dot_fn fn = cpu::get_vec_dot(t.quant);
        std::vector<float> x(K), w(K);
        for (auto& v : x) v = nd(rng);
        std::vector<block_q8_K> xq(K / QK_K);
        quantize_row_q8_K_ref(x.data(), xq.data(), K);
        double max_rel = 0, max_rel_fp = 0;
        const int64_t nrows = (int64_t)(t.shape.size() / K);
        for (int64_t r = 0; r < std::min<int64_t>(nrows, 256); r++) {
            const uint8_t* row = p.tensor_data(t) + (size_t)r * rb;
            dequantize_row_ref(t.quant, row, w.data(), K);
            double ref = 0, ref_fp = 0, mag = 0;
            for (int k = 0; k < K; k++) {
                const double xk = (double)xq[k / QK_K].d * xq[k / QK_K].qs[k % QK_K];
                ref += (double)w[k] * xk;
                ref_fp += (double)w[k] * x[k];
                mag += std::fabs((double)w[k] * x[k]);
            }
            const float y = fn(K, row, xq.data());
            max_rel = std::max(max_rel, std::fabs(y - ref) / mag);
            max_rel_fp = std::max(max_rel_fp, std::fabs(y - ref_fp) / mag);
        }
        bool ok = max_rel < 1e-6;
        // multi-column kernels: every column bit-identical to the single-column kernel
        const int nbk = K / QK_K;
        std::vector<block_q8_K> x4(8 * nbk);
        {
            std::vector<float> xc(K);
            for (int b = 0; b < 8; b++) {
                for (auto& v : xc) v = nd(rng) * (b + 1);
                quantize_row_q8_K_ref(xc.data(), x4.data() + b * nbk, K);
            }
            int bad_n = 0;
            for (int nbc = 1; nbc <= 8; nbc++) {
                cpu::vec_dot_n_fn fnn = cpu::get_vec_dot_n(t.quant, nbc);
                for (int64_t r = 0; r < std::min<int64_t>(nrows, 64); r++) {
                    const uint8_t* row = p.tensor_data(t) + (size_t)r * rb;
                    float yn[8];
                    fnn(K, row, x4.data(), nbk, yn);
                    for (int b = 0; b < nbc; b++) {
                        const float y1 = fn(K, row, x4.data() + b * nbk);
                        if (memcmp(&y1, &yn[b], 4)) bad_n++;
                    }
                }
            }
            if (bad_n) printf("  %s: %d multi-column results differ from the single-column kernel\n", quant_name(t.quant), bad_n);
            ok = ok && bad_n == 0;
            // R4 / R2 re-encodings: every result bit-identical to the source kernel
            if (cpu::re_supported(t.quant)) {
                std::vector<uint8_t> r4(cpu::re_row_bytes(t.quant, K));
                int bad_r4 = 0, bad_pack = 0;
                for (int64_t r = 0; r < std::min<int64_t>(nrows, 256); r++) {
                    const uint8_t* row = p.tensor_data(t) + (size_t)r * rb;
                    if (!cpu::re_repack_row(t.quant, row, K, r4.data())) { bad_pack++; continue; }
                    for (int nbc = 1; nbc <= 8; nbc++) {
                        float ya[8], yb[8];
                        cpu::get_vec_dot_n(t.quant, nbc)(K, row, x4.data(), nbk, ya);
                        cpu::get_vec_dot_re_n(t.quant, nbc)(K, r4.data(), x4.data(), nbk, yb);
                        bad_r4 += memcmp(ya, yb, 4 * nbc) != 0;
                    }
                    const float y1 = fn(K, row, x4.data()), y2 = cpu::get_vec_dot_re(t.quant)(K, r4.data(), x4.data());
                    bad_r4 += memcmp(&y1, &y2, 4) != 0;
                }
                if (bad_pack || bad_r4) printf("  %s: re-encoding failures %d, results differing %d\n", quant_name(t.quant), bad_pack, bad_r4);
                ok = ok && bad_pack == 0 && bad_r4 == 0;
            }
        }
        failures += !ok;
        double gcols[9] = {};
        double g1 = 0, gn = 0;
        if (bench) {
            // concatenate all same-K tensors of this format (up to 512 MB) into anonymous huge-page memory
            size_t total = 0;
            for (const TensorInfo* ti : kv.second)
                if ((int)ti->shape.dims[0] == K && total + ti->size_bytes <= (512ull << 20)) total += ti->size_bytes;
            const size_t alloc = (total + (2 << 20) - 1) / (2 << 20) * (2 << 20);
            uint8_t* W = (uint8_t*)aligned_alloc(2 << 20, alloc);
            madvise(W, alloc, MADV_HUGEPAGE);
            size_t off = 0;
            for (const TensorInfo* ti : kv.second)
                if ((int)ti->shape.dims[0] == K && off + ti->size_bytes <= total) {
                    memcpy(W + off, p.tensor_data(*ti), ti->size_bytes);
                    off += ti->size_bytes;
                }
            const int64_t rows = (int64_t)(total / rb);
            std::vector<float> y(rows);
            BenchCtx c{fn, W, rb, K, rows, xq.data(), y.data(), 1};
            pool.run(bench_task, &c);  // warm (page faults)
            auto t0 = std::chrono::steady_clock::now();
            c.reps = 3;
            pool.run(bench_task, &c);
            gn = (double)total * 3 / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9;
            const int64_t rows1 = std::min<int64_t>(rows, (int64_t)((96ull << 20) / rb));
            t0 = std::chrono::steady_clock::now();
            for (int64_t r = 0; r < rows1; r++) y[r] = fn(K, W + (size_t)r * rb, xq.data());
            g1 = (double)rows1 * rb / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9;
            // multi-column passes: weight GB/s with nb tokens per pass
            for (int nbc = 1; nbc <= 8; nbc++) {
                BenchCtxN cn{cpu::get_vec_dot_n(t.quant, nbc), W, rb, K, rows, x4.data(), (size_t)nbk, y.data()};
                pool.run(bench_task_n, &cn);
                t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < 3; r++) pool.run(bench_task_n, &cn);
                gcols[nbc] = (double)total * 3 / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9;
            }
            if (cpu::re_supported(t.quant)) {   // re-encoded copy: GB/s in source-format bytes (comparable with cols1-4)
                const size_t rb4 = cpu::re_row_bytes(t.quant, K);
                const size_t alloc4 = ((size_t)rows * rb4 + (2 << 20) - 1) / (2 << 20) * (2 << 20);
                uint8_t* W4 = (uint8_t*)aligned_alloc(2 << 20, alloc4);
                madvise(W4, alloc4, MADV_HUGEPAGE);
                for (int64_t r = 0; r < rows; r++) cpu::re_repack_row(t.quant, W + (size_t)r * rb, K, W4 + (size_t)r * rb4);
                printf("%-8s RE (%.2f bpw) cols1-8 GB/s of source bytes:", quant_name(t.quant), rb4 * 8.0 / K);
                for (int nbc = 1; nbc <= 8; nbc++) {
                    BenchCtxN cn{cpu::get_vec_dot_re_n(t.quant, nbc), W4, rb4, K, rows, x4.data(), (size_t)nbk, y.data()};
                    pool.run(bench_task_n, &cn);
                    t0 = std::chrono::steady_clock::now();
                    for (int r = 0; r < 3; r++) pool.run(bench_task_n, &cn);
                    printf(" %6.2f", (double)total * 3 / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9);
                }
                printf("\n");
                free(W4);
            }
            free(W);
        }
        printf("%-8s %6d %10.2e %10.2e %9.2f %9.2f", quant_name(t.quant), K, max_rel, max_rel_fp, g1, gn);
        if (bench) { printf("   cols1-8 GB/s"); for (int c = 1; c <= 8; c++) printf(" %6.2f", gcols[c]); }
        printf(" %s\n", ok ? "" : "FAIL");
    }
    if (failures) { printf("test_cpu_gemv: %d failures\n", failures); return 1; }
    printf("test_cpu_gemv: all formats match the reference\n");
    return 0;
}
