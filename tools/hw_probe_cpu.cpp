// hw_probe_cpu.cpp — CPU DRAM read bandwidth (AVX2), linked into hw_probe.
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstddef>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// CPU DRAM read bandwidth with nthreads (AVX2 loads, xor-reduce)
double cpu_read_bw(const uint8_t* buf, size_t bytes, int nthreads, int reps) {
    double best = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        unsigned long long sink = 0;
#pragma omp parallel num_threads(nthreads) reduction(^:sink)
        {
            int tid = omp_get_thread_num(), nt = omp_get_num_threads();
            size_t chunk = bytes / nt;
            const __m256i* p = (const __m256i*)(buf + tid * chunk);
            size_t n = chunk / 32;
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
            for (size_t i = 0; i + 4 <= n; i += 4) {
                a0 = _mm256_xor_si256(a0, _mm256_load_si256(p + i));
                a1 = _mm256_xor_si256(a1, _mm256_load_si256(p + i + 1));
                a2 = _mm256_xor_si256(a2, _mm256_load_si256(p + i + 2));
                a3 = _mm256_xor_si256(a3, _mm256_load_si256(p + i + 3));
            }
            a0 = _mm256_xor_si256(_mm256_xor_si256(a0, a1), _mm256_xor_si256(a2, a3));
            sink ^= (unsigned long long)_mm256_extract_epi64(a0, 0);
        }
        double dt = now_s() - t0;
        if (sink == 42) printf(" ");
        best = std::max(best, bytes / dt / 1e9);
    }
    return best;
}

