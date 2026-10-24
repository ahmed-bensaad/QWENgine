#pragma once
/// @file thread_pool.h
/// Persistent worker pool for the CPU share of decode. Workers are pinned to physical cores
/// (CPU i -> core i on this machine; i+8 are the hyperthread siblings) and spin on a generation
/// counter, so dispatch latency is ~1 us instead of a futex wake-up. The calling thread runs
/// task 0 itself.

#include <pthread.h>
#include <atomic>
#include <thread>
#include <vector>
#include <immintrin.h>

namespace q38rt {
namespace cpu {

class ThreadPool {
public:
    using Fn = void (*)(void* ctx, int tid, int nthreads);

    explicit ThreadPool(int nthreads, int first_cpu = 0) : n_(nthreads) {
        pin(first_cpu);  // caller is worker 0
        for (int i = 1; i < n_; i++) {
            threads_.emplace_back([this, i, first_cpu] {
                pin(first_cpu + i);
                uint64_t seen = 0;
                while (true) {
                    uint64_t g;
                    int spins = 0;
                    while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                        if (++spins > 4096) { std::this_thread::yield(); spins = 0; } else _mm_pause();
                    }
                    seen = g;
                    if (stop_.load(std::memory_order_relaxed)) return;
                    fn_(ctx_, i, n_);
                    done_.fetch_add(1, std::memory_order_acq_rel);
                }
            });
        }
    }

    ~ThreadPool() {
        stop_.store(true);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& t : threads_) t.join();
    }

    /// Run fn(ctx, tid, n) on all workers (tid 0 on the caller); returns when all finished.
    void run(Fn fn, void* ctx) {
        fn_ = fn;
        ctx_ = ctx;
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        fn(ctx, 0, n_);
        while (done_.load(std::memory_order_acquire) != n_ - 1) _mm_pause();
    }

    /// Start fn(ctx, tid, n) on the workers only (tid 1..n-1) and return at once; join() waits for them.
    void start(Fn fn, void* ctx) {
        fn_ = fn;
        ctx_ = ctx;
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
    }
    void join() {
        while (done_.load(std::memory_order_acquire) != n_ - 1) _mm_pause();
    }

    int size() const { return n_; }

    static void pin(int cpu) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(cpu, &s);
        pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
    }

private:
    int n_;
    std::vector<std::thread> threads_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
    Fn fn_ = nullptr;
    void* ctx_ = nullptr;
};

} // namespace cpu
} // namespace q38rt
