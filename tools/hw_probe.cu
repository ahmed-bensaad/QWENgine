// hw_probe.cu — measure the actual machine (GPU, PCIe, VRAM, CPU DRAM).
// Build: g++-12 -O3 -mavx2 -fopenmp -c tools/hw_probe_cpu.cpp -o build/hw_probe_cpu.o &&
//        nvcc -O3 -arch=sm_75 -ccbin g++-12 tools/hw_probe.cu build/hw_probe_cpu.o -Xcompiler -fopenmp -o build/hw_probe
// Every number printed here is measured, not looked up.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

__global__ void k_empty() {}

// Read-only bandwidth kernel: each thread sums uint4 loads; result written once per block.
__global__ void k_read(const uint4* __restrict__ p, size_t n, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;  // practically never; prevents DCE
}

__global__ void k_spin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

static double gpu_copy_bw(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, int reps) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaMemcpy(dst, src, bytes, kind));  // warm
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; i++) CK(cudaMemcpyAsync(dst, src, bytes, kind));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    return (double)bytes * reps / (ms * 1e-3) / 1e9;
}

double cpu_read_bw(const uint8_t* buf, size_t bytes, int nthreads, int reps);  // hw_probe_cpu.cpp

int main() {
    int dev = 0;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    int drv, rt;
    CK(cudaDriverGetVersion(&drv)); CK(cudaRuntimeGetVersion(&rt));
    int clk_khz = 0, memclk_khz = 0, busw = 0, l2 = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&memclk_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
    size_t freeb, totb;
    CK(cudaMemGetInfo(&freeb, &totb));
    printf("== GPU ==\n");
    printf("name=%s cc=%d.%d SMs=%d sm_clock=%d MHz mem_clock=%d MHz bus=%d bit L2=%d KiB\n",
           p.name, p.major, p.minor, p.multiProcessorCount, clk_khz / 1000, memclk_khz / 1000, busw, l2 / 1024);
    printf("theoretical VRAM BW = %.1f GB/s\n", 2.0 * memclk_khz * 1e3 * busw / 8 / 1e9);
    printf("VRAM total=%.0f MiB free=%.0f MiB  smem/block optin=%zu KiB regs/SM=%d\n",
           totb / 1048576.0, freeb / 1048576.0, p.sharedMemPerBlockOptin / 1024, p.regsPerMultiprocessor);
    printf("driver=%d runtime=%d asyncEngines=%d\n", drv, rt, p.asyncEngineCount);

    // ---- launch latency
    for (int i = 0; i < 100; i++) k_empty<<<1, 32>>>();
    CK(cudaDeviceSynchronize());
    const int NL = 20000;
    double t0 = now_s();
    for (int i = 0; i < NL; i++) k_empty<<<1, 32>>>();
    CK(cudaDeviceSynchronize());
    double t_launch = (now_s() - t0) / NL * 1e6;
    t0 = now_s();
    for (int i = 0; i < 2000; i++) { k_empty<<<1, 32>>>(); CK(cudaStreamSynchronize(0)); }
    double t_sync = (now_s() - t0) / 2000 * 1e6;
    // graph of 100 empty kernels
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < 100; i++) k_empty<<<1, 32, 0, s>>>();
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    t0 = now_s();
    for (int i = 0; i < 200; i++) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    double t_graph = (now_s() - t0) / (200 * 100) * 1e6;
    printf("\n== launch ==\nback-to-back launch: %.2f us/kernel | launch+sync roundtrip: %.2f us | in CUDA graph: %.2f us/kernel\n",
           t_launch, t_sync, t_graph);

    // ---- VRAM read bandwidth
    size_t vbytes = 1ull << 30;
    uint4* d; unsigned* dout;
    CK(cudaMalloc(&d, vbytes)); CK(cudaMalloc(&dout, 4));
    CK(cudaMemset(d, 1, vbytes));
    printf("\n== VRAM ==\n");
    cudaEvent_t ea, eb; CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
    for (int blocks : {p.multiProcessorCount * 4, p.multiProcessorCount * 8, p.multiProcessorCount * 16}) {
        k_read<<<blocks, 256>>>(d, vbytes / 16, dout);
        CK(cudaEventRecord(ea));
        for (int r = 0; r < 10; r++) k_read<<<blocks, 256>>>(d, vbytes / 16, dout);
        CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
        float ms; CK(cudaEventElapsedTime(&ms, ea, eb));
        printf("read kernel %d blocks x256: %.1f GB/s\n", blocks, 10.0 * vbytes / (ms * 1e-3) / 1e9);
    }
    // small working sets (L2-resident-ish / GEMV-sized)
    for (size_t sz : {(size_t)2 << 20, (size_t)16 << 20, (size_t)64 << 20}) {
        int blocks = p.multiProcessorCount * 8;
        CK(cudaEventRecord(ea));
        for (int r = 0; r < 200; r++) k_read<<<blocks, 256>>>(d, sz / 16, dout);
        CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
        float ms; CK(cudaEventElapsedTime(&ms, ea, eb));
        printf("read %3zu MiB x200 (incl. launch): %.1f GB/s, %.1f us/pass\n", sz >> 20, 200.0 * sz / (ms * 1e-3) / 1e9, ms * 1e3 / 200);
    }
    uint4* d2; CK(cudaMalloc(&d2, 256 << 20));
    printf("D2D memcpy 256 MiB: %.1f GB/s (read+write counted once)\n", gpu_copy_bw(d2, d, 256 << 20, cudaMemcpyDeviceToDevice, 10));

    // ---- PCIe
    printf("\n== PCIe ==\n");
    size_t hb = 256ull << 20;
    void* hpin; CK(cudaHostAlloc(&hpin, hb, cudaHostAllocDefault));
    void* hpage = aligned_alloc(4096, hb);
    memset(hpin, 1, hb); memset(hpage, 1, hb);
    for (size_t sz : {(size_t)1 << 20, (size_t)8 << 20, (size_t)64 << 20, (size_t)256 << 20}) {
        double h2d_pin = gpu_copy_bw(d, hpin, sz, cudaMemcpyHostToDevice, 8);
        double d2h_pin = gpu_copy_bw(hpin, d, sz, cudaMemcpyDeviceToHost, 8);
        double h2d_pg = gpu_copy_bw(d, hpage, sz, cudaMemcpyHostToDevice, 4);
        double d2h_pg = gpu_copy_bw(hpage, d, sz, cudaMemcpyDeviceToHost, 4);
        printf("%4zu MiB: H2D pinned %.2f | D2H pinned %.2f | H2D pageable %.2f | D2H pageable %.2f GB/s\n",
               sz >> 20, h2d_pin, d2h_pin, h2d_pg, d2h_pg);
    }
    // small transfer latency (activation-sized)
    {
        const int N = 2000;
        t0 = now_s();
        for (int i = 0; i < N; i++) { CK(cudaMemcpyAsync(d, hpin, 20480, cudaMemcpyHostToDevice, s)); CK(cudaStreamSynchronize(s)); }
        double h2d_small = (now_s() - t0) / N * 1e6;
        t0 = now_s();
        for (int i = 0; i < N; i++) { CK(cudaMemcpyAsync(hpin, d, 20480, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); }
        double d2h_small = (now_s() - t0) / N * 1e6;
        printf("20 KiB roundtrip (copy+sync): H2D %.1f us, D2H %.1f us\n", h2d_small, d2h_small);
    }
    // overlap: H2D on one stream while a long kernel runs on another
    {
        cudaStream_t sc; CK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
        CK(cudaEventRecord(ea, s));
        k_spin<<<1, 1, 0, sc>>>((long long)clk_khz * 1000 / 10);  // ~100 ms
        for (int r = 0; r < 4; r++) CK(cudaMemcpyAsync(d, hpin, hb, cudaMemcpyHostToDevice, s));
        CK(cudaEventRecord(eb, s)); CK(cudaEventSynchronize(eb));
        float ms; CK(cudaEventElapsedTime(&ms, ea, eb));
        CK(cudaDeviceSynchronize());
        printf("H2D pinned while kernel runs on other stream: %.2f GB/s\n", 4.0 * hb / (ms * 1e-3) / 1e9);
    }

    // ---- CPU DRAM
    printf("\n== CPU DRAM ==\n");
    size_t cb = 2ull << 30;
    uint8_t* cbuf = (uint8_t*)aligned_alloc(64, cb);
    memset(cbuf, 3, cb);
    for (int nt : {1, 2, 4, 6, 8, 12, 16})
        printf("read %2d threads: %.1f GB/s\n", nt, cpu_read_bw(cbuf, cb, nt, 3));

    // ---- contention: CPU read (8 threads) concurrent with pinned H2D DMA
    {
        std::atomic<bool> stop{false};
        std::atomic<long long> copied{0};
        std::thread dma([&] {
            CK(cudaSetDevice(0));
            while (!stop.load()) {
                CK(cudaMemcpyAsync(d, hpin, hb, cudaMemcpyHostToDevice, s));
                CK(cudaStreamSynchronize(s));
                copied += hb;
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        long long c0 = copied.load();
        double tA = now_s();
        double cpu_bw = cpu_read_bw(cbuf, cb, 8, 5);
        double tB = now_s();
        long long c1 = copied.load();
        stop = true; dma.join();
        printf("contention: CPU read 8thr %.1f GB/s while H2D pinned DMA %.2f GB/s\n", cpu_bw, (c1 - c0) / (tB - tA) / 1e9);
    }
    return 0;
}
