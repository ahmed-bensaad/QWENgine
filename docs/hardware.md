# Target hardware

All figures were measured on the target laptop (`build/hw_probe`, micro-benchmarks, `nvidia-smi`) unless marked
"spec". They are the constants the engine is designed around.

## GPU: Quadro RTX 4000 Mobile (TU104GLM, Turing)

| Property | Value |
|---|---|
| Compute capability | 7.5 (sm_75), 40 SMs, 64 FP32 lanes and 8 tensor cores per SM |
| Clocks | SM 1560 MHz max, memory 7001 MHz (14 Gbps GDDR6), 256-bit bus |
| VRAM | 8192 MiB physical, 7778 MiB visible, ~0.5-0.7 GB used by the desktop |
| VRAM read bandwidth | 416 GB/s measured (448 theoretical) |
| VRAM write bandwidth | 410 GB/s `cudaMemset`; 357 GB/s when a warp writes 2 KB contiguously, 315 GB/s with coalesced 16 B stores, 233 GB/s when each thread stores 64 contiguous bytes |
| L2 / shared memory | 4 MiB L2; 64 KiB shared memory per block (opt-in), 64K registers per SM |
| Tensor cores | `mma.m16n8k8` FP16 → FP32: 64 TFLOPS theoretical at 1560 MHz; cuBLAS reaches 40-45 TFLOPS on well-shaped GEMMs (row counts multiples of 512), 11-18 TFLOPS on badly shaped ones |
| Power | 110 W limit; long prefills hit it (~102 W) and the SM clock drops to ~1250-1350 MHz |
| Kernel launch | 1.66 µs back-to-back, 6.2 µs launch + synchronize |
| Missing features | no `cp.async`, no `mma.m16n8k16` FP16, no FP8, no BF16 tensor cores (sm_80+) |

## PCIe

| Property | Value |
|---|---|
| Link | Gen3, **x8** (the GPU supports x16, the board trains x8); idles at Gen1 |
| Host → device, pinned | 6.47 GB/s (pageable 6.3) |
| Device → host, pinned | 6.55 GB/s |
| Round trip of a 20 KiB copy + synchronize | ~9 µs |
| Zero-copy reads by GPU kernels from mapped host memory | ~4.4 GB/s (80 KiB in ~18 µs, independent of parallelism) |
| Copy while a kernel runs | 6.45 GB/s (copy engines independent of SMs) |

## CPU: Intel Core i9-9880H

| Property | Value |
|---|---|
| Cores | 8 cores / 16 threads (CPU i and i+8 share core i), 2.3 GHz base, 4.8 GHz turbo |
| ISA | AVX2, FMA, F16C, BMI2 (`pdep`/`pext`); no AVX-512, no VNNI. `vpgather` is fast (no gather-data-sampling microcode mitigation on this CPU) |
| Caches | 32 KiB L1d and 256 KiB L2 per core, 16 MiB shared L3 |
| Sustained clock under decode | 2.5-2.7 GHz at the 100 °C thermal limit (package temperature reaches ~98 °C within one benchmark) |

## Host memory

| Property | Value |
|---|---|
| Capacity | 32 GB DDR4 (30 GiB usable) |
| Read bandwidth | 23.7 GB/s (1 thread), 33.8 (2), 37.3 (4), 36.0 (8), 33.7 (16 threads) |
| Read bandwidth while PCIe DMA runs | 28.7 GB/s CPU + 6.4 GB/s DMA (same DRAM) |

## Consequences for the design

- ~9.7 GB of weights per pass vs ~6.5 GB of VRAM for weights: ~2.8 GB are read from host memory every pass, at
  ~29-36 GB/s by the CPU or 6.5 GB/s over PCIe. This sets decode speed; see `docs/ARCHITECTURE.md` §1.
- The CPU is thermally throttled and some quantization formats are compute-bound on it, so the CPU kernels'
  instruction count matters as much as their memory traffic.
- Small CPU↔GPU transfers cost microseconds of latency: activations go through mapped pinned memory, and the host
  synchronizes only on events.
- Benchmarks drift ±5-10% with temperature; comparisons use `bench/ab.py` (cool-down + ABBA order).
