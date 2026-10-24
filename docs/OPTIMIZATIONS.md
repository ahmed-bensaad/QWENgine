# How QWENgine is tuned to its hardware

QWENgine runs a 27-billion-parameter model whose weights take 9.7 GB on a laptop whose GPU has 8 GB. On that
machine and with that exact GGUF, llama.cpp decodes at 3.89 tokens/s and prefills at 234 tokens/s. QWENgine decodes
at ~9 tokens/s without speculation and ~15 tokens/s with it, and prefills at ~490 tokens/s.

There is no single trick behind this. The engine was built from measurements of this one machine: what each
resource can deliver, and which one is the limit at each step. This page goes through those limits one by one, with
the technique that answers each and what it gained, measured end to end. Implementation details are in
[ARCHITECTURE.md](ARCHITECTURE.md), and the raw measurements of the machine in [hardware.md](hardware.md).

---

## 1. The machine as a set of bandwidths

| Resource | Measured | What it limits |
|---|---|---|
| VRAM | 7778 MiB visible, 0.5-0.7 GB of it used by the desktop | ~6.5 GB of weights fit; ~2.8 GB stay in host memory |
| GPU memory bandwidth | 416 GB/s | reading the resident weights costs ~16 ms per pass |
| Host memory bandwidth | 36 GB/s (8 threads); 29 GB/s while PCIe DMA runs | reading the host weights costs ≥ 80 ms per pass |
| PCIe | Gen3 **x8** (the board does not train x16): 6.5 GB/s | streaming the host weights to the GPU is 5x slower than reading them on the CPU |
| CPU | i9-9880H, 8 cores, AVX2 / FMA / F16C, no AVX-512, no VNNI; 2.5-2.7 GHz sustained at the 100 °C limit | some weight formats are compute-bound on the CPU |
| GPU | Quadro RTX 4000 Mobile, Turing sm_75, 40 SMs; FP16 tensor cores (`mma.m16n8k8`), `dp4a`; no `cp.async`, no BF16/FP8; 110 W power cap | kernel design; prefill speed |

Generating a token reads every weight once. With 6.5 GB in VRAM and 2.8 GB in host memory, the GPU part takes
~16 ms and the CPU part at least 80 ms. **Decode speed is therefore set by the host share**, and the decode design
does three things about it:

1. keep the host share small and well chosen (§2.1-2.2);
2. read it as fast as the DRAM allows (§2.3-2.5);
3. read it once for several tokens (§2.7).

---

## 2. Decode

### 2.1 Split every matrix, and run both halves at once

General-purpose runtimes offload whole layers to the GPU (llama.cpp's `-ngl`): the CPU layers and GPU layers then
take turns, and each device idles while the other works. QWENgine splits **every large matrix by rows**: the first
rows are resident in VRAM and computed by the GPU, the remaining rows are in host memory and computed by the CPU,
**at the same time**. A layer has three such groups: input projections, output projection, FFN. Each group costs
`max(T_gpu, T_cpu)` plus a short serial chain, so the split is sized to make both sides finish together.

The FFN is split by blocks of 256 intermediate channels: the same blocks are taken from the gate and up rows and
from the down columns. The CPU then computes `silu(gate) * up` and its partial `down` product on its own blocks, and
the two halves never exchange the 17408-wide intermediate vector.

### 2.2 Choose which rows go where by per-format cost

The GGUF mixes ten quantization formats, chosen per tensor by the quantizer (GSQ-RCO). On the GPU, every format
decodes at close to memory bandwidth. On the throttled CPU, decoding cost per byte differs a lot between formats.
IQ4_XS, a simple table lookup, runs at 31.6 GB/s; IQ1_M, with grid lookups and more arithmetic per byte, runs at
8.7 GB/s. A byte of IQ1_M in host memory thus costs 3.6x more CPU time than a byte of IQ4_XS.

The residency planner (`src/runtime/planner.cpp`) uses the measured rates of every format on both devices. Starting
from mandatory minimums, it moves to VRAM the units that save the most CPU time per byte of VRAM, without pushing
any group past its balance point. Result: the IQ1/IQ2 formats live mostly in VRAM, IQ4_XS mostly in host memory.
The CPU rates are planned at 0.8x their cool-machine value, because the CPU spends decode at its thermal limit
(+2%).

### 2.3 CPU kernels: AVX2 without VNNI, gathers that are fast on this CPU

The CPU computes integer dot products between quantized weights and 8-bit quantized activations (ggml's Q8_K
format), as exact integer sums with float scales. Without AVX-512 VNNI, each 32-weight sub-block goes through
`maddubs` (u8 × s8 → s16), `madd` with the sub-block scale (→ s32) and an add.

- **Gathers for the IQ grids.** The IQ formats store indices into small codebooks ("grids"). This CPU has no
  microcode mitigation for Gather Data Sampling ("Downfall"), so AVX2 `vpgather` is fast on it. The kernels gather
  grid entries straight from L1-resident tables, about 2x the speed of ggml's IQ kernels on the same tensors.
- **Several token columns per weight block.** Speculative verify passes multiply each weight block by 3-4
  activation columns. The kernels are templated on the column count: a block is decoded once and the per-column
  work is only load, sign, `maddubs`, `madd`, add.
- **Exact re-encodings (R4, R2).** With 3-4 columns the gathers themselves become the bottleneck. At load, the
  host rows of the IQ3 formats are re-encoded into a 4-bit code per weight (decoded with one `pshufb`), and those
  of the IQ2 formats into 2-bit magnitude codes plus sign bits. Both layouts reproduce the gather kernels' integer
  products exactly. They cost 22-60% more bytes and run 20-40% faster: worth it only because the throttled CPU is
  compute-bound on these formats (+3.6% and +1.6% end to end).
- **Register barriers.** In the 3-4 column templates GCC reassociated the accumulations and spilled them to the
  stack. An empty `asm` with a register constraint after each accumulation keeps one register chain per column
  (-3.0% verify-pass time).

### 2.4 CPU threads under a thermal limit

- 8 worker threads pinned to the 8 physical cores, spinning on a generation counter: about 1 µs to dispatch a job,
  ~2-3 times per layer. Hyperthreads gave nothing (16 threads) or lost ~5% (12 threads): at the thermal limit, energy
  per byte matters as much as hiding latency.
- At 100 °C the cores do not run at equal speeds, and per-row time estimates are imperfect. Work is cut into chunks
  taken from an atomic counter. The first chunks take 1/(2 × threads) of the estimated time left, later ones stay
  small, so all threads finish within microseconds of each other (guided self-scheduling, -4.8% verify-pass time).

### 2.5 PCIe: stream what the CPU cannot hide, exchange activations without copies

**Weight streaming.** The GPU's copy engines run independently of its compute units, so the PCIe link is free
bandwidth during a pass. Each pass streams a slice of the host rows to the GPU through a ring of 8 slots of 8 MB,
and the GPU computes those rows instead of the CPU. This only pays while the copy finishes before the GPU needs the
rows. It also takes DRAM bandwidth from the CPU: 29 GB/s for the CPU plus 6.4 GB/s of DMA, instead of 36 GB/s.
The streaming planner simulates the copy timeline of a pass (ring slots, per-slice overheads, per-group CPU/GPU
times) and adds slices while the simulated pass gets faster. About 450 MiB per pass end up streamed (+2.9% at short
context, +5.2% at 100K).

**Activations.** A pass synchronizes CPU and GPU a few times per layer, 64 layers deep, and a small copy round trip
costs ~9 µs. Activations therefore never go through `cudaMemcpy`:
- the CPU writes its outputs into mapped pinned memory, and the next GPU kernel reads them directly;
- the GPU's norm kernel writes the CPU's next input, already quantized to Q8_K, into the CPU's buffer;
- the host waits only on events, by spinning.

These fused chains gave +2.1%; handing over Q8_K instead of FP32 another +0.6%, within noise.

### 2.6 VRAM is worth more than anything else

Every megabyte of VRAM holds weights that would otherwise be read at 36 GB/s by the CPU instead of 416 GB/s by the
GPU. Measured: 200 MiB more resident weights gave +5.9% decode. Hence:

- The KV cache is allocated for the current context only, and doubles on demand (`Engine::resize` re-plans the
  weights). The server starts at 4096 tokens.
- The 64 MiB decode streaming ring doubles as the FP16 dequantization scratch of prefill.
- Prefill places its activation buffers in the VRAM of the resident LM-head rows, which prefill does not use, and
  restores those rows from a pinned host copy afterwards (~30 ms). This let the VRAM reserve drop from 260 to 60 MiB.
- The cuBLAS handle is created before planning so its memory is accounted for.
- The desktop's 0.5-0.7 GB of VRAM is the largest remaining reserve. Driving the display from the integrated GPU
  (hybrid graphics mode) or running headless would give it to the weights.

### 2.7 Speculative decoding suits the split especially well

The fixed cost of a pass is reading the host weights. Verifying several tokens at once multiplies them with more
activation columns but reads the weights only once: a pass costs ~110 ms with one token and ~144 ms with ~3.3 tokens.
The model ships a multi-token-prediction (MTP) block, one extra layer trained to predict the token after next.
QWENgine uses it as a draft model:

- **Cheap drafts.** The MTP layer is kept fully in VRAM, requantized at load from Q6_K to Q4_K (exact port of
  ggml's quantizer; acceptance unchanged, +2%). Its output head is the LM head's resident 64K-row prefix, so a
  draft never touches the CPU.
- **Exact output.** The longest draft prefix matching the target's choices is accepted, plus the target's own next
  token. Greedy output is identical to plain decoding, and sampled output follows the exact target distribution.
- **DeltaNet rollback without copies.** The DeltaNet recurrent state, 3 MiB per layer, cannot be restored cheaply.
  A verify pass computes the drafts' outputs without advancing the stored state and keeps their inputs. The next
  pass first replays the accepted tokens into the state.
- **Adaptive draft length, up to 6 drafts.** Each cycle, k ∈ {1, ..., 6} is chosen to maximize expected tokens per
  second, from per-depth acceptance and measured pass times. On code edits the MTP block's drafts match the target
  90-97% of the time at every depth, so a pass of 7 tokens (210 ms instead of 146 ms for 4) yields ~5.5 tokens:
  agent-style generation runs ~10% faster than with at most 3 drafts. Prose stays at k = 1-4.
- **Overlapped drafting.** The GPU finishes its LM-head rows ~10 ms before the CPU. The engine guesses the cycle's
  outcome from the GPU's rows and drafts the next cycle in that window. The guess is almost always right, since the
  frequent tokens are in the GPU's rows (+3.0%).

Speculation took the corpus average from ~9 to ~15 tokens/s; code-like text reaches ~20.

### 2.8 GPU kernels written for Turing

- **Matrix-vector products:** one warp per row; activations quantized to int8 per 32 elements; weights decoded per
  format into int8 and multiplied with `dp4a`; grid tables in shared memory; 1-7 columns sharing each decoded block.
- **Decode attention:** tensor-core flash-decoding with `mma.m16n8k8`, the FP16 MMA shape Turing supports. A
  16-row tile holds 2 tokens × the 6 query heads of a KV group, and K fragments load straight from the cache. KV
  dequantization uses the `0x6400 | x` magic-number trick, so one `HMUL2` by the scale rounds once.
- **No `cp.async` on sm_75:** the prefill attention kernel uses 2 producer warps that load and dequantize the next
  K/V tile while 6 compute warps work on the current one (2.0-2.3x kernel speed).

---

## 3. Prefill

Prefill processes many tokens per weight read, so it is compute-bound and runs entirely on the GPU's tensor cores.

- **FP16 GEMMs on dequantized weight blocks.** Resident rows are dequantized to FP16 block by block and multiplied
  with cuBLAS. On Turing, row blocks that are multiples of 512 run at 31-45 TFLOPS, other shapes at 13-18; aligning
  the blocks gave +31-39%.
- **Host weights streamed once per chunk.** The 2.8 GB of host rows cross PCIe once per chunk (~0.44 s), in 24 MB
  pieces through a two-slot staging ring, overlapped with compute. Larger chunks amortize this and the
  dequantization: 1024-token chunks, made possible by packing the activations of all phases of a layer into one
  shared region (186 KB per token), gave +10-13%.
- **Attention:** the producer/consumer flash-attention kernel above (+12-18% at 16-32K tokens).
- **DeltaNet recurrence:** sequential over tokens with the state in registers. 4 lanes share each state column, so a
  dot product reduces over 4 lanes instead of 32 (+3.9%).
- **Power:** long prefills hit the GPU's 110 W cap (~102 W) and the SM clock falls from 1560 to ~1300 MHz. Part
  of every kernel speedup turns into lower clocks instead.
- **Reproducibility under a moving VRAM budget:** the desktop's VRAM use changes from second to second, so prefill
  buffers sometimes do not fit. They then move into the borrowed LM-head rows rather than shrinking the staging
  pieces: the piece size sets the cuBLAS shapes, and other shapes round differently.

At 4K tokens, GEMMs take ~68% of prefill time, DeltaNet ~12%, dequantization ~10%, attention ~4%. Attention grows
with the square of the prompt and takes about half of a 100K-token prefill.

---

## 4. Running on other hardware

QWENgine was tuned on one machine. The model's shapes are compile-time constants, and the kernels were written
for sm_75 (`-DCMAKE_CUDA_ARCHITECTURES=native` builds them for another GPU of compute capability 7.5 or newer) and
AVX2 (`-march=native`). The planner adapts to free VRAM, but its device rates are measurements of this CPU and GPU
(`DeviceRates::measured_defaults`). If you try another machine, these are the parts that change, and please share
the result in [BENCHMARKS.md](../BENCHMARKS.md):

| Difference | Effect |
|---|---|
| More VRAM (12 GB or more) | most or all weights resident; decode becomes GPU-bound and much faster; the CPU kernels matter less |
| PCIe Gen3/4 x16 | 2-4x the streaming bandwidth; the streaming planner would stream more rows, and prefill would wait less on copies |
| AVX-512 VNNI CPU | `vpdpbusd` would replace the `maddubs` + `madd` chain; new kernels needed for the gain |
| CPU with the Downfall mitigation | `vpgather` becomes several times slower; the R4/R2 re-encodings would matter more |
| Ampere or newer GPU | `cp.async`, `mma.m16n8k16`, BF16 tensor cores: the attention kernels would be written differently |
| Different desktop VRAM use, thermals | the planner adapts to free VRAM, but the CPU rates (x0.8 thermal factor) are tuned here |

The method transfers better than the code: measure every resource, find the binding limit, change one thing, and
keep it only if an A/B run on the end-to-end metric says so ([DEVELOPMENT.md](DEVELOPMENT.md)).
