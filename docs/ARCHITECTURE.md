# QWENgine architecture

QWENgine runs one model, **Qwen3.8-27B** in the GGUF quantization
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS` (optionally with its MTP block), on one machine: a laptop with a
**Quadro RTX 4000 Mobile (8 GB, Turing sm_75)**, an **i9-9880H** and **32 GB DDR4**, up to **100K tokens of
context**. Everything in the code (tensor shapes, formats, tile sizes, cost constants, thread placement) is
specialized for that pair; nothing is meant to be portable.

This document describes what the engine computes and how. Measurements quoted here were made on that machine,
whose characteristics are in `docs/hardware.md`.

---

## 1. The problem in numbers

| Quantity | Value | Consequence |
|---|---|---|
| Weight bytes read per generated token (everything but the embedding table) | ~9.68 GB | does not fit in VRAM |
| VRAM | 7778 MiB visible, ~0.5-0.7 GB used by the desktop | ~6.5 GB of weights can be resident |
| GPU memory bandwidth | 416 GB/s measured | the resident part costs ~16 ms per pass |
| Host DRAM bandwidth | 36-37 GB/s (≥4 threads), 29 GB/s while PCIe DMA runs | the host part (~2.8 GB) costs ≥ 80 ms per pass |
| PCIe | Gen3 x8 (the board does not train x16): 6.5 GB/s | streaming host weights to the GPU is slower than reading them on the CPU |
| CPU | 8 cores, AVX2/FMA/F16C (no AVX-512), sustains 2.5-2.7 GHz at 100 °C | the CPU share is partly compute-bound |

A pass over the weights is therefore dominated by the ~2.8 GB that stay in host memory. The design follows from
that:

1. **Split every matrix** between GPU (VRAM-resident rows) and CPU (host rows) and run both halves at the same
   time, sized so both finish together (residency planner, §4).
2. **Stream** a further slice of the host rows to the GPU over PCIe during each pass, using link bandwidth the
   CPU does not need (streaming planner, §4.3).
3. **Verify several tokens per pass** (speculative decoding with the model's own MTP block, §7): the host bytes
   are read once for 2-7 tokens.
4. Make the CPU kernels reach the memory limit even with several token columns (§6.2), and keep all other work
   (attention, recurrence, norms) on the GPU so the CPU only does matrix-vector products.

Reference point: llama.cpp on the same machine and GGUF decodes at 3.89 tok/s (2.89 at 8K depth) and prefills
pp512 at 234 tok/s. QWENgine decodes at ~9 tok/s without speculation and ~15 tok/s with it (corpus average),
and prefills at ~470-500 tok/s at 4K.

---

## 2. The model as implemented

Architecture `qwen35` (verified against the GGUF and llama.cpp's `src/models/qwen35.cpp`):

| Parameter | Value |
|---|---|
| Hidden size | 5120 |
| Layers | 64: layer `il` is full attention when `(il + 1) % 4 == 0` (16 layers), Gated DeltaNet otherwise (48) |
| FFN | SwiGLU, intermediate 17408: `down(silu(gate(x)) * up(x))` |
| Attention | 24 query heads, 4 KV heads (GQA group 6), head dim 256 |
| Gated DeltaNet | 16 Q/K heads × 128, 48 V heads × 128 (V head `h` uses Q/K head `h % 16`), causal conv width 4 over 10240 channels |
| Vocabulary | 248,320 (GPT-2 byte-level BPE, pre-tokenizer `qwen35`), EOS 248046 `<|im_end|>`, no BOS |
| Norms | RMSNorm, eps 1e-6 (GGUF weights already include Qwen's "+1") |
| RoPE | IMROPE with sections [11, 11, 10, 0], 64 rotary dims of 256, θ = 1e7; for text positions it equals NeoX RoPE on the first 64 dims |

Per layer (`x` is the residual stream):

```
xn = rmsnorm(x) * attn_norm
DeltaNet layer:                                  Full-attention layer:
  qkv = W_qkv xn            (10240)                q|gate = W_q xn        (24 x [256 q | 256 gate])
  z   = W_gate xn           (6144)                 k = W_k xn, v = W_v xn (4 x 256 each)
  beta  = sigmoid(W_beta xn)          (48, BF16)   q, k: per-head RMSNorm, RoPE at the token position
  g     = softplus(W_alpha xn + dt_bias) * ssm_a   append k, v to the KV cache
  q, k, v = silu(causal_conv4(qkv))                o = softmax(q K^T / 16) V   per head
  q, k: L2-normalized; decay = e^g                 o = o * sigmoid(gate)
  S = decay*S + k (beta (v - decay*S^T k))^T       y = W_o o
  o = S^T q / sqrt(128)
  o = rmsnorm(o) * ssm_norm * silu(z)  (per head)
  y = W_ssm_out o
x = x + y
xn = rmsnorm(x) * post_attention_norm
x = x + W_down(silu(W_gate xn) * W_up xn)
```

After the last layer, `logits = W_output · rmsnorm(x)`. The embedding table (`token_embd`, IQ2_S, 388 MiB) is
read one row per token on the CPU and never uploaded.

**MTP block** (only in the `-mtp.gguf` variant, as `blk.64`): one extra decoder layer used for drafting. Input
for position q is `eh_proj([enorm(embed(x_q)) | hnorm(h_{q-1})])` (10240 -> 5120), where `h_{q-1}` is the main
model's final hidden state (after `output_norm`) at the previous position; then a full-attention layer with its
own KV cache, an FFN and `shared_head_norm`, followed by the shared LM head.

**Weight formats.** The GGUF mixes ten quantization formats per tensor (GSQ-RCO non-uniform mixed precision),
e.g. layer 0 has an IQ4_XS qkv, IQ1_S gate, IQ1_M up and IQ2_XS down projection. Totals: IQ3_S 2159 MiB, IQ3_XXS
1949, IQ2_S 1908, IQ4_XS 1383, Q2_K 594, IQ2_XS 588, IQ2_XXS 541, Q4_K 366, IQ1_M 38, IQ1_S 35, plus BF16
(DeltaNet alpha/beta) and F32 (norms, conv). The MTP block is Q6_K (332 MiB) and is requantized at load (§7.1).
Exact inventory: `docs/gguf_inventory.md`.

All weight formats are consumed as stored (bit layouts of ggml, `src/quant/quant_formats.h`); the reference
dequantizers in `src/quant/quant_ref.cpp` are bit-exact with ggml (`tests/unit/test_quant_ref`).

---

## 3. Memory

### 3.1 VRAM (default: MTP GGUF, 4096-token context, Q8_0 KV)

| Region | Size | Contents |
|---|---|---|
| State, KV, activations (`d_misc_`) | 383 MiB | KV cache of 17 attention layers (16 + MTP; Q8_0: 2176 B per layer and token), DeltaNet state (48 layers × (3 MiB recurrent state + 0.23 MiB conv windows)), lazy-commit save buffers, decode activations for 4 token columns, logits, attention / top-k work buffers, small GPU-only weights (norms, BF16 alpha/beta, conv kernels) |
| cuBLAS handle | ~12 MiB | created before planning so its memory is accounted |
| Decode streaming ring (`d_slots_`) | 64 MiB | 8 slots × 8 MiB for weights streamed during decode; reused as prefill FP16 scratch |
| Weight arena (`d_arena_`) | 6552 MiB | the GPU share of every split matrix, LM-head rows `[0, 65536)` last |
| Free | 60 MiB reserve + 64 MiB slack | left to the desktop / driver |

The KV cache is sized for the current context capacity only (it grows on demand, §9), because every unused KV
byte would otherwise be VRAM taken from the weights.

### 3.2 Host memory

| Region | Size (default) | Contents |
|---|---|---|
| GGUF mapping | 9.7 GB (page cache) | `mmap` of the model file; source for uploads |
| Host weight shares (`h_weights_`) | 2849 MiB + 170 MiB | the CPU share of every split matrix, plus a copy of the resident LM-head rows; anonymous huge-page memory registered with CUDA (pinned) so it can be DMA'd |
| Re-encoded copies (`h_re_`) | 1738 MiB | R4/R2 re-encodings of the IQ3/IQ2 host rows, read only by the CPU kernels (§6.2) |
| Pinned activation buffers (`h_base_`) | a few MiB | mapped into the GPU address space: CPU outputs read directly by GPU kernels, GPU outputs written directly into CPU buffers |

Peak RSS is ~14-18 GB (17.7 GiB at 100K context, where the host share is larger).

---

## 4. Residency and streaming planners

### 4.1 Units and sync groups

A decode pass is a fixed sequence of **sync groups**: three per layer (input projections; output projection;
FFN) plus the LM head, plus the MTP block's groups. Inside a group the GPU part and the CPU part run concurrently,
so a group costs `max(T_gpu, T_cpu)` plus a fixed chain (§5).

Each large matrix is a **plan item** split at a granularity of:

- single rows, for the attention / DeltaNet projections;
- 256-wide blocks of the intermediate dimension for the FFN: the same blocks are taken from the gate and up rows
  (row split) and from the down columns (column split), so the CPU computes `silu(gate)*up` and its partial
  `down` product on its own blocks without exchanging the intermediate vector;
- 64-row units for the LM head, with at least `draft_rows = 65536` rows resident (the MTP draft head reads them).

### 4.2 Residency planner (`src/runtime/planner.cpp`, `Engine::build_plan`)

Cost model: per format, measured GEMV throughput on each device (`DeviceRates::measured_defaults`: e.g. CPU
IQ4_XS 31.6 GB/s, IQ1_M 8.7 GB/s; GPU 85-320 GB/s), CPU rates scaled by 0.8 for the thermally throttled
steady state, and measured multi-column factors (time of a pass over the weights with n = 2..4 token columns
relative to one; e.g. IQ4_XS 1.02 / 1.05 / 1.18 on the CPU). With the MTP block the plan is made for speculative
cycles of `spec_k = 2` drafts: trunk and LM head cost 3 columns, MTP groups are weighted by 2 executions.

Given the VRAM budget (free VRAM minus reserve, slack and the streaming ring), the planner greedily moves to the
GPU the units that save the most CPU time per byte, never pushing a group past its balance point, starting from
the mandatory minimums. The result lists, per matrix, the number of resident rows (`SplitMat::gpu_part`).
Formats that are slow on the CPU (IQ2_*, IQ1_*) end up mostly in VRAM; IQ4_XS, the fastest CPU format, mostly in
host memory.

### 4.3 Streaming planner (`Engine::plan_streaming`)

Candidates are the first rows (32-row units) or FFN blocks of each matrix's host share. Adding a candidate moves its CPU time
to the GPU but requires a PCIe copy into one of the 8 ring slots before the GPU can compute it. The planner
simulates a steady-state pass: the copy stream (6.0 GB/s model), the ring (a slot is reused only after the
kernel reading it finished), per-slice overheads and the per-group CPU/GPU times. It adds candidates while the
simulated pass gets faster. Typical result: ~450 MiB per pass streamed in 79 slices. The streamed rows keep
their place in the host buffer (`SplitMat::stream_part` rows at the start of the host share); slices of the FFN
down columns use pinned contiguous duplicates.

---

## 5. Decode pass (`Engine::decode_batch`)

`decode_batch(tokens, n)` processes `n ≤ 7` consecutive tokens with one pass over the weights; every weight
sub-block is decoded once and multiplied with all `n` activation columns. Its results are bit-identical to `n`
calls of `decode()` (`integration.decode_batch.*`).

### 5.1 Data flow of one layer

```
GPU stream                                         CPU (8 pinned threads)
-----------------------------------------------    -------------------------------------------
residual + RMSNorm kernel
  -> xn as int8/32 (GPU) and as Q8_K into the
     CPU's pinned buffer; event
GPU share of the input projections                 wait(event) ── input projections, CPU rows
  + streamed slices (ring)                           (Q8_K activations, AVX2 kernels)
DeltaNet recurrence / attention: reads the CPU     ── outputs into mapped pinned memory
  rows straight from mapped host memory; event
GPU share of the output projection                 wait(event) ── output projection, CPU rows
residual + RMSNorm (adds the CPU rows read from
  mapped memory); event
GPU share of gate/up + SwiGLU + down (+ streamed)  wait(event) ── gate/up rows, SwiGLU, down partial
                                                      sums over its FFN blocks
next layer's norm adds the CPU partial sums
```

After the last layer: final norm, LM head (GPU: resident and streamed rows, top-k on the GPU; CPU: remaining
rows, top-k per thread), merged top-k per token (k ≤ 32) or the full logits on request.

Points that make this fast:

- **No copies for activations.** CPU outputs are written into mapped pinned memory and the next GPU kernel reads
  them directly (`cuda::RowSrc` = device head + host tail); the norm kernel writes the CPU's Q8_K activations
  straight into its buffer. Weight DMA never queues in front of activation transfers.
- **GPU work ahead of the CPU.** All GPU kernels of a group are enqueued before the CPU waits; the host only
  synchronizes on events (spin-wait), ~2-3 per layer.
- **One quantization per consumer.** GPU GEMVs take int8 activations with a scale and a sum per 32 elements
  (`ActQ8`); CPU kernels take ggml's Q8_K (256-element blocks with 16-element sums), produced bit-exactly by the
  GPU norm kernel.

### 5.2 CPU side

- `cpu::ThreadPool` (`src/cpu/thread_pool.h`): 8 workers pinned to the 8 physical cores, spinning on a
  generation counter (~1 µs dispatch); the calling thread is worker 0.
- A CPU job is a list of segments (matrix rows × token columns). Rows are laid end to end by estimated time
  (bytes × per-format cost × column factor) and cut into chunks taken dynamically from an atomic counter
  (guided self-scheduling): each of the first chunks takes 1/(2 × threads) of the estimated time still left, then
  chunks keep a minimum of max(4 µs, job/512). Large chunks first keep the counter traffic low; small chunks at
  the end let every thread finish at nearly the same time despite mixed formats and thermal noise.
- Kernels are compiled with register barriers (`Q38_KEEP`, an empty `asm` with a register constraint) after each
  per-column accumulation: without them GCC reassociated the 3-4 column accumulations and spilled them to the
  stack.
- Kernels: §6.2.

### 5.3 Where the time goes

Speculative verify pass with ~3.3 token columns at 4K context (~144 ms): CPU matrix-vector work ~126 ms (FFN
gate/up ~60% of it, LM head ~13%), CPU waiting on the GPU ~10 ms (PCIe-bound norm transfers, the sequential
DeltaNet recurrence), thread imbalance ~5 ms. The CPU kernels run at 18-31 GB/s depending on format and column
count, against a DRAM limit of ~29 GB/s while DMA runs. Plain single-token decode: ~110 ms per token.

---

## 6. Kernels

### 6.1 GPU matrix-vector (`src/cuda/gemv.cu`, `quant_dot.cuh`)

One warp per row at a time, lanes over 32-weight sub-blocks; activations staged in shared memory per CTA. Each
format has a device decoder (`QDot<T>`) that turns a sub-block into int8 values and computes `dp4a` dot products
with the int8 activations, applying sub-block scales in float (not truncated as in ggml-cuda). The multi-column
variant (`dotn<NB>`, NB = 1..7) decodes once and accumulates per column with `__fmul_rn`/`__fadd_rn` so every
column is bit-identical to the single-column kernel. Grid tables (IQ formats) live in shared memory.

### 6.2 CPU matrix-vector (`src/cpu/cpu_kernels.cpp`)

AVX2 kernels per format against Q8_K activations. Per 32-weight sub-block: build 32 weight bytes (grid formats:
`vpgather` from the L1-resident grid; LUT formats: `pshufb`), apply the weight signs to the activations
(`_mm256_sign_epi8`), `maddubs` (u8 × s8 → s16 pairs), `madd` with the sub-block scale (→ s32); the float scale is
applied once per 256-weight super-block. Integer arithmetic is exact, so results equal the reference dot product
up to float summation order. Templated on NB = 1..8 columns: the decode is shared, the per-column work is load,
sign, `maddubs`, `madd`, add; columns are bit-identical to NB = 1.

Per-block work is vectorized (scales unpacked once per super-block and broadcast with one `pshufb`; Q2_K / Q4_K
min corrections as one `madd` over the 16 bsums; IQ2_XXS / IQ2_XS / IQ3_XXS sign groups from a 128-entry table).

**Exact re-encodings.** At load, the host shares of grid formats are re-encoded into layouts that decode without
gathers, kept next to the originals (which streaming and prefill still use):

- **R4** (IQ3_S, IQ3_XXS): 134 bytes per 256 weights — fp16 d, 8 scale nibbles, a 4-bit code per weight (level
  index 0-7 of the grid byte, +8 for negative weights); decoded with one `pshufb` from a 16-entry signed table.
- **R2** (IQ2_XXS, IQ2_XS, IQ2_S; grids use only the magnitudes 8, 25, 43): 106 bytes per 256 weights — fp16 d, a
  scale nibble per 16 weights, 2-bit magnitude codes in the Q2_K layout, explicit sign bits.

Both reproduce the gather kernels' integer products exactly (bit-identical results), and run 20-40% faster with
3-4 columns at the cost of 22-60% more bytes for those rows.

### 6.3 Decode attention (`src/cuda/ops.cu`)

Tensor-core flash-decoding with `mma.m16n8k8` (FP16 in, FP32 accumulate). MMA rows are (query token, head) pairs:
a 16-row tile holds 2 tokens × the 6 query heads of one KV group (up to 4 tiles for 7 tokens); head dimensions are
permuted so the K fragments load directly from the KV cache without shared memory. Keys are split into 256-key chunks processed by independent warps (online softmax per chunk), then
combined; KV dequantization (Q8_0 / Q4_0 → FP16) uses the `0x6400 | x` magic-number trick, so one `HMUL2` by the
scale rounds once. Templated on the KV type.

### 6.4 DeltaNet decode (`src/cuda/ops.cu`, `k_deltanet`)

One CTA per V head (48), 512 threads; the 128 × 128 state of the head lives in registers for the whole call.
Per token: causal conv + SiLU, L2 norms, gating scalars, the delta-rule update and output, gated RMSNorm. The call
first replays the previous pass's committed tokens (lazy commit, §7.3), then processes the new ones; the new
tokens' CPU-computed inputs are prefetched from mapped host memory once.

### 6.5 Other

Residual + RMSNorm (`k_add_rmsnorm`: device + mapped-host sources, outputs fp32 / int8 / Q8_K), attention prep
(Q/K norms, RoPE, KV append), SwiGLU with int8 output, two-stage top-k over the GPU logits.

---

## 7. Speculative decoding with the MTP block

### 7.1 The draft model

The MTP block runs as one extra layer (`Engine::mtp_rows`). Its Q6_K weights (332 MiB) are requantized at load to
Q4_K (228 MiB) with a bit-exact port of ggml's `quantize_row_q4_K_ref`, cached in `data/cache/` (drafts do not
affect the output, only the acceptance rate, which did not change). Its draft head is the resident LM-head prefix
(token ids < 65536), computed on the GPU. The MTP layer keeps its own KV cache of (h_{q-1}, x_q) pairs:

- `mtp_catchup()` appends the pairs whose token is known (positions already decoded);
- `mtp_draft(next, k)` appends the pair for the sampled-but-not-decoded token `next`, drafts one token, then
  chains k-1 more steps on its own output state.

### 7.2 Verification

A cycle: draft k tokens; run `decode_batch([next, d_1..d_k], speculative = true)`; accept the longest prefix where
each draft equals the target's choice at the previous position; the target's token after the accepted prefix
comes for free. So a cycle yields 1..k+1 tokens for one pass over the weights. `commit(m)` keeps m tokens.

With greedy decoding the output is identical to plain greedy decoding token for token (`integration.spec.*`).
With sampling the generator uses speculative sampling with deterministic drafts: draft d is accepted with
probability p(d) under the target distribution (temperature, top-p over the top-32 logits); on rejection the
token is drawn from p with d removed — exact for that distribution.

### 7.3 Lazy DeltaNet commit

Attention layers need no rollback: the KV rows of rejected tokens are simply overwritten later. The DeltaNet state
cannot be rolled back cheaply (3 MiB per layer), so a speculative pass computes the new tokens' outputs without
advancing the stored state; their inputs (qkv, beta, alpha) are saved per layer. `commit(m)` records that m of
them are accepted, and the next pass first replays those m tokens into the state, then processes its own. Conv
windows are double-buffered, the buffer index flipping whenever the state is written back.

### 7.4 Overlapped drafting

In a verify pass the GPU finishes its LM-head rows (the resident 64K-row prefix and streamed rows) ~10 ms before
the CPU finishes its own. `lm_head` copies the GPU top-k as soon as it is ready and calls the generator's guess
callback (`SpecGuessFn`), which applies the acceptance rule to this partial result with a copy of the sampling RNG.
If the guess gives the next token, the calling thread drafts the next cycle on the GPU (`mtp_draft_at`, the MTP
block is kept fully resident for this, `EngineConfig::mtp_overlap`) and then joins the CPU threads. The next cycle
uses these drafts only if the real outcome matches the guess (`take_drafts`: same position, same next token, enough
drafts); otherwise they are discarded and the MTP state rolled back. Drafts are only proposals, so the output does
not depend on this. The guess is almost always right because the frequent tokens are in the GPU's rows; the MTP
time per cycle drops from 3.8 to 0.5 ms (+3% end to end).

### 7.5 Draft length

`DraftController` picks k ∈ {1, ..., 6} each cycle by maximizing expected tokens per cycle divided by cycle time,
using per-depth acceptance rates (exponential decay, memory of ~12 cycles) and measured cycle times per k. The
acceptance at depth d > 1 is blended with the estimate of depth d - 1: conditional acceptance is nearly constant
over depths for a given text (`q38_eval draftstats`: 0.97 ... 0.90 on code edits, 0.75 ... 0.70 on prose), so a
depth that has not been drafted yet inherits the last measured rate, and long drafts are tried exactly where they
can pay. Pass times grow with the token count (4K context: 104, 115, 129, 146, 164, 188, 210 ms for 1-7 tokens).
Code edits run at k = 6 (~5.5 tokens per pass), prose at k = 1-4. The server keeps one controller across requests
(seeded sampling uses a fresh one, so its output does not depend on earlier requests).

---

## 8. Prefill (`Engine::prefill_batched`)

The whole prompt runs on the GPU in chunks of up to 1024 tokens; for each chunk and layer:

1. **GEMMs** (`gemm_split`): resident rows are dequantized to FP16 in row blocks that are multiples of 512 rows
   (badly shaped blocks ran at 11-18 TFLOPS instead of 30-45) and multiplied with cuBLAS (FP16 in, FP32 out). The
   dequantization kernel reuses the GEMV decoders (`QDot<T>::decode`: int8 values and per-8 scale / offset, exact in
   float, so the FP16 weights equal the reference rounded once) and stores each warp's 2 KB as coalesced rows; it
   is bound by the FP16 writes.
   Host rows are streamed in 24 MiB pieces through a two-slot staging ring on a copy stream, issued two pieces
   ahead and consumed in the exact order of the layer computation, then dequantized and multiplied the same way.
2. **Attention**: tensor-core flash attention (`k_attn_prefill_mma`): a CTA handles 16 queries × the 6 heads of a
   KV group; 2 producer warps dequantize the next 16-key K/V tile into the other half of a double buffer while 6
   compute warps process the current one (~20 TFLOPS with Q8_0 KV).
3. **DeltaNet**: token-parallel conv, norms and gating; the recurrence is sequential over tokens with the state
   in registers (4 lanes per state column, so each dot product reduces over 4 lanes); gated norm.
4. **FFN**: gate and up GEMMs, SwiGLU in place (FP16 written over the gate rows), down GEMM.
5. With the MTP block, its KV cache is filled for the chunk (eh_proj, K/V projections and RoPE only).

Logits are computed only for the requested rows (normally the last token), after the chunk loop.

**Memory.** Activations of a chunk use one region shared by the phases of a layer (attention, DeltaNet, FFN, MTP
never live at the same time; 136 KB per token) plus X, Y and the FP16 normed input (50 KB per token): 186 KB per
token, 238 MiB for 1024 tokens including staging. The FP16 dequant scratch is the decode streaming ring (64 MiB,
idle during prefill). When free VRAM does not fit the rest, the largest buffers are placed in the resident LM-head
rows (unused by the prefill layers; 170 MiB) and those rows are restored from a pinned host copy afterwards
(~30 ms); a failed `cudaMalloc` also leads there. Only if memory is still short do the staging pieces shrink to
12 MiB, then the chunk is halved. The order matters for reproducibility: host rows are multiplied piece by piece,
so the piece size sets the cuBLAS shapes and the rounding, while borrowing only moves buffers.

**Performance.** ~470-500 tok/s at 4K, ~375 at 32K, 261 tok/s averaged over a 100K prompt. GEMMs are ~70% of the
time at 4K and ~55% at 64K; the GPU is power-capped (~102 of 110 W, SM clock ~1300 MHz instead of 1560).

---

## 9. KV cache and context growth

Per attention layer and KV head, values and scales are stored separately (`cuda::KVLayer`): F16 (512 B per row of
256), Q8_0 (256 B + 8 fp16 scales) or Q4_0 (128 B + 8 fp16 scales). Per token over the 16 trunk layers: 65.5 KB
(F16), 34.8 KB (Q8_0), 18.4 KB (Q4_0); 100K tokens of Q4_0 take 1.84 GB.

The context is limited to 262144 positions, the model's training context and the decode attention kernel's limit
(`Engine::max_context_limit()`). The KV capacity starts small and doubles on demand. `Engine::resize(new_ctx)` parks the live state (KV rows,
DeltaNet states and conv windows, MTP rings) in host memory, tears down the device allocations, re-plans the
weights for the smaller budget, reloads, and restores the state: a few seconds, with outputs bit-identical to never
resizing (`integration.resize`).

---

## 10. Image input (`src/runtime/vision.*`, `vision_encode.cpp`, `src/cuda/vision.*`)

**The encoder.** The mmproj GGUF holds a Qwen3-VL vision transformer in BF16 (0.9 GB), loaded by `VisionModel`:

- **Preprocessing (CPU, ~5 ms).** stb_image decodes the file. The smart resize of the Hugging Face Qwen2-VL
  processor picks sides that are multiples of 32 with an area between the token limits (default 8 to 1024 tokens
  of 32x32 pixels). The image is stretched to that size with Pillow's bicubic filter, ported byte for byte
  (`unit.image_resize`), normalized to (v/255 - 0.5)/0.5 and cut into 16x16 patches, in merge order (the 4 patches
  of each 2x2 group consecutive).
- **Network.** Patch embedding: one GEMM with K = 768, the two temporal kernels summed for a still image. Learned
  48x48 position embeddings are resized bilinearly (align corners) to the patch grid. Then 27 pre-norm blocks:
  - LayerNorm (eps 1e-6);
  - QKV, with 2D rotary positions on head-dim pairs (j, j+36): the first 18 frequencies turn with the patch row,
    the next 18 with the column;
  - 16-head attention, output projection;
  - LayerNorm, tanh-GELU MLP of width 4304.

  A final LayerNorm follows. Each 2x2 group is concatenated (4608) and projected to 5120 by a GELU MLP: one token
  per 32x32 pixels.
- **On the GPU (`Engine::encode_image`).** The weights live in pinned host memory, converted once to FP16
  (870 MiB). Each layer's matrices (30.5 MB) are copied into one half of the decode streaming ring, which is idle
  between passes, while the GPU computes the previous layer from the other half. The projector goes through in row
  pieces. Activations:
  - the residual stream is FP32, GEMM inputs are FP16 (cuBLAS, FP32 accumulation);
  - attention runs per head and per chunk of 1024 queries: FP32 scores, softmax to FP16, then P V;
  - scratch (~110 MB for a 1024-token image) comes from free VRAM, or from the resident LM-head rows like the
    prefill buffers.

  An image costs ~150 ms at 300 tokens and ~0.6 s at 1024.
- **Accuracy.** Against llama.cpp's mtmd on the CPU (`golden.vision.*`, images that need no resize): mean cosine per
  token ≥ 0.99996. llama.cpp's own CUDA path reaches 0.9995 against its CPU path. The lowest token, 0.982 on the
  1024-token test image, is ill-conditioned: llama.cpp's CUDA and CPU paths agree there only to 0.927. Splitting Q
  and K into FP16 hi + lo parts (near-FP32 scores) did not move it.

**In the language model.** The chat template marks an image as `<|vision_start|><|image_pad|><|vision_end|>`. The
pad expands to the image's nx * ny tokens, and `prefill_batched` takes `PrefillImage` spans whose rows use the image
embeddings instead of token embeddings. The MTP block sees the same embeddings, so drafting after an image uses the
same inputs as the trunk. Positions follow Qwen3-VL's interleaved M-RoPE:

- frequency i of the 32 rotary frequencies turns with the t, h or w position for i mod 3 = 0, 1, 2 (sections
  11 / 11 / 10, checked at load);
- a text token has t = h = w = its position, which is plain RoPE;
- the image token in row r, column c gets (p, p + r, p + c), where p is the position of the image's first token;
- text after the image continues at p + max(nx, ny), although the image fills nx * ny cache rows.

The engine keeps the image spans of its cache. The rotary position of text is its cache row plus `rope_delta()`,
the sum over images of max(nx, ny) - nx * ny. The decode attention prep takes the rotary position apart from the
cache row. A prefill chunk with image tokens builds its RoPE table from per-token (t, h, w) positions
(`cuda::rope_table_pos`). The spans travel with snapshots and context growth. Attention stays causal in cache order,
which is raster order within an image, as in Hugging Face transformers and llama.cpp. Speculative greedy output
after an image equals plain greedy (`integration.spec.image`, with the image spread over several prefill chunks).

**In the server.** `chat::render` renders image parts as the placeholder and returns their URLs in prompt order.
`api.cpp` decodes the `data:` URLs, preprocesses the images and expands the pads. The generator encodes only the
images whose tokens must be processed. It compares the images of the cached conversation by a hash of their bytes,
and prompt reuse never starts or ends inside an image.

---

## 11. Numerics and what is guaranteed

| Property | Guarantee | Checked by |
|---|---|---|
| Quantized weight values | exactly ggml's dequantized values | `unit.quant_ref` (vs libggml) |
| Tokenizer | id-for-id equal to llama.cpp | `unit.tokenizer` |
| CPU / GPU dot products | exact integer sums, float scales; within 1e-6 relative of the reference in double | `kernels.cpu_gemv`, `kernels.gemv` |
| Multi-column kernels, R4 / R2 re-encodings, GPU Q8_K quantizer | bit-identical to the single-column / gather / CPU versions | `kernels.cpu_gemv`, golden `--save` comparisons |
| `decode_batch(n)` vs n × `decode()`, speculative passes with commits | bit-identical | `integration.decode_batch.*` |
| Speculative greedy output | identical to plain greedy | `integration.spec.*` |
| Context growth | bit-identical | `integration.resize` |
| Logits vs llama.cpp and the FP32 reference | top-1 ≥ 30/32, KL ≤ 0.002 on four prompts | `golden.*` |
| Image resize | byte-identical to Pillow's `Image.resize(BICUBIC)` | `unit.image_resize` |
| Image embeddings vs llama.cpp's mtmd (CPU) | mean cosine per token ≥ 0.9999 on three images | `golden.vision.*` |
| Speculative greedy output after an image | identical to plain greedy | `integration.spec.image` |

Approximations relative to an FP32 forward pass: int8 activations for the matrix-vector products (int8 per 32 on
the GPU, Q8_K on the CPU, as llama.cpp does), FP16 tensor-core GEMMs and attention in prefill, the quantized KV
cache. The FP32 reference implementation (`src/reference/ref_model.cpp`) is the ground truth for the golden tests.

---

## 12. Server (`tools/q38_server.cpp`, `src/server/`)

Usage and the full API are in `docs/SERVER.md`. The layers:

- **`http.cpp`.** HTTP/1.1 with one thread per connection, at most 64 connections; `Content-Length` bodies;
  plain responses or Server-Sent Events, with optional `event:` names; CORS; `Connection: close`. A non-blocking
  peek detects clients that disconnect during non-streamed requests.
- **`json.cpp`.** A strict parser plus two writers:
  - compact output for API responses;
  - Python `json.dumps` style for the chat template: `", "` / `": "` separators, Python float repr, non-ASCII
    kept, members in insertion order.
- **`chat.cpp`.** A port of the GGUF's Jinja chat template, checked byte for byte against jinja2 on 12
  conversations (`unit.chat`). It covers:
  - the reasoning-effort instructions;
  - the tools block;
  - assistant turns with reasoning and XML tool calls;
  - tool results grouped into user turns;
  - `preserve_thinking`, and the generation prompt with or without thinking;
  - image parts, rendered as the vision placeholder (section 10).

  It also parses output. Tool-call blocks become OpenAI tool calls, with arguments typed by each tool's JSON
  schema. Streaming splitters separate reasoning, content and tool-call text, holding back partial markers.
- **`api.cpp`.** The endpoints:
  - Chat Completions with tools, `tool_choice` (forced calls are prefilled), `parallel_tool_calls`, reasoning
    effort and streamed tool-call deltas;
  - the Responses API: input items are converted to chat messages, output items and events are built from the
    same turn; `previous_response_id` uses an in-memory store;
  - Completions; models.

  A mutex serializes generation. The prompt is tokenized as two parts, the conversation and the generation
  prompt. Special tokens delimit tokenization, so this gives the same tokens as the whole text (checked in
  `unit.chat`), and it tells the generator where the next request is expected to diverge.
- **`generator.cpp`.** Prompt processing:
  - Reuse the engine state when the prompt extends everything already processed.
  - Otherwise, restore the engine snapshot from the previous request if the prompt still starts with that
    request's conversation.
  - Otherwise start over.

  `Engine::save_snapshot` / `restore_snapshot` copy the DeltaNet recurrent and convolution states and the MTP
  hidden-state ring to pinned host memory and back: 25 ms each. The KV cache needs no copy, since restoring only
  moves the position back. A snapshot is taken after the conversation part of each prompt, before the 4-7 tokens
  of the generation prompt are processed; `integration.snapshot` checks that a restore is bit-identical.

  Generation:
  - grows the KV cache up to `--ctx`;
  - runs the speculative loop with `DraftController`;
  - greedy or sampled output: temperature, top-k and top-p over the engine's top-32 logits; the defaults come
    from the GGUF's `general.sampling.*`;
  - stop strings and UTF-8-safe streaming deltas.

  It reports progress to an observer (`GenEvent`): the prompt-reuse decision, every prefill chunk
  (`Engine::prefill_batched` calls back when the GPU has finished a chunk, so the reported progress is never
  ahead of the work), the end of the prompt, every decode or verify pass, and every KV-cache growth. Growth time
  is measured on its own and kept out of the prefill and decode times.
- **`metrics.cpp`.** `Monitor` turns those events into the request log on stderr (task-numbered lines, prompt
  progress, periodic decode speed, a llama.cpp-format timing block) and into the counters and gauges of
  `GET /metrics` (Prometheus text, llama.cpp names plus `q38:*`). Each request adds only the change since its
  previous event, so the counters advance during a request and count every token once. One mutex guards the
  state, and a `/metrics` query never waits for the generation mutex.

---

## 13. Configuration

`EngineConfig` (`src/runtime/engine.h`): `max_context`, `kv_type` (F16 / Q8_0 / Q4_0), `cpu_threads` (8),
`vram_reserve_mib` (60), `gpu_weight_cap_mib` (testing: fixes the plan), `stream_mib` (cap on decode streaming per
pass, 2000), `mtp`, `draft_rows` (65536), `mtp_q4k`, `cpu_r4`, `cpu_r2`, `spec_k` (2), `mtp_overlap` (true).

Environment variables:

| Variable | Effect |
|---|---|
| `Q38_PROFILE_WAITS=1` / `2` | decode wait profile per site / per group (`q38_eval bench`, `specset`) |
| `Q38_PROFILE_PREFILL=1` | prefill GPU time per phase (`q38_eval pbench`) |
| `Q38_CPU_RATE_SCALE` | CPU rate scale of the planner (default 0.8) |
| `Q38_PREFILL_BORROW=1` | test: always place prefill buffers in the LM-head rows |
| `Q38_PLAN_NO_RE=1` | test: plan with the gather-kernel rates while running the re-encoded kernels |
| `Q38_GUIDED=0`, `Q38_GUIDED_MIN_US` | test: equal-time CPU chunks instead of guided scheduling; minimum guided chunk |
| `Q38_PLAN_DUMP=<text>` | print the plan items whose name contains the text |

---

## 14. Source map

| Path | Responsibility |
|---|---|
| `include/q38rt/q38rt_types.h` | model constants, ggml type ids and block sizes, tensor metadata, `ModelConfig` |
| `third_party/frokenizer` | [Frokenizer](https://github.com/yassa9/frokenizer) (git submodule): its re2c-generated Qwen pre-tokenizer lexer |
| `src/runtime/gguf_parser.*` | GGUF v3 parser over an `mmap` of the file; model config and tensor table |
| `src/runtime/tokenizer.*` | byte-level BPE with vocabulary and merges from the GGUF |
| `src/runtime/planner.*` | per-format device rates, column factors, residency planner |
| `src/runtime/engine.*` | engine: loading, planning, uploads, re-encodings, decode pass, MTP, speculative bookkeeping, streaming, prefill, resize, image spans (M-RoPE) |
| `src/runtime/vision.*`, `vision_encode.cpp` | vision encoder: mmproj loading, image decoding and preprocessing, the GPU forward (`Engine::encode_image`) |
| `include/third_party/stb_image.h` | image decoding (vendored) |
| `src/cpu/cpu_kernels.*`, `thread_pool.h` | AVX2 dot products (all formats, 1-8 columns, R4/R2), Q8_K quantizer, thread pool |
| `src/cuda/gemv.*`, `quant_dot.cuh`, `iq_tables_dev.cuh` | GPU matrix-vector kernels |
| `src/cuda/ops.*` | decode kernels: norms, DeltaNet, attention, SwiGLU, top-k, copies |
| `src/cuda/prefill.*` | prefill kernels: dequant to FP16, cuBLAS wrapper, attention, DeltaNet, norms, RoPE tables |
| `src/cuda/vision.*` | vision encoder kernels: LayerNorm, GELU, 2D RoPE, softmax, residuals |
| `src/quant/` | ggml block layouts, IQ grid tables, reference (de)quantizers, Q4_K quantizer |
| `src/reference/ref_model.*` | FP32 reference forward pass |
| `src/server/` | JSON, HTTP, chat template and output parsing, generation loop, OpenAI API endpoints, metrics and request log |
| `tools/` | `q38_server`, `serve.sh`, `q38_eval` (tests and benchmarks), `inspect_gguf`, `hw_probe`, `ref_compare`, `ref_llama_logits`, `ref_mtmd_embd`, `bench_cpu_ggml`, `gguf_inventory.py` |
| `tests/` | unit, kernel, golden and integration tests (`docs/DEVELOPMENT.md`) |
| `bench/` | `ab.py` (controlled A/B), `run_bench.py`, `results/` (immutable records) |
