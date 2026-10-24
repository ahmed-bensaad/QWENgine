# QWENgine

A from-scratch inference engine for **Qwen3.8-27B** that runs a 9.7 GB model on a laptop with an **8 GB GPU** at
**~15 tokens/s** (llama.cpp: 3.9 tokens/s on the same machine), with an OpenAI-compatible API for agent harnesses
and image input through the model's vision encoder.

QWENgine is deliberately not general-purpose. It runs one model file,
[`Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`](#1-get-the-model) (Gated DeltaNet + gated attention, ten mixed i-quant /
K-quant formats), and it was tuned on one machine:

- a **Quadro RTX 4000 Mobile** (8 GB, Turing sm_75);
- an **Intel i9-9880H** (8 cores, AVX2);
- **32 GB DDR4**.

Every design choice comes from measuring that machine. [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) explains them
for readers interested in how inference can be pushed on constrained hardware.

**Built by a human and an AI.** The idea came from [Strata](https://github.com/niko1221/strata). I started QWENgine
with my own research, design and codebase, then developed it further with an AI coding agent (Claude, in Claude
Code) under my direction.
[How this was built](#how-this-was-built-human-and-ai) says who did what.

**Other machines are welcome.** QWENgine should also run on other Linux machines with an NVIDIA GPU (Turing or
newer, 8 GB or more) and an AVX2 CPU, but it has not been measured on any yet. If you try it, please run the
benchmark script and share your numbers, good or bad, in [BENCHMARKS.md](BENCHMARKS.md): see
[Other hardware](#other-hardware).

## Performance

| | QWENgine | llama.cpp (same machine, same GGUF) |
|---|---|---|
| Decode, speculative (16-prompt corpus average, 4K context) | ~15 tok/s (code ~20, stories ~11) | — |
| Decode, speculative, code-editing prompts (agent-style, `data/agent_prompts.txt`) | ~26 tok/s | — |
| Decode, plain (4K context) | ~9 tok/s | 3.89 tok/s |
| Decode at 150K tokens of context (Q4_0 KV cache, speculative) | ~9 tok/s | — |
| Prefill, 4K / 32K prompt | ~490 / ~375 tok/s | 234 tok/s (pp512) |
| Prefill, average over a 150K prompt | 317 tok/s | — |

These are cool-start numbers on the reference machine: they drift ±5-10% with the laptop's temperature, and
sustained use runs ~10% lower. Numbers from other machines: [BENCHMARKS.md](BENCHMARKS.md).
Greedy output with speculative decoding is identical token for token to plain decoding. Logits are checked against
llama.cpp and an FP32 reference implementation by the test suite.

## How it works, in short

- **The weights do not fit in VRAM, so decode is hybrid.** A residency planner splits every large matrix between
  VRAM and host memory, using measured per-format costs on each device. The GPU and the CPU (AVX2 kernels) then
  compute their halves at the same time. Part of the host share is also streamed to the GPU over PCIe during each
  pass. Attention, the DeltaNet recurrence, norms and the KV cache stay on the GPU.
- **Speculative decoding with the model's own MTP block.** The `-mtp.gguf` variant carries a multi-token-prediction
  layer. QWENgine uses it to draft up to 6 tokens and verifies them in one pass over the weights. The host weights
  are read once for 2-7 tokens, which suits the hybrid split especially well.
- **Prefill runs entirely on the GPU:** FP16 tensor-core GEMMs on weight blocks dequantized on the fly, with host
  weights streamed over PCIe once per 1024-token chunk, and tensor-core flash attention over the quantized KV cache.
- **Exact by construction.** Multi-token passes, speculative decoding, context growth and state snapshots are
  bit-identical to the plain paths, and the test suite checks it.
- **Images** go through the model's vision transformer (the mmproj GGUF) on the GPU, with its weights streamed from
  host memory per image so they take no VRAM from the language model: ~150 ms for a 640x480 photo.

## Requirements

| | |
|---|---|
| GPU | NVIDIA, compute capability 7.5 or newer (Turing, Ampere, Ada, Hopper, Blackwell), 8 GB of VRAM or more. Tuned on a Quadro RTX 4000 Mobile; other GPUs are untested so far (see [Other hardware](#other-hardware)) |
| CPU | x86-64 with AVX2, FMA and F16C (checked when configuring); tuned on an i9-9880H |
| Memory | 32 GB of RAM (peak use ~14-18 GB, ~1 GB more with the vision encoder) |
| OS | Linux |
| Toolchain | CUDA 12.4 (nvcc, cuBLAS; the tested version), with `g++-12` as nvcc's host compiler when installed (nvcc 12.4 rejects GCC 14+; `-DCMAKE_CUDA_HOST_COMPILER=...` changes it), a C++17 compiler for host code, CMake ≥ 3.24, Ninja, Python 3 for tests and benchmarks |
| Disk | ~21 GB for the two model files (+0.9 GB for the optional vision encoder) |

## Other hardware

QWENgine is tuned for the machine above, but it adapts to others: at startup the residency planner measures the
free VRAM, keeps as many weights there as fit and gives the rest to the CPU. It should run on any Linux machine
with the [requirements](#requirements) above. Build for your GPU with `-DCMAKE_CUDA_ARCHITECTURES=native` (or its
number, e.g. `86` or `89`).

It has only been measured on the reference machine so far, so **please measure yours and share the result**.
`python3 bench/community_bench.py --tests` describes your machine, measures its bandwidths, runs the standard
benchmarks and the test suite, and prints a row for [BENCHMARKS.md](BENCHMARKS.md). Open a pull request with it;
[BENCHMARKS.md](BENCHMARKS.md#add-your-machine) has step-by-step instructions. Failed tests and slow numbers are as
useful as good ones.

What to expect:

| Your machine has | Expected effect |
|---|---|
| More VRAM (12-24 GB) | more weights on the GPU and faster decode; from about 12 GB the whole model fits and decode becomes GPU-bound |
| Faster CPU memory (DDR5, more channels) | faster decode: the CPU reads its share of the weights from RAM on every pass |
| PCIe 4.0 or 5.0 x16 | more weights streamed to the GPU per pass, faster prefill; a x4 or x8 link (eGPUs, some laptops) is slower |
| A desktop GPU and CPU | higher power limits than this laptop (the GPU is capped at 110 W, the CPU at 45 W) |
| Less than 8 GB of VRAM | untested: more weights on the CPU (slower), and the prefill buffers may not fit |

Known limitations on other hardware:

- **The planner's cost model was measured here.** The per-format CPU and GPU throughputs, the PCIe rate and the
  thermal factor in `src/runtime/planner.cpp` and `engine.h` come from the reference machine. Elsewhere the CPU/GPU
  split works but is not optimal; calibrating these numbers for other machines is a welcome contribution.
- **The kernels are tuned for Turing.** Tile sizes and register budgets were chosen for the RTX 4000, and some grids
  for its 40 SMs. The kernels are correct on newer GPUs but do not use Ampere and later features (`cp.async`, BF16
  tensor cores, larger shared memory). GTX 16-series cards (Turing without tensor cores, 6 GB) are untested.
- **The CPU kernels are AVX2 only** (AVX-512 and VNNI are not used) and rely on fast AVX2 gathers. With the Gather
  Data Sampling ("Downfall") mitigation active, `cat /sys/devices/system/cpu/vulnerabilities/gather_data_sampling`
  shows "Mitigation", and these instructions get several times slower. They are also slower on older AMD Zen cores.
- **Not supported:** AMD and Intel GPUs, Apple Silicon, ARM CPUs, CPU-only machines, Windows (WSL2 is untested).
- Only Linux with NVIDIA driver 595 and CUDA 12.4 has been tested.

Changes that make QWENgine better on other hardware are welcome, as long as they do not slow down the reference
machine ([CONTRIBUTING.md](CONTRIBUTING.md#other-hardware)).

## Quick start

### 1. Get the model

QWENgine reads the GGUF files of
[ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) on Hugging Face:
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized by ISTA DASLab (GSQ-RCO, non-uniform mixed
precision). Download these files from it:

| File | Size | Used for |
|---|---|---|
| `Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf` | 10.1 GB | plain decoding, most tests |
| `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` | 10.4 GB | the same model plus the MTP block (`blk.64`): speculative decoding, the server |
| `mmproj-Qwen3.8-27B-BF16.gguf` (optional) | 0.9 GB | the vision encoder: image input in the server |

Place them in `models/` at the repository root (ignored by git), for example with the Hugging Face CLI
(`pip install -U huggingface_hub`):

```bash
hf download ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf \
    Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf mmproj-Qwen3.8-27B-BF16.gguf --local-dir models
```

Files you already have elsewhere can be linked instead:
`mkdir -p models && ln -s /path/to/Qwen3.8-27B-GSQ-RCO-IQ3_XXS*.gguf models/`.

Another location works too: pass `-DQ38_MODEL_DIR=/path` to CMake and set `Q38_MTP_MODEL` for `tools/serve.sh`.
[docs/gguf_inventory.md](docs/gguf_inventory.md) lists the exact contents and sizes of both files.

### 2. Build

Clone with the submodule (`git clone --recursive <repository URL>`), or run `git submodule update --init` in an
existing clone: the tokenizer uses [Frokenizer](https://github.com/yassa9/frokenizer)'s lexer from
`third_party/frokenizer`. Then:

```bash
cmake -S . -B build -G Ninja      # Release, -march=native, sm_75
cmake --build build
```

On a GPU other than a Turing card, add `-DCMAKE_CUDA_ARCHITECTURES=native` to the first command.

### 3. Serve

```bash
tools/serve.sh                    # http://127.0.0.1:8080/v1, model id "qwen3.8-27b"
```

Loading takes ~10-15 seconds. After that the server logs each request: prompt progress, decode speed, timings. Try
it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Write a haiku about GPUs."}],"stream":true}'
```

Point any OpenAI-compatible client or agent harness at base URL `http://127.0.0.1:8080/v1` with model
`qwen3.8-27b` and any API key (or the one set with `--api-key`). [docs/SERVER.md](docs/SERVER.md) covers:

- the endpoints: Chat Completions with tools and streaming, a Responses API subset, Completions, Prometheus
  `/metrics`;
- image input (`image_url` parts with base64 `data:` URLs), enabled when `models/` holds the mmproj file;
- reasoning control (`reasoning_effort`) and tool calling;
- how agent loops reuse the engine state between turns;
- ready-made configurations for Aider, Continue, Cline, opencode, Codex CLI, the OpenAI SDKs and others;
- the request log and the metrics.

Useful options: `--effort low` (shorter reasoning), `--ctx 100352 --kv q4_0` (100K context), `--api-key KEY`,
`--host 0.0.0.0` (expose on the network; set a key).

#### Best settings

The fastest configuration measured on the reference machine, with everything on (MTP speculative decoding, image
input):

```bash
build/q38_server models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf --mmproj models/mmproj-Qwen3.8-27B-BF16.gguf --ctx 32768 --ctx-init 4096 --kv q8_0 --threads 8 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0
```

`tools/serve.sh --temp 0.6` is the same: everything else is its default. For long contexts (coding agents that send
whole files and histories), raise the limit and use the Q4_0 KV cache:

```bash
tools/serve.sh --temp 0.6 --ctx 131072 --kv q4_0
```

| Setting | Why |
|---|---|
| the `-mtp.gguf` model | speculative decoding with the model's MTP block: ~15 tok/s on mixed prompts and ~26-30 on code edits, against ~9 without it |
| `--mmproj` | image input. The encoder lives in host RAM (~1 GB) and uses VRAM only while it runs, so decode speed does not change. |
| `--ctx-init 4096` | the KV cache starts small and doubles on demand up to `--ctx`, so the VRAM it does not use holds weights (faster decode). Each doubling costs ~6.5 s, once; `--ctx-init 16384` avoids these pauses for long agent prompts, at some decode speed. |
| `--kv q8_0` | KV-cache precision. At long context use `q4_0`: it halves the KV cache's VRAM, which then holds weights. At 64K tokens it decoded 21% faster than `q8_0` (9.5 against 7.8 tok/s on the coding benchmark). |
| `--threads 8` | one thread per physical core (8 here: hyperthreads, 12 or 16 threads, brought no gain); on another CPU, use its number of physical cores |
| `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0` | The model's recommended sampling (the GGUF's defaults, which the server uses when no option is given) is the same with temperature 1.0. At 0.6, more drafts are accepted: ~4% faster overall, ~7% on prose and the same on code. Greedy (0) is not faster than 0.6. |

Two more things help:

- **Shorter reasoning:** `--effort low`, or `reasoning_effort` per request, gives faster answers through fewer
  tokens, at the same tok/s.
- **Free VRAM:** close GPU programs, or run headless. On the reference machine, freeing 200 MiB for weights gave
  +5.9% decode speed.

### 4. Test

```bash
cd build && ctest; echo $?        # ~7 minutes; most tests load the model
```

Tests that need files you do not have are disabled: ctest lists them and still exits 0 (re-run `cmake` after adding
the files). The reference tests compare against llama.cpp.
They need a llama.cpp build with shared libraries (`-DQ38_GGML_DIR=<llama.cpp>/build/bin`) and golden data
generated by `tests/golden/make_golden.sh`, `make_ref.sh` and, for the vision encoder, `make_vision_ref.sh`.
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) lists every test and what it guarantees.

## Measuring

`q38_eval` runs the benchmarks and the correctness modes (`MTP=models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`):

```bash
build/q38_eval $MTP specset --draft auto --gen 80 --nprompts 16    # speculative decode over 16 corpus prompts
build/q38_eval $MTP pbench --depth 8192                            # prefill 8K tokens, then decode at that depth
build/q38_eval $MTP needle --depth 100000 --kv q4_0 --ctx 100352   # 100K-token retrieval test
python3 bench/ab.py --a "CMD_A" --b "CMD_B" --metric specset_tok_s # controlled A/B comparison of two builds
python3 bench/community_bench.py --tests                           # the standard benchmark of a machine (BENCHMARKS.md)
```

Profiling: `Q38_PROFILE_WAITS=1` (decode CPU/GPU waits), `Q38_PROFILE_PREFILL=1` (prefill phases), `nsys`.

## Documentation

| Document | Contents |
|---|---|
| [BENCHMARKS.md](BENCHMARKS.md) | results from every machine, and how to add yours |
| [CONTRIBUTING.md](CONTRIBUTING.md) | what contributions fit, how changes are tested and measured, pull requests |
| [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) | why the engine is fast on this hardware: each resource limit and the technique that answers it, with measured gains |
| [docs/SERVER.md](docs/SERVER.md) | the OpenAI-compatible server: options, endpoints, tool calling, reasoning, conversation reuse, request log, metrics, client configurations |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | what the engine computes and how: model, memory plan, planners, decode pass, kernels, speculative decoding, prefill, KV cache, numerics, server, source map |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | build options, every test and what it guarantees, golden data, benchmarking method, profiling, pitfalls |
| [docs/optimization_log.md](docs/optimization_log.md) | every performance experiment since the first release, with measurements and the decision (kept or reverted) |
| [docs/hardware.md](docs/hardware.md) | measured characteristics of the target machine |
| [docs/gguf_inventory.md](docs/gguf_inventory.md) | exact contents of the two model files |

## Repository layout

```
include/            model constants and types; vendored stb_image
third_party/        Frokenizer, the Qwen pre-tokenizer lexer (git submodule)
src/runtime/        GGUF parser, tokenizer, residency planner, engine (decode, MTP, streaming, prefill, resize),
                    vision encoder (image preprocessing and GPU forward)
src/cpu/            AVX2 quantized dot products (all formats, 1-8 columns, R4/R2 re-encodings), thread pool
src/cuda/           GPU matrix-vector kernels, decode ops, prefill kernels, vision encoder kernels
src/quant/          ggml block layouts, IQ grids, reference (de)quantizers
src/reference/      FP32 reference forward pass (ground truth for golden tests)
src/server/         JSON, HTTP, chat template and output parsing, generation loop (prompt reuse, speculative
                    decoding, sampling), OpenAI API endpoints, Prometheus metrics and request log
tools/              q38_server, serve.sh, q38_eval, inspect_gguf, hw_probe, ref_compare, ref_llama_logits,
                    ref_mtmd_embd, bench_cpu_ggml, gguf_inventory.py, ci.sh (the CI steps)
tests/              unit, kernel, golden and integration tests
bench/              ab.py, run_bench.py, results/ (immutable benchmark records); community_bench.py and
                    community/ (results from other machines, BENCHMARKS.md)
data/               benchmark prompts; cache/ (requantized MTP block, created on first load)
models/             the GGUF files (not in git)
docs/               documentation
```

## Contributing

Contributions are welcome:

- bug reports and fixes, and correctness tests;
- performance work;
- support and tuning for other hardware;
- server and API improvements, and documentation;
- **benchmark results from your machine**, added to [BENCHMARKS.md](BENCHMARKS.md) through a pull request.

The engine is fast because it is specific: changes for other hardware must not slow down the reference machine, and
changes that make it generic at that cost are better kept in a fork (we are happy to link forks).

The rules in short:

- `ctest` must stay green. GitHub CI builds every pull request and runs the tests that need no GPU; the full suite
  needs the GPU and the model files.
- Performance changes are measured end to end with `bench/ab.py`, and the record goes into `bench/results/`.
- Every experiment, kept or reverted, gets an entry in `docs/optimization_log.md`.

[CONTRIBUTING.md](CONTRIBUTING.md) has the details: what fits, how to measure, code style, pull requests and issue
reports.

## How this was built: human and AI

The idea came when I saw [Strata](https://github.com/niko1221/strata), which runs a 125-billion-parameter model
(Qwen3.8-Flash-Next) on a gaming PC with a 12 GB graphics card. I wondered whether the same was possible, even to
an extent, with a smaller good model like Qwen3.8-27B, on smaller GPUs: QWENgine is the answer for an 8 GB laptop.

I started QWENgine from a codebase I wrote, then developed it with an AI coding agent, Claude Opus 5.5 by Anthropic
running in Claude Code, in sessions between 2026-10-15 and 2026-10-23. I directed the work; the agent implemented
most of it.

**What I did:**

- **The original codebase:**
  - the research behind the engine (`docs/research.md`): an analysis of the model's Gated DeltaNet and attention
    layers and of its MTP block, KV-cache sizing for 100K tokens, a study of an existing engine (NInfer), and the
    PCIe and memory analysis of the laptop;
  - the hardware notes and the GGUF analysis;
  - the first code: the GGUF parser and the model types (still in use), the tokenizer built on the
    Frokenizer lexer (still the tokenizer today), first CUDA kernels, and the engine, KV-cache, streaming,
    API and CLI modules.
- **The targets and the initial design:** one model on one laptop, 10-15 tokens/s decode, a 100K context with a Q4
  KV cache, weights partly resident and partly streamed over PCIe by residency class, GPU-resident recurrent state,
  FP16 tensor-core compute.
- **The specification and the rules** I gave the agent: correctness first, then a measured optimization loop;
  end-to-end speed decides; benchmark records are never edited; no approximations for malformed values.
- **The direction:** I chose what to build next (the OpenAI-compatible server, metrics and logging, publication,
  image input, long-context benchmarks) and what to change.
- **Real-world testing** with coding agents. A 200K-token session exposed a decode attention bug beyond 131072
  positions, and these sessions gave the long-context speeds the benchmarks were built to reproduce.
- The machine, the model files, and the decisions to accept the work.

**What the AI agent did:**

- **The audit and repairs.** My original code did not compile yet, its inference path was still a placeholder, and
  the GGUF parser needed fixes. The agent audited it, repaired the parser and built the engine out.
- **Most of today's code:**
  - the CUDA and AVX2 kernels for all ten weight formats;
  - the hybrid decode, where the GPU and CPU compute their parts of every matrix at the same time and part of the
    host share is streamed over PCIe (a development of the original residency and streaming design);
  - the residency and streaming planners, MTP speculative decoding, batched prefill;
  - the server, the vision encoder and the tools.
- **The tests** and the comparisons against llama.cpp, ggml and the FP32 reference.
- **The measurements:** every benchmark, A/B comparison and experiment, with the decisions to keep or revert
  each change under my rules.
- **Debugging, profiling, and the documentation** in its current form.

In lines of code: of the ~22,500 lines in the repository today (vendored files excluded), about 440 still come
line for line from my original codebase, mostly the GGUF parser and the model types. My research, design
direction and project structure carried over further than the lines.

**What this means for you:**

- Most of today's code was written by an AI. What establishes its correctness is the test suite: bit-exact checks
  against ggml, logits against llama.cpp and an FP32 reference, bit-identity checks between code paths, end-to-end
  server tests ([docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) lists them).
- The performance claims are measurements: `q38_eval` and `bench/community_bench.py` reproduce them, and the
  reference machine's record is in [bench/community/](bench/community/).
- Reviews, bug reports and corrections from human readers are especially welcome.

## License and acknowledgements

QWENgine is released under the [MIT License](LICENSE). It builds on the work of others (see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)):

- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT): the quantization formats, their reference
  (de)quantizers and lookup tables, ported bit-exactly. llama.cpp is also the reference implementation in the tests.
- [Frokenizer](https://github.com/yassa9/frokenizer) by Yassa Sfen (MIT): the Qwen pre-tokenizer lexer, included as
  a git submodule.
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0): the model and its chat template, ported to C++.
- [ISTA DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) (Apache-2.0): the GSQ-RCO quantization
  of the model and its vision encoder GGUF.

The model weights are not part of this repository and keep their own license.
