# Development guide

How to build, test, measure and change QWENgine. The design itself is described in `docs/ARCHITECTURE.md`.

## Build

Requirements on the reference machine: CUDA 12.4 (nvcc), `g++-12` as nvcc's host compiler (nvcc 12.4 rejects
GCC 15; `CMakeLists.txt` uses it when installed, `-DCMAKE_CUDA_HOST_COMPILER=...` overrides it), a recent `g++` for
the C++ code, CMake ≥ 3.24, Ninja, cuBLAS. Configuring stops with a message on CPUs without AVX2, FMA and F16C.

```bash
cmake -S . -B build -G Ninja      # Release by default; -march=native, sm_75
cmake --build build
```

Other GPUs: `-DCMAKE_CUDA_ARCHITECTURES=native` (or a list such as `"86;89"`; compute capability 7.5 or newer).

**CI.** `.github/workflows/ci.yml` runs `tools/ci.sh` on GitHub's GPU-less runners. The script installs the CUDA
12.4 compiler and libraries, builds everything for sm_75, runs the tests that need neither a GPU nor the model
(`unit.json`, `unit.chat`, `unit.metrics`, `unit.image_resize`) and checks the syntax of the Python and shell
scripts. To reproduce it in a clean container:
`docker run --rm -v "$PWD":/src:ro ubuntu:22.04 bash /src/tools/ci.sh /src /tmp/build`.

CMake cache variables:

| Variable | Default | Used for |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `75` | GPU architectures to compile for (`native` = the GPU in this machine) |
| `Q38_MODEL_DIR` | `models/` in the repository | directory of the two GGUF files |
| `Q38_MODEL` | `${Q38_MODEL_DIR}/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf` | model-level tests |
| `Q38_MTP_MODEL` | `${Q38_MODEL_DIR}/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` | MTP / speculative / server tests |
| `Q38_MMPROJ` | `${Q38_MODEL_DIR}/mmproj-Qwen3.8-27B-BF16.gguf` | vision encoder tests (optional) |
| `Q38_GGML_DIR` | empty | `build/bin` of a llama.cpp build with shared libraries (`libggml-base`, `libggml-cpu`, `libllama`): reference tests and tools |
| `Q38_LLAMA_INC` | `${Q38_GGML_DIR}/../..` | llama.cpp source dir (headers) |

Without the llama.cpp libraries the reference tests (`unit.quant_ref`, `unit.tokenizer`) and the tools
`ref_llama_logits`, `bench_cpu_ggml`, `ref_mtmd_embd` (the last also needs `libmtmd`) are not built. Tests whose
model file or golden data is missing when `cmake` runs are disabled (ctest lists them and exits 0; re-run `cmake`
after adding the files). `unit.json`, `unit.chat`, `unit.metrics` and `unit.image_resize` need neither a model nor
a GPU (`unit.image_resize` passes without checking when Pillow is not installed).

Data files: `data/corpus_prompts.txt` (40 prompts for `q38_eval specset`); `data/agent_prompts.txt` (8
code-editing prompts, one per line with `\n` escapes: `specset --prompts-file data/agent_prompts.txt`);
`data/longctx_code.txt` (a fixed snapshot of this repository's sources from October 2026, 149K tokens: the context
of `q38_eval longctx`; frozen so that results stay comparable across versions) and `data/longctx_tasks.txt` (its four
coding tasks);
`data/cache/` (gitignored) holds the requantized MTP block, created on first load.

## Tests

`cd build && ctest` runs 35 tests (~7 minutes; most load the model). Always check the exit status (`ctest > log;
echo $?`), not a piped `tail`.

| Test | What it guarantees |
|---|---|
| `unit.json` | JSON parser/writer (escapes, surrogate pairs, numbers, depth limit, insertion order, Python-style output) |
| `unit.chat` | chat-template port byte-identical to jinja2 on 12 conversations, image parts included (`tests/unit/data/chat_cases.json`, regenerate with `tests/unit/make_chat_cases.py <model.gguf>`); tool-call parsing; streaming splitters fed byte by byte; Responses input conversion; prompt split into conversation + generation prompt tokenizes like the whole prompt |
| `unit.image_resize` | image resize byte-identical to Pillow's `Image.resize(BICUBIC)` (down- and upscaling, one or both axes) |
| `unit.gguf_parser` | parser facts (offsets, sizes, config) match an independent Python implementation |
| `unit.quant_ref`, `.mtp` | reference dequantizers bit-exact vs libggml for every format; Q4_K quantizer vs ggml |
| `unit.tokenizer` | token ids identical to llama.cpp's tokenizer on the golden prompts |
| `kernels.cpu_gemv`, `.mtp` | CPU kernels vs a double-precision reference on real tensors; 1-8 column kernels and R4/R2 re-encodings bit-identical to the single-column / gather kernels; Q8_K quantizer bit-exact |
| `kernels.gemv`, `.mtp` | GPU GEMV kernels vs reference, multi-column bit-identity |
| `unit.metrics` | metrics of the server: counters advance once per event, queue / processing gauges, Prometheus exposition format |
| `kernels.ops` | GPU decode ops (norms, SwiGLU, DeltaNet, attention, top-k) vs CPU math in double, including decode attention at 140000 positions (beyond the kernel's old 131072 limit) |
| `kernels.attn_prefill` | tensor-core prefill attention vs the CUDA-core version, all KV types, partial tiles |
| `golden.{short,chat,code,multilingual}` | engine logits vs llama.cpp golden and FP32 reference (top-1 ≥ 30/32, KL ≤ 0.002) |
| `golden.prefill.{short,code}` | same with the prompt through batched prefill (multi-chunk) |
| `golden.prefill.borrow` | same with prefill buffers forced into the borrowed LM-head rows |
| `golden.vision.{768x512,320x640,1024x1024}` | vision encoder embeddings vs llama.cpp's mtmd on the CPU: mean cosine per token ≥ 0.9999, minimum per image (see ARCHITECTURE.md, image input) |
| `integration.decode_batch.{q8_0,q4_0}` | `decode_batch` (every pass width 1-7) and speculative passes with random commits bit-identical to sequential decode |
| `integration.spec.{k1,k3,kauto}` | speculative greedy output identical to plain greedy |
| `integration.spec.prefill`, `.borrow` | same after a batched prefill (MTP KV filled across chunks) |
| `integration.spec.image` | same with an image before the prompt, spread over several prefill chunks (M-RoPE positions, image embeddings in the MTP block) |
| `integration.resize` | context growth mid-sequence bit-identical to no growth |
| `integration.snapshot` | engine snapshot save / restore around speculative generation: logits and continuation bit-identical |
| `integration.server` | `q38_server` end to end: models, API key, chat plain and streamed (reasoning separated, usage chunk), conversation reuse, tool-call round trip, thinking agent loop reusing the snapshot, streamed tool calls, forced tool choice, seeded sampling, Responses API (function call, `previous_response_id`, streamed events), completions with stop, error cases; with the mmproj: image questions (Chat and Responses), image reuse in a follow-up, a different image of the same size not taken from the cache, image errors; OpenAI SDK checks when `openai` is installed |

### Golden data

`tests/golden/data/` (gitignored, ~32 MB per file) holds full-vocabulary logits:

```bash
tests/golden/make_golden.sh   # llama.cpp (build/ref_llama_logits): <p>.llama_cpu.bin (FP32 KV), <p>.llama_gpu.bin
tests/golden/make_ref.sh      # FP32 reference model (build/ref_compare): <p>.ref.bin + comparison report
tests/golden/make_vision_ref.sh   # test images + llama.cpp mtmd embeddings on the CPU (build/ref_mtmd_embd)
```

### Bit-identity checks between builds

Many changes must not change any result. Compare two builds under the same weight plan (`--cap-mib` fixes the
budget, so the plan does not depend on free VRAM):

```bash
build/q38_eval $MTP golden tests/golden/data/chat.llama_gpu.bin --kv q8_0 --ctx 4096 --cap-mib 5000 --save a.bin
other/build/q38_eval $MTP golden ... --save b.bin && cmp a.bin b.bin
```

A plan change alone changes the logits slightly (GPU and CPU kernels round differently), so bit-identity is only
meaningful under the same plan.

## Benchmarks

`build/q38_eval <gguf> <mode> [options]` (full list in `tools/q38_eval.cpp`):

| Mode | Measures |
|---|---|
| `bench --depth N \| --synthetic-depth N --gen N` | plain decode speed (`RESULT decode_tok_s=...`) |
| `pbench --depth N` | batched prefill of N tokens, then decode at that depth |
| `spec --prompt code\|chat\|story\|facts --draft auto` | speculative decoding on one prompt; checks output == greedy |
| `specset --draft auto --gen 80 --nprompts 16` | speculative throughput aggregated over corpus prompts |
| `longctx --depth 65536 --ctx 66560 --kv q4_0 --gen 256` | long-context coding benchmark: code snapshot as context, coding tasks from an engine snapshot; prefill, speculative and plain decode speed at depth |
| `needle --depth 100000 --kv q4_0 --ctx 100352` | 100K retrieval test + prefill / decode speed at depth |
| `draftstats --prompts-file F --draft-max K` | MTP draft acceptance per depth on greedy text, and the tokens per cycle a draft limit of 1..K would give |
| `batchtest`, `resizetest`, `snaptest`, `golden` | correctness modes used by ctest (`batchtest` also times 1-7 token passes, `snaptest` the snapshot save / restore) |

**Machine benchmark.** `python3 bench/community_bench.py [--quick] [--tests]` runs `hw_probe`, `longctx` at 64K
and 128K tokens of context (Q4_0 KV) and the image encoder, optionally `ctest`. It writes a JSON record to `bench/community/` and prints the row for [BENCHMARKS.md](../BENCHMARKS.md).
That is the format for results from other machines; `bench/results/` holds the records of performance
experiments (`docs/optimization_log.md`).

### Comparing two versions

Laptop thermals move results by ±5-10% between consecutive runs, and a change of the weight plan changes
near-tie numerics and therefore the generated text and the draft acceptance of any single prompt. Two rules:

1. Compare speculative decoding on the aggregate of many prompts (`specset`), not on single prompts.
2. Use `bench/ab.py`, which runs ABBA-ordered pairs and waits for the CPU package to cool to a fixed temperature
   before every run:

```bash
python3 bench/ab.py --a "base/build/q38_eval $MTP specset --ctx 4096 --kv q8_0 --draft auto --gen 80 --nprompts 16" \
                    --b "build/q38_eval $MTP specset --ctx 4096 --kv q8_0 --draft auto --gen 80 --nprompts 16" \
                    --metric specset_tok_s --rounds 2 --cool-to 70 --label my_change
```

It prints every run, the per-pair B/A ratios and their geometric mean, and writes a JSON record to
`bench/results/`. Build the baseline in a scratch `git worktree`. Both a sequential comparison and a single-prompt
comparison have each pointed the wrong way at least once in this project.

`bench/run_bench.py --label ...` records a plain decode benchmark with GPU/CPU utilization samples.

### Profiling

| Tool | Use |
|---|---|
| `Q38_PROFILE_WAITS=1` (or `2`) with `bench` / `specset` | CPU job time, CPU waits on the GPU per site ("behind": GPU GEMVs still running; "chain": serial GPU work after them), thread imbalance, GPU stalls on streamed slices; `2` adds a per-group table |
| `Q38_PROFILE_PREFILL=1` with `pbench` | prefill GPU time per phase (dequant, GEMM, copy waits, attention, DeltaNet) |
| `nsys profile --trace=cuda ...` | per-kernel times. If only a `.qdstrm` file is produced, convert it with `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i file.qdstrm`, then `nsys stats --report cuda_gpu_kern_sum` |
| `test_cpu_gemv --bench`, `test_gemv --bench`, `test_attn_prefill --bench` | kernel throughput per format / column count / context |
| `bench_cpu_ggml` | ggml's own CPU kernels on the same tensors (baseline) |
| `hw_probe` | GPU, PCIe and DRAM characteristics (`docs/hardware.md`) |

Nsight Compute hardware counters are not available on this machine (`RmProfilingAdminOnly=1`); use SASS
(`cuobjdump -sass`), `-Xptxas -v` (registers, spills) and micro-benchmarks.

## Working conventions

- Every experiment gets an entry in `docs/optimization_log.md`: hypothesis, change, before/after measurements,
  correctness checks, decision (kept or reverted), including failed ones.
- `bench/results/` is append-only: never modify or delete a recorded result.
- A change is kept only if end-to-end decode (or prefill) improves measurably and all tests pass; otherwise it is
  reverted and logged.
- Bugs get a regression test.

## Known pitfalls

- nvcc 12.4 miscompiled stores through `uint16` members of packed structs (low byte lost): in device code, store
  through explicitly aligned pointers.
- GPU float code meant to be bit-identical across code paths must use `__fmul_rn` / `__fadd_rn` (or identical
  expression structure): FMA contraction differs between unrolled variants.
- Template GPU kernels on the KV type: a runtime type inside unrolled loops doubled the instruction footprint.
- Never select between two `std::initializer_list` temporaries with `?:` (dangling backing array).
- Python edit scripts: assert that every anchor string is non-empty and found exactly once (an empty anchor once
  inserted text between every character of a source file).
- `pgrep -f` / `pkill -f` match their own command line; wait on a PID instead.
- The prefill chunk size silently drops when its buffers do not fit; check the `prefill buffers B=` log line
  before comparing prefill numbers.
- Prefill results depend on the GEMM shapes: the weight plan (made from the free VRAM at start, which the desktop
  changes) and the staging piece size select different cuBLAS kernels and therefore different roundings. Two runs
  are bit-comparable only under the same plan (`--cap-mib`) and the same piece size. When the 24 MB staging
  buffers do not fit, prefill borrows the LM-head rows before falling back to 12 MB pieces, so within a process the
  piece size practically never changes (the 12 MB fallback made `integration.snapshot` flaky).
- A kernel that is skipped on an out-of-range argument leaves stale outputs that the next kernels consume without
  error. Kernels abort instead (decode attention beyond its position limit once produced plausible but wrong text
  past 131072 tokens).
