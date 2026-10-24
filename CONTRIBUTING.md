# Contributing to QWENgine

Thanks for your interest. QWENgine is unusual: it is an inference engine for **one model**
(Qwen3.8-27B-GSQ-RCO-IQ3_XXS), **tuned on one machine** (a Quadro RTX 4000 Mobile + i9-9880H laptop), and it is fast
because of that. It should run on other NVIDIA + x86 machines too, and finding out how well is one of the most
useful contributions right now. This guide says what kinds of contributions fit, and how changes are tested and
measured before they are merged.

## What fits

| Contribution | Welcome? | Notes |
|---|---|---|
| Bug reports and fixes (crashes, wrong output, server/API incompatibilities) | yes | please include a reproduction; fixes come with a regression test |
| Correctness: tests, numerical checks against llama.cpp or the FP32 reference | yes | see "Correctness" below |
| Performance on the reference hardware | yes | needs an A/B measurement (see "Performance changes") |
| **Benchmark results from your machine** | **yes, as a pull request** | run `bench/community_bench.py`, add the record and a row to [BENCHMARKS.md](BENCHMARKS.md) (see "Benchmark results") |
| Support and tuning for other hardware | yes | build fixes, calibrated planner rates, kernels for other GPU or CPU generations; must not slow down the reference machine (see "Other hardware") |
| Server and API: OpenAI compatibility, agent-harness support, observability | yes | the engine is serialized: one generation at a time |
| Documentation, explanations, examples of client configurations | yes | |
| Other models, or generic rewrites | as a fork | changes that make the code generic at the cost of speed on the reference machine will not be merged; a fork that documents what it changed is the better place, and we are happy to link it |

If you plan a large change, open an issue first to discuss it.

## Development setup

Follow the [README](README.md) to get the model files and build. Then read:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): what the engine computes and how;
- [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md): why it is built the way it is;
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): tests, benchmarks, profiling, pitfalls.

On another NVIDIA GPU, configure with `-DCMAKE_CUDA_ARCHITECTURES=native`. Without a GPU you can still build the
code and run the tests that need no GPU or model (`unit.json`, `unit.chat`, `unit.metrics`, `unit.image_resize`):
`tools/ci.sh` does exactly that, and GitHub CI runs it on every pull request.

## Correctness

Every change must keep `ctest` green on the reference machine: `cd build && ctest; echo $?`. Check the exit status,
not a piped `tail`. GitHub CI builds every pull request and runs the tests that need no GPU (`tools/ci.sh`); the
full suite needs the GPU and the model files. The suite (~7 minutes) checks:

- quantized weights against libggml;
- tokenizer output against llama.cpp;
- CPU and GPU kernels against double-precision references;
- logits against llama.cpp and the FP32 reference implementation;
- that multi-token decode, speculative decoding, context growth and snapshots are **bit-identical** to the plain
  paths;
- the server end to end.

The reference tests need a llama.cpp build (`-DQ38_GGML_DIR=<llama.cpp>/build/bin`, see
[DEVELOPMENT.md](docs/DEVELOPMENT.md)) and golden data generated with `tests/golden/make_golden.sh` and
`make_ref.sh`. Without them those tests are disabled (re-run `cmake` once you have them).

Many optimizations must not change any result. For those, compare the logits of two builds under the same weight
plan (`golden --save` with `--cap-mib`, then `cmp`; see DEVELOPMENT.md). A change that alters results on purpose
(different summation order, a new kernel) must still pass the golden thresholds, and the PR must say so.

## Benchmark results

Numbers from other machines show where the engine's assumptions hold. To contribute yours, follow
[BENCHMARKS.md, Add your machine](BENCHMARKS.md#add-your-machine). In short:

1. Build, with `-DCMAKE_CUDA_ARCHITECTURES=native` on a non-Turing GPU, and put the model files in `models/`.
2. Close GPU programs, plug in a laptop, and run
   `python3 bench/community_bench.py --tests --name <handle> --ram "<type, channels>" --notes "<anything unusual>"`.
3. Open a pull request titled `Benchmark: <GPU> + <CPU>` that adds `bench/community/<record>.json` and the printed
   row to the table in BENCHMARKS.md.

Submit failed tests and slow results too. For a failure, also open an issue with the output of
`ctest --test-dir build --output-on-failure -R <test>`.

## Other hardware

QWENgine was built and tuned on one machine, but improvements for other hardware are welcome:

- build and portability fixes (compilers, CUDA versions, distributions);
- planner calibration: the device rates in `src/runtime/planner.cpp` and the streaming constants in `engine.h` were
  measured on the reference machine; measuring them elsewhere, or measuring them automatically at startup, would
  make the CPU/GPU split better everywhere;
- kernels for other GPU generations (`cp.async`, BF16 tensor cores, more SMs) or CPU features (AVX-512, VNNI),
  selected at compile time or at runtime.

The conditions:

- **The reference machine must not get slower.** Paths for other hardware should be selected by the architecture
  or CPU feature. A change that affects the shared code path needs a `bench/ab.py` measurement on the reference
  machine; if you do not have one, say so in the pull request and the maintainer will run it.
- **Show the gain on your machine** with `bench/community_bench.py` records before and after (or `bench/ab.py`).
- **`ctest` passes** on your machine and on the reference machine.

## Performance changes

End-to-end throughput decides, not micro-benchmarks. The laptop's thermal state moves single runs by ±5-10%, so:

1. Build the baseline in a separate `git worktree`.
2. Compare with `bench/ab.py`. It runs ABBA-ordered pairs and lets the CPU cool to a fixed temperature before
   every run:
   ```bash
   python3 bench/ab.py --a "base/build/q38_eval $MTP specset --draft auto --gen 80 --nprompts 16" \
                       --b "build/q38_eval $MTP specset --draft auto --gen 80 --nprompts 16" \
                       --metric specset_tok_s --rounds 2 --cool-to 70 --label my_change
   ```
   For prefill use `pbench --depth N` with `--metric prefill_tok_s`.
3. Keep the change only if the geometric mean of the pair ratios improves and all pairs point the same way, or the
   change strictly removes work.
4. Add the JSON record `ab.py` writes to `bench/results/`. That directory is append-only: never edit or delete a
   record.
5. Add an entry to [docs/optimization_log.md](docs/optimization_log.md): hypothesis, change, measurements before and
   after, correctness checks, decision. Failed experiments are welcome too: knowing what does not work saves
   everyone time.

Never run two GPU jobs at the same time (benchmarks, tests, a server): they compete for VRAM and the weight plan
changes.

## Code style

- C++17 and CUDA, built with `-Wall`; no new warnings.
- Match the surrounding code: naming, comment density, and the idioms of the file you change. Comments explain
  *why* (a measurement, a hardware constraint) rather than restating the code.
- Hot paths are specialized on purpose (model constants, formats, tile sizes). Do not generalize them without a
  measurement showing no loss.
- In GPU code meant to be bit-identical across paths, use `__fmul_rn` / `__fadd_rn` or identical expression
  structure. DEVELOPMENT.md lists other pitfalls.
- Python tools use the standard library only, except the optional OpenAI SDK checks.

## Pull requests

- One topic per PR, with a description of what changed and why.
- State what you ran: `ctest` result, `ab.py` record (performance changes), bit-identity check (no-change
  refactors).
- Update the documentation that describes what you changed (ARCHITECTURE.md, SERVER.md, DEVELOPMENT.md).
- Commit messages: a short summary line, then the reasoning and the measurement.

## Reporting issues

Please include:

- what you ran (command line or API request) and what happened, with the server log (`q38_server` writes one
  block per request) or the `q38_eval` output;
- `nvidia-smi` output, CPU model, and whether you are on the reference hardware;
- the commit you built (`git rev-parse HEAD`).

For performance reports, include the `q38_eval specset` / `pbench` lines and the GPU / CPU temperatures if you have
them, or better, a `bench/community_bench.py` record.
