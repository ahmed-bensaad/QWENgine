# Benchmarks

QWENgine is tuned on one machine, a laptop with a Quadro RTX 4000 Mobile (8 GB) and an i9-9880H. Nobody knows yet
how it behaves on yours: a different GPU, CPU, memory or PCIe link moves the bottlenecks, and only measurements can
tell by how much. This file collects those measurements. **Results from any machine are welcome, slow ones and
failed tests included**: they show where the engine's assumptions hold and where they do not.

## Results

One row per machine and configuration, measured with [`bench/community_bench.py`](bench/community_bench.py) (see
[Add your machine](#add-your-machine)). The first row is the reference machine. Each row links to its JSON record in [`bench/community/`](bench/community/),
which holds the full machine description, the `hw_probe` bandwidth measurements and every run.

| Date | GPU | CPU | RAM | Decode at 64K (tok/s) | Decode at 128K (tok/s) | Plain decode at 64K (tok/s) | Prefill 64K (tok/s) | Prefill 128K (tok/s) | Image, 300 tokens (ms) | Tests | Commit | Record | Submitter, notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23 | Quadro RTX 4000 (laptop) 8 GB, PCIe 3.0 x16 | Intel Core i9-9880H (8 cores) | 30.7 GiB DDR4 | 10.2 | 7.2 | 6.6 | 350 | 319 | 170 | pass | `initial release` | [json](bench/community/2026-10-23_quadro-rtx-4000_i9-9880h.json) | reference machine (QWENgine is tuned on it); desktop session on the GPU |
<!-- add your row below this line (keep the rows in date order) -->

The benchmark is a coding agent's workload at long context (`q38_eval longctx`): a fixed snapshot of C++ / CUDA
sources ([`data/longctx_code.txt`](data/longctx_code.txt)) fills the context, then four coding tasks
([`data/longctx_tasks.txt`](data/longctx_tasks.txt): write a unit test, add a server option, write a log parser,
refactor a class) are answered from it, each as the next turn of the conversation (the context is processed once and
restored from a snapshot). The KV cache is Q4_0, sized to the context.

| Column | What it measures |
|---|---|
| Decode at 64K | the main number: speculative decoding (MTP drafts, greedy) of the four answers, 256 tokens each, after 64K tokens of code |
| Decode at 128K | the same after 127K tokens of code, filling a 131072-token context |
| Plain decode at 64K | one token per pass, no drafting: the raw speed of the CPU / GPU / PCIe split at that depth |
| Prefill 64K / 128K | average prompt processing speed over the whole context (GPU tensor cores, host weights streamed over PCIe) |
| Image, 300 tokens | the vision encoder on a 640x480 image (when the mmproj file is present) |
| Tests | `ctest` on that machine: `pass`, the number of failures, or `not run` |

The JSON records also carry the measured bandwidths that explain most differences between machines: VRAM read,
PCIe host-to-device, CPU DRAM read alone and during DMA.

## Add your machine

1. **Build** following the [README](README.md#quick-start). On a GPU other than a Turing card, build for it:
   `cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build`.
2. **Get the model files** into `models/`: the `-mtp.gguf` file is required; the mmproj file is optional (image
   timing).
3. **Prepare the machine.** Close programs that use the GPU (browsers with hardware acceleration, games, other
   inference servers). Plug a laptop in and use its performance power mode. Let it cool down for a minute.
4. **Run** (about 30 minutes with the tests; the 128K context alone takes ~12):
   ```bash
   python3 bench/community_bench.py --tests --name <your GitHub handle> --ram "DDR5-5600, 2 channels" \
           --notes "desktop, Ubuntu 24.04, GPU also drives the display"
   ```
   `--ram` and `--notes` are free text; the script cannot read the memory type without root. It waits for the CPU
   to cool before each run, writes `bench/community/<date>_<gpu>_<cpu>.json` and prints the table row. `--quick`
   (~6 minutes: 64K only, two tasks, no tests) is fine for a first look, but please submit full runs. With more
   VRAM you may also try `--kv q8_0` (recorded in the notes column).
5. **Open a pull request** titled `Benchmark: <GPU> + <CPU>` that adds the JSON file and the printed row to the
   table above. One machine per pull request; several configurations of the same machine are welcome (a different
   `--kv`, power mode, driver, ...: say what differs in the notes).

Please:

- **Submit failed tests too.** The record lists the failing tests; also open an issue with `ctest --test-dir build
  --output-on-failure -R <test>`. A failure on new hardware is exactly what this collection is for.
- **Run unmodified code**, or say what you changed in the notes (the commit column ends in `+modified` when the
  working tree has changes). Changes that make QWENgine faster on your machine are welcome as separate pull
  requests (see [CONTRIBUTING.md](CONTRIBUTING.md#other-hardware)).
- **Do not edit other people's rows or records.** If a record looks wrong, open an issue.

## What to expect on other machines

The engine runs the same way everywhere: it measures the free VRAM, keeps as many weights there as fit and computes
the rest on the CPU, with some of it streamed over PCIe. What changes is which resource is the bottleneck, and the
planner's cost model, which was measured on the reference machine. The README's
[Other hardware](README.md#other-hardware) section lists what helps, what hurts, and the known limitations.
