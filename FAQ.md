## Frequently Asked Questions

### Can I use another quantization of Qwen3.8-27B (Q4_K_M, Q8_0, Unsloth, bartowski, ...)?

**The other GSQ-RCO files** of the same
[repository](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) (IQ2_XS, IQ2_S and IQ3_S, each with an
`-mtp` variant) use only formats QWENgine has kernels for, according to their published tensor allocations. They
should load and run, but they are untested. The 2.5 and 2.75 bpw files leave more weights in VRAM (faster, at a
quality cost); the 3.5 bpw IQ3_S file leaves more to the CPU (slower). Please share results in
[BENCHMARKS.md](BENCHMARKS.md).

**Other quantizations** work only if every tensor is in a format QWENgine has kernels for:

- weight matrices: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q2_K, Q4_K or Q6_K;
- norms: F32;
- the DeltaNet alpha / beta projections: BF16.

The engine checks this when loading and names the tensors it cannot run. Many llama.cpp quantizations use Q3_K,
Q5_K or Q8_0 for some tensors, and those are refused. A new format needs GPU and CPU kernels, and contributions
are welcome.

Two more caveats:

- **Bigger files are slower.** With 8 GB of VRAM, the weights that do not fit are read by the CPU on every pass:
  about 2.7 GB for the 9.7 GB GSQ-RCO file, but about 9 GB for a 16 GB Q4_K_M. Expect much slower decode.
- **Speculative decoding needs the MTP block** (`blk.64.*`, the `nextn` layer). Many conversions drop it.

### Can I use another model (Qwen3.6, Qwen3.8-35B-A3B, Llama, ...)?

No. The shapes of Qwen3.8-27B (64 Gated DeltaNet and gated-attention layers, the MTP block, 248K vocabulary) are
compile-time constants, and the loader refuses other files. Another model needs new kernels and planner constants,
so a fork is the better place.

### Will it run on my GPU?

It should if your GPU is an NVIDIA card with compute capability 7.5 or newer and 8 GB or more: RTX 20, 30, 40 and
50 series, Quadro RTX, RTX A-series, T4, L4, A100, H100. Build with `-DCMAKE_CUDA_ARCHITECTURES=native` (the RTX
50 series needs CUDA 12.8 or newer for that). It is untested so far:
[please share your results](BENCHMARKS.md#add-your-machine). Pascal and older (GTX 10-series) is not supported:
the kernels use Turing's tensor-core instructions.

### What about AMD or Intel GPUs, Apple Silicon, Windows, or CPU only?

None of these are supported: the GPU code is CUDA, and the CPU kernels are x86 AVX2. On Windows, WSL2 might work
but is untested.

### Does it need this exact CPU? What about AMD CPUs?

No. Any x86-64 CPU with AVX2, FMA and F16C works; the build checks for them. Speed depends mostly on memory
bandwidth. AVX2 gathers, which the CPU kernels for the IQ formats use, are slower with the Downfall mitigation and
on older AMD Zen cores (see [Other hardware](#other-hardware)).

### How much RAM do I need?

32 GB is recommended. Peak use is ~14-18 GB, more at long contexts, plus ~1 GB with the vision encoder. 16 GB is
untested and probably too tight.

### Why is it faster than llama.cpp on this machine?

llama.cpp places whole layers on the GPU or the CPU and runs them one after the other. QWENgine does three things
differently:

- it splits every matrix between the GPU and the CPU, and computes both halves at the same time;
- it streams part of the CPU's share over PCIe;
- it verifies up to 7 tokens per pass over the weights, with the model's own MTP block as the drafter.

[docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) explains each technique with its measured gain.

### Is the output identical to llama.cpp's?

Not bit for bit, but close. The tests compare logits with llama.cpp and with an FP32 reference implementation:
top-1 agreement ≥ 30/32 and KL ≤ 0.002. Within QWENgine, these paths are bit-identical to plain decoding:
multi-token passes, speculative decoding, context growth and snapshots.

### Does speculative decoding change the answers?

No. Greedy output is identical token for token to plain decoding, and sampled output follows exactly the same
distribution (speculative sampling).

### I only have the GGUF without the MTP block. Does it work?

Yes, with plain decoding (~9 tok/s on the reference machine): `build/q38_server models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`.

### How long can the context be?

Up to 262144 tokens, the model's training context. 100K and 150K were tested with `--kv q4_0`: about 9 tok/s, and
retrieval tests passed. A larger KV cache leaves less VRAM for the weights, so decode slows down as the context
grows.

### Can several users or requests run at once?

One generation runs at a time and other requests wait their turn. The engine keeps one conversation cached (plus
a snapshot), so clients that alternate evict each other's state.

### Images, video, audio?

Images are supported with the mmproj file ([docs/SERVER.md](docs/SERVER.md#images)). Video and audio are not.

### Does it work with agents and tool calling?

Yes. Tool calling, the Responses API and reasoning control are supported. [docs/SERVER.md](docs/SERVER.md) has
ready-made configurations for Aider, Continue, Cline, opencode, Codex CLI, the OpenAI SDKs and others.

### Which sampling parameters are supported?

These are supported: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop` and `max_tokens`, plus reasoning
effort. Sampling uses the engine's top-32 logits. These are not: repetition, presence and frequency penalties
(ignored), `logprobs`, `n > 1`, and enforcement of `response_format`.

### Why does a request sometimes pause for a few seconds?

The KV cache grew. It doubles on demand, and each doubling re-plans and re-uploads the weights (~6.5 s, once per
size). `--ctx-init` sets the starting size.

### How do I check that it works correctly on my machine?

Run `cd build && ctest; echo $?`. Tests whose model files or reference data are missing are skipped. The reference
comparisons need a llama.cpp build ([docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)). `bench/community_bench.py --tests`
records the outcome together with the benchmark.
