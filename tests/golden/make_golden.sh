#!/usr/bin/env bash
# Regenerate golden logits with the trusted llama.cpp implementation.
# Full-vocabulary dumps go to tests/golden/data/ (gitignored, ~32 MB each).
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${Q38_MODEL:-../../models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf}
BIN=../../build/ref_llama_logits
mkdir -p data
for p in short chat code multilingual; do
  $BIN "$MODEL" $p.txt data/$p.llama_cpu.bin --ngl 0 --gen 16 --last 16 --kv f32 --threads 8 2> data/$p.llama_cpu.log
  $BIN "$MODEL" $p.txt data/$p.llama_gpu.bin --ngl 36 --gen 16 --last 16 --kv f16 --threads 8 2> data/$p.llama_gpu.log
  grep -h "greedy continuation" data/$p.llama_cpu.log data/$p.llama_gpu.log
done
