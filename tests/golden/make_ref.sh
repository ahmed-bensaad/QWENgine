#!/usr/bin/env bash
# Run the FP32 reference model on every golden prompt; saves data/<p>.ref.bin and a comparison report.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${Q38_MODEL:-../../models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf}
for p in ${@:-short chat code multilingual}; do
  ../../build/ref_compare "$MODEL" data/$p.llama_cpu.bin --golden2 data/$p.llama_gpu.bin --save data/$p.ref.bin 2>&1 \
    | grep -v INFO | sed "s/^/[$p] /"
done
