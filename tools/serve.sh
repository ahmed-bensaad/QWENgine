#!/usr/bin/env bash
# Start the OpenAI-compatible server on the MTP model (speculative decoding). See docs/SERVER.md.
#
#   tools/serve.sh [q38_server options]        e.g.  tools/serve.sh --port 8080 --effort low
#
# Environment: Q38_MTP_MODEL (model path; default models/ in the repository), Q38_MMPROJ (vision encoder for image
# input; default models/mmproj-Qwen3.8-27B-BF16.gguf, used when the file exists; empty: no image input), Q38_API_KEY
# (require this key from clients).
# Default options: 127.0.0.1:8080, context up to 32768 tokens, Q8_0 KV cache; any option given here overrides them.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${Q38_MTP_MODEL:-$root/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf}"
bin="$root/build/q38_server"
if [[ ! -x "$bin" ]]; then
    echo "q38_server is not built: cmake -S $root -B $root/build -G Ninja && cmake --build $root/build" >&2
    exit 1
fi
if [[ ! -f "$model" ]]; then
    echo "model not found: $model (set Q38_MTP_MODEL)" >&2
    exit 1
fi
mmproj="${Q38_MMPROJ-$root/models/mmproj-Qwen3.8-27B-BF16.gguf}"
extra=()
if [[ -n "$mmproj" && -f "$mmproj" && " $* " != *" --mmproj "* ]]; then
    extra=(--mmproj "$mmproj")
fi
exec "$bin" "$model" "${extra[@]}" "$@"
