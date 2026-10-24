#!/usr/bin/env bash
# Reference image embeddings for the vision encoder test (golden.vision.*), from llama.cpp's multimodal library on the
# CPU (FP32; its CUDA path is noisier: per-token cosine down to 0.965 against the CPU on a 768x768 image).
# Test images are generated here (deterministic, binary PPM) at sizes that are multiples of 32 within the token limits,
# so neither side resizes them: llama.cpp pads and resamples bilinearly where the engine stretches with Pillow's bicubic
# filter (the Hugging Face processor), so resized images are not comparable.
# Output in tests/golden/data/ (gitignored): vision_<name>.ppm and vision_<name>.ref.bin (~2-21 MB each).
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${Q38_MODEL:-../../models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf}
MMPROJ=${Q38_MMPROJ:-../../models/mmproj-Qwen3.8-27B-BF16.gguf}
BIN=../../build/ref_mtmd_embd
mkdir -p data
python3 - <<'PY'
import math
def make(name, w, h, seed):
    # smooth color fields, rings and a checkerboard (structure at several scales) plus deterministic noise
    s = seed
    px = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            s = (s * 1103515245 + 12345) & 0x7fffffff
            n = (s >> 16) % 25 - 12
            r = 128 + 100 * math.sin(x / 37.0 + seed) * math.cos(y / 53.0)
            g = 128 + 100 * math.sin(math.hypot(x - w / 3, y - h / 2) / 19.0)
            b = 60 if ((x // 64) + (y // 64)) % 2 else 200
            px[(y * w + x) * 3:(y * w + x) * 3 + 3] = bytes(max(0, min(255, int(v + n))) for v in (r, g, b))
    with open(f"data/vision_{name}.ppm", "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h) + bytes(px))
make("768x512", 768, 512, 1)
make("320x640", 320, 640, 2)
make("1024x1024", 1024, 1024, 3)
PY
for n in 768x512 320x640 1024x1024; do
  $BIN "$MODEL" "$MMPROJ" data/vision_$n.ppm data/vision_$n.ref.bin --cpu 2> data/vision_$n.log
done
