#!/usr/bin/env python3
"""Image preprocessing: resize_bicubic (runtime/vision.h) must equal Pillow's Image.resize(BICUBIC) byte for byte
(the Qwen2-VL image processor resizes with Pillow). Random images, down- and upscaling, both axes and one axis only.
Skipped (passes) when Pillow is not installed.
Usage: test_image_resize.py <test_image binary>"""
import os, random, subprocess, sys, tempfile

try:
    from PIL import Image
except ImportError:
    print("skip: Pillow is not installed")
    sys.exit(0)

binary = sys.argv[1]
rng = random.Random(7)
cases = [((640, 488), (640, 480)), ((1920, 1080), (1344, 768)), ((333, 211), (416, 256)), ((97, 1200), (64, 1024)),
         ((800, 600), (800, 608)), ((1024, 77), (1024, 96))]
fails = 0
with tempfile.TemporaryDirectory() as d:
    for (iw, ih), (ow, oh) in cases:
        src = Image.frombytes("RGB", (iw, ih), bytes(rng.getrandbits(8) for _ in range(iw * ih * 3)))
        # smooth content too (random noise alone exercises the clamping, not the interpolation)
        src = Image.blend(src, src.resize((max(1, iw // 8), max(1, ih // 8))).resize((iw, ih), Image.BILINEAR), 0.7)
        p_in, p_out = os.path.join(d, "in.ppm"), os.path.join(d, "out.ppm")
        src.save(p_in)
        subprocess.run([binary, p_in, str(ow), str(oh), p_out], check=True)
        ours = Image.open(p_out).convert("RGB").tobytes()
        ref = src.resize((ow, oh), Image.BICUBIC).tobytes()
        diff = sum(1 for a, b in zip(ours, ref) if a != b)
        print(("ok   " if diff == 0 else "FAIL ") + f"{iw}x{ih} -> {ow}x{oh}: {diff} of {len(ref)} bytes differ")
        fails += diff != 0
print("test_image_resize:", "all identical to Pillow" if fails == 0 else f"{fails} failures")
sys.exit(1 if fails else 0)
