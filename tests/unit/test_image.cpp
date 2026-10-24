/// @file test_image.cpp
/// Image helper for tests/unit/test_image_resize.py: decode an image file, resize it with resize_bicubic (Pillow's
/// BICUBIC) and write the result as a binary PPM.
/// Usage: test_image <in> <width> <height> <out.ppm>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "runtime/vision.h"

using namespace q38rt;

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s <in> <width> <height> <out.ppm>\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    std::vector<uint8_t> bytes;
    uint8_t buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) bytes.insert(bytes.end(), buf, buf + n);
    fclose(f);
    Image in, out;
    std::string err;
    if (!decode_image(bytes.data(), bytes.size(), in, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    resize_bicubic(in, atoi(argv[2]), atoi(argv[3]), out);
    FILE* o = fopen(argv[4], "wb");
    if (!o) return 1;
    fprintf(o, "P6\n%d %d\n255\n", out.width, out.height);
    fwrite(out.rgb.data(), 1, out.rgb.size(), o);
    fclose(o);
    return 0;
}
