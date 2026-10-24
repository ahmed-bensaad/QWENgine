/// @file ref_mtmd_embd.cpp
/// Reference image embeddings from llama.cpp's multimodal library (libmtmd), for the vision encoder test: loads the
/// text model's metadata and the mmproj, encodes one image with the same token limits, and writes the embeddings.
///
///   ref_mtmd_embd <model.gguf> <mmproj.gguf> <image> <out.bin> [--min-tokens N] [--max-tokens N] [--cpu]
///
/// Output: "Q38V", int32 nx, ny, dim, then nx * ny * dim floats (merged tokens in raster order).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.gguf> <mmproj.gguf> <image> <out.bin> [--min-tokens N] [--max-tokens N] [--cpu]\n", argv[0]);
        return 2;
    }
    int min_tokens = 8, max_tokens = 1024;
    bool cpu = false;
    for (int i = 5; i < argc; i++) {
        if (!strcmp(argv[i], "--min-tokens") && i + 1 < argc) min_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) cpu = true;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = !cpu;
    cp.warmup = false;
    cp.n_threads = 8;
    cp.image_min_tokens = min_tokens;
    cp.image_max_tokens = max_tokens;
    mtmd_context* ctx = mtmd_init_from_file(argv[2], model, cp);
    if (!ctx) { fprintf(stderr, "cannot load %s\n", argv[2]); return 1; }
    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, argv[3], false);
    if (!bw.bitmap) { fprintf(stderr, "cannot load image %s\n", argv[3]); return 1; }
    const std::string prompt = mtmd_default_marker();
    mtmd_input_text text{prompt.c_str(), prompt.size(), false, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[] = {bw.bitmap};
    if (mtmd_tokenize(ctx, chunks, &text, bitmaps, 1) != 0) { fprintf(stderr, "mtmd_tokenize failed\n"); return 1; }
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        if (mtmd_encode_chunk(ctx, ch) != 0) { fprintf(stderr, "mtmd_encode_chunk failed\n"); return 1; }
        const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ch);
        const int n = (int)mtmd_input_chunk_get_n_tokens(ch), dim = llama_model_n_embd(model);
        const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, n - 1);   // (col, row) of the last token
        const int nx = (int)last.x + 1, ny = (int)last.y + 1;
        if (n != nx * ny) { fprintf(stderr, "unexpected token count %d for %dx%d\n", n, nx, ny); return 1; }
        FILE* f = fopen(argv[4], "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
        const int hdr[3] = {nx, ny, dim};
        fwrite("Q38V", 1, 4, f);
        fwrite(hdr, 4, 3, f);
        fwrite(mtmd_get_output_embd(ctx), 4, (size_t)n * dim, f);
        fclose(f);
        printf("image %s: %d x %d merged tokens of %d -> %s\n", argv[3], nx, ny, dim, argv[4]);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
        mtmd_free(ctx);
        llama_model_free(model);
        return 0;
    }
    fprintf(stderr, "no image chunk\n");
    return 1;
}
