/// @file vision.cpp
/// See vision.h. Host side: image decoding, preprocessing, mmproj loading. The GPU forward is Engine::encode_image
/// (vision_encode.cpp).

#include "runtime/vision.h"
#include <cuda_runtime.h>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "common/utils.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "third_party/stb_image.h"
#pragma GCC diagnostic pop

namespace q38rt {

bool decode_image(const void* data, size_t bytes, Image& out, std::string& err) {
    int w = 0, h = 0, n = 0;
    uint8_t* px = stbi_load_from_memory((const stbi_uc*)data, (int)bytes, &w, &h, &n, 3);
    if (!px) {
        err = std::string("cannot decode the image: ") + stbi_failure_reason();
        return false;
    }
    out.width = w;
    out.height = h;
    out.rgb.assign(px, px + (size_t)w * h * 3);
    stbi_image_free(px);
    return true;
}

namespace {

// BF16 / F16 / F32 tensor -> FP16 (round to nearest even). False if a value does not fit FP16.
bool to_fp16(const GGUFParser& g, const TensorInfo& t, uint16_t* dst) {
    const uint8_t* src = g.tensor_data(t);
    const size_t n = t.shape.size();
    float maxabs = 0;
    size_t i = 0;
    auto put8 = [&](__m256 v, size_t at) {
        maxabs = std::max(maxabs, [&] {
            alignas(32) float f[8];
            _mm256_store_ps(f, _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v));
            return *std::max_element(f, f + 8);
        }());
        _mm_storeu_si128((__m128i*)(dst + at), _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
    };
    switch (t.quant) {
        case QuantFormat::BF16:
            for (; i + 8 <= n; i += 8) {
                const __m128i b = _mm_loadu_si128((const __m128i*)(src + 2 * i));
                put8(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b), 16)), i);
            }
            break;
        case QuantFormat::FP32:
            for (; i + 8 <= n; i += 8) put8(_mm256_loadu_ps((const float*)src + i), i);
            break;
        case QuantFormat::FP16:
            memcpy(dst, src, n * 2);
            return true;
        default:
            LOG_ERROR("mmproj tensor %s: unsupported type %s", t.name.c_str(), quant_name(t.quant));
            return false;
    }
    if (i != n) { LOG_ERROR("mmproj tensor %s: %zu elements, not a multiple of 8", t.name.c_str(), n); return false; }
    if (maxabs > 65504.0f) { LOG_ERROR("mmproj tensor %s: value %g does not fit FP16", t.name.c_str(), maxabs); return false; }
    return true;
}

bool to_f32(const GGUFParser& g, const TensorInfo& t, float* dst) {
    if (t.quant != QuantFormat::FP32) { LOG_ERROR("mmproj tensor %s: expected F32, got %s", t.name.c_str(), quant_name(t.quant)); return false; }
    memcpy(dst, g.tensor_data(t), t.shape.size() * 4);
    return true;
}

} // namespace

VisionModel::~VisionModel() {
    for (uint16_t* p : {h_layers_, h_patch_, h_mm0_, h_mm2_}) if (p) cudaFreeHost(p);
}

bool VisionModel::load(const std::string& path, int min_tokens, int max_tokens) {
    if (!gguf_.parse(path, false, "clip")) return false;
    min_tokens_ = std::max(1, min_tokens);
    max_tokens_ = std::max(min_tokens_, max_tokens);
    const GGUFParser& g = gguf_;
    auto need_u = [&](const char* key, uint64_t want) {
        if (g.get_kv_uint(key, 0) == want) return true;
        LOG_ERROR("mmproj %s = %llu, this runtime expects %llu (Qwen3.8-27B's vision projector)", key,
                  (unsigned long long)g.get_kv_uint(key, 0), (unsigned long long)want);
        return false;
    };
    if (g.get_kv_string("clip.projector_type") != "qwen3vl_merger") {
        LOG_ERROR("mmproj projector '%s', expected qwen3vl_merger", g.get_kv_string("clip.projector_type").c_str());
        return false;
    }
    if (!need_u("clip.vision.projection_dim", OUT) || !need_u("clip.vision.embedding_length", D) ||
        !need_u("clip.vision.block_count", LAYERS) || !need_u("clip.vision.attention.head_count", HEADS) ||
        !need_u("clip.vision.patch_size", PATCH) || !need_u("clip.vision.feed_forward_length", FF) ||
        !need_u("clip.vision.spatial_merge_size", MERGE))
        return false;
    if (const GGUFValue* ds = g.find_kv("clip.vision.is_deepstack_layers"))
        for (uint8_t b : ds->arr_raw) if (b) { LOG_ERROR("mmproj with deepstack layers is not supported"); return false; }
    for (const char* key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
        const GGUFValue* v = g.find_kv(key);
        if (!v || v->arr_raw.size() != 12) { LOG_ERROR("mmproj: %s missing", key); return false; }
        float f[3];
        memcpy(f, v->arr_raw.data(), 12);
        if (f[0] != 0.5f || f[1] != 0.5f || f[2] != 0.5f) { LOG_ERROR("mmproj: %s is not 0.5 (preprocessing assumes it)", key); return false; }
    }
    eps_ = g.get_kv_float("clip.vision.attention.layer_norm_epsilon", 1e-6f);

    auto tensor = [&](const std::string& name, std::initializer_list<uint64_t> dims) -> const TensorInfo* {
        const TensorInfo* t = g.find_tensor(name);
        if (!t) { LOG_ERROR("mmproj tensor %s missing", name.c_str()); return nullptr; }
        size_t i = 0;
        for (uint64_t d : dims) {
            if (i >= t->shape.ndim || t->shape.dims[i] != d) { LOG_ERROR("mmproj tensor %s: unexpected shape", name.c_str()); return nullptr; }
            i++;
        }
        return t;
    };
    auto alloc = [&](uint16_t*& p, size_t elems) {
        if (cudaHostAlloc((void**)&p, elems * 2, cudaHostAllocDefault) != cudaSuccess) {
            (void)cudaGetLastError();
            p = nullptr;
            LOG_ERROR("mmproj: cannot allocate %.0f MiB of pinned memory", elems * 2 / 1048576.0);
            return false;
        }
        return true;
    };
    if (!alloc(h_layers_, LAYER_ELEMS * LAYERS) || !alloc(h_patch_, (size_t)D * 3 * PATCH * PATCH) ||
        !alloc(h_mm0_, (size_t)MERGED * MERGED) || !alloc(h_mm2_, (size_t)OUT * MERGED))
        return false;
    small_.assign(LAYER_SMALL * LAYERS, 0.0f);
    for (int il = 0; il < LAYERS; il++) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        uint16_t* w = h_layers_ + (size_t)il * LAYER_ELEMS;
        float* s = small_.data() + (size_t)il * LAYER_SMALL;
        const TensorInfo *qkv = tensor(p + "attn_qkv.weight", {D, 3 * D}), *out = tensor(p + "attn_out.weight", {D, D}),
                         *up = tensor(p + "ffn_up.weight", {D, FF}), *down = tensor(p + "ffn_down.weight", {FF, D});
        const TensorInfo *ln1w = tensor(p + "ln1.weight", {D}), *ln1b = tensor(p + "ln1.bias", {D}),
                         *qkvb = tensor(p + "attn_qkv.bias", {3 * D}), *outb = tensor(p + "attn_out.bias", {D}),
                         *ln2w = tensor(p + "ln2.weight", {D}), *ln2b = tensor(p + "ln2.bias", {D}),
                         *upb = tensor(p + "ffn_up.bias", {FF}), *downb = tensor(p + "ffn_down.bias", {D});
        if (!qkv || !out || !up || !down || !ln1w || !ln1b || !qkvb || !outb || !ln2w || !ln2b || !upb || !downb) return false;
        if (!to_fp16(g, *qkv, w) || !to_fp16(g, *out, w + QKV_ELEMS) || !to_fp16(g, *up, w + QKV_ELEMS + OUT_ELEMS) ||
            !to_fp16(g, *down, w + QKV_ELEMS + OUT_ELEMS + UP_ELEMS))
            return false;
        float* o = s;
        for (const TensorInfo* t : {ln1w, ln1b, qkvb, outb, ln2w, ln2b, upb, downb}) {
            if (!to_f32(g, *t, o)) return false;
            o += t->shape.size();
        }
    }
    // patch embedding: the two temporal kernels [16][16][3][1152] (ggml order: x fastest) see the same still image
    {
        const TensorInfo *k0 = tensor("v.patch_embd.weight", {PATCH, PATCH, 3, D}),
                         *k1 = tensor("v.patch_embd.weight.1", {PATCH, PATCH, 3, D}),
                         *b = tensor("v.patch_embd.bias", {D});
        if (!k0 || !k1 || !b) return false;
        std::vector<float> a(k0->shape.size()), c(k1->shape.size());
        if (!to_f32(g, *k0, a.data()) || !to_f32(g, *k1, c.data())) return false;
        for (size_t i = 0; i < a.size(); i++) a[i] += c[i];
        for (size_t i = 0; i < a.size(); i += 8)
            _mm_storeu_si128((__m128i*)(h_patch_ + i), _mm256_cvtps_ph(_mm256_loadu_ps(a.data() + i), _MM_FROUND_TO_NEAREST_INT));
        patch_bias_.resize(D);
        if (!to_f32(g, *b, patch_bias_.data())) return false;
    }
    {
        const TensorInfo* t = tensor("v.position_embd.weight", {D, POS_SIDE * POS_SIDE});
        if (!t) return false;
        pos_.resize(t->shape.size());
        if (!to_f32(g, *t, pos_.data())) return false;
    }
    {
        const TensorInfo *w = tensor("v.post_ln.weight", {D}), *b = tensor("v.post_ln.bias", {D});
        if (!w || !b) return false;
        post_ln_.resize(2 * D);
        if (!to_f32(g, *w, post_ln_.data()) || !to_f32(g, *b, post_ln_.data() + D)) return false;
    }
    {
        const TensorInfo *w0 = tensor("mm.0.weight", {MERGED, MERGED}), *b0 = tensor("mm.0.bias", {MERGED}),
                         *w2 = tensor("mm.2.weight", {MERGED, OUT}), *b2 = tensor("mm.2.bias", {OUT});
        if (!w0 || !b0 || !w2 || !b2) return false;
        if (!to_fp16(g, *w0, h_mm0_) || !to_fp16(g, *w2, h_mm2_)) return false;
        mm_bias_.resize(MERGED + OUT);
        if (!to_f32(g, *b0, mm_bias_.data()) || !to_f32(g, *b2, mm_bias_.data() + MERGED)) return false;
    }
    LOG_INFO("vision encoder: %s (%.0f MiB of FP16 weights in pinned memory), images of %d to %d tokens", path.c_str(),
             (LAYER_ELEMS * LAYERS + (size_t)MERGED * (MERGED + OUT)) * 2 / 1048576.0, min_tokens_, max_tokens_);
    return true;
}

// llama.cpp img_tool::calc_size_preserved_ratio (transformers' smart_resize): sides multiples of 32 near the input's,
// area within [min_tokens, max_tokens] * 32 * 32 pixels.
void VisionModel::target_size(int width, int height, int& out_w, int& out_h) const {
    const int f = PATCH * MERGE;
    const int min_pixels = min_tokens_ * f * f, max_pixels = max_tokens_ * f * f;
    auto round_f = [f](float x) { return (int)std::round(x / (float)f) * f; };
    auto ceil_f = [f](float x) { return (int)std::ceil(x / (float)f) * f; };
    auto floor_f = [f](float x) { return (int)std::floor(x / (float)f) * f; };
    int h = std::max(f, round_f((float)height)), w = std::max(f, round_f((float)width));
    if (h * w > max_pixels) {
        const float beta = std::sqrt((float)(height * width) / max_pixels);
        h = std::max(f, floor_f(height / beta));
        w = std::max(f, floor_f(width / beta));
    } else if (h * w < min_pixels) {
        const float beta = std::sqrt((float)min_pixels / (height * width));
        h = ceil_f(height * beta);
        w = ceil_f(width * beta);
    }
    out_w = w;
    out_h = h;
}

namespace {

// Pillow's Image.resize(BICUBIC) (ImagingResample): separable, horizontal then vertical; bicubic kernel a = -0.5,
// widened by the scale factor when downsampling; 22-bit fixed-point weights rounded half away from zero.
// The Hugging Face Qwen2-VL image processor, used for the model's training, resizes this way.
void resize_bicubic_pillow(const std::vector<uint8_t>& in, int iw, int ih, std::vector<uint8_t>& out, int ow, int oh) {
    constexpr int PREC = 22;
    auto filter = [](double x) {
        constexpr double a = -0.5;
        x = std::fabs(x);
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    };
    struct Coeffs { int ksize; std::vector<int> bounds; std::vector<int32_t> k; };
    auto coeffs = [&](int in_size, int out_size) {
        Coeffs c;
        const double scale = (double)in_size / out_size, fscale = std::max(1.0, scale), support = 2.0 * fscale;
        c.ksize = (int)std::ceil(support) * 2 + 1;
        c.bounds.resize(2 * out_size);
        c.k.assign((size_t)out_size * c.ksize, 0);
        std::vector<double> w(c.ksize);
        for (int xx = 0; xx < out_size; xx++) {
            const double center = (xx + 0.5) * scale;
            int xmin = (int)(center - support + 0.5), xmax = (int)(center + support + 0.5);
            xmin = std::max(xmin, 0);
            xmax = std::min(xmax, in_size) - xmin;
            double ww = 0;
            for (int x = 0; x < xmax; x++) { w[x] = filter((x + xmin - center + 0.5) / fscale); ww += w[x]; }
            for (int x = 0; x < xmax; x++) {
                const double v = ww != 0.0 ? w[x] / ww : 0.0;
                c.k[(size_t)xx * c.ksize + x] = (int32_t)(v * (1 << PREC) + (v < 0 ? -0.5 : 0.5));
            }
            c.bounds[2 * xx] = xmin;
            c.bounds[2 * xx + 1] = xmax;
        }
        return c;
    };
    auto clip8 = [](int32_t v) { return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); };
    std::vector<uint8_t> tmp;
    const std::vector<uint8_t>* src = &in;
    int w = iw;
    if (ow != iw) {   // horizontal
        const Coeffs c = coeffs(iw, ow);
        tmp.resize((size_t)ow * ih * 3);
        for (int y = 0; y < ih; y++)
            for (int xx = 0; xx < ow; xx++) {
                const int xmin = c.bounds[2 * xx], xcnt = c.bounds[2 * xx + 1];
                const int32_t* k = c.k.data() + (size_t)xx * c.ksize;
                for (int ch = 0; ch < 3; ch++) {
                    int32_t ss = 1 << (PREC - 1);
                    for (int x = 0; x < xcnt; x++) ss += (*src)[((size_t)y * iw + xmin + x) * 3 + ch] * k[x];
                    tmp[((size_t)y * ow + xx) * 3 + ch] = clip8(ss >> PREC);
                }
            }
        src = &tmp;
        w = ow;
    }
    if (oh != ih) {   // vertical
        const Coeffs c = coeffs(ih, oh);
        out.resize((size_t)w * oh * 3);
        for (int yy = 0; yy < oh; yy++) {
            const int ymin = c.bounds[2 * yy], ycnt = c.bounds[2 * yy + 1];
            const int32_t* k = c.k.data() + (size_t)yy * c.ksize;
            for (int x = 0; x < w; x++)
                for (int ch = 0; ch < 3; ch++) {
                    int32_t ss = 1 << (PREC - 1);
                    for (int y = 0; y < ycnt; y++) ss += (*src)[((size_t)(ymin + y) * w + x) * 3 + ch] * k[y];
                    out[((size_t)yy * w + x) * 3 + ch] = clip8(ss >> PREC);
                }
        }
    } else {
        out = *src;
    }
}

} // namespace

void resize_bicubic(const Image& in, int width, int height, Image& out) {
    out.width = width;
    out.height = height;
    if (width == in.width && height == in.height) { out.rgb = in.rgb; return; }
    resize_bicubic_pillow(in.rgb, in.width, in.height, out.rgb, width, height);
}

ImagePatches VisionModel::preprocess(const Image& img) const {
    ImagePatches ip;
    target_size(img.width, img.height, ip.width, ip.height);
    ip.nx = ip.width / (PATCH * MERGE);
    ip.ny = ip.height / (PATCH * MERGE);
    // stretched to the target size (as the Qwen2-VL processor; llama.cpp pads to keep the aspect ratio instead)
    std::vector<uint8_t> rs;
    const std::vector<uint8_t>* px = &img.rgb;
    if (ip.width != img.width || ip.height != img.height) {
        resize_bicubic_pillow(img.rgb, img.width, img.height, rs, ip.width, ip.height);
        px = &rs;
    }
    // normalize ((v / 255 - 0.5) / 0.5) and cut into patches in merge order
    ip.patch.resize((size_t)ip.n_patches() * 3 * PATCH * PATCH);
    for (int my = 0; my < ip.ny; my++)
        for (int mx = 0; mx < ip.nx; mx++)
            for (int dy = 0; dy < MERGE; dy++)
                for (int dx = 0; dx < MERGE; dx++) {
                    const size_t t = (((size_t)my * ip.nx + mx) * MERGE + dy) * MERGE + dx;
                    float* o = ip.patch.data() + t * 3 * PATCH * PATCH;
                    const int x0 = (mx * MERGE + dx) * PATCH, y0 = (my * MERGE + dy) * PATCH;
                    for (int c = 0; c < 3; c++)
                        for (int ky = 0; ky < PATCH; ky++)
                            for (int kx = 0; kx < PATCH; kx++) {
                                const float v = (float)(*px)[((size_t)(y0 + ky) * ip.width + x0 + kx) * 3 + c] / 255.0f;
                                o[c * PATCH * PATCH + ky * PATCH + kx] = (v - 0.5f) / 0.5f;
                            }
                }
    return ip;
}

// ggml_interpolate(BILINEAR | ALIGN_CORNERS) of the 48x48 grid to the image's patch grid, then merge order
std::vector<float> VisionModel::position_embeddings(int nx, int ny) const {
    const int W = nx * MERGE, H = ny * MERGE;
    std::vector<float> grid((size_t)W * H * D);   // raster order
    if (W == POS_SIDE && H == POS_SIDE) {
        grid = pos_;
    } else {
        const float sf0 = W > 1 ? (float)(W - 1) / (POS_SIDE - 1) : (float)W / POS_SIDE;
        const float sf1 = H > 1 ? (float)(H - 1) / (POS_SIDE - 1) : (float)H / POS_SIDE;
        for (int y = 0; y < H; y++) {
            const float fy = (float)y / sf1;
            int y0 = (int)std::floor(fy), y1 = y0 + 1;
            y0 = std::max(0, std::min(y0, POS_SIDE - 1));
            y1 = std::max(0, std::min(y1, POS_SIDE - 1));
            const float dy = std::max(0.0f, std::min(fy - (float)y0, 1.0f));
            for (int x = 0; x < W; x++) {
                const float fx = (float)x / sf0;
                int x0 = (int)std::floor(fx), x1 = x0 + 1;
                x0 = std::max(0, std::min(x0, POS_SIDE - 1));
                x1 = std::max(0, std::min(x1, POS_SIDE - 1));
                const float dx = std::max(0.0f, std::min(fx - (float)x0, 1.0f));
                const float* a = pos_.data() + ((size_t)y0 * POS_SIDE + x0) * D;
                const float* b = pos_.data() + ((size_t)y0 * POS_SIDE + x1) * D;
                const float* c = pos_.data() + ((size_t)y1 * POS_SIDE + x0) * D;
                const float* d = pos_.data() + ((size_t)y1 * POS_SIDE + x1) * D;
                float* o = grid.data() + ((size_t)y * W + x) * D;
                for (int e = 0; e < D; e++)
                    o[e] = a[e] * (1 - dx) * (1 - dy) + b[e] * dx * (1 - dy) + c[e] * (1 - dx) * dy + d[e] * dx * dy;
            }
        }
    }
    std::vector<float> out(grid.size());
    for (int my = 0; my < ny; my++)
        for (int mx = 0; mx < nx; mx++)
            for (int dy = 0; dy < MERGE; dy++)
                for (int dx = 0; dx < MERGE; dx++) {
                    const size_t t = (((size_t)my * nx + mx) * MERGE + dy) * MERGE + dx;
                    const size_t src = (size_t)(my * MERGE + dy) * W + mx * MERGE + dx;
                    memcpy(out.data() + t * D, grid.data() + src * D, D * 4);
                }
    return out;
}

} // namespace q38rt
