#pragma once
/// @file vision.h
/// Image input: the vision encoder of Qwen3.8-27B's mmproj GGUF and the preprocessing in front of it.
///
/// The encoder is a Qwen3-VL vision transformer: 16x16-pixel patches embedded to width 1152 (two temporal patch kernels,
/// summed for a still image), plus learned 48x48 position embeddings resized bilinearly to the image's patch grid;
/// 27 pre-norm blocks (LayerNorm, 16-head attention with 2D rotary positions, GELU MLP of width 4304); a final
/// LayerNorm; then each 2x2 group of patches is concatenated (4608) and projected to the language model's width 5120 by
/// a 2-layer GELU MLP. One merged token per 32x32 pixels. The network follows llama.cpp's mtmd (clip_graph_qwen3vl),
/// which the tests compare against; the preprocessing follows the Hugging Face Qwen2-VL image processor used in
/// training (smart resize to multiples of 32, stretched with Pillow's bicubic filter), where llama.cpp pads the image to
/// keep its aspect ratio and resamples bilinearly.
///
/// Weights stay in pinned host memory (converted once from BF16 to FP16) and are streamed to the GPU layer by layer
/// for each image (Engine::encode_image): the VRAM belongs to the language model's weights.

#include <cstdint>
#include <string>
#include <vector>
#include "runtime/gguf_parser.h"

namespace q38rt {

/// An image decoded to 8-bit RGB.
struct Image {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;   // [height][width][3]
};

/// Decode PNG, JPEG, BMP, GIF (first frame), TGA, PSD, HDR or PNM bytes (stb_image). False with a message on failure.
bool decode_image(const void* data, size_t bytes, Image& out, std::string& err);
/// Resize like Pillow's Image.resize(..., BICUBIC), byte for byte (tests/unit/test_image_resize.py).
void resize_bicubic(const Image& in, int width, int height, Image& out);

/// An image prepared for the encoder: resized to multiples of 32 pixels (smart resize: sides rounded to multiples of 32,
/// area scaled into the token limits, aspect ratio approximately kept), normalized, cut into 16x16 patches in merge
/// order (the 4 patches of each 2x2 group consecutive, groups in raster order).
struct ImagePatches {
    int width = 0, height = 0;   // resized size in pixels
    int nx = 0, ny = 0;          // merged-token grid: width / 32, height / 32
    std::vector<float> patch;    // [4 * nx * ny][768]: value of channel c at (y, x) of the patch at c * 256 + y * 16 + x
    int n_tokens() const { return nx * ny; }
    int n_patches() const { return 4 * nx * ny; }
};

class VisionModel {
public:
    static constexpr int PATCH = 16, MERGE = 2, D = 1152, HEADS = 16, HEAD_DIM = 72, FF = 4304, LAYERS = 27,
                         POS_SIDE = 48, OUT = 5120;
    static constexpr int MERGED = D * MERGE * MERGE;   // 4608

    VisionModel() = default;
    ~VisionModel();
    VisionModel(const VisionModel&) = delete;
    VisionModel& operator=(const VisionModel&) = delete;

    /// Load the mmproj GGUF (architecture clip, projector qwen3vl_merger, projection 5120). Image size limits in merged
    /// tokens (32x32-pixel areas): images are scaled to hold between min_tokens and max_tokens.
    bool load(const std::string& path, int min_tokens = 8, int max_tokens = 1024);
    bool loaded() const { return h_layers_ != nullptr; }
    int max_tokens() const { return max_tokens_; }

    /// Merged-token grid (nx, ny) and resized pixel size of a width x height image (smart resize).
    void target_size(int width, int height, int& out_w, int& out_h) const;
    ImagePatches preprocess(const Image& img) const;

    // ---- weights, for the GPU forward (Engine::encode_image) ----
    /// Per layer, FP16 in pinned host memory, contiguous: qkv [3456][1152] | out [1152][1152] | up [4304][1152] |
    /// down [1152][4304] (row = output feature).
    static constexpr size_t QKV_ELEMS = (size_t)3 * D * D, OUT_ELEMS = (size_t)D * D, UP_ELEMS = (size_t)FF * D,
                            DOWN_ELEMS = (size_t)D * FF;
    static constexpr size_t LAYER_ELEMS = QKV_ELEMS + OUT_ELEMS + UP_ELEMS + DOWN_ELEMS;
    const uint16_t* layer_weights(int il) const { return h_layers_ + (size_t)il * LAYER_ELEMS; }
    /// Per layer, FP32: ln1 w, b | qkv bias (3456) | out bias | ln2 w, b | up bias (4304) | down bias.
    static constexpr size_t LAYER_SMALL = (size_t)D * 2 + 3 * D + D + D * 2 + FF + D;
    const std::vector<float>& small() const { return small_; }   // [LAYERS][LAYER_SMALL]
    const uint16_t* patch_weight() const { return h_patch_; }     // [1152][768] FP16 (both temporal kernels summed)
    const std::vector<float>& patch_bias() const { return patch_bias_; }
    const std::vector<float>& post_ln() const { return post_ln_; }   // w [1152] | b [1152]
    const uint16_t* mm0() const { return h_mm0_; }   // [4608][4608] FP16
    const uint16_t* mm2() const { return h_mm2_; }   // [5120][4608] FP16
    const std::vector<float>& mm_bias() const { return mm_bias_; }   // mm0 bias [4608] | mm2 bias [5120]
    /// Learned position embeddings resized to an nx x ny merged grid, in patch merge order: [4 * nx * ny][1152].
    std::vector<float> position_embeddings(int nx, int ny) const;
    float eps() const { return eps_; }

private:
    GGUFParser gguf_;
    int min_tokens_ = 8, max_tokens_ = 1024;
    float eps_ = 1e-6f;
    uint16_t* h_layers_ = nullptr;   // pinned
    uint16_t* h_patch_ = nullptr;
    uint16_t* h_mm0_ = nullptr;
    uint16_t* h_mm2_ = nullptr;
    std::vector<float> small_, patch_bias_, post_ln_, mm_bias_, pos_;   // pos_: [48 * 48][1152]
};

} // namespace q38rt
