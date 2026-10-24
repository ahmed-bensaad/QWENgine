#pragma once
/// @file q38rt_types.h
/// Model constants, the exact ggml storage types, tensor metadata and the model configuration read from the GGUF.

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>

namespace q38rt {

// ============================================================================
// Model Architecture Constants (Qwen3.8-27B)
// ============================================================================

static constexpr int32_t NUM_HIDDEN         = 5120;   // qwen35.embedding_length
static constexpr int32_t NUM_LAYERS         = 64;     // qwen35.block_count
static constexpr int32_t NUM_LINEAR_LAYERS  = 48;     // Gated DeltaNet layers ((il+1) % 4 != 0)
static constexpr int32_t NUM_FULL_ATTN      = 16;     // Full attention layers (il = 3, 7, ..., 63)
static constexpr int32_t FULL_ATTN_INTERVAL = 4;      // qwen35.full_attention_interval
static constexpr int32_t NUM_ATTENTION_HEADS    = 24; // qwen35.attention.head_count
static constexpr int32_t NUM_KEY_VALUE_HEADS    = 4;  // qwen35.attention.head_count_kv (GQA group 6)
static constexpr int32_t HEAD_DIM             = 256;  // qwen35.attention.key_length / value_length
static constexpr int32_t VALUE_HEADS          = 48;   // qwen35.ssm.time_step_rank (DeltaNet V heads)
static constexpr int32_t FFN_HIDDEN          = 17408; // qwen35.feed_forward_length
static constexpr int32_t VOCAB_SIZE           = 248320;

// DeltaNet recurrence state
static constexpr int32_t DELTA_NET_STATE_DIM  = 128;  // qwen35.ssm.state_size (head k/v dim)
static constexpr int32_t CONV_WINDOW          = 4;    // qwen35.ssm.conv_kernel

/// Tensor storage type. Values are the exact ggml_type ids (ggml/include/ggml.h)
/// so the parser never maps one binary layout onto another.
enum class QuantFormat : uint32_t {
    FP32    = 0,
    FP16    = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    INVALID = 0xFFFFFFFFu,
};

/// Block layout of a ggml type: elements per block and bytes per block.
struct QuantBlockInfo {
    const char* name;
    uint32_t block_elems;  // 0 = unsupported/unknown type id
    uint32_t block_bytes;
};

inline QuantBlockInfo quant_block_info(QuantFormat t) {
    switch (t) {
        case QuantFormat::FP32:    return {"F32", 1, 4};
        case QuantFormat::FP16:    return {"F16", 1, 2};
        case QuantFormat::Q4_0:    return {"Q4_0", 32, 18};
        case QuantFormat::Q4_1:    return {"Q4_1", 32, 20};
        case QuantFormat::Q5_0:    return {"Q5_0", 32, 22};
        case QuantFormat::Q5_1:    return {"Q5_1", 32, 24};
        case QuantFormat::Q8_0:    return {"Q8_0", 32, 34};
        case QuantFormat::Q8_1:    return {"Q8_1", 32, 36};
        case QuantFormat::Q2_K:    return {"Q2_K", 256, 84};
        case QuantFormat::Q3_K:    return {"Q3_K", 256, 110};
        case QuantFormat::Q4_K:    return {"Q4_K", 256, 144};
        case QuantFormat::Q5_K:    return {"Q5_K", 256, 176};
        case QuantFormat::Q6_K:    return {"Q6_K", 256, 210};
        case QuantFormat::Q8_K:    return {"Q8_K", 256, 292};
        case QuantFormat::IQ2_XXS: return {"IQ2_XXS", 256, 66};
        case QuantFormat::IQ2_XS:  return {"IQ2_XS", 256, 74};
        case QuantFormat::IQ3_XXS: return {"IQ3_XXS", 256, 98};
        case QuantFormat::IQ1_S:   return {"IQ1_S", 256, 50};
        case QuantFormat::IQ4_NL:  return {"IQ4_NL", 32, 18};
        case QuantFormat::IQ3_S:   return {"IQ3_S", 256, 110};
        case QuantFormat::IQ2_S:   return {"IQ2_S", 256, 82};
        case QuantFormat::IQ4_XS:  return {"IQ4_XS", 256, 136};
        case QuantFormat::I8:      return {"I8", 1, 1};
        case QuantFormat::I16:     return {"I16", 1, 2};
        case QuantFormat::I32:     return {"I32", 1, 4};
        case QuantFormat::I64:     return {"I64", 1, 8};
        case QuantFormat::F64:     return {"F64", 1, 8};
        case QuantFormat::IQ1_M:   return {"IQ1_M", 256, 56};
        case QuantFormat::BF16:    return {"BF16", 1, 2};
        default:                   return {"INVALID", 0, 0};
    }
}

inline const char* quant_name(QuantFormat t) { return quant_block_info(t).name; }

/// Bytes of one row of `n` elements (n must be a multiple of the block size).
inline size_t quant_row_bytes(QuantFormat t, int64_t n) {
    QuantBlockInfo b = quant_block_info(t);
    return b.block_elems ? (size_t)(n / b.block_elems) * b.block_bytes : 0;
}

// ============================================================================
// Tensor Metadata
// ============================================================================

struct TensorShape {
    uint32_t ndim = 0;
    uint64_t dims[4] = {1, 1, 1, 1};   // ggml order: dims[0] = ne0 = contiguous row length

    uint64_t size() const {
        uint64_t s = 1;
        for (uint32_t i = 0; i < ndim; i++) s *= dims[i];
        return s;
    }

    std::string name() const {
        std::string s = "[";
        for (uint32_t i = 0; i < ndim && i < 4; i++) {
            if (i) s += ", ";
            s += std::to_string(dims[i]);
        }
        s += "]";
        return s;
    }
};

struct TensorInfo {
    std::string name;
    QuantFormat quant = QuantFormat::FP32;  // exact ggml_type
    TensorShape shape;
    uint64_t rel_offset = 0;  // offset relative to the tensor-data section (as stored in GGUF)
    uint64_t offset = 0;      // absolute byte offset in the GGUF file
    uint64_t size_bytes = 0;  // exact storage size from the ggml block layout
    uint64_t padding = 0;     // alignment padding after this tensor (to next tensor / EOF)
    int32_t layer_index = -1; // blk.N index (-1 for model-level tensors)
};

// ============================================================================
// Model Config
// ============================================================================

struct ModelConfig {
    // Values below are defaults; GGUFParser::build_config() overwrites them from
    // the GGUF metadata (qwen35.*) and fails if the file disagrees with what the
    // specialized kernels support.
    int32_t num_hidden_layers    = NUM_LAYERS;
    int32_t hidden_size          = NUM_HIDDEN;
    int32_t num_attention_heads  = NUM_ATTENTION_HEADS;
    int32_t num_key_value_heads  = NUM_KEY_VALUE_HEADS;
    int32_t head_dim             = HEAD_DIM;
    int32_t ffn_hidden           = FFN_HIDDEN;
    int32_t vocab_size           = VOCAB_SIZE;
    int32_t num_layers           = NUM_LAYERS;   // trunk layers (block_count - nextn_predict_layers)
    int32_t num_nextn_layers     = 0;            // MTP blocks after the trunk (blk.<num_layers>)
    int32_t num_linear_layers    = NUM_LINEAR_LAYERS;
    int32_t num_full_attn        = NUM_FULL_ATTN;
    int32_t full_attn_interval   = FULL_ATTN_INTERVAL;
    int32_t train_context        = 262144;        // qwen35.context_length
    float   rms_eps              = 1e-6f;         // qwen35.attention.layer_norm_rms_epsilon

    // RoPE (IMROPE; for text positions it reduces to NeoX RoPE on the first rope_dims dims)
    float rope_theta           = 10000000.0f;   // qwen35.rope.freq_base
    int32_t rope_dims          = 64;            // qwen35.rope.dimension_count
    float rope_partial_ratio   = 0.25f;         // rope_dims / head_dim
    std::vector<int32_t> rope_mrope_sections = {11, 11, 10, 0}; // qwen35.rope.dimension_sections

    // Gated DeltaNet
    int32_t delta_net_state_dim = DELTA_NET_STATE_DIM; // qwen35.ssm.state_size
    int32_t ssm_group_count     = 16;                  // qwen35.ssm.group_count (Q/K heads)
    int32_t value_heads         = VALUE_HEADS;         // qwen35.ssm.time_step_rank
    int32_t ssm_inner_size      = 6144;                // qwen35.ssm.inner_size
    int32_t conv_window         = CONV_WINDOW;         // qwen35.ssm.conv_kernel

    // Tokenizer
    int32_t bos_token_id = 248044;
    int32_t eos_token_id = 248046;
    int32_t pad_token_id = 248044;
    bool add_bos_token   = false;

    bool is_full_attention(int layer) const { return (layer + 1) % full_attn_interval == 0; }
};

} // namespace q38rt
