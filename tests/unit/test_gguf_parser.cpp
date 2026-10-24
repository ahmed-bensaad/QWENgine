/// @file test_gguf_parser.cpp
/// Regression test for the GGUF parser against facts verified independently
/// (tools/gguf_inventory.py, a separate Python implementation of the spec).

#include <cstdio>
#include <cstring>
#include <string>
#include "runtime/gguf_parser.h"

using namespace q38rt;

static int g_fail = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 2; }
    GGUFParser p;
    if (!p.parse(argv[1])) { fprintf(stderr, "FAIL: parse\n"); return 1; }

    const GGUFHeader& h = p.get_header();
    EXPECT(h.version == 3, "version %u", h.version);
    EXPECT(h.tensor_count == 851, "tensor_count %llu", (unsigned long long)h.tensor_count);
    EXPECT(h.tensor_kv_count == 52, "kv count %llu", (unsigned long long)h.tensor_kv_count);
    EXPECT(p.get_kv_list().size() == 52, "parsed kv %zu", p.get_kv_list().size());
    EXPECT(p.get_tensors().size() == 851, "parsed tensors %zu", p.get_tensors().size());
    EXPECT(p.metadata_end() == 0xa7c47d, "metadata end 0x%llx", (unsigned long long)p.metadata_end());
    EXPECT(p.data_offset() == 0xa7c480, "data offset 0x%llx", (unsigned long long)p.data_offset());
    EXPECT(p.file_size() == 10094357632ull, "file size");

    // Metadata types that the inherited parser got wrong.
    const GGUFValue* topk = p.find_kv("general.sampling.top_k");
    EXPECT(topk && topk->type == GGUFType::INT64 && topk->i64 == 20, "general.sampling.top_k INT64 20");
    const GGUFValue* sec = p.find_kv("qwen35.rope.dimension_sections");
    EXPECT(sec && sec->type == GGUFType::ARRAY && sec->arr_type == GGUFType::INT32 && sec->arr_count == 4, "sections");
    const GGUFValue* toks = p.find_kv("tokenizer.ggml.tokens");
    EXPECT(toks && toks->arr_type == GGUFType::STRING && toks->arr_str.size() == 248320, "tokens");
    EXPECT(toks && toks->arr_str[0] == "!" && toks->arr_str[248319] == "[PAD248319]", "token strings");
    const GGUFValue* merges = p.find_kv("tokenizer.ggml.merges");
    EXPECT(merges && merges->arr_str.size() == 247587, "merges");
    EXPECT(p.get_kv_uint("qwen35.block_count") == 64, "block_count");
    EXPECT(p.get_kv_uint("tokenizer.ggml.eos_token_id") == 248046, "eos");
    EXPECT(p.get_kv_bool("tokenizer.ggml.add_bos_token", true) == false, "add_bos");
    EXPECT(p.get_kv_string("general.architecture") == "qwen35", "arch");

    // Exact sizes / types / absolute offsets of representative tensors (from the Python inventory).
    struct Ref { const char* name; QuantFormat q; uint64_t d0, d1, off, bytes; };
    const Ref refs[] = {
        {"output.weight", QuantFormat::IQ4_XS, 5120, 248320, 10994816, 675430400},
        {"output_norm.weight", QuantFormat::FP32, 5120, 1, 686425216, 20480},
        {"token_embd.weight", QuantFormat::IQ2_S, 5120, 248320, 686445696, 407244800},
        {"blk.0.attn_qkv.weight", QuantFormat::IQ4_XS, 5120, 10240, 1107227776, 27852800},
        {"blk.0.ffn_gate.weight", QuantFormat::IQ1_S, 5120, 17408, 1160844416, 17408000},
        {"blk.0.ffn_up.weight", QuantFormat::IQ1_M, 5120, 17408, 1178252416, 19496960},
        {"blk.0.ssm_alpha.weight", QuantFormat::BF16, 5120, 48, 1197770048, 491520},
        {"blk.0.ssm_conv1d.weight", QuantFormat::FP32, 4, 10240, 1198753088, 163840},
        {"blk.3.attn_q.weight", QuantFormat::Q2_K, 5120, 12288, 1460798208, 20643840},
        {"blk.3.attn_v.weight", QuantFormat::Q4_K, 5120, 1024, 1481443072, 2949120},
        {"blk.4.ffn_down.weight", QuantFormat::IQ3_XXS, 17408, 5120, 1594042112, 34119680},
        {"blk.7.attn_v.weight", QuantFormat::IQ3_S, 5120, 1024, 1997052288, 2252800},
        {"blk.1.ffn_up.weight", QuantFormat::IQ2_XXS, 5120, 17408, 1281513472, 22978560},
        {"blk.0.ffn_down.weight", QuantFormat::IQ2_XS, 17408, 5120, 1135080576, 25763840},
    };
    for (const Ref& r : refs) {
        const TensorInfo* t = p.find_tensor(r.name);
        EXPECT(t != nullptr, "missing %s", r.name);
        if (!t) continue;
        EXPECT(t->quant == r.q, "%s type %s", r.name, quant_name(t->quant));
        EXPECT(t->shape.dims[0] == r.d0 && t->shape.dims[1] == r.d1, "%s dims %s", r.name, t->shape.name().c_str());
        EXPECT(t->offset == r.off, "%s offset %llu", r.name, (unsigned long long)t->offset);
        EXPECT(t->size_bytes == r.bytes, "%s bytes %llu", r.name, (unsigned long long)t->size_bytes);
    }
    // Last tensor ends exactly at EOF (inherited parser gave it size 0).
    const TensorInfo* last = p.find_tensor("blk.63.post_attention_norm.weight");
    EXPECT(last && last->offset + last->size_bytes == p.file_size(), "last tensor ends at EOF");

    uint64_t total = 0;
    int n_layer_tensors = 0;
    for (const TensorInfo& t : p.get_tensors()) {
        total += t.size_bytes;
        if (t.layer_index >= 0) n_layer_tensors++;
    }
    EXPECT(total == 10083362816ull, "total tensor bytes %llu", (unsigned long long)total);
    EXPECT(n_layer_tensors == 848, "layer tensors %d", n_layer_tensors);

    const ModelConfig& c = p.get_config();
    EXPECT(c.num_full_attn == 16 && c.num_linear_layers == 48, "layer split");
    EXPECT(c.num_attention_heads == 24 && c.num_key_value_heads == 4 && c.head_dim == 256, "attn shape");
    EXPECT(c.is_full_attention(3) && !c.is_full_attention(4) && c.is_full_attention(63), "pattern");

    if (g_fail) { fprintf(stderr, "%d failures\n", g_fail); return 1; }
    printf("test_gguf_parser: all checks passed\n");
    return 0;
}
