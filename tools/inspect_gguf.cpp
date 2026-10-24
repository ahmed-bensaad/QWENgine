/// @file inspect_gguf.cpp
/// Metadata-only GGUF inspection using the runtime's own GGUFParser.
/// Usage: inspect_gguf <file.gguf> [--csv out.csv] [--quiet]
/// Prints header, every KV (index, key, type, scalar value / array summary) and
/// every tensor (index, name, ndim, dims, ggml type, absolute offset, bytes).
/// Tensor data is never read.

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include "../src/runtime/gguf_parser.h"

using namespace q38rt;

static std::string value_summary(const GGUFValue& v) {
    char buf[256];
    switch (v.type) {
        case GGUFType::UINT8: case GGUFType::UINT16: case GGUFType::UINT32: case GGUFType::UINT64:
            snprintf(buf, sizeof buf, "%llu", (unsigned long long)v.u64); return buf;
        case GGUFType::BOOL: return v.u64 ? "true" : "false";
        case GGUFType::INT8: case GGUFType::INT16: case GGUFType::INT32: case GGUFType::INT64:
            snprintf(buf, sizeof buf, "%lld", (long long)v.i64); return buf;
        case GGUFType::FLOAT32: case GGUFType::FLOAT64:
            snprintf(buf, sizeof buf, "%.9g", v.f64); return buf;
        case GGUFType::STRING: {
            std::string s = v.str.substr(0, 80);
            for (char& c : s) if (c == '\n') c = ' ';
            return "\"" + s + (v.str.size() > 80 ? "...\"" : "\"");
        }
        case GGUFType::ARRAY: {
            snprintf(buf, sizeof buf, "array<%s>[%llu]", gguf_type_name(v.arr_type), (unsigned long long)v.arr_count);
            std::string s = buf;
            if (v.arr_type == GGUFType::INT32 && v.arr_count <= 16) {
                s += " = {";
                for (uint64_t i = 0; i < v.arr_count; i++) {
                    int32_t x; memcpy(&x, v.arr_raw.data() + 4 * i, 4);
                    s += (i ? ", " : "") + std::to_string(x);
                }
                s += "}";
            } else if (v.arr_type == GGUFType::STRING && !v.arr_str.empty()) {
                s += " first=\"" + v.arr_str.front().substr(0, 20) + "\" last=\"" + v.arr_str.back().substr(0, 20) + "\"";
            }
            return s;
        }
        default: return "?";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.gguf> [--csv out.csv] [--quiet]\n", argv[0]);
        return 1;
    }
    const char* csv = nullptr;
    bool quiet = false;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!strcmp(argv[i], "--quiet")) quiet = true;
    }
    GGUFParser p;
    if (!p.parse(argv[1])) return 2;
    const GGUFHeader& h = p.get_header();
    printf("GGUF version: %u\ntensor count: %llu\nKV count: %llu\n", h.version,
           (unsigned long long)h.tensor_count, (unsigned long long)h.tensor_kv_count);
    printf("metadata end: 0x%llx  data offset: 0x%llx  alignment: %u  file size: %llu\n",
           (unsigned long long)p.metadata_end(), (unsigned long long)p.data_offset(), p.alignment(),
           (unsigned long long)p.file_size());
    if (!quiet) {
        printf("\n-- metadata --\n");
        int i = 0;
        for (const auto& kv : p.get_kv_list())
            printf("KV[%2d] @0x%08llx %-42s %-8s %s\n", i++, (unsigned long long)kv.file_offset, kv.key.c_str(),
                   gguf_type_name(kv.value.type), value_summary(kv.value).c_str());
        printf("\n-- tensors --\n");
    }
    std::map<std::string, std::pair<int, uint64_t>> by_type;
    uint64_t total = 0;
    int i = 0;
    FILE* fc = csv ? fopen(csv, "w") : nullptr;
    if (fc) fprintf(fc, "name,type,dims,abs_offset,nbytes,pad\n");
    for (const auto& t : p.get_tensors()) {
        std::string dims;
        for (uint32_t d = 0; d < t.shape.ndim; d++) dims += (d ? "x" : "") + std::to_string(t.shape.dims[d]);
        if (!quiet)
            printf("T[%3d] %-36s ndim=%u %-14s %-8s off=0x%010llx bytes=%llu\n", i, t.name.c_str(), t.shape.ndim,
                   dims.c_str(), quant_name(t.quant), (unsigned long long)t.offset, (unsigned long long)t.size_bytes);
        if (fc)
            fprintf(fc, "%s,%s,%s,%llu,%llu,%llu\n", t.name.c_str(), quant_name(t.quant), dims.c_str(),
                    (unsigned long long)t.offset, (unsigned long long)t.size_bytes, (unsigned long long)t.padding);
        auto& e = by_type[quant_name(t.quant)];
        e.first++;
        e.second += t.size_bytes;
        total += t.size_bytes;
        i++;
    }
    if (fc) fclose(fc);
    printf("\n-- totals --\n");
    for (const auto& kv : by_type)
        printf("%-8s %4d tensors %12llu bytes (%7.1f MiB)\n", kv.first.c_str(), kv.second.first,
               (unsigned long long)kv.second.second, kv.second.second / 1048576.0);
    printf("all      %4d tensors %12llu bytes (%7.3f GiB)\n", i, (unsigned long long)total, total / 1073741824.0);
    const ModelConfig& c = p.get_config();
    printf("\nconfig: layers=%d (full=%d, linear=%d) hidden=%d ffn=%d heads=%d kv_heads=%d head_dim=%d vocab=%d "
           "rope_dims=%d theta=%.0f eps=%g ssm(state=%d groups=%d v_heads=%d inner=%d conv=%d) bos=%d eos=%d\n",
           c.num_layers, c.num_full_attn, c.num_linear_layers, c.hidden_size, c.ffn_hidden, c.num_attention_heads,
           c.num_key_value_heads, c.head_dim, c.vocab_size, c.rope_dims, c.rope_theta, c.rms_eps,
           c.delta_net_state_dim, c.ssm_group_count, c.value_heads, c.ssm_inner_size, c.conv_window,
           c.bos_token_id, c.eos_token_id);
    return 0;
}
