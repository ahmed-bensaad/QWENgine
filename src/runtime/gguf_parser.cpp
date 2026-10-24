#include "gguf_parser.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace q38rt {

namespace {
// Sanity limits for values read from the file before they drive allocations.
constexpr uint64_t MAX_TENSORS       = 1000000;
constexpr uint64_t MAX_KV            = 100000;
constexpr uint64_t MAX_STRING_BYTES  = 1ull << 30;   // 1 GiB
constexpr uint64_t MAX_ARRAY_ELEMS   = 1ull << 32;
constexpr uint32_t MAX_DIMS          = 4;            // GGML_MAX_DIMS
constexpr int      MAX_ARRAY_NESTING = 4;

size_t scalar_size(GGUFType t) {
    switch (t) {
        case GGUFType::UINT8: case GGUFType::INT8: case GGUFType::BOOL: return 1;
        case GGUFType::UINT16: case GGUFType::INT16: return 2;
        case GGUFType::UINT32: case GGUFType::INT32: case GGUFType::FLOAT32: return 4;
        case GGUFType::UINT64: case GGUFType::INT64: case GGUFType::FLOAT64: return 8;
        default: return 0;  // STRING / ARRAY / invalid are not fixed-size
    }
}
} // namespace

const char* gguf_type_name(GGUFType t) {
    switch (t) {
        case GGUFType::UINT8: return "UINT8";
        case GGUFType::INT8: return "INT8";
        case GGUFType::UINT16: return "UINT16";
        case GGUFType::INT16: return "INT16";
        case GGUFType::UINT32: return "UINT32";
        case GGUFType::INT32: return "INT32";
        case GGUFType::FLOAT32: return "FLOAT32";
        case GGUFType::BOOL: return "BOOL";
        case GGUFType::STRING: return "STRING";
        case GGUFType::ARRAY: return "ARRAY";
        case GGUFType::UINT64: return "UINT64";
        case GGUFType::INT64: return "INT64";
        case GGUFType::FLOAT64: return "FLOAT64";
        default: return "INVALID";
    }
}

GGUFParser::GGUFParser() {}

GGUFParser::~GGUFParser() {
    if (base_) munmap(const_cast<uint8_t*>(base_), file_size_);
    if (fd_ >= 0) close(fd_);
}

// ---------------------------------------------------------------------------
// Bounds-checked readers
// ---------------------------------------------------------------------------

void GGUFParser::need(uint64_t n, const char* what) const {
    if (pos_ > file_size_ || n > file_size_ - pos_) {
        LOG_ERROR("Unexpected EOF reading %s (%llu bytes) at offset 0x%llx (file size 0x%llx)",
                  what, (unsigned long long)n, (unsigned long long)pos_, (unsigned long long)file_size_);
        throw std::runtime_error(std::string("Unexpected EOF reading ") + what);
    }
}

template <typename T>
T GGUFParser::read_pod(const char* what) {
    need(sizeof(T), what);
    T v;
    memcpy(&v, base_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
}

std::string GGUFParser::read_string() {
    const uint64_t at = pos_;
    const uint64_t length = read_uint64();
    if (length > MAX_STRING_BYTES) {
        LOG_ERROR("Invalid GGUF string length=%llu at offset=0x%llx",
                  (unsigned long long)length, (unsigned long long)at);
        throw std::runtime_error("Invalid GGUF string length");
    }
    need(length, "string bytes");
    std::string s(reinterpret_cast<const char*>(base_ + pos_), length);
    pos_ += length;
    return s;
}

bool GGUFParser::read_array(GGUFValue& out, int depth) {
    const uint64_t at = pos_;
    // GGUF ARRAY encoding: uint32 element type, then uint64 count, then elements.
    const uint32_t raw_type = read_uint32();
    const uint64_t count = read_uint64();
    out.arr_type = static_cast<GGUFType>(raw_type);
    out.arr_count = count;
    if (count > MAX_ARRAY_ELEMS) {
        LOG_ERROR("Array count %llu too large at offset 0x%llx", (unsigned long long)count, (unsigned long long)at);
        return false;
    }
    if (out.arr_type == GGUFType::STRING) {
        out.arr_str.reserve(count);
        for (uint64_t i = 0; i < count; i++) out.arr_str.push_back(read_string());
        return true;
    }
    if (out.arr_type == GGUFType::ARRAY) {
        // Nested arrays are legal in the spec; validate and skip them (unused here).
        if (depth >= MAX_ARRAY_NESTING) {
            LOG_ERROR("Array nesting too deep at offset 0x%llx", (unsigned long long)at);
            return false;
        }
        for (uint64_t i = 0; i < count; i++) {
            GGUFValue inner;
            if (!read_array(inner, depth + 1)) return false;
        }
        return true;
    }
    const size_t es = scalar_size(out.arr_type);
    if (es == 0) {
        LOG_ERROR("Unsupported GGUF array element type %u at offset 0x%llx", raw_type, (unsigned long long)at);
        return false;
    }
    const uint64_t bytes = count * es;  // count <= 2^32, es <= 8: no overflow
    need(bytes, "array elements");
    out.arr_raw.assign(base_ + pos_, base_ + pos_ + bytes);
    pos_ += bytes;
    return true;
}

bool GGUFParser::read_value(GGUFType type, GGUFValue& out, int depth) {
    out.type = type;
    switch (type) {
        case GGUFType::UINT8:   out.u64 = read_uint8(); return true;
        case GGUFType::INT8:    out.i64 = read_int8(); return true;
        case GGUFType::UINT16:  out.u64 = read_uint16(); return true;
        case GGUFType::INT16:   out.i64 = read_int16(); return true;
        case GGUFType::UINT32:  out.u64 = read_uint32(); return true;
        case GGUFType::INT32:   out.i64 = read_int32(); return true;   // exactly 4 bytes
        case GGUFType::FLOAT32: out.f64 = read_float(); return true;
        case GGUFType::BOOL:    out.u64 = read_uint8(); return true;   // 1 byte
        case GGUFType::STRING:  out.str = read_string(); return true;
        case GGUFType::ARRAY:   return read_array(out, depth);
        case GGUFType::UINT64:  out.u64 = read_uint64(); return true;
        case GGUFType::INT64:   out.i64 = read_int64(); return true;
        case GGUFType::FLOAT64: out.f64 = read_double(); return true;
        default:
            LOG_ERROR("Unsupported GGUF metadata type %u at offset 0x%llx",
                      static_cast<unsigned>(type), (unsigned long long)pos_);
            return false;
    }
}

// ---------------------------------------------------------------------------
// Parse
// ---------------------------------------------------------------------------

bool GGUFParser::parse(const std::string& path, bool verbose, const std::string& arch) {
    verbose_ = verbose;
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        LOG_ERROR("Cannot open GGUF file: %s", path.c_str());
        return false;
    }
    struct stat st;
    if (fstat(fd_, &st) != 0) {
        LOG_ERROR("fstat failed: %s", path.c_str());
        return false;
    }
    file_size_ = static_cast<uint64_t>(st.st_size);
    void* m = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (m == MAP_FAILED) {
        LOG_ERROR("mmap failed: %s", path.c_str());
        return false;
    }
    base_ = static_cast<const uint8_t*>(m);
    pos_ = 0;

    try {
        if (!parse_header()) return false;
        if (!parse_kv()) return false;
        if (!parse_tensors()) return false;
        if (!finalize_tensors()) return false;
        if (arch == "qwen35") {
            if (!build_config()) return false;
        } else if (get_kv_string("general.architecture") != arch) {
            LOG_ERROR("%s: architecture '%s', expected '%s'", path.c_str(), get_kv_string("general.architecture").c_str(),
                      arch.c_str());
            return false;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("GGUF parse failed at offset 0x%llx: %s", (unsigned long long)pos_, e.what());
        return false;
    }
    LOG_INFO("GGUF v%u: %llu KV, %llu tensors, data at 0x%llx (align %u), file %llu bytes",
             header_.version, (unsigned long long)header_.tensor_kv_count,
             (unsigned long long)header_.tensor_count, (unsigned long long)data_offset_,
             alignment_, (unsigned long long)file_size_);
    return true;
}

bool GGUFParser::parse_header() {
    header_.magic = read_uint32();
    if (header_.magic != Q38_GGUF_MAGIC) {
        LOG_ERROR("Invalid GGUF magic 0x%08x (expected 0x%08x)", header_.magic, Q38_GGUF_MAGIC);
        return false;
    }
    header_.version = read_uint32();          // uint32, not uint64
    if (header_.version < 2 || header_.version > 3) {
        LOG_ERROR("Unsupported GGUF version %u", header_.version);
        return false;
    }
    header_.tensor_count = read_uint64();
    header_.tensor_kv_count = read_uint64();
    if (header_.tensor_count > MAX_TENSORS) {
        LOG_ERROR("Invalid tensor count: %llu", (unsigned long long)header_.tensor_count);
        return false;
    }
    if (header_.tensor_kv_count > MAX_KV) {
        LOG_ERROR("Invalid metadata count: %llu", (unsigned long long)header_.tensor_kv_count);
        return false;
    }
    return true;
}

bool GGUFParser::parse_kv() {
    kv_list_.reserve(header_.tensor_kv_count);
    for (uint64_t i = 0; i < header_.tensor_kv_count; i++) {
        GGUFKVPair kv;
        kv.file_offset = pos_;
        kv.key = read_string();
        const uint32_t raw_type = read_uint32();
        if (!read_value(static_cast<GGUFType>(raw_type), kv.value, 0)) {
            LOG_ERROR("Failed to read value of KV %llu '%s' (type %u)",
                      (unsigned long long)i, kv.key.c_str(), raw_type);
            return false;
        }
        if (verbose_) {
            LOG_DEBUG("KV[%llu] @0x%llx key='%s' type=%s", (unsigned long long)i,
                      (unsigned long long)kv.file_offset, kv.key.c_str(), gguf_type_name(kv.value.type));
        }
        if (kv_index_.count(kv.key)) {
            LOG_ERROR("Duplicate GGUF key '%s'", kv.key.c_str());
            return false;
        }
        kv_index_[kv.key] = kv_list_.size();
        kv_list_.push_back(std::move(kv));
    }
    if (const GGUFValue* a = find_kv("general.alignment")) {
        if (a->type != GGUFType::UINT32 || a->u64 == 0 || (a->u64 & (a->u64 - 1)) != 0) {
            LOG_ERROR("Invalid general.alignment");
            return false;
        }
        alignment_ = static_cast<uint32_t>(a->u64);
    }
    return true;
}

bool GGUFParser::parse_tensors() {
    tensors_.reserve(header_.tensor_count);
    for (uint64_t i = 0; i < header_.tensor_count; i++) {
        TensorInfo t;
        const uint64_t at = pos_;
        t.name = read_string();
        const uint32_t ndim = read_uint32();      // uint32, not uint64
        if (ndim == 0 || ndim > MAX_DIMS) {
            LOG_ERROR("Invalid tensor ndim %u for '%s' at offset 0x%llx", ndim, t.name.c_str(), (unsigned long long)at);
            return false;
        }
        t.shape.ndim = ndim;
        for (uint32_t j = 0; j < ndim; j++) {
            t.shape.dims[j] = read_uint64();
            if (t.shape.dims[j] == 0 || t.shape.dims[j] > (1ull << 40)) {
                LOG_ERROR("Invalid dim %u=%llu for '%s'", j, (unsigned long long)t.shape.dims[j], t.name.c_str());
                return false;
            }
        }
        const uint32_t raw_type = read_uint32();
        t.quant = static_cast<QuantFormat>(raw_type);   // exact ggml_type id, never remapped
        t.rel_offset = read_uint64();
        const QuantBlockInfo bi = quant_block_info(t.quant);
        if (bi.block_elems == 0) {
            LOG_ERROR("Unknown ggml type %u for tensor '%s'", raw_type, t.name.c_str());
            return false;
        }
        if (t.shape.dims[0] % bi.block_elems != 0) {
            LOG_ERROR("Tensor '%s' row length %llu not a multiple of %s block size %u", t.name.c_str(),
                      (unsigned long long)t.shape.dims[0], bi.name, bi.block_elems);
            return false;
        }
        t.size_bytes = t.shape.size() / bi.block_elems * bi.block_bytes;

        // block index of "blk.N.<role>" tensors (model-level tensors keep -1)
        if (t.name.compare(0, 4, "blk.") == 0) {
            const size_t dot = t.name.find('.', 4);
            if (dot == std::string::npos) {
                LOG_ERROR("Malformed tensor name '%s'", t.name.c_str());
                return false;
            }
            t.layer_index = std::stoi(t.name.substr(4, dot - 4));
        }

        if (verbose_) {
            LOG_DEBUG("tensor[%llu] @0x%llx '%s' ndim=%u %s %s rel_off=0x%llx", (unsigned long long)i,
                      (unsigned long long)at, t.name.c_str(), ndim, t.shape.name().c_str(), bi.name,
                      (unsigned long long)t.rel_offset);
        }
        if (tensor_index_.count(t.name)) {
            LOG_ERROR("Duplicate tensor name '%s'", t.name.c_str());
            return false;
        }
        tensor_index_[t.name] = tensors_.size();
        tensors_.push_back(std::move(t));
    }
    meta_end_ = pos_;
    data_offset_ = (meta_end_ + alignment_ - 1) / alignment_ * alignment_;
    return true;
}

bool GGUFParser::finalize_tensors() {
    if (data_offset_ > file_size_) {
        LOG_ERROR("Data section offset 0x%llx beyond file size", (unsigned long long)data_offset_);
        return false;
    }
    const uint64_t data_bytes = file_size_ - data_offset_;
    // GGUF does not require the info table to be in offset order: validate on a sorted view.
    std::vector<size_t> order(tensors_.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return tensors_[a].rel_offset < tensors_[b].rel_offset; });
    for (size_t k = 0; k < order.size(); k++) {
        TensorInfo& t = tensors_[order[k]];
        if (t.rel_offset % alignment_ != 0) {
            LOG_ERROR("Tensor '%s' offset 0x%llx not %u-aligned", t.name.c_str(),
                      (unsigned long long)t.rel_offset, alignment_);
            return false;
        }
        const uint64_t end = t.rel_offset + t.size_bytes;
        const uint64_t next = (k + 1 < order.size()) ? tensors_[order[k + 1]].rel_offset : data_bytes;
        if (end > next) {
            LOG_ERROR("Tensor '%s' [0x%llx, 0x%llx) overlaps next tensor/EOF at 0x%llx", t.name.c_str(),
                      (unsigned long long)t.rel_offset, (unsigned long long)end, (unsigned long long)next);
            return false;
        }
        t.padding = next - end;
        t.offset = data_offset_ + t.rel_offset;
    }
    return true;
}

bool GGUFParser::build_config() {
    const std::string arch = get_kv_string("general.architecture");
    if (arch != "qwen35") {
        LOG_ERROR("Unsupported architecture '%s' (this runtime is specialized for qwen35)", arch.c_str());
        return false;
    }
    ModelConfig& c = config_;
    auto u = [&](const char* k, int32_t& dst) {
        const GGUFValue* v = find_kv(k);
        if (!v) { LOG_ERROR("Missing required key %s", k); throw std::runtime_error(k); }
        dst = static_cast<int32_t>(get_kv_uint(k));
    };
    u("qwen35.block_count", c.num_hidden_layers);
    // MTP variants append next-token-prediction blocks after the trunk (blk.64 here)
    c.num_nextn_layers = find_kv("qwen35.nextn_predict_layers") ? (int32_t)get_kv_uint("qwen35.nextn_predict_layers") : 0;
    c.num_layers = c.num_hidden_layers - c.num_nextn_layers;
    u("qwen35.context_length", c.train_context);
    u("qwen35.embedding_length", c.hidden_size);
    u("qwen35.feed_forward_length", c.ffn_hidden);
    u("qwen35.attention.head_count", c.num_attention_heads);
    u("qwen35.attention.head_count_kv", c.num_key_value_heads);
    u("qwen35.attention.key_length", c.head_dim);
    u("qwen35.ssm.conv_kernel", c.conv_window);
    u("qwen35.ssm.state_size", c.delta_net_state_dim);
    u("qwen35.ssm.group_count", c.ssm_group_count);
    u("qwen35.ssm.time_step_rank", c.value_heads);
    u("qwen35.ssm.inner_size", c.ssm_inner_size);
    u("qwen35.full_attention_interval", c.full_attn_interval);
    u("qwen35.rope.dimension_count", c.rope_dims);
    c.rms_eps = get_kv_float("qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    c.rope_theta = get_kv_float("qwen35.rope.freq_base", 1e7f);
    c.rope_partial_ratio = (float)c.rope_dims / (float)c.head_dim;
    if (const GGUFValue* s = find_kv("qwen35.rope.dimension_sections")) {
        if (s->type == GGUFType::ARRAY && s->arr_type == GGUFType::INT32 && s->arr_count == 4) {
            c.rope_mrope_sections.resize(4);
            memcpy(c.rope_mrope_sections.data(), s->arr_raw.data(), 16);
        }
    }
    int32_t value_len = 0;
    u("qwen35.attention.value_length", value_len);
    c.num_full_attn = 0;
    for (int i = 0; i < c.num_layers; i++) c.num_full_attn += c.is_full_attention(i) ? 1 : 0;
    c.num_linear_layers = c.num_layers - c.num_full_attn;

    if (const TensorInfo* emb = find_tensor("token_embd.weight")) c.vocab_size = (int32_t)emb->shape.dims[1];
    c.bos_token_id = (int32_t)get_kv_uint("tokenizer.ggml.bos_token_id", c.bos_token_id);
    c.eos_token_id = (int32_t)get_kv_uint("tokenizer.ggml.eos_token_id", c.eos_token_id);
    c.pad_token_id = (int32_t)get_kv_uint("tokenizer.ggml.padding_token_id", c.pad_token_id);
    c.add_bos_token = get_kv_bool("tokenizer.ggml.add_bos_token", false);

    // The kernels are specialized: refuse files that do not match the compiled shapes.
    const bool ok = c.num_layers == NUM_LAYERS && c.hidden_size == NUM_HIDDEN && c.ffn_hidden == FFN_HIDDEN &&
                    c.num_attention_heads == NUM_ATTENTION_HEADS && c.num_key_value_heads == NUM_KEY_VALUE_HEADS &&
                    c.head_dim == HEAD_DIM && value_len == HEAD_DIM && c.delta_net_state_dim == DELTA_NET_STATE_DIM &&
                    c.value_heads == VALUE_HEADS && c.ssm_group_count == 16 && c.ssm_inner_size == 6144 &&
                    c.conv_window == CONV_WINDOW && c.full_attn_interval == FULL_ATTN_INTERVAL &&
                    c.rope_dims == 64 && c.vocab_size == VOCAB_SIZE && c.num_full_attn == NUM_FULL_ATTN &&
                    c.rope_mrope_sections == std::vector<int32_t>{11, 11, 10, 0} &&   // cuda::rope_table_pos
                    (c.num_nextn_layers == 0 || c.num_nextn_layers == 1);
    if (!ok) {
        LOG_ERROR("GGUF hyperparameters differ from the specialized Qwen3.8-27B shapes");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

const TensorInfo* GGUFParser::find_tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const GGUFValue* GGUFParser::find_kv(const std::string& key) const {
    auto it = kv_index_.find(key);
    return it == kv_index_.end() ? nullptr : &kv_list_[it->second].value;
}

std::string GGUFParser::get_kv_string(const std::string& key, const std::string& def) const {
    const GGUFValue* v = find_kv(key);
    return (v && v->type == GGUFType::STRING) ? v->str : def;
}

uint64_t GGUFParser::get_kv_uint(const std::string& key, uint64_t def) const {
    const GGUFValue* v = find_kv(key);
    if (!v) return def;
    switch (v->type) {
        case GGUFType::UINT8: case GGUFType::UINT16: case GGUFType::UINT32:
        case GGUFType::UINT64: case GGUFType::BOOL: return v->u64;
        case GGUFType::INT8: case GGUFType::INT16: case GGUFType::INT32:
        case GGUFType::INT64: return v->i64 < 0 ? def : (uint64_t)v->i64;
        default: return def;
    }
}

int64_t GGUFParser::get_kv_int(const std::string& key, int64_t def) const {
    const GGUFValue* v = find_kv(key);
    if (!v) return def;
    switch (v->type) {
        case GGUFType::INT8: case GGUFType::INT16: case GGUFType::INT32: case GGUFType::INT64: return v->i64;
        case GGUFType::UINT8: case GGUFType::UINT16: case GGUFType::UINT32:
        case GGUFType::UINT64: case GGUFType::BOOL: return (int64_t)v->u64;
        default: return def;
    }
}

float GGUFParser::get_kv_float(const std::string& key, float def) const {
    const GGUFValue* v = find_kv(key);
    if (!v) return def;
    if (v->type == GGUFType::FLOAT32 || v->type == GGUFType::FLOAT64) return (float)v->f64;
    return def;
}

bool GGUFParser::get_kv_bool(const std::string& key, bool def) const {
    const GGUFValue* v = find_kv(key);
    return (v && v->type == GGUFType::BOOL) ? v->u64 != 0 : def;
}

std::vector<uint8_t> GGUFParser::get_tensor_data(const std::string& name, size_t offset, size_t length) const {
    const TensorInfo* t = find_tensor(name);
    if (!t) {
        LOG_ERROR("Tensor not found: %s", name.c_str());
        return {};
    }
    if (offset > t->size_bytes) return {};
    const size_t n = length ? std::min<size_t>(length, t->size_bytes - offset) : t->size_bytes - offset;
    const uint8_t* p = tensor_data(*t) + offset;
    return std::vector<uint8_t>(p, p + n);
}

} // namespace q38rt
