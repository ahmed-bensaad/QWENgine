#pragma once
/// @file gguf_parser.h
/// GGUF v3 parser for Qwen3.8-27B (spec: ggml/docs/gguf.md).
///
/// The file is memory-mapped; metadata is parsed with a bounds-checked cursor so
/// every field consumes exactly the number of bytes the format defines and any
/// out-of-range read is an error with a file offset (never an allocation from
/// an unchecked length). Tensor offsets in GGUF are relative to the start of the
/// tensor-data section, which begins at the first `general.alignment` (default
/// 32) boundary after the tensor-info table.

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include "../common/utils.h"
#include "../../include/q38rt/q38rt_types.h"

namespace q38rt {

static constexpr uint32_t Q38_GGUF_MAGIC = 0x46554747; // "GGUF" little-endian

/// GGUF metadata value types — exact values from the GGUF spec.
enum class GGUFType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

const char* gguf_type_name(GGUFType t);

struct GGUFHeader {
    uint32_t magic = 0;
    uint32_t version = 0;          // uint32 in GGUF v2/v3
    uint64_t tensor_count = 0;
    uint64_t tensor_kv_count = 0;
};

/// One metadata value. Scalars are widened into the matching 64-bit field;
/// arrays keep their raw little-endian element bytes (fixed-size element types)
/// or a vector of strings (STRING arrays).
struct GGUFValue {
    GGUFType type = GGUFType::UINT8;
    uint64_t u64 = 0;              // unsigned integer types and BOOL
    int64_t  i64 = 0;              // signed integer types
    double   f64 = 0.0;            // FLOAT32 / FLOAT64
    std::string str;               // STRING
    GGUFType arr_type = GGUFType::UINT8;
    uint64_t arr_count = 0;
    std::vector<uint8_t> arr_raw;  // fixed-size element arrays
    std::vector<std::string> arr_str; // STRING arrays
};

struct GGUFKVPair {
    std::string key;
    uint64_t file_offset = 0;      // where the key's length field starts
    GGUFValue value;
};

class GGUFParser {
public:
    GGUFParser();
    ~GGUFParser();
    GGUFParser(const GGUFParser&) = delete;
    GGUFParser& operator=(const GGUFParser&) = delete;

    /// Map and parse the file. Tensor data is not touched (the mapping is lazy). `arch`: the expected
    /// general.architecture; the model configuration (get_config) is built for "qwen35" only (the vision projector
    /// file is "clip").
    bool parse(const std::string& path, bool verbose = false, const std::string& arch = "qwen35");

    const GGUFHeader& get_header() const { return header_; }
    const ModelConfig& get_config() const { return config_; }
    const std::vector<TensorInfo>& get_tensors() const { return tensors_; }
    const std::vector<GGUFKVPair>& get_kv_list() const { return kv_list_; }

    const TensorInfo* find_tensor(const std::string& name) const;
    const GGUFValue* find_kv(const std::string& key) const;

    std::string get_kv_string(const std::string& key, const std::string& def = "") const;
    uint64_t get_kv_uint(const std::string& key, uint64_t def = 0) const;
    int64_t get_kv_int(const std::string& key, int64_t def = 0) const;
    float get_kv_float(const std::string& key, float def = 0.0f) const;
    bool get_kv_bool(const std::string& key, bool def = false) const;

    /// Pointer to the tensor's bytes inside the read-only mapping.
    const uint8_t* tensor_data(const TensorInfo& t) const { return base_ + t.offset; }
    /// Copy of the tensor's bytes (or a sub-range).
    std::vector<uint8_t> get_tensor_data(const std::string& name, size_t offset = 0, size_t length = 0) const;

    uint64_t file_size() const { return file_size_; }
    uint64_t data_offset() const { return data_offset_; }
    uint64_t metadata_end() const { return meta_end_; }
    uint32_t alignment() const { return alignment_; }
    const uint8_t* mapping() const { return base_; }

private:
    GGUFHeader header_;
    ModelConfig config_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
    std::vector<GGUFKVPair> kv_list_;
    std::unordered_map<std::string, size_t> kv_index_;

    int fd_ = -1;
    const uint8_t* base_ = nullptr;
    uint64_t file_size_ = 0;
    uint64_t pos_ = 0;              // parse cursor
    uint64_t meta_end_ = 0;         // end of tensor-info table
    uint64_t data_offset_ = 0;      // aligned start of tensor data
    uint32_t alignment_ = 32;
    bool verbose_ = false;

    bool parse_header();
    bool parse_kv();
    bool parse_tensors();
    bool finalize_tensors();
    bool build_config();

    // Bounds-checked readers: throw std::runtime_error (with offset) on EOF.
    void need(uint64_t n, const char* what) const;
    template <typename T> T read_pod(const char* what);
    uint8_t  read_uint8()  { return read_pod<uint8_t>("uint8"); }
    int8_t   read_int8()   { return read_pod<int8_t>("int8"); }
    uint16_t read_uint16() { return read_pod<uint16_t>("uint16"); }
    int16_t  read_int16()  { return read_pod<int16_t>("int16"); }
    uint32_t read_uint32() { return read_pod<uint32_t>("uint32"); }
    int32_t  read_int32()  { return read_pod<int32_t>("int32"); }
    uint64_t read_uint64() { return read_pod<uint64_t>("uint64"); }
    int64_t  read_int64()  { return read_pod<int64_t>("int64"); }
    float    read_float()  { return read_pod<float>("float32"); }
    double   read_double() { return read_pod<double>("float64"); }
    std::string read_string();
    bool read_value(GGUFType type, GGUFValue& out, int depth);
    bool read_array(GGUFValue& out, int depth);
};

} // namespace q38rt
