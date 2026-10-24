#pragma once
/// @file ref_model.h
/// Slow, straightforward FP32 reference implementation of the Qwen3.8-27B forward pass
/// (qwen35 graph as implemented by llama.cpp src/models/qwen35.cpp). Weights are
/// dequantized exactly (quant_ref.cpp); activations stay FP32 (no activation
/// quantization), attention/KV in FP32. It is the numerical ground truth for the
/// optimized engine and is itself validated against llama.cpp golden logits.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "runtime/gguf_parser.h"

namespace q38rt {

class RefModel {
public:
    /// Optional per-layer probe: (layer, name, data, T, row_len) for intermediate activations.
    using Probe = std::function<void(int layer, const char* name, const float* data, int T, int n)>;

    explicit RefModel(const GGUFParser& gguf);

    /// Evaluate T tokens at positions [pos(), pos()+T). Writes logits (vocab floats per row) for the rows
    /// listed in `logit_rows` (indices into this batch) to `logits` (logit_rows.size() x vocab).
    void forward(const int32_t* tokens, int T, const std::vector<int>& logit_rows, std::vector<float>& logits);

    int pos() const { return pos_; }
    void reset();
    void set_probe(Probe p) { probe_ = std::move(p); }
    void set_threads(int n) { threads_ = n; }

private:
    const GGUFParser& g_;
    const ModelConfig& c_;
    int pos_ = 0;
    int threads_ = 8;
    Probe probe_;

    // state
    std::vector<std::vector<float>> kcache_, vcache_;   // [16][pos][1024]
    std::vector<std::vector<float>> conv_state_;        // [48][3][10240] (oldest first)
    std::vector<std::vector<float>> ssm_state_;         // [48][48 heads][128 k][128 v]

    const TensorInfo& T(const std::string& name) const;
    std::vector<float> load_f32(const std::string& name) const;   // small tensors (norms etc.)
    /// y[t][r] = sum_k W[r][k] x[t][k] for W of shape [K, N] in GGUF order (ne0 = K).
    void matmul(const std::string& wname, const float* x, int T, int K, float* y, int N) const;
    void layer_attention(int il, int L_attn, std::vector<float>& xn, int T, std::vector<float>& out);
    void layer_deltanet(int il, int L_lin, std::vector<float>& xn, int T, std::vector<float>& out);
    void probe(int layer, const char* name, const float* d, int T, int n) const {
        if (probe_) probe_(layer, name, d, T, n);
    }
};

} // namespace q38rt
