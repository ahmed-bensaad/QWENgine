/// @file ref_model.cpp
/// FP32 reference forward pass. Clarity over speed: every step mirrors one op of
/// llama.cpp's qwen35 graph. Parallelized with OpenMP only where it is trivially safe.

#include "reference/ref_model.h"
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "quant/quant_formats.h"

namespace q38rt {

namespace {

constexpr int D_MODEL = 5120;
constexpr int N_HEAD = 24, N_KV = 4, HD = 256, N_ROT = 64;
constexpr int DN_K_HEADS = 16, DN_V_HEADS = 48, DN_HD = 128;
constexpr int DN_CONV = 10240, DN_QK = 2048, DN_V = 6144;
constexpr int FFN = 17408;

float dot_f32(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    float buf[8];
    _mm256_storeu_ps(buf, s);
    double acc = 0;
    for (int j = 0; j < 8; j++) acc += buf[j];
    for (; i < n; i++) acc += (double)a[i] * b[i];
    return (float)acc;
}

void rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * scale * (w ? w[i] : 1.0f);
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
inline float siluf(float x) { return x / (1.0f + expf(-x)); }
inline float softplusf(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

// NeoX-style RoPE on the first N_ROT dims of a head (IMROPE with text positions p,p,p,0).
void rope_neox(float* x, int pos, float theta_base) {
    for (int i = 0; i < N_ROT / 2; i++) {
        const float theta = (float)pos * powf(theta_base, -2.0f * i / N_ROT);
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = x[i], x1 = x[i + N_ROT / 2];
        x[i] = x0 * c - x1 * s;
        x[i + N_ROT / 2] = x0 * s + x1 * c;
    }
}

} // namespace

RefModel::RefModel(const GGUFParser& gguf) : g_(gguf), c_(gguf.get_config()) { reset(); }

void RefModel::reset() {
    pos_ = 0;
    kcache_.assign(16, {});
    vcache_.assign(16, {});
    conv_state_.assign(48, std::vector<float>(3 * DN_CONV, 0.0f));
    ssm_state_.assign(48, std::vector<float>((size_t)DN_V_HEADS * DN_HD * DN_HD, 0.0f));
}

const TensorInfo& RefModel::T(const std::string& name) const {
    const TensorInfo* t = g_.find_tensor(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    return *t;
}

std::vector<float> RefModel::load_f32(const std::string& name) const {
    const TensorInfo& t = T(name);
    std::vector<float> v(t.shape.size());
    if (!dequantize_row_ref(t.quant, g_.tensor_data(t), v.data(), (int64_t)v.size()))
        throw std::runtime_error("unsupported type for " + name);
    return v;
}

void RefModel::matmul(const std::string& wname, const float* x, int Tn, int K, float* y, int N) const {
    const TensorInfo& t = T(wname);
    if ((int)t.shape.dims[0] != K || (int)t.shape.dims[1] != N)
        throw std::runtime_error("shape mismatch for " + wname);
    const size_t row_bytes = quant_row_bytes(t.quant, K);
    const uint8_t* base = g_.tensor_data(t);
#pragma omp parallel num_threads(threads_)
    {
        std::vector<float> w(K);
#pragma omp for schedule(dynamic, 32)
        for (int r = 0; r < N; r++) {
            dequantize_row_ref(t.quant, base + (size_t)r * row_bytes, w.data(), K);
            for (int tt = 0; tt < Tn; tt++) y[(size_t)tt * N + r] = dot_f32(w.data(), x + (size_t)tt * K, K);
        }
    }
}

void RefModel::layer_attention(int il, int La, std::vector<float>& xn, int Tn, std::vector<float>& out) {
    const std::string p = "blk." + std::to_string(il) + ".";
    std::vector<float> qg((size_t)Tn * N_HEAD * HD * 2), k((size_t)Tn * N_KV * HD), v((size_t)Tn * N_KV * HD);
    matmul(p + "attn_q.weight", xn.data(), Tn, D_MODEL, qg.data(), N_HEAD * HD * 2);
    matmul(p + "attn_k.weight", xn.data(), Tn, D_MODEL, k.data(), N_KV * HD);
    matmul(p + "attn_v.weight", xn.data(), Tn, D_MODEL, v.data(), N_KV * HD);
    const std::vector<float> qn = load_f32(p + "attn_q_norm.weight");
    const std::vector<float> kn = load_f32(p + "attn_k_norm.weight");

    std::vector<float> q((size_t)Tn * N_HEAD * HD), gate((size_t)Tn * N_HEAD * HD);
    for (int t = 0; t < Tn; t++) {
        for (int h = 0; h < N_HEAD; h++) {
            const float* src = &qg[((size_t)t * N_HEAD + h) * 2 * HD];
            float* qd = &q[((size_t)t * N_HEAD + h) * HD];
            rmsnorm(src, qn.data(), qd, HD, c_.rms_eps);
            memcpy(&gate[((size_t)t * N_HEAD + h) * HD], src + HD, HD * sizeof(float));
            rope_neox(qd, pos_ + t, c_.rope_theta);
        }
        for (int h = 0; h < N_KV; h++) {
            float* kd = &k[((size_t)t * N_KV + h) * HD];
            std::vector<float> tmp(kd, kd + HD);
            rmsnorm(tmp.data(), kn.data(), kd, HD, c_.rms_eps);
            rope_neox(kd, pos_ + t, c_.rope_theta);
        }
    }
    probe(il, "q_roped", q.data(), Tn, N_HEAD * HD);
    probe(il, "k_roped", k.data(), Tn, N_KV * HD);
    probe(il, "v", v.data(), Tn, N_KV * HD);
    std::vector<float>& KC = kcache_[La];
    std::vector<float>& VC = vcache_[La];
    KC.insert(KC.end(), k.begin(), k.end());
    VC.insert(VC.end(), v.begin(), v.end());

    std::vector<float> attn((size_t)Tn * N_HEAD * HD);
    const float scale = 1.0f / sqrtf((float)HD);
#pragma omp parallel for collapse(2) num_threads(threads_) schedule(dynamic)
    for (int t = 0; t < Tn; t++) {
        for (int h = 0; h < N_HEAD; h++) {
            const int kvh = h / (N_HEAD / N_KV);
            const int npos = pos_ + t + 1;                 // causal
            const float* qh = &q[((size_t)t * N_HEAD + h) * HD];
            std::vector<double> s(npos);
            double mx = -1e300;
            for (int j = 0; j < npos; j++) {
                s[j] = (double)dot_f32(qh, &KC[((size_t)j * N_KV + kvh) * HD], HD) * scale;
                mx = std::max(mx, s[j]);
            }
            double den = 0;
            for (int j = 0; j < npos; j++) { s[j] = std::exp(s[j] - mx); den += s[j]; }
            float* o = &attn[((size_t)t * N_HEAD + h) * HD];
            for (int d = 0; d < HD; d++) {
                double acc = 0;
                for (int j = 0; j < npos; j++) acc += s[j] * VC[((size_t)j * N_KV + kvh) * HD + d];
                o[d] = (float)(acc / den);
            }
        }
    }
    for (size_t i = 0; i < attn.size(); i++) attn[i] *= sigmoidf(gate[i]);
    probe(il, "attn_gated", attn.data(), Tn, N_HEAD * HD);
    out.assign((size_t)Tn * D_MODEL, 0.0f);
    matmul(p + "attn_output.weight", attn.data(), Tn, N_HEAD * HD, out.data(), D_MODEL);
}

void RefModel::layer_deltanet(int il, int Ll, std::vector<float>& xn, int Tn, std::vector<float>& out) {
    const std::string p = "blk." + std::to_string(il) + ".";
    std::vector<float> qkv((size_t)Tn * DN_CONV), z((size_t)Tn * DN_V), b((size_t)Tn * DN_V_HEADS),
        a((size_t)Tn * DN_V_HEADS);
    matmul(p + "attn_qkv.weight", xn.data(), Tn, D_MODEL, qkv.data(), DN_CONV);
    matmul(p + "attn_gate.weight", xn.data(), Tn, D_MODEL, z.data(), DN_V);
    matmul(p + "ssm_beta.weight", xn.data(), Tn, D_MODEL, b.data(), DN_V_HEADS);
    matmul(p + "ssm_alpha.weight", xn.data(), Tn, D_MODEL, a.data(), DN_V_HEADS);
    probe(il, "qkv_mixed", qkv.data(), Tn, DN_CONV);
    probe(il, "z", z.data(), Tn, DN_V);
    const std::vector<float> conv_w = load_f32(p + "ssm_conv1d.weight");  // [10240][4]
    const std::vector<float> dt_bias = load_f32(p + "ssm_dt.bias");
    const std::vector<float> ssm_a = load_f32(p + "ssm_a");
    const std::vector<float> ssm_norm = load_f32(p + "ssm_norm.weight");

    std::vector<float>& cs = conv_state_[Ll];   // [3][10240]
    std::vector<float>& S = ssm_state_[Ll];     // [48][128 k][128 v]
    std::vector<float> o((size_t)Tn * DN_V);
    std::vector<float> conv((size_t)DN_CONV);
    for (int t = 0; t < Tn; t++) {
        const float* x = &qkv[(size_t)t * DN_CONV];
        for (int c = 0; c < DN_CONV; c++) {
            const float* w = &conv_w[(size_t)c * 4];
            const float acc = w[0] * cs[0 * DN_CONV + c] + w[1] * cs[1 * DN_CONV + c] + w[2] * cs[2 * DN_CONV + c] +
                              w[3] * x[c];
            conv[c] = siluf(acc);
            cs[0 * DN_CONV + c] = cs[1 * DN_CONV + c];
            cs[1 * DN_CONV + c] = cs[2 * DN_CONV + c];
            cs[2 * DN_CONV + c] = x[c];
        }
        if (t == Tn - 1) probe(il, "conv_out_last", conv.data(), 1, DN_CONV);
        // L2-normalize q and k per head
        float qn[DN_QK], kn[DN_QK];
        for (int h = 0; h < DN_K_HEADS; h++) {
            double sq = 0, sk = 0;
            for (int d = 0; d < DN_HD; d++) {
                sq += (double)conv[h * DN_HD + d] * conv[h * DN_HD + d];
                sk += (double)conv[DN_QK + h * DN_HD + d] * conv[DN_QK + h * DN_HD + d];
            }
            const float iq = 1.0f / std::max((float)std::sqrt(sq), c_.rms_eps);
            const float ik = 1.0f / std::max((float)std::sqrt(sk), c_.rms_eps);
            for (int d = 0; d < DN_HD; d++) {
                qn[h * DN_HD + d] = conv[h * DN_HD + d] * iq;
                kn[h * DN_HD + d] = conv[DN_QK + h * DN_HD + d] * ik;
            }
        }
        const float* vv = &conv[2 * DN_QK];
        const float qscale = 1.0f / sqrtf((float)DN_HD);
#pragma omp parallel for num_threads(threads_)
        for (int h = 0; h < DN_V_HEADS; h++) {
            const int kh = h % DN_K_HEADS;      // ggml_repeat tiling (GGUF V-heads are pre-permuted)
            const float beta = sigmoidf(b[(size_t)t * DN_V_HEADS + h]);
            const float g = softplusf(a[(size_t)t * DN_V_HEADS + h] + dt_bias[h]) * ssm_a[h];
            const float decay = expf(g);
            float* Sh = &S[(size_t)h * DN_HD * DN_HD];   // Sh[i*128 + j]: i = k index, j = v index
            const float* kk = &kn[kh * DN_HD];
            const float* qq = &qn[kh * DN_HD];
            const float* vh = vv + h * DN_HD;
            float kv[DN_HD];
            for (int j = 0; j < DN_HD; j++) kv[j] = 0.0f;
            for (int i = 0; i < DN_HD; i++) {
                for (int j = 0; j < DN_HD; j++) {
                    Sh[i * DN_HD + j] *= decay;
                    kv[j] += Sh[i * DN_HD + j] * kk[i];
                }
            }
            float delta[DN_HD];
            for (int j = 0; j < DN_HD; j++) delta[j] = (vh[j] - kv[j]) * beta;
            float oh[DN_HD];
            for (int j = 0; j < DN_HD; j++) oh[j] = 0.0f;
            for (int i = 0; i < DN_HD; i++) {
                for (int j = 0; j < DN_HD; j++) {
                    Sh[i * DN_HD + j] += kk[i] * delta[j];
                    oh[j] += Sh[i * DN_HD + j] * qq[i];
                }
            }
            // gated RMSNorm: rmsnorm(o_h) * ssm_norm * silu(z_h)
            float* od = &o[(size_t)t * DN_V + h * DN_HD];
            for (int j = 0; j < DN_HD; j++) oh[j] *= qscale;
            rmsnorm(oh, ssm_norm.data(), od, DN_HD, c_.rms_eps);
            const float* zh = &z[(size_t)t * DN_V + h * DN_HD];
            for (int j = 0; j < DN_HD; j++) od[j] *= siluf(zh[j]);
        }
    }
    probe(il, "dn_gated_out", o.data(), Tn, DN_V);
    out.assign((size_t)Tn * D_MODEL, 0.0f);
    matmul(p + "ssm_out.weight", o.data(), Tn, DN_V, out.data(), D_MODEL);
}

void RefModel::forward(const int32_t* tokens, int Tn, const std::vector<int>& logit_rows, std::vector<float>& logits) {
    // embeddings
    std::vector<float> x((size_t)Tn * D_MODEL);
    {
        const TensorInfo& e = T("token_embd.weight");
        const size_t rb = quant_row_bytes(e.quant, D_MODEL);
        for (int t = 0; t < Tn; t++)
            dequantize_row_ref(e.quant, g_.tensor_data(e) + (size_t)tokens[t] * rb, &x[(size_t)t * D_MODEL], D_MODEL);
    }
    probe(-1, "embd", x.data(), Tn, D_MODEL);
    std::vector<float> xn((size_t)Tn * D_MODEL), y;
    int La = 0, Ll = 0;
    for (int il = 0; il < c_.num_layers; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const std::vector<float> an = load_f32(p + "attn_norm.weight");
        for (int t = 0; t < Tn; t++)
            rmsnorm(&x[(size_t)t * D_MODEL], an.data(), &xn[(size_t)t * D_MODEL], D_MODEL, c_.rms_eps);
        if (c_.is_full_attention(il)) layer_attention(il, La++, xn, Tn, y);
        else layer_deltanet(il, Ll++, xn, Tn, y);
        probe(il, "attn_out", y.data(), Tn, D_MODEL);
        for (size_t i = 0; i < x.size(); i++) x[i] += y[i];
        const std::vector<float> pn = load_f32(p + "post_attention_norm.weight");
        for (int t = 0; t < Tn; t++)
            rmsnorm(&x[(size_t)t * D_MODEL], pn.data(), &xn[(size_t)t * D_MODEL], D_MODEL, c_.rms_eps);
        std::vector<float> gt((size_t)Tn * FFN), up((size_t)Tn * FFN);
        matmul(p + "ffn_gate.weight", xn.data(), Tn, D_MODEL, gt.data(), FFN);
        matmul(p + "ffn_up.weight", xn.data(), Tn, D_MODEL, up.data(), FFN);
        for (size_t i = 0; i < gt.size(); i++) gt[i] = siluf(gt[i]) * up[i];
        y.assign((size_t)Tn * D_MODEL, 0.0f);
        matmul(p + "ffn_down.weight", gt.data(), Tn, FFN, y.data(), D_MODEL);
        for (size_t i = 0; i < x.size(); i++) x[i] += y[i];
        probe(il, "l_out", x.data(), Tn, D_MODEL);
    }
    const std::vector<float> on = load_f32("output_norm.weight");
    std::vector<float> sel((size_t)logit_rows.size() * D_MODEL);
    for (size_t r = 0; r < logit_rows.size(); r++)
        rmsnorm(&x[(size_t)logit_rows[r] * D_MODEL], on.data(), &sel[r * D_MODEL], D_MODEL, c_.rms_eps);
    logits.assign(logit_rows.size() * (size_t)c_.vocab_size, 0.0f);
    if (!logit_rows.empty())
        matmul("output.weight", sel.data(), (int)logit_rows.size(), D_MODEL, logits.data(), c_.vocab_size);
    pos_ += Tn;
}

} // namespace q38rt
