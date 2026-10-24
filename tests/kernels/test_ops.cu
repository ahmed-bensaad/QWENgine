/// @file test_ops.cu
/// GPU decode ops vs straightforward CPU math (double accumulation) using real layer parameters.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "cuda/ops.cuh"
#include "quant/quant_formats.h"
#include "runtime/gguf_parser.h"
#include "common/utils.h"

using namespace q38rt;

static int g_fail = 0;
static void check(const char* what, double err, double tol) {
    const bool ok = err <= tol;
    printf("%-48s max_rel_err %.3e (tol %.1e) %s\n", what, err, tol, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

template <typename T> static T* dev(const std::vector<T>& h) {
    T* d;
    CHECK_CUDA(cudaMalloc(&d, h.size() * sizeof(T) + 16));
    CHECK_CUDA(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T> static std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    CHECK_CUDA(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
static std::vector<float> load(const GGUFParser& p, const char* name) {
    const TensorInfo* t = p.find_tensor(name);
    std::vector<float> v(t->shape.size());
    dequantize_row_ref(t->quant, p.tensor_data(*t), v.data(), (int64_t)v.size());
    return v;
}
static double rel_err(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < b.size(); i++) { num = std::max(num, std::fabs(a[i] - b[i])); den = std::max(den, std::fabs(b[i])); }
    return num / (den > 0 ? den : 1);
}
static cuda::ActQ8 alloc_act(int K) {
    cuda::ActQ8 a;
    CHECK_CUDA(cudaMalloc(&a.q, K));
    CHECK_CUDA(cudaMalloc(&a.d, K / 32 * 4));
    CHECK_CUDA(cudaMalloc(&a.s, K / 32 * 4));
    return a;
}
static double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double silu(double x) { return x * sig(x); }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 2; }
    GGUFParser p;
    if (!p.parse(argv[1])) return 1;
    const float eps = 1e-6f;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    cudaStream_t st;
    CHECK_CUDA(cudaStreamCreate(&st));

    // ---------------- add_rmsnorm + ActQ8 ----------------
    {
        const int n = 5120;
        std::vector<float> x(n), a(n), b(n);
        for (int i = 0; i < n; i++) { x[i] = nd(rng) * 3; a[i] = nd(rng); b[i] = nd(rng) * 0.5f; }
        const std::vector<float> w = load(p, "blk.0.post_attention_norm.weight");
        float *dx = dev(x), *da = dev(a), *db = dev(b), *dw = dev(w), *dxn;
        CHECK_CUDA(cudaMalloc(&dxn, n * 4));
        cuda::ActQ8 q = alloc_act(n);
        cuda::add_rmsnorm(dx, da, db, dw, n, eps, dxn, &q, st);
        CHECK_CUDA(cudaStreamSynchronize(st));
        std::vector<double> xr(n), xnr(n);
        double ss = 0;
        for (int i = 0; i < n; i++) { xr[i] = (double)x[i] + a[i] + b[i]; ss += xr[i] * xr[i]; }
        const double sc = 1.0 / std::sqrt(ss / n + eps);
        for (int i = 0; i < n; i++) xnr[i] = xr[i] * sc * w[i];
        check("add_rmsnorm: residual", rel_err(host(dx, n), xr), 1e-6);
        check("add_rmsnorm: normed", rel_err(host(dxn, n), xnr), 1e-5);
        std::vector<int8_t> qq = host(q.q, n);
        std::vector<float> qd = host(q.d, n / 32), qs = host(q.s, n / 32), deq(n);
        double serr = 0;
        for (int i = 0; i < n; i++) deq[i] = qd[i / 32] * qq[i];
        for (int bk = 0; bk < n / 32; bk++) {
            double s = 0, mag = 0;
            for (int j = 0; j < 32; j++) { s += qd[bk] * qq[bk * 32 + j]; mag += std::fabs(qd[bk] * qq[bk * 32 + j]); }
            serr = std::max(serr, std::fabs(s - qs[bk]) / (mag + 1e-12));   // relative to the summed magnitudes
        }
        check("add_rmsnorm: ActQ8 dequant vs normed", rel_err(deq, xnr), 1.0 / 127 + 1e-4);
        check("add_rmsnorm: ActQ8 block sums", serr, 1e-5);
    }

    // ---------------- swiglu ----------------
    {
        const int n = 17408 - 256 * 10;
        std::vector<float> g(n), u(n);
        for (int i = 0; i < n; i++) { g[i] = nd(rng) * 2; u[i] = nd(rng); }
        float *dg = dev(g), *du = dev(u), *dh;
        CHECK_CUDA(cudaMalloc(&dh, n * 4));
        cuda::swiglu(dg, du, n, dh, nullptr, st);
        CHECK_CUDA(cudaStreamSynchronize(st));
        std::vector<double> r(n);
        for (int i = 0; i < n; i++) r[i] = silu(g[i]) * u[i];
        check("swiglu", rel_err(host(dh, n), r), 1e-5);
    }

    // ---------------- Gated DeltaNet: 6 steps on layer 0 parameters ----------------
    {
        const int CH = 10240, HD = 128, VH = 48, KH = 16;
        const std::vector<float> conv_w = load(p, "blk.0.ssm_conv1d.weight"), dt = load(p, "blk.0.ssm_dt.bias"),
                                 sa = load(p, "blk.0.ssm_a"), sn = load(p, "blk.0.ssm_norm.weight");
        cuda::DeltaNetParams dp;
        dp.conv_w = dev(conv_w); dp.dt_bias = dev(dt); dp.ssm_a = dev(sa); dp.ssm_norm = dev(sn);
        CHECK_CUDA(cudaMalloc(&dp.conv_state, 2 * 3 * CH * 4));
        CHECK_CUDA(cudaMemset(dp.conv_state, 0, 2 * 3 * CH * 4));
        CHECK_CUDA(cudaMalloc(&dp.ssm_state, (size_t)VH * HD * HD * 4));
        CHECK_CUDA(cudaMemset(dp.ssm_state, 0, (size_t)VH * HD * HD * 4));
        std::vector<double> cs(3 * CH, 0.0), S((size_t)VH * HD * HD, 0.0);   // S[h][i][j]
        float *dqkv, *dz, *db, *da, *dout;
        CHECK_CUDA(cudaMalloc(&dqkv, CH * 4)); CHECK_CUDA(cudaMalloc(&dz, 6144 * 4));
        CHECK_CUDA(cudaMalloc(&db, VH * 4)); CHECK_CUDA(cudaMalloc(&da, VH * 4)); CHECK_CUDA(cudaMalloc(&dout, 6144 * 4));
        cuda::ActQ8 oq = alloc_act(6144);
        double worst = 0;
        for (int step = 0; step < 6; step++) {
            std::vector<float> qkv(CH), z(6144), b(VH), a(VH);
            for (auto& v : qkv) v = nd(rng) * 0.5f;
            for (auto& v : z) v = nd(rng);
            for (auto& v : b) v = nd(rng);
            for (auto& v : a) v = nd(rng) * 2;
            CHECK_CUDA(cudaMemcpy(dqkv, qkv.data(), CH * 4, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dz, z.data(), 6144 * 4, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(db, b.data(), VH * 4, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(da, a.data(), VH * 4, cudaMemcpyHostToDevice));
            cuda::deltanet_step(dp, dqkv, dz, db, da, step & 1, eps, dout, &oq, st);
            CHECK_CUDA(cudaStreamSynchronize(st));
            CHECK_CUDA(cudaGetLastError());
            // CPU reference step
            std::vector<double> conv(CH);
            for (int c = 0; c < CH; c++) {
                const double acc = conv_w[c * 4] * cs[c] + conv_w[c * 4 + 1] * cs[CH + c] + conv_w[c * 4 + 2] * cs[2 * CH + c] +
                                   conv_w[c * 4 + 3] * (double)qkv[c];
                conv[c] = silu(acc);
                cs[c] = cs[CH + c]; cs[CH + c] = cs[2 * CH + c]; cs[2 * CH + c] = qkv[c];
            }
            std::vector<double> o(6144);
            for (int h = 0; h < VH; h++) {
                const int kh = h % KH;
                double nq = 0, nk = 0;
                for (int d = 0; d < HD; d++) { nq += conv[kh * HD + d] * conv[kh * HD + d]; nk += conv[2048 + kh * HD + d] * conv[2048 + kh * HD + d]; }
                const double iq = 1.0 / std::max(std::sqrt(nq), (double)eps), ik = 1.0 / std::max(std::sqrt(nk), (double)eps);
                const double beta = sig(b[h]);
                const double av = a[h] + dt[h];
                const double g = (av > 20 ? av : std::log1p(std::exp(av))) * sa[h];
                const double decay = std::exp(g);
                double* Sh = &S[(size_t)h * HD * HD];
                std::vector<double> kv(HD, 0.0), oh(HD, 0.0);
                for (int i = 0; i < HD; i++) for (int j = 0; j < HD; j++) kv[j] += Sh[i * HD + j] * conv[2048 + kh * HD + i] * ik;
                for (int j = 0; j < HD; j++) {
                    const double delta = (conv[4096 + h * HD + j] - decay * kv[j]) * beta;
                    for (int i = 0; i < HD; i++) {
                        Sh[i * HD + j] = decay * Sh[i * HD + j] + conv[2048 + kh * HD + i] * ik * delta;
                        oh[j] += Sh[i * HD + j] * conv[kh * HD + i] * iq;
                    }
                }
                double ss = 0;
                for (int j = 0; j < HD; j++) { oh[j] /= std::sqrt((double)HD); ss += oh[j] * oh[j]; }
                const double rs = 1.0 / std::sqrt(ss / HD + eps);
                for (int j = 0; j < HD; j++) o[h * HD + j] = oh[j] * rs * sn[j] * silu(z[h * HD + j]);
            }
            worst = std::max(worst, rel_err(host(dout, 6144), o));
        }
        check("deltanet_step: outputs over 6 steps", worst, 2e-4);
        std::vector<float> Sg = host(dp.ssm_state, (size_t)VH * HD * HD);   // [h][j][i]
        std::vector<float> Sgt(Sg.size());
        std::vector<double> Sd(S.begin(), S.end());
        for (int h = 0; h < VH; h++) for (int i = 0; i < HD; i++) for (int j = 0; j < HD; j++)
            Sgt[((size_t)h * HD + i) * HD + j] = Sg[((size_t)h * HD + j) * HD + i];
        check("deltanet_step: recurrent state after 6 steps", rel_err(Sgt, Sd), 2e-4);
        std::vector<float> csg = host(dp.conv_state + 0 * 3 * CH, 3 * CH);   // after 6 steps parity 0 holds latest
        check("deltanet_step: conv state", rel_err(csg, cs), 1e-7);
    }

    // ---------------- attention: 300 positions on layer 3 norms ----------------
    for (cuda::KVType kvt : {cuda::KVType::F16, cuda::KVType::Q8_0, cuda::KVType::Q4_0}) {
        const int NPOS = 300, AH = 24, AKV = 4, AD = 256;
        const float theta = 1e7f;
        const std::vector<float> qn = load(p, "blk.3.attn_q_norm.weight"), kn = load(p, "blk.3.attn_k_norm.weight");
        float* dqn = dev(qn);
        float* dkn = dev(kn);
        cuda::KVLayer kv;
        kv.capacity = 512;
        const size_t rb = cuda::kv_value_bytes(kvt);
        CHECK_CUDA(cudaMalloc(&kv.k, AKV * kv.capacity * rb));
        CHECK_CUDA(cudaMalloc(&kv.v, AKV * kv.capacity * rb));
        CHECK_CUDA(cudaMalloc(&kv.ks, AKV * kv.capacity * 16));
        CHECK_CUDA(cudaMalloc(&kv.vs, AKV * kv.capacity * 16));
        // host-side decode of one stored row (split layout) -> 256 floats
        auto read_row = [&](const uint8_t* vals, const uint16_t* scales, size_t row, float* out) {
            std::vector<uint8_t> vb(rb);
            uint16_t sc[8] = {0};
            CHECK_CUDA(cudaMemcpy(vb.data(), vals + row * rb, rb, cudaMemcpyDeviceToHost));
            if (kvt != cuda::KVType::F16) CHECK_CUDA(cudaMemcpy(sc, scales + row * 8, 16, cudaMemcpyDeviceToHost));
            for (int d = 0; d < AD; d++) {
                if (kvt == cuda::KVType::F16) out[d] = fp16_to_fp32(((const uint16_t*)vb.data())[d]);
                else if (kvt == cuda::KVType::Q8_0) out[d] = fp16_to_fp32(sc[d / 32]) * (float)((const int8_t*)vb.data())[d];
                else out[d] = fp16_to_fp32(sc[d / 32]) * (float)((int)((vb[d / 2] >> (4 * (d & 1))) & 0xF) - 8);
            }
        };
        float *dqg, *dk, *dv, *dq, *dgate, *dout;
        CHECK_CUDA(cudaMalloc(&dqg, AH * 512 * 4)); CHECK_CUDA(cudaMalloc(&dk, AKV * AD * 4)); CHECK_CUDA(cudaMalloc(&dv, AKV * AD * 4));
        CHECK_CUDA(cudaMalloc(&dq, AH * AD * 4)); CHECK_CUDA(cudaMalloc(&dgate, AH * AD * 4)); CHECK_CUDA(cudaMalloc(&dout, AH * AD * 4));
        void* work;
        CHECK_CUDA(cudaMalloc(&work, cuda::attn_work_bytes(NPOS)));
        // tensor-core path: outputs of the last 3 single-query calls (for the multi-query exactness check)
        float *dq3, *dg3, *dout3, *dsimple;
        CHECK_CUDA(cudaMalloc(&dq3, 3 * AH * AD * 4)); CHECK_CUDA(cudaMalloc(&dg3, 3 * AH * AD * 4));
        CHECK_CUDA(cudaMalloc(&dout3, 3 * AH * AD * 4)); CHECK_CUDA(cudaMalloc(&dsimple, AH * AD * 4));
        std::vector<std::vector<float>> single_out(3, std::vector<float>(AH * AD));
        int multi_bad = 0, multi_checked = 0;
        std::vector<double> KC, VC;   // reference fp64 caches [pos][4][256]
        double worst = 0, worst_kernel = 0, worst_simple = 0;
        int vbytes_bad = 0;
        const QuantFormat qf = kvt == cuda::KVType::Q8_0 ? QuantFormat::Q8_0 : QuantFormat::Q4_0;
        for (int pos = 0; pos < NPOS; pos++) {
            std::vector<float> qg(AH * 512), k(AKV * AD), v(AKV * AD);
            for (auto& x : qg) x = nd(rng);
            for (auto& x : k) x = nd(rng);
            for (auto& x : v) x = nd(rng);
            CHECK_CUDA(cudaMemcpy(dqg, qg.data(), qg.size() * 4, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
            cuda::attn_prep(dqg, dk, dv, dqn, dkn, pos, theta, eps, kvt, kv, dq, dgate, st);
            cuda::attn_decode(dq, dgate, kv, kvt, pos + 1, work, dout, nullptr, st);
            cuda::attn_decode_simple(dq, dgate, kv, kvt, pos + 1, work, dsimple, nullptr, st);
            // keep this position's q/gate/output in slot pos % 3
            CHECK_CUDA(cudaMemcpyAsync(dq3 + (pos % 3) * AH * AD, dq, AH * AD * 4, cudaMemcpyDeviceToDevice, st));
            CHECK_CUDA(cudaMemcpyAsync(dg3 + (pos % 3) * AH * AD, dgate, AH * AD * 4, cudaMemcpyDeviceToDevice, st));
            CHECK_CUDA(cudaMemcpy(single_out[pos % 3].data(), dout, AH * AD * 4, cudaMemcpyDeviceToHost));
            if (pos >= 2 && pos % 7 == 0) {   // queries pos-2, pos-1, pos in one call == the three single calls
                float *qc, *gc;
                CHECK_CUDA(cudaMalloc(&qc, 3 * AH * AD * 4)); CHECK_CUDA(cudaMalloc(&gc, 3 * AH * AD * 4));
                for (int t = 0; t < 3; t++) {
                    const int slot = (pos - 2 + t) % 3;
                    CHECK_CUDA(cudaMemcpy(qc + t * AH * AD, dq3 + slot * AH * AD, AH * AD * 4, cudaMemcpyDeviceToDevice));
                    CHECK_CUDA(cudaMemcpy(gc + t * AH * AD, dg3 + slot * AH * AD, AH * AD * 4, cudaMemcpyDeviceToDevice));
                }
                cuda::attn_decode_multi(qc, gc, kv, kvt, pos - 2, 3, work, dout3, nullptr, st);
                std::vector<float> m3(3 * AH * AD);
                CHECK_CUDA(cudaMemcpy(m3.data(), dout3, m3.size() * 4, cudaMemcpyDeviceToHost));
                for (int t = 0; t < 3; t++)
                    multi_bad += memcmp(m3.data() + t * AH * AD, single_out[(pos - 2 + t) % 3].data(), AH * AD * 4) != 0;
                multi_checked += 3;
                cudaFree(qc); cudaFree(gc);
            }
            auto norm_rope = [&](const float* src, const std::vector<float>& w, std::vector<double>& dst) {
                double ss = 0;
                for (int d = 0; d < AD; d++) ss += (double)src[d] * src[d];
                const double sc = 1.0 / std::sqrt(ss / AD + eps);
                dst.resize(AD);
                for (int d = 0; d < AD; d++) dst[d] = src[d] * sc * w[d];
                for (int i = 0; i < 32; i++) {
                    const double ang = pos * std::pow((double)theta, -2.0 * i / 64);
                    const double c = std::cos(ang), s = std::sin(ang), x0 = dst[i], x1 = dst[i + 32];
                    dst[i] = x0 * c - x1 * s;
                    dst[i + 32] = x0 * s + x1 * c;
                }
            };
            for (int g = 0; g < AKV; g++) {
                std::vector<double> kr;
                norm_rope(&k[g * AD], kn, kr);
                KC.insert(KC.end(), kr.begin(), kr.end());
                VC.insert(VC.end(), v.begin() + g * AD, v.begin() + (g + 1) * AD);
                if (kvt != cuda::KVType::F16) {   // V rows are stored unmodified: values must equal ggml's quantizer
                    CHECK_CUDA(cudaStreamSynchronize(st));
                    std::vector<float> got(AD), want(AD);
                    read_row(kv.v, kv.vs, (size_t)g * kv.capacity + pos, got.data());
                    std::vector<uint8_t> blk(AD / 32 * sizeof(block_q8_0));
                    if (kvt == cuda::KVType::Q8_0) quantize_row_q8_0_ref(&v[g * AD], (block_q8_0*)blk.data(), AD);
                    else quantize_row_q4_0_ref(&v[g * AD], (block_q4_0*)blk.data(), AD);
                    dequantize_row_ref(qf, blk.data(), want.data(), AD);
                    vbytes_bad += memcmp(got.data(), want.data(), AD * 4) != 0;
                }
            }
            if (pos % 37 != 0 && pos != NPOS - 1) continue;   // check a subset of positions
            CHECK_CUDA(cudaStreamSynchronize(st));
            // dequantized cache as actually stored on the GPU (kernel-correctness reference)
            std::vector<double> KQ((size_t)(pos + 1) * AKV * AD), VQ(KQ.size());
            {
                std::vector<float> tmp(AD);
                for (int g = 0; g < AKV; g++)
                    for (int j = 0; j <= pos; j++)
                        for (int pass = 0; pass < 2; pass++) {
                            read_row(pass ? kv.v : kv.k, pass ? kv.vs : kv.ks, (size_t)g * kv.capacity + j, tmp.data());
                            double* dstp = (pass ? VQ.data() : KQ.data()) + ((size_t)j * AKV + g) * AD;
                            for (int d = 0; d < AD; d++) dstp[d] = tmp[d];
                        }
            }
            auto attend = [&](const std::vector<double>& K, const std::vector<double>& V) {
                std::vector<double> out(AH * AD);
                for (int h = 0; h < AH; h++) {
                    std::vector<double> qr;
                    norm_rope(&qg[h * 512], qn, qr);
                    const int g = h / 6;
                    std::vector<double> sc(pos + 1);
                    double mx = -1e300;
                    for (int j = 0; j <= pos; j++) {
                        double s = 0;
                        for (int d = 0; d < AD; d++) s += qr[d] * K[((size_t)j * AKV + g) * AD + d];
                        sc[j] = s / 16.0;
                        mx = std::max(mx, sc[j]);
                    }
                    double den = 0;
                    for (int j = 0; j <= pos; j++) { sc[j] = std::exp(sc[j] - mx); den += sc[j]; }
                    for (int d = 0; d < AD; d++) {
                        double acc = 0;
                        for (int j = 0; j <= pos; j++) acc += sc[j] * V[((size_t)j * AKV + g) * AD + d];
                        out[h * AD + d] = acc / den * sig(qg[h * 512 + 256 + d]);
                    }
                }
                return out;
            };
            const std::vector<float> got = host(dout, AH * AD);
            const std::vector<double> ref_q = attend(KQ, VQ);
            worst = std::max(worst, rel_err(got, attend(KC, VC)));
            worst_kernel = std::max(worst_kernel, rel_err(got, ref_q));
            worst_simple = std::max(worst_simple, rel_err(host(dsimple, AH * AD), ref_q));
        }
        {
            char name[96];
            const char* tn = kvt == cuda::KVType::F16 ? "F16" : kvt == cuda::KVType::Q8_0 ? "Q8_0" : "Q4_0";
            snprintf(name, sizeof name, "attention (tensor cores, fp16) vs stored-KV ref, KV %s", tn);
            check(name, worst_kernel, 2e-3);
            snprintf(name, sizeof name, "attention (fp32 simple) vs stored-KV ref, KV %s", tn);
            check(name, worst_simple, 5e-5);
            snprintf(name, sizeof name, "attention 3 queries/call == single calls, KV %s (bad of %d)", tn, multi_checked);
            check(name, multi_bad, 0);
            if (kvt != cuda::KVType::F16) {
                snprintf(name, sizeof name, "V rows == ggml quantizer values, KV %s (bad rows)", tn);
                check(name, vbytes_bad, 0);
            }
        }
        // format error vs exact fp64 KV (property of the KV format on random data; informational bound)
        const double tol = kvt == cuda::KVType::F16 ? 3e-3 : kvt == cuda::KVType::Q8_0 ? 3e-2 : 3e-1;
        char name[64];
        snprintf(name, sizeof name, "attention vs exact KV (format error), KV %s", kvt == cuda::KVType::F16 ? "F16" : kvt == cuda::KVType::Q8_0 ? "Q8_0" : "Q4_0");
        check(name, worst, tol);
    }

    // ---------------- attention beyond 131072 positions (the kernel's old limit: decode then skipped attention) ------
    {
        const int NPOS = 140000, AH = 24, AKV = 4, AD = 256, needle = 135000;
        const cuda::KVType kvt = cuda::KVType::Q8_0;
        const size_t rb = cuda::kv_value_bytes(kvt);
        cuda::KVLayer kv;
        kv.capacity = NPOS;
        std::vector<float> q(AH * AD), gate(AH * AD);
        for (auto& x : q) x = nd(rng);
        for (auto& x : gate) x = nd(rng);
        // random K / V rows, quantized with ggml's Q8_0 quantizer into the split layout (values | fp16 scales); the K row
        // at `needle` points along each group's query so that the far end of the cache dominates the output
        std::vector<uint8_t> hk((size_t)AKV * NPOS * rb), hv(hk.size());
        std::vector<uint16_t> hks((size_t)AKV * NPOS * 8), hvs(hks.size());
        std::vector<float> row(AD);
        block_q8_0 blk[AD / 32];
        auto store = [&](std::vector<uint8_t>& vals, std::vector<uint16_t>& sc, size_t r) {
            quantize_row_q8_0_ref(row.data(), blk, AD);
            for (int b = 0; b < AD / 32; b++) {
                memcpy(&vals[r * rb + 32 * b], blk[b].qs, 32);
                memcpy(&sc[r * 8 + b], &blk[b].d, 2);
            }
        };
        for (int g = 0; g < AKV; g++)
            for (int j = 0; j < NPOS; j++) {
                for (int d = 0; d < AD; d++) row[d] = j == needle ? 0.4f * q[(g * 6) * AD + d] : nd(rng);
                store(hk, hks, (size_t)g * NPOS + j);
                for (auto& x : row) x = nd(rng);
                store(hv, hvs, (size_t)g * NPOS + j);
            }
        kv.k = dev(hk);
        kv.v = dev(hv);
        kv.ks = dev(hks);
        kv.vs = dev(hvs);
        float* dq = dev(q);
        float* dgate = dev(gate);
        float* dout;
        void* work;
        CHECK_CUDA(cudaMalloc(&dout, AH * AD * 4));
        CHECK_CUDA(cudaMalloc(&work, cuda::attn_work_bytes(NPOS)));
        cuda::attn_decode(dq, dgate, kv, kvt, NPOS, work, dout, nullptr, st);
        CHECK_CUDA(cudaStreamSynchronize(st));
        const std::vector<float> got = host(dout, AH * AD);
        // reference over the dequantized cache, in double
        auto deq = [&](const std::vector<uint8_t>& vals, const std::vector<uint16_t>& sc, size_t r, int d) {
            return (double)fp16_to_fp32(sc[r * 8 + d / 32]) * (double)(int8_t)vals[r * rb + d];
        };
        std::vector<double> ref(AH * AD);
        for (int h = 0; h < AH; h++) {
            const int g = h / 6;
            std::vector<double> sc(NPOS);
            double mx = -1e300;
            for (int j = 0; j < NPOS; j++) {
                double s = 0;
                for (int d = 0; d < AD; d++) s += q[h * AD + d] * deq(hk, hks, (size_t)g * NPOS + j, d);
                sc[j] = s / 16.0;
                mx = std::max(mx, sc[j]);
            }
            double den = 0;
            for (int j = 0; j < NPOS; j++) { sc[j] = std::exp(sc[j] - mx); den += sc[j]; }
            for (int d = 0; d < AD; d++) {
                double acc = 0;
                for (int j = 0; j < NPOS; j++) acc += sc[j] * deq(hv, hvs, (size_t)g * NPOS + j, d);
                ref[h * AD + d] = acc / den * sig(gate[h * AD + d]);
            }
        }
        check("attention at 140000 positions vs stored-KV ref, KV Q8_0", rel_err(got, ref), 2e-3);
        cudaFree(kv.k); cudaFree(kv.v); cudaFree(kv.ks); cudaFree(kv.vs); cudaFree(dq); cudaFree(dgate); cudaFree(dout);
        cudaFree(work);
    }

    // ---------------- top-k ----------------
    {
        for (int n : {248320, 100000, 4096, 1000}) {
            std::vector<float> x(n);
            for (auto& v : x) v = nd(rng) * 4;
            float* dx = dev(x);
            void* work;
            CHECK_CUDA(cudaMalloc(&work, cuda::topk_work_bytes(n)));
            float* dvals; int* didx;
            CHECK_CUDA(cudaMalloc(&dvals, 32 * 4)); CHECK_CUDA(cudaMalloc(&didx, 32 * 4));
            for (int k : {1, 20, 32}) {
                cuda::topk(dx, n, k, 1000, dvals, didx, work, st);
                CHECK_CUDA(cudaStreamSynchronize(st));
                std::vector<int> gi = host(didx, k);
                std::vector<int> order(n);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) { return x[a] > x[b]; });
                int bad = 0;
                for (int i = 0; i < k; i++) bad += gi[i] != order[i] + 1000;
                char name[64];
                snprintf(name, sizeof name, "topk n=%d k=%d (mismatched ranks)", n, k);
                check(name, bad, 0);
            }
        }
    }
    if (g_fail) { printf("test_ops: %d failures\n", g_fail); return 1; }
    printf("test_ops: all ops match\n");
    return 0;
}
