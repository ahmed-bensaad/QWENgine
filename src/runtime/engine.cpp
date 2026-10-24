/// @file engine.cpp
/// Hybrid CPU+GPU decode engine (see engine.h).

#include "runtime/engine.h"
#include "cuda/prefill.cuh"
#include <sys/mman.h>
#include <sys/stat.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include "common/utils.h"
#include "quant/quant_formats.h"

namespace q38rt {

namespace {

constexpr int D = 5120, FF = 17408, VOCAB = 248320;
constexpr int DN_CH = 10240, DN_V = 6144, DN_VH = 48;
constexpr int A_Q = 24 * 512, A_KV = 1024, A_O = 24 * 256;

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// ---------------------------------------------------------------------------
// CPU job: byte-balanced partition of row ranges of several matrices over the pool.
// ---------------------------------------------------------------------------
struct CpuSeg {
    cpu::vec_dot_fn fn = nullptr;
    cpu::vec_dot_n_fn fn_n = nullptr;
    const uint8_t* W = nullptr;
    size_t rb = 0;
    int K = 0, rows = 0;
    const block_q8_K* x = nullptr;
    float* y = nullptr;
    int nb = 1;          // tokens
    size_t xs = 0;       // activation stride between tokens (blocks)
    int y_ld = 0;        // output stride between tokens
    double row_cost = 0; // estimated seconds per row (rows are split across threads by estimated time)
};

constexpr int kChunksPerThread = 4;   // equal-time chunks per thread (Q38_GUIDED=0)
// Guided self-scheduling: chunk sizes shrink geometrically (each takes 1/(2*threads) of what is left), down to a
// floor, so a job ends with small chunks and threads finish together even when cores run at different speeds.
bool g_guided = true;
double g_guided_min_s = 4e-6;   // smallest chunk (estimated seconds)

// estimated CPU seconds per byte of a format with nb token columns (formats differ ~2x in speed)
double cpu_s_per_byte(QuantFormat t, int nb, bool re) {   // per source-format byte; re: re-encoded rows
    static const DeviceRates rates[2] = {DeviceRates::measured_defaults(0), DeviceRates::measured_defaults(1)};
    // measured GB/s of this format's kernel on this CPU
    const double g = rates[re].cpu_gbs[(int)t] > 0 ? rates[re].cpu_gbs[(int)t] : 20.0;
    // nb columns cost less than nb times one column (weights read once)
    return DeviceRates::cpu_cols_factor(t, nb, re) / (g * 1e9);
}
struct CpuJob {
    CpuSeg seg[4];
    int n = 0;
    mutable std::atomic<int> next{0};   // dynamic scheduling: next chunk to take
    // CPU range after decode streaming, or the whole host share (row split only) if !after_stream.
    // nb tokens: activations of token t at x + t*xs (blocks), outputs at y + t*y_ld.
    void add(const SplitMat& m, const block_q8_K* x, float* y, bool after_stream = true, int nb = 1, size_t xs = 0,
             int y_ld = 0) {
        // what remains for the CPU once the streamed part is taken off
        const int rows = after_stream ? m.cpu2_n() : m.cpu_n(), K = after_stream ? m.cpu2_k() : m.cpu_k();
        if (rows <= 0 || K <= 0) return;
        CpuSeg& s = seg[n++];
        // re-encoded (R4 / R2) copy of the rows: faster CPU kernels, identical results
        const bool re = m.cpu_re != nullptr;
        s.fn = re ? cpu::get_vec_dot_re(m.type) : cpu::get_vec_dot(m.type);
        s.fn_n = re ? cpu::get_vec_dot_re_n(m.type, nb) : cpu::get_vec_dot_n(m.type, nb);
        s.W = re ? (after_stream ? m.cpu2_ptr_re() : m.cpu_re) : (after_stream ? m.cpu2_ptr() : m.cpu);
        s.rb = re ? m.cpu_re_rb : m.cpu_rb;
        s.K = K;
        s.rows = rows;
        s.x = x;
        s.y = y;
        s.nb = nb;
        s.xs = xs;
        s.y_ld = y_ld;
        // estimated seconds per row: rows of different formats are balanced by time, not count
        s.row_cost = (double)(after_stream ? m.cpu2_k() : m.cpu_k()) / m.K * (double)quant_row_bytes(m.type, m.K) *
                     cpu_s_per_byte(m.type, nb, re);
    }
};
// profiling (Q38_PROFILE_WAITS): busy time of each thread in the current job
bool g_prof_jobs = false;
double g_thread_busy[64], g_thread_t0[64], g_thread_t1[64];
int g_thread_chunks[64];
// Rows of all segments laid end to end by estimated time, cut into kChunks equal-time chunks taken dynamically
// (an atomic counter): threads that finish early take more, so mixed formats and thermal noise balance out.
void cpu_job_task(void* ctx, int tid, int nth) {
    const double t_start = g_prof_jobs ? now_s() : 0.0;
    struct Busy {
        double t0; int tid;
        ~Busy() { if (g_prof_jobs) { const double t = now_s(); g_thread_busy[tid] = t - t0; g_thread_t0[tid] = t0; g_thread_t1[tid] = t; } }
    } busy{t_start, tid};
    const CpuJob& j = *(const CpuJob*)ctx;
    if (g_prof_jobs) g_thread_chunks[tid] = 0;
    double total = 0;
    // the whole job in estimated seconds
    for (int i = 0; i < j.n; i++) total += (double)j.seg[i].rows * j.seg[i].row_cost;
    // chunk ch covers [bound(ch), bound(ch + 1)) of the estimated time
    int nch = nth * kChunksPerThread, ngeo = 0;
    double lq = 0, tmin = 0, bgeo = 0;
    if (g_guided && total > 0) {
        lq = std::log1p(-1.0 / (2 * nth));                                   // log of the shrink factor
        tmin = std::max(g_guided_min_s, total / 512);
        // number of chunks before the geometric sizes fall below tmin
        ngeo = std::max(0, (int)std::floor(std::log(tmin * 2 * nth / total) / lq) + 1);   // geometric chunks >= tmin
        bgeo = total * -std::expm1(ngeo * lq);
        nch = ngeo + (int)std::ceil((total - bgeo) / tmin - 1e-9);   // then equal chunks of tmin to the end
    }
    auto bound = [&](int c) {
        if (!g_guided || total <= 0) return total * c / (nth * kChunksPerThread);
        if (c <= ngeo) return total * -std::expm1(c * lq);
        return std::min(total, bgeo + (c - ngeo) * tmin);
    };
    // dynamic scheduling: each thread takes the next chunk when it is free
    for (int ch = j.next.fetch_add(1, std::memory_order_relaxed); ch < nch; ch = j.next.fetch_add(1, std::memory_order_relaxed)) {
        if (g_prof_jobs) g_thread_chunks[tid]++;
        const double b0 = bound(ch), b1 = ch + 1 == nch ? total : bound(ch + 1);
        double acc = 0;
        for (int i = 0; i < j.n; i++) {
            const CpuSeg& s = j.seg[i];
            const double sb = (double)s.rows * s.row_cost;
            if (b1 > acc && b0 < acc + sb) {
                // the chunk's time interval mapped back to rows of this segment
                const int r0 = std::clamp((int)std::lround((b0 - acc) / s.row_cost), 0, s.rows);
                const int r1 = std::clamp((int)std::lround((b1 - acc) / s.row_cost), 0, s.rows);
                if (s.nb == 1) {
                    // one dot product per weight row
                    for (int r = r0; r < r1; r++) s.y[r] = s.fn(s.K, s.W + (size_t)r * s.rb, s.x);
                } else {
                    float out[Engine::MAX_BATCH];
                    for (int r = r0; r < r1; r++) {
                        // row read once, dotted with every token's activations
                        s.fn_n(s.K, s.W + (size_t)r * s.rb, s.x, s.xs, out);
                        for (int c = 0; c < s.nb; c++) s.y[(size_t)c * s.y_ld + r] = out[c];
                    }
                }
            }
            acc += sb;
        }
    }
}

// Per-thread top-k over a slice of the CPU logits (k <= 32), merged on the caller.
struct TopkCtx {
    const float* x;      // logits of the CPU share
    int n, k, offset;    // count, k, vocabulary index of x[0]
    float vals[16][32];
    int ids[16][32];
};
void topk_task(void* ctx, int tid, int nth) {
    TopkCtx& c = *(TopkCtx*)ctx;
    const int i0 = (int)((int64_t)c.n * tid / nth), i1 = (int)((int64_t)c.n * (tid + 1) / nth);
    float* v = c.vals[tid];
    int* id = c.ids[tid];
    int cnt = 0;
    for (int i = i0; i < i1; i++) {
        const float x = c.x[i];
        if (cnt == c.k && !(x > v[c.k - 1])) continue;   // not better than the current k-th: skip
        int j = cnt < c.k ? cnt++ : c.k - 1;
        while (j > 0 && v[j - 1] < x) { v[j] = v[j - 1]; id[j] = id[j - 1]; j--; }   // insertion into the sorted list
        v[j] = x;
        id[j] = c.offset + i;
    }
    for (int j = cnt; j < c.k; j++) { v[j] = -3.0e38f; id[j] = -1; }
}

struct SwigluCtx { const float* g; const float* u; float* h; int n; };
void swiglu_task(void* ctx, int tid, int nth) {
    SwigluCtx& c = *(SwigluCtx*)ctx;
    const int i0 = (int)((int64_t)c.n * tid / nth), i1 = (int)((int64_t)c.n * (tid + 1) / nth);
    for (int i = i0; i < i1; i++) c.h[i] = c.g[i] / (1.0f + expf(-c.g[i])) * c.u[i];
}

} // namespace

Engine::Engine() {}

Engine::~Engine() {
    teardown();
    if (h_snap_) cudaFreeHost(h_snap_);
}

void Engine::teardown() {
    if (st_) cudaStreamSynchronize(st_);
    if (cst_) cudaStreamSynchronize(cst_);
    prefill_free();
    if (cublas_) { cuda::cublas_destroy(cublas_); cublas_ = nullptr; }
    if (d_arena_) cudaFree(d_arena_);
    if (d_misc_) cudaFree(d_misc_);
    if (host_registered_) cudaHostUnregister(h_weights_);
    if (h_weights_) munmap(h_weights_, h_weights_size_);
    if (h_re_) munmap(h_re_, h_re_size_);
    h_re_ = nullptr;
    h_re_size_ = 0;
    for (int i = 0; i < 2; i++) {
        if (ev_copied_[i]) cudaEventDestroy(ev_copied_[i]);
        if (ev_used_[i]) cudaEventDestroy(ev_used_[i]);
        if (ev_emb_[i]) cudaEventDestroy(ev_emb_[i]);
        ev_copied_[i] = ev_used_[i] = ev_emb_[i] = nullptr;
    }
    for (int i = 0; i < NSLOT; i++) {
        if (ev_scopied_[i]) cudaEventDestroy(ev_scopied_[i]);
        if (ev_sused_[i]) cudaEventDestroy(ev_sused_[i]);
        ev_scopied_[i] = ev_sused_[i] = nullptr;
        sslot_used_[i] = false;
    }
    if (d_slots_) cudaFree(d_slots_);
    if (h_strdup_) cudaFreeHost(h_strdup_);
    if (cst_) cudaStreamDestroy(cst_);
    if (h_base_) cudaFreeHost(h_base_);
    if (ev_) cudaEventDestroy(ev_);
    if (ev_lmtopk_) cudaEventDestroy(ev_lmtopk_);
    ev_lmtopk_ = nullptr;
    for (cudaEvent_t& e : ev_g_) { if (e) cudaEventDestroy(e); e = nullptr; }
    for (cudaEvent_t e : sprof_ev_) if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : gprof_ev_) if (e) cudaEventDestroy(e);
    if (st_) cudaStreamDestroy(st_);
    d_arena_ = d_misc_ = d_slots_ = h_weights_ = h_strdup_ = h_base_ = nullptr;
    host_registered_ = false;
    h_weights_size_ = 0;
    cst_ = st_ = nullptr;
    ev_ = nullptr;
    sprof_ev_.clear(); sprof_stall_.clear(); gprof_ev_.clear(); gprof_site_.clear(); gcpu_last_.clear(); ggpu_last_.clear();
    gprof_n_ = 0;
    spieces_.clear(); pieces_.clear();
    sseq_ = sissued_ = 0;
    L_.clear();
    lm_ = SplitMat();
    plan_ = Plan();
    vram_used_ = host_bytes_ = 0;
}

int Engine::max_context_limit() const {
    // 262144: the training context, also the attention kernels' limit
    return std::min((int)gguf_.get_config().train_context, cuda::attn_max_positions());
}

bool Engine::resize(int new_ctx) {
    if (new_ctx <= cfg_.max_context) return true;
    if (new_ctx > max_context_limit()) { LOG_ERROR("resize: %d positions exceed the limit of %d", new_ctx, max_context_limit()); return false; }
    if (spec_open_) { LOG_ERROR("resize: speculative pass not committed"); return false; }
    discard_predraft();
    flush_pending();
    CHECK_CUDA(cudaStreamSynchronize(st_));
    const double t0 = now_s();
    // ---- park the live state in host memory ----
    const int P = pos_;
    const size_t vbytes = cuda::kv_value_bytes(cfg_.kv_type), sbytes = cuda::kv_scale_bytes(cfg_.kv_type);
    const int old_cap = cfg_.max_context;
    struct LayerState { std::vector<uint8_t> k, v, ks, vs, conv, ssm; };
    std::vector<LayerState> st(L_.size());
    for (size_t il = 0; il < L_.size(); il++) {
        const LayerWeights& w = L_[il];
        LayerState& ls = st[il];
        if (w.full) {
            ls.k.resize((size_t)4 * P * vbytes); ls.v.resize(ls.k.size());   // only the P positions in use, per KV head
            ls.ks.resize((size_t)4 * P * sbytes); ls.vs.resize(ls.ks.size());
            for (int h = 0; h < 4 && P > 0; h++) {
                // rows of head h start at h * capacity: copy the used prefix
                CHECK_CUDA(cudaMemcpy(ls.k.data() + (size_t)h * P * vbytes, w.kv.k + (size_t)h * old_cap * vbytes, (size_t)P * vbytes, cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(ls.v.data() + (size_t)h * P * vbytes, w.kv.v + (size_t)h * old_cap * vbytes, (size_t)P * vbytes, cudaMemcpyDeviceToHost));
                if (sbytes) {
                    CHECK_CUDA(cudaMemcpy(ls.ks.data() + (size_t)h * P * sbytes, (const uint8_t*)w.kv.ks + (size_t)h * old_cap * sbytes, (size_t)P * sbytes, cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(ls.vs.data() + (size_t)h * P * sbytes, (const uint8_t*)w.kv.vs + (size_t)h * old_cap * sbytes, (size_t)P * sbytes, cudaMemcpyDeviceToHost));
                }
            }
        } else {
            ls.conv.resize((size_t)2 * 3 * DN_CH * 4);   // both conv window buffers (the current one is conv_cur_)
            ls.ssm.resize((size_t)DN_VH * 128 * 128 * 4);
            CHECK_CUDA(cudaMemcpy(ls.conv.data(), w.dn.conv_state, ls.conv.size(), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(ls.ssm.data(), w.dn.ssm_state, ls.ssm.size(), cudaMemcpyDeviceToHost));
        }
    }
    std::vector<float> hring((size_t)HRING * D);
    if (has_mtp_) CHECK_CUDA(cudaMemcpy(hring.data(), d_hring_, hring.size() * 4, cudaMemcpyDeviceToHost));
    const int conv_cur = conv_cur_, mtp_pos = mtp_pos_, rope_delta = rope_delta_;
    const std::vector<ImageSpan> images = images_;
    const DecodeStats stats = stats_;
    const bool snap_valid = snap_valid_;   // init() resets; the snapshot stays valid (KV rows are carried over)
    // ---- rebuild for the new capacity ----
    teardown();   // free everything: the new KV size changes the weight plan
    EngineConfig c2 = cfg_;
    c2.max_context = new_ctx;
    if (!init(c2)) return false;   // re-plan and reload the weights for the smaller budget
    for (size_t il = 0; il < L_.size(); il++) {
        LayerWeights& w = L_[il];
        const LayerState& ls = st[il];
        if (w.full) {
            for (int h = 0; h < 4 && P > 0; h++) {
                // same rows at the new head stride
                CHECK_CUDA(cudaMemcpy(w.kv.k + (size_t)h * new_ctx * vbytes, ls.k.data() + (size_t)h * P * vbytes, (size_t)P * vbytes, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(w.kv.v + (size_t)h * new_ctx * vbytes, ls.v.data() + (size_t)h * P * vbytes, (size_t)P * vbytes, cudaMemcpyHostToDevice));
                if (sbytes) {
                    CHECK_CUDA(cudaMemcpy((uint8_t*)w.kv.ks + (size_t)h * new_ctx * sbytes, ls.ks.data() + (size_t)h * P * sbytes, (size_t)P * sbytes, cudaMemcpyHostToDevice));
                    CHECK_CUDA(cudaMemcpy((uint8_t*)w.kv.vs + (size_t)h * new_ctx * sbytes, ls.vs.data() + (size_t)h * P * sbytes, (size_t)P * sbytes, cudaMemcpyHostToDevice));
                }
            }
        } else {
            CHECK_CUDA(cudaMemcpy(w.dn.conv_state, ls.conv.data(), ls.conv.size(), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(w.dn.ssm_state, ls.ssm.data(), ls.ssm.size(), cudaMemcpyHostToDevice));
        }
    }
    if (has_mtp_) CHECK_CUDA(cudaMemcpy(d_hring_, hring.data(), hring.size() * 4, cudaMemcpyHostToDevice));
    pos_ = P;
    conv_cur_ = conv_cur;
    mtp_pos_ = mtp_pos;
    images_ = images;
    rope_delta_ = rope_delta;
    stats_ = stats;
    snap_valid_ = snap_valid;
    if (cfg_.verbose) LOG_INFO("context resized %d -> %d positions at position %d in %.1f s", old_cap, new_ctx, P, now_s() - t0);
    return true;
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

const TensorInfo& Engine::wt(const std::string& name) const {
    auto it = woverride_.find(name);
    // substituted tensors (the requantized MTP block) take precedence
    return it != woverride_.end() ? it->second.info : *gguf_.find_tensor(name);
}

const uint8_t* Engine::wdata(const TensorInfo& t) const {
    auto it = woverride_.find(t.name);
    return it != woverride_.end() ? it->second.data.data() : gguf_.tensor_data(t);
}

// MTP block Q6_K -> Q4_K (ggml's reference quantizer on the exact Q6_K values), once per process, threaded.
void Engine::requantize_mtp() {
    if (!woverride_.empty()) return;
    const double t0 = now_s();
    const std::string p = "blk." + std::to_string(n_trunk_) + ".";
    size_t in_bytes = 0, out_bytes = 0;
    // disk cache in the repo's data/cache (gitignored): <model basename>.mtp_q4k, validated by the model file size
    // and the tensor sizes
    const std::string base = cfg_.model_path.substr(cfg_.model_path.find_last_of('/') + 1);
    const std::string cache_dir = std::string(Q38_DATA_DIR) + "/cache";
    mkdir(cache_dir.c_str(), 0755);
    const std::string cache_path = cache_dir + "/" + base + ".mtp_q4k";
    const uint64_t model_size = gguf_.file_size();
    auto try_load = [&]() -> bool {
        FILE* f = fopen(cache_path.c_str(), "rb");
        if (!f) return false;
        char magic[8] = {};
        uint64_t msize = 0, count = 0;
        // cache valid only for the same model file (size check)
        bool ok = fread(magic, 1, 8, f) == 8 && !memcmp(magic, "Q38MTPQ4", 8) && fread(&msize, 8, 1, f) == 1 &&
                  msize == model_size && fread(&count, 8, 1, f) == 1 && count > 0 && count < 64;
        for (uint64_t i = 0; ok && i < count; i++) {
            uint32_t nlen = 0;
            uint64_t bytes = 0;
            ok = fread(&nlen, 4, 1, f) == 1 && nlen < 256;
            std::string name(nlen, '\0');
            ok = ok && fread(name.data(), 1, nlen, f) == nlen && fread(&bytes, 8, 1, f) == 1;
            const TensorInfo* t = ok ? gguf_.find_tensor(name) : nullptr;
            ok = ok && t && bytes == (uint64_t)t->shape.dims[1] * quant_row_bytes(QuantFormat::Q4_K, (int)t->shape.dims[0]);
            if (!ok) break;
            WOverride& o = woverride_[name];
            o.info = *t;
            o.info.quant = QuantFormat::Q4_K;
            o.info.size_bytes = bytes;
            o.data.resize(bytes);
            ok = fread(o.data.data(), 1, bytes, f) == bytes;
            in_bytes += t->size_bytes;
            out_bytes += bytes;
        }
        fclose(f);
        if (!ok) { woverride_.clear(); in_bytes = out_bytes = 0; }
        return ok;
    };
    if (try_load()) {
        if (cfg_.verbose) LOG_INFO("MTP block Q4_K (cached %s): %.0f MiB", cache_path.c_str(), out_bytes / 1048576.0);
        return;
    }
    for (const char* nm : {"nextn.eh_proj.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                           "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
        const TensorInfo* t = gguf_.find_tensor(p + nm);
        if (!t || t->quant != QuantFormat::Q6_K) continue;   // only Q6_K MTP tensors are converted
        WOverride& o = woverride_[t->name];
        o.info = *t;
        o.info.quant = QuantFormat::Q4_K;
        const int K = (int)t->shape.dims[0], N = (int)t->shape.dims[1];
        const size_t irb = quant_row_bytes(QuantFormat::Q6_K, K), orb = quant_row_bytes(QuantFormat::Q4_K, K);
        o.info.size_bytes = (size_t)N * orb;
        o.data.resize(o.info.size_bytes);
        struct Ctx { const uint8_t* src; uint8_t* dst; size_t irb, orb; int K, N; } c{gguf_.tensor_data(*t), o.data.data(), irb, orb, K, N};
        pool_->run([](void* vc, int tid, int nth) {
            const Ctx& c = *(const Ctx*)vc;
            std::vector<float> row(c.K);
            for (int r = (int)((int64_t)c.N * tid / nth); r < (int)((int64_t)c.N * (tid + 1) / nth); r++) {
                // exact Q6_K values ...
                dequantize_row_ref(QuantFormat::Q6_K, c.src + (size_t)r * c.irb, row.data(), c.K);
                // ... requantized with ggml's reference Q4_K quantizer: 31% fewer bytes per draft step
                quantize_row_q4_K_ref(row.data(), (block_q4_K*)(c.dst + (size_t)r * c.orb), c.K);
            }
        }, &c);
        in_bytes += t->size_bytes;
        out_bytes += o.info.size_bytes;
    }
    if (cfg_.verbose && !woverride_.empty())
        LOG_INFO("MTP block requantized Q6_K -> Q4_K: %.0f -> %.0f MiB in %.1f s", in_bytes / 1048576.0, out_bytes / 1048576.0,
                 now_s() - t0);
    if (FILE* f = fopen(cache_path.c_str(), "wb")) {   // best effort
        const uint64_t count = woverride_.size();
        fwrite("Q38MTPQ4", 1, 8, f);
        fwrite(&model_size, 8, 1, f);
        fwrite(&count, 8, 1, f);
        for (const auto& kv : woverride_) {
            const uint32_t nlen = (uint32_t)kv.first.size();
            const uint64_t bytes = kv.second.data.size();
            fwrite(&nlen, 4, 1, f);
            fwrite(kv.first.data(), 1, nlen, f);
            fwrite(&bytes, 8, 1, f);
            fwrite(kv.second.data.data(), 1, bytes, f);
        }
        fclose(f);
    }
}

bool Engine::build_plan(double budget) {
    const DeviceRates rates = DeviceRates::measured_defaults();
    plan_ = Plan();
    plan_.budget_bytes = budget;
    const ModelConfig& c = gguf_.get_config();
    auto item = [&](const char* name, int group, int units, int unit_rows, double bytes_total, double cpu_s_total,
                    double gpu_s_total) {
        PlanItem it;
        it.name = name;
        it.group = group;
        it.units = units;
        it.unit_rows = unit_rows;
        it.bytes_per_unit = bytes_total / units;
        it.cpu_s_per_unit = cpu_s_total / units;
        it.gpu_s_per_unit = gpu_s_total / units;
        plan_.items.push_back(it);
    };
    // speculative planning: trunk + LM head run once per cycle on k+1 token columns, the MTP block k times on one
    // costs for the expected pass width: k drafts + 1 verified token
    const int ncols = has_mtp_ && cfg_.spec_k > 0 ? std::min(cfg_.spec_k + 1, MAX_BATCH) : 1;
    auto cost = [&](const TensorInfo& t, double& b, double& cs, double& gs) {
        b = (double)t.size_bytes;
        const int n = t.layer_index >= n_trunk_ ? 1 : ncols;   // the MTP block always runs one token per step
        // CPU seconds if all of it were computed on the CPU
        cs = b / (rates.cpu_gbs[(int)t.quant] * 1e9) * DeviceRates::cpu_cols_factor(t.quant, n);
        // GPU seconds if all of it were resident
        gs = b / (rates.gpu_gbs[(int)t.quant] * 1e9) * DeviceRates::gpu_cols_factor(n);
    };
    int group = 0;
    for (int il = 0; il < c.num_layers; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        double b, cs, gs;
        const bool full = c.is_full_attention(il);
        // group A: input projections; GPU-only fixed work of the layer core counted here
        // fixed GPU work of the group (norms, attention / DeltaNet core)
        plan_.groups.push_back({full ? 60e-6 : 40e-6});
        // NB: never select between two initializer_lists with ?: — the backing array dangles (UB).
        const std::vector<const char*> group_a = full ? std::vector<const char*>{"attn_q.weight", "attn_k.weight", "attn_v.weight"}
                                                      : std::vector<const char*>{"attn_qkv.weight", "attn_gate.weight"};
        for (const char* nm : group_a) {
            const TensorInfo& t = wt(p + nm);
            cost(t, b, cs, gs);
            item((p + nm).c_str(), group, (int)t.shape.dims[1], 1, b, cs, gs);
        }
        group++;
        // group B: output projection
        plan_.groups.push_back({10e-6});
        {
            const std::string nm = full ? "attn_output.weight" : "ssm_out.weight";
            const TensorInfo& t = wt(p + nm);
            cost(t, b, cs, gs);
            item((p + nm).c_str(), group, (int)t.shape.dims[1], 1, b, cs, gs);
        }
        group++;
        // group C: FFN, split in 256-wide blocks of the intermediate dimension
        plan_.groups.push_back({10e-6});
        {
            double bt = 0, ct = 0, gt = 0;
            for (const char* nm : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
                cost(wt(p + nm), b, cs, gs);
                bt += b; ct += cs; gt += gs;
            }
            // gate, up and down move together in blocks of 256 intermediate rows
            item((p + "ffn").c_str(), group, FF / 256, 256, bt, ct, gt);
        }
        group++;
    }
    plan_.groups.push_back({20e-6});
    {
        double b, cs, gs;
        const TensorInfo& t = wt("output.weight");
        cost(t, b, cs, gs);
        item("output.weight", group, VOCAB / 64, 64, b, cs, gs);   // 64-row units
        // a resident vocabulary prefix of the LM head: read by the MTP draft head, and its VRAM is what prefill borrows
        // for its buffers (prefill_alloc), so it is kept with or without the MTP block
        plan_.items.back().min_gpu_units = std::min(cfg_.draft_rows, VOCAB) / 64;
    }
    group++;
    if (has_mtp_) {   // MTP block: eh_proj | q k v | o | FFN, once per draft step
        const std::string p = "blk." + std::to_string(n_trunk_) + ".";
        const double mw = cfg_.spec_k;   // MTP passes per cycle (0: not planned for, gets no VRAM)
        // overlapped drafting runs the MTP block while the pool computes the LM head: no CPU share
        const bool mtp_vram = cfg_.mtp_overlap && cfg_.spec_k > 0;
        auto add = [&](const std::vector<const char*>& names, double fixed) {
            plan_.groups.push_back({fixed, mw});
            for (const char* nm : names) {
                double b, cs, gs;
                const TensorInfo& t = wt(p + nm);
                cost(t, b, cs, gs);
                item((p + nm).c_str(), group, (int)t.shape.dims[1], 1, b, cs, gs);
                // fully resident: drafting runs on the GPU while the CPU computes the LM head
                if (mtp_vram) plan_.items.back().min_gpu_units = plan_.items.back().units;
            }
            group++;
        };
        add({"nextn.eh_proj.weight"}, 10e-6);
        add({"attn_q.weight", "attn_k.weight", "attn_v.weight"}, 60e-6);
        add({"attn_output.weight"}, 10e-6);
        plan_.groups.push_back({10e-6, mw});
        double bt = 0, ct = 0, gt = 0;
        for (const char* nm : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
            double b, cs, gs;
            cost(wt(p + nm), b, cs, gs);
            bt += b; ct += cs; gt += gs;
        }
        item((p + "ffn").c_str(), group, FF / 256, 256, bt, ct, gt);
        if (mtp_vram) plan_.items.back().min_gpu_units = plan_.items.back().units;
        group++;
    }
    plan_residency(plan_);   // greedy placement: the bytes that save the most time per byte go to VRAM first
    return true;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

// Re-encoded copies of the IQ3 (R4) / IQ2 (R2) host shares: the CPU kernels decode them without grid
// gathers (bit-identical results, 20-40% faster with 2-4 token columns). The source rows stay in place for decode
// streaming and prefill. Built in parallel at load (~0.5 s).
void Engine::build_re() {
    auto wanted = [&](QuantFormat t) {
        const bool iq3 = t == QuantFormat::IQ3_S || t == QuantFormat::IQ3_XXS;
        return cpu::re_supported(t) && (iq3 ? cfg_.cpu_r4 : cfg_.cpu_r2);
    };
    std::vector<SplitMat*> ms;
    for (LayerWeights& w : L_)
        for (SplitMat* m : {&w.q, &w.k, &w.v, &w.o, &w.qkv, &w.z, &w.ssm_out, &w.gate, &w.up, &w.down, &w.eh_proj})
            // only host shares are re-encoded: the GPU uses the original format
            if (m->cpu && wanted(m->type) && m->cpu_n() > 0 && m->cpu_k() > 0) ms.push_back(m);
    if (ms.empty()) return;
    size_t total = 0;
    for (SplitMat* m : ms) total += align_up((size_t)m->cpu_n() * cpu::re_row_bytes(m->type, m->cpu_k()), 64);
    h_re_size_ = align_up(total, 2 << 20);
    h_re_ = (uint8_t*)mmap(nullptr, h_re_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (h_re_ == MAP_FAILED) { h_re_ = nullptr; h_re_size_ = 0; LOG_WARN("re-encoding allocation failed: gather kernels"); return; }
    madvise(h_re_, h_re_size_, MADV_HUGEPAGE);   // 2 MB pages: fewer TLB misses while streaming through GBs of weights
    struct Row { const SplitMat* m; int r; };
    std::vector<Row> rows;
    uint8_t* cur = h_re_;
    for (SplitMat* m : ms) {
        m->cpu_re = cur;
        m->cpu_re_rb = cpu::re_row_bytes(m->type, m->cpu_k());
        cur += align_up((size_t)m->cpu_n() * m->cpu_re_rb, 64);
        for (int r = 0; r < m->cpu_n(); r++) rows.push_back({m, r});
    }
    struct Ctx { const std::vector<Row>* rows; std::atomic<int> bad{0}; } ctx{&rows};
    pool_->run([](void* p, int tid, int nth) {
        Ctx& c = *(Ctx*)p;
        const size_t n = c.rows->size(), r0 = n * tid / nth, r1 = n * (tid + 1) / nth;
        for (size_t i = r0; i < r1; i++) {
            const Row& rw = (*c.rows)[i];
            const SplitMat& m = *rw.m;
            // fails only if a weight is outside the expected grid level set
            if (!cpu::re_repack_row(m.type, m.cpu + (size_t)rw.r * m.cpu_rb, m.cpu_k(), m.cpu_re + (size_t)rw.r * m.cpu_re_rb))
                c.bad.fetch_add(1);
        }
    }, &ctx);
    if (ctx.bad.load()) { LOG_ERROR("re-encoding: %d rows with values outside the grid level sets", ctx.bad.load()); std::abort(); }
    if (cfg_.verbose) LOG_INFO("re-encoded copies of %zu IQ2/IQ3 host matrices: %.0f MiB", ms.size(), total / 1048576.0);
}

void Engine::upload_split(SplitMat& m, const TensorInfo& t, uint8_t*& dcur, uint8_t*& hcur) {
    m.name = t.name;
    m.type = t.quant;
    m.K = (int)t.shape.dims[0];
    m.N = (int)t.shape.dims[1];
    const size_t full_rb = quant_row_bytes(t.quant, m.K);
    const uint8_t* src = wdata(t);
    if (!m.col_split) {
        m.gpu_rb = m.cpu_rb = full_rb;
        if (m.gpu_part > 0) {   // row split: the first gpu_part rows to VRAM ...
            m.gpu = dcur;
            CHECK_CUDA(cudaMemcpy(m.gpu, src, (size_t)m.gpu_part * full_rb, cudaMemcpyHostToDevice));
            dcur += align_up((size_t)m.gpu_part * full_rb, 256);
        }
        if (m.gpu_part < m.N) {   // ... the remaining rows to host memory
            m.cpu = hcur;
            memcpy(m.cpu, src + (size_t)m.gpu_part * full_rb, (size_t)(m.N - m.gpu_part) * full_rb);
            hcur += align_up((size_t)(m.N - m.gpu_part) * full_rb, 64);
        }
    } else {
        // column split (FFN down): each row's first gpu_part columns to VRAM
        m.gpu_rb = quant_row_bytes(t.quant, m.gpu_part);
        m.cpu_rb = full_rb - m.gpu_rb;
        if (m.gpu_part > 0) {
            m.gpu = dcur;
            // strided copy of the leading column blocks
            CHECK_CUDA(cudaMemcpy2D(m.gpu, m.gpu_rb, src, full_rb, m.gpu_rb, m.N, cudaMemcpyHostToDevice));
            dcur += align_up((size_t)m.N * m.gpu_rb, 256);
        }
        if (m.gpu_part < m.K) {
            m.cpu = hcur;
            for (int r = 0; r < m.N; r++) memcpy(m.cpu + (size_t)r * m.cpu_rb, src + (size_t)r * full_rb + m.gpu_rb, m.cpu_rb);
            hcur += align_up((size_t)m.N * m.cpu_rb, 64);
        }
    }
}

// The kernels cover the formats of the target GGUF and its MTP variant. Other quantizations of the same model can use
// formats without kernels (Q3_K, Q5_K, Q8_0, F16, ...) or store the small tensors differently: refuse them at load,
// by name, instead of failing in a kernel.
bool Engine::check_formats() const {
    std::map<std::string, std::vector<std::string>> bad;   // expected format -> offending tensors
    for (const TensorInfo& t : gguf_.get_tensors()) {
        const std::string& n = t.name;
        const bool alpha_beta = n.find("ssm_alpha") != std::string::npos || n.find("ssm_beta") != std::string::npos;
        std::string want;
        if (t.shape.ndim == 1 || n.find("ssm_conv1d") != std::string::npos) {
            if (t.quant != QuantFormat::FP32) want = "F32";
        } else if (n == "token_embd.weight") {   // rows dequantized on the host (dequantize_row_ref)
            std::vector<float> row(t.shape.dims[0]);
            if (!dequantize_row_ref(t.quant, gguf_.tensor_data(t), row.data(), (int)row.size()))
                want = "a format of the reference dequantizer";
        } else if (alpha_beta) {
            if (t.quant != QuantFormat::BF16) want = "BF16";
        } else if (!cuda::gemv_q8_supported(t.quant)) {   // GPU and CPU kernels and the prefill dequantizer share the set
            want = "IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q2_K, Q4_K or Q6_K";
        }
        if (!want.empty()) bad[want].push_back(n + " (" + quant_name(t.quant) + ")");
    }
    for (const auto& [want, names] : bad) {
        std::string list;
        for (size_t i = 0; i < names.size() && i < 5; i++) list += (i ? ", " : "") + names[i];
        if (names.size() > 5) list += ", ... (" + std::to_string(names.size()) + " tensors)";
        LOG_ERROR("unsupported tensor format: %s; expected %s. This engine has kernels for the formats of "
                  "Qwen3.8-27B-GSQ-RCO-IQ3_XXS only (see the README's FAQ)", list.c_str(), want.c_str());
    }
    return bad.empty();
}

bool Engine::init(const EngineConfig& cfg) {
    cfg_ = cfg;
    // planner + CPU job costs use the re-encoded rates for IQ2 / IQ3 host rows (Q38_PLAN_NO_RE: plan with the gather
    // rates while running re-encoded kernels, to compare both kernels under one plan)
    if (getenv("Q38_PLAN_NO_RE")) DeviceRates::set_cpu_reencode(false, false);
    else DeviceRates::set_cpu_reencode(cfg.cpu_r4, cfg.cpu_r2);
    if (!parsed_) {   // first init: device flags, context, GGUF (kept across resize())
        // spin-wait on events: lower latency than yielding, the decode loop syncs ~50 times per token
        CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        CHECK_CUDA(cudaFree(nullptr));   // create context
        if (!gguf_.parse(cfg.model_path) || !check_formats()) return false;
        parsed_ = true;
    }
    const ModelConfig& c = gguf_.get_config();
    if (cfg.max_context > max_context_limit()) {
        LOG_ERROR("context of %d positions exceeds the limit of %d (model training context %d, attention kernels %d)",
                  cfg.max_context, max_context_limit(), c.train_context, cuda::attn_max_positions());
        return false;
    }
    tok_embd_ = gguf_.find_tensor("token_embd.weight");
    CHECK_CUDA(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventCreateWithFlags(&ev_lmtopk_, cudaEventDisableTiming));
    h_xq_lm_.resize((size_t)MAX_BATCH * (D / QK_K));
    // cuBLAS (prefill GEMMs) lives as long as the engine: created before the VRAM plan so its ~12 MiB are accounted
    if (!cublas_ && !(cublas_ = cuda::cublas_create())) return false;
    if (const char* e = getenv("Q38_PROFILE_WAITS")) prof_waits_ = atoi(e) != 0;
    if (const char* e = getenv("Q38_GUIDED")) g_guided = atoi(e) != 0;
    if (const char* e = getenv("Q38_GUIDED_MIN_US")) g_guided_min_s = atof(e) * 1e-6;
    g_prof_jobs = prof_waits_;
    for (cudaEvent_t& e : ev_g_) CHECK_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    if (!pool_) pool_.reset(new cpu::ThreadPool(cfg.cpu_threads));

    // ---- fixed device allocations (state, KV, activations) ----
    // 4 KV heads x capacity rows: allocated up front for the whole context
    const size_t kv_vals = (size_t)4 * cfg.max_context * cuda::kv_value_bytes(cfg.kv_type);   // per K or V
    const size_t kv_scales = (size_t)4 * cfg.max_context * 8 * 2;
    size_t misc = 0;
    auto reserve = [&](size_t bytes) { size_t off = misc; misc += align_up(bytes, 256); return off; };
    n_trunk_ = c.num_layers;
    has_mtp_ = cfg.mtp && c.num_nextn_layers > 0;
    if (has_mtp_ && cfg.mtp_q4k) requantize_mtp();   // before planning: the plan sizes the Q4_K copies
    const int n_all = n_trunk_ + (has_mtp_ ? 1 : 0);
    auto is_attn = [&](int il) { return il >= n_trunk_ || c.is_full_attention(il); };   // the MTP block attends
    std::vector<size_t> off_conv(n_all), off_ssm(n_all), off_k(n_all), off_v(n_all), off_ks(n_all), off_vs(n_all);
    for (int il = 0; il < n_all; il++) {
        if (is_attn(il)) {
            off_k[il] = reserve(kv_vals); off_v[il] = reserve(kv_vals);
            if (cuda::kv_scale_bytes(cfg.kv_type)) { off_ks[il] = reserve(kv_scales); off_vs[il] = reserve(kv_scales); }
        }
        // DeltaNet layer: 2 conv windows (ping-pong) + the 48 x 128 x 128 FP32 state
        else { off_conv[il] = reserve(2 * 3 * DN_CH * 4); off_ssm[il] = reserve((size_t)DN_VH * 128 * 128 * 4); }
    }
    // activations: MAX_BATCH token rows each (decode_batch)
    const size_t MB = MAX_BATCH;
    const size_t o_x = reserve(MB * D * 4), o_xn = reserve(MB * D * 4), o_qkv = reserve(MB * DN_CH * 4),
                 o_z = reserve(MB * DN_V * 4), o_beta = reserve(MB * 64 * 4), o_alpha = reserve(MB * 64 * 4),
                 o_o = reserve(MB * A_O * 4), o_y = reserve(MB * D * 4), o_y2 = reserve(MB * D * 4),
                 o_g = reserve(MB * FF * 4), o_u = reserve(MB * FF * 4), o_qg = reserve(MB * A_Q * 4),
                 o_k = reserve(MB * A_KV * 4), o_v = reserve(MB * A_KV * 4), o_q = reserve(MB * A_O * 4),
                 o_gate = reserve(MB * A_O * 4), o_logits = reserve(MB * VOCAB * 4), o_tkv = reserve(MB * 64 * 4),
                 o_tki = reserve(MB * 64 * 4), o_attw = reserve(cuda::attn_work_bytes(cfg.max_context)),
                 o_tkw = reserve(cuda::topk_work_bytes(VOCAB));
    const size_t o_ax = reserve(MB * (D + D / 32 * 8)), o_ao = reserve(MB * (A_O + A_O / 32 * 8)),
                 o_ah = reserve(MB * (FF + FF / 32 * 8));
    // inputs of uncommitted speculative tokens, replayed on commit
    const size_t o_dnsave = reserve((size_t)2 * DN_LAYERS * MB * DNS_TOK * 4);   // lazy-commit save buffers
    const size_t o_hring = reserve((size_t)HRING * D * 4), o_hmtp = reserve((size_t)D * 4);   // MTP inputs
    // small GPU-only weights: norms, ssm params, bf16 alpha/beta
    size_t small = 0;
    for (const TensorInfo& t : gguf_.get_tensors()) {
        if (t.shape.ndim == 1 || t.quant == QuantFormat::BF16 || t.name.find("ssm_conv1d") != std::string::npos)
            small += align_up(t.size_bytes, 256);
    }
    const size_t o_small = reserve(small);
    CHECK_CUDA(cudaMalloc(&d_misc_, misc));
    CHECK_CUDA(cudaMemset(d_misc_, 0, misc));
    vram_used_ = misc;

    // ---- weight budget and plan ----
    size_t free_b, total_b;
    CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
    // whatever VRAM is free after the fixed buffers goes to weights
    double budget = (double)free_b - cfg.vram_reserve_mib * 1048576.0 - 64.0 * 1048576.0;   // arena slack
    // 64 MB ring for weights streamed over PCIe during decode
    if (cfg.stream_mib > 0) budget -= (double)NSLOT * SLOT_BYTES;   // decode streaming ring
    if (cfg.gpu_weight_cap_mib >= 0) budget = std::min(budget, cfg.gpu_weight_cap_mib * 1048576.0);
    budget = std::max(budget, 0.0);
    build_plan(budget);
    if (cfg.verbose) {
        LOG_INFO("VRAM: free %.0f MiB after %.0f MiB of state/KV/activations; weight budget %.0f MiB", free_b / 1048576.0,
                 misc / 1048576.0, budget / 1048576.0);
        fprintf(stderr, "%s", plan_summary(plan_).c_str());
    }
    if (const char* e = getenv("Q38_PLAN_DUMP")) {   // items whose name contains the given text
        for (const PlanItem& it : plan_.items)
            if (it.name.find(e) != std::string::npos)
                fprintf(stderr, "plan %-36s group %3d units %6d gpu %6d (%7.2f MiB GPU, %7.2f MiB CPU) %s\n", it.name.c_str(),
                        it.group, it.units, it.gpu_units, it.gpu_units * it.bytes_per_unit / 1048576.0,
                        (it.units - it.gpu_units) * it.bytes_per_unit / 1048576.0, it.reason.c_str());
    }

    // ---- apply the plan to the split matrices ----
    L_.resize(n_all);
    std::vector<int> gpu_part_by_item(plan_.items.size());
    std::map<std::string, const PlanItem*> by_name;
    for (const PlanItem& it : plan_.items) by_name[it.name] = &it;
    size_t dev_bytes = 0, host_bytes = 0;
    auto sizes = [&](SplitMat& m, const TensorInfo& t, bool col, int part) {
        m.col_split = col;
        m.gpu_part = part;
        const int K = (int)t.shape.dims[0], N = (int)t.shape.dims[1];
        const size_t rb = quant_row_bytes(t.quant, K);
        if (!col) {
            dev_bytes += align_up((size_t)part * rb, 256);
            host_bytes += align_up((size_t)(N - part) * rb, 64);
        } else {
            const size_t grb = quant_row_bytes(t.quant, part);
            dev_bytes += align_up((size_t)N * grb, 256);
            host_bytes += align_up((size_t)N * (rb - grb), 64);
        }
    };
    for (int il = 0; il < n_all; il++) {
        LayerWeights& lw = L_[il];
        lw.full = is_attn(il);
        lw.mtp = il >= n_trunk_;
        const std::string p = "blk." + std::to_string(il) + ".";
        // GPU rows the planner gave this matrix
        auto rows_of = [&](const std::string& nm) { const PlanItem* it = by_name.at(p + nm); return it->gpu_units * it->unit_rows; };
        if (lw.mtp) sizes(lw.eh_proj, wt(p + "nextn.eh_proj.weight"), false, rows_of("nextn.eh_proj.weight"));
        if (lw.full) {
            sizes(lw.q, wt(p + "attn_q.weight"), false, rows_of("attn_q.weight"));
            sizes(lw.k, wt(p + "attn_k.weight"), false, rows_of("attn_k.weight"));
            sizes(lw.v, wt(p + "attn_v.weight"), false, rows_of("attn_v.weight"));
            sizes(lw.o, wt(p + "attn_output.weight"), false, rows_of("attn_output.weight"));
        } else {
            sizes(lw.qkv, wt(p + "attn_qkv.weight"), false, rows_of("attn_qkv.weight"));
            sizes(lw.z, wt(p + "attn_gate.weight"), false, rows_of("attn_gate.weight"));
            sizes(lw.ssm_out, wt(p + "ssm_out.weight"), false, rows_of("ssm_out.weight"));
        }
        const int fpart = rows_of("ffn");
        sizes(lw.gate, wt(p + "ffn_gate.weight"), false, fpart);
        sizes(lw.up, wt(p + "ffn_up.weight"), false, fpart);
        // down is split by columns: the same intermediate blocks as gate / up rows
        sizes(lw.down, wt(p + "ffn_down.weight"), true, fpart);
    }
    size_t lm_copy = 0;   // pinned host copy of the resident LM-head rows: prefill borrows their VRAM
    {
        const PlanItem* it = by_name.at("output.weight");
        const int part = it->gpu_units * it->unit_rows;
        sizes(lm_, wt("output.weight"), false, part);
        lm_copy = align_up((size_t)part * quant_row_bytes(wt("output.weight").quant, D), 64);
    }
    if (dev_bytes) CHECK_CUDA(cudaMalloc(&d_arena_, dev_bytes));
    vram_used_ += dev_bytes;
    // host weights in one 2 MB-aligned anonymous mapping (huge pages)
    h_weights_size_ = align_up(std::max<size_t>(host_bytes + lm_copy, 1), 2 << 20);
    h_weights_ = (uint8_t*)mmap(nullptr, h_weights_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (h_weights_ == MAP_FAILED) { LOG_ERROR("host weight allocation failed"); return false; }
    madvise(h_weights_, h_weights_size_, MADV_HUGEPAGE);
    host_bytes_ = host_bytes;

    // ---- upload ----
    const double t0 = now_s();
    uint8_t* dcur = d_arena_;
    uint8_t* hcur = h_weights_;
    for (int il = 0; il < n_all; il++) {
        LayerWeights& lw = L_[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        if (lw.mtp) upload_split(lw.eh_proj, wt(p + "nextn.eh_proj.weight"), dcur, hcur);
        if (lw.full) {
            upload_split(lw.q, wt(p + "attn_q.weight"), dcur, hcur);
            upload_split(lw.k, wt(p + "attn_k.weight"), dcur, hcur);
            upload_split(lw.v, wt(p + "attn_v.weight"), dcur, hcur);
            upload_split(lw.o, wt(p + "attn_output.weight"), dcur, hcur);
        } else {
            upload_split(lw.qkv, wt(p + "attn_qkv.weight"), dcur, hcur);
            upload_split(lw.z, wt(p + "attn_gate.weight"), dcur, hcur);
            upload_split(lw.ssm_out, wt(p + "ssm_out.weight"), dcur, hcur);
        }
        upload_split(lw.gate, wt(p + "ffn_gate.weight"), dcur, hcur);
        upload_split(lw.up, wt(p + "ffn_up.weight"), dcur, hcur);
        upload_split(lw.down, wt(p + "ffn_down.weight"), dcur, hcur);
    }
    upload_split(lm_, wt("output.weight"), dcur, hcur);
    h_lm_gpu_copy_ = nullptr;
    if (lm_copy && lm_.gpu_part > 0) {
        h_lm_gpu_copy_ = hcur;
        // copy kept to restore the LM-head rows after prefill borrows their VRAM
        memcpy(hcur, wdata(wt("output.weight")), (size_t)lm_.gpu_part * lm_.gpu_rb);
        hcur += lm_copy;
    }
    if (cfg.cpu_r4 || cfg.cpu_r2) build_re();
    // pin the CPU-resident weights: prefill (and decode streaming) copy them to the GPU with async DMA
    // pinned: DMA straight from the weight memory, no staging copy
    if (host_bytes + lm_copy > 0 && cudaHostRegister(h_weights_, h_weights_size_, cudaHostRegisterDefault) == cudaSuccess)
        host_registered_ = true;
    else
        LOG_WARN("cudaHostRegister of host weights failed: prefill copies will be synchronous");
    CHECK_CUDA(cudaStreamCreateWithFlags(&cst_, cudaStreamNonBlocking));
    if (cfg.stream_mib > 0 && host_registered_) {
        CHECK_CUDA(cudaMalloc(&d_slots_, (size_t)NSLOT * SLOT_BYTES));
        vram_used_ += (size_t)NSLOT * SLOT_BYTES;
        for (int i = 0; i < NSLOT; i++) {
            CHECK_CUDA(cudaEventCreateWithFlags(&ev_scopied_[i], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&ev_sused_[i], cudaEventDisableTiming));
        }
        plan_streaming(cfg.stream_mib * 1048576.0);   // choose which host rows are streamed to the GPU every token
        // column slices of FFN down are strided in the host share (tiny rows: DMA-hostile); keep a contiguous
        // pinned duplicate of each streamed column piece so every piece is one linear transfer
        size_t dup = 0;
        for (const SPiece& p : spieces_) if (p.m->col_split) dup += align_up(p.bytes, 256);
        if (dup) {
            CHECK_CUDA(cudaHostAlloc((void**)&h_strdup_, dup, cudaHostAllocDefault));
            uint8_t* cur = h_strdup_;
            for (SPiece& p : spieces_) {
                if (!p.m->col_split) continue;
                const SplitMat& m = *p.m;
                const size_t off_b = quant_row_bytes(m.type, p.off);
                // gather the strided column slice into one contiguous block
                for (int r = 0; r < m.N; r++) memcpy(cur + (size_t)r * p.rb, m.cpu + (size_t)r * m.cpu_rb + off_b, p.rb);
                p.src = cur;
                cur += align_up(p.bytes, 256);
            }
        }
        if (cfg.verbose) LOG_INFO("decode streaming: %zu slices, %.0f MiB per token over PCIe", spieces_.size(),
                                  streamed_bytes_per_token_ / 1048576.0);
    }

    // small GPU-only tensors
    uint8_t* scur = d_misc_ + o_small;
    auto up_small = [&](const std::string& name) -> void* {
        const TensorInfo* t = gguf_.find_tensor(name);
        if (!t) { LOG_ERROR("missing tensor %s", name.c_str()); return nullptr; }
        void* d = scur;
        CHECK_CUDA(cudaMemcpy(d, gguf_.tensor_data(*t), t->size_bytes, cudaMemcpyHostToDevice));
        scur += align_up(t->size_bytes, 256);
        return d;
    };
    for (int il = 0; il < n_all; il++) {
        LayerWeights& lw = L_[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        if (lw.mtp) {
            lw.enorm = (float*)up_small(p + "nextn.enorm.weight");
            lw.hnorm = (float*)up_small(p + "nextn.hnorm.weight");
            lw.head_norm = (float*)up_small(p + "nextn.shared_head_norm.weight");
        }
        lw.attn_norm = (float*)up_small(p + "attn_norm.weight");
        lw.post_norm = (float*)up_small(p + "post_attention_norm.weight");
        if (lw.full) {
            lw.q_norm = (float*)up_small(p + "attn_q_norm.weight");
            lw.k_norm = (float*)up_small(p + "attn_k_norm.weight");
            lw.kv.k = d_misc_ + off_k[il];
            lw.kv.v = d_misc_ + off_v[il];
            lw.kv.ks = (uint16_t*)(d_misc_ + off_ks[il]);
            lw.kv.vs = (uint16_t*)(d_misc_ + off_vs[il]);
            lw.kv.capacity = cfg.max_context;   // row stride between KV heads
        } else {
            lw.alpha = (uint16_t*)up_small(p + "ssm_alpha.weight");
            lw.beta = (uint16_t*)up_small(p + "ssm_beta.weight");
            lw.dn.conv_w = (const float*)up_small(p + "ssm_conv1d.weight");
            lw.dn.dt_bias = (const float*)up_small(p + "ssm_dt.bias");
            lw.dn.ssm_a = (const float*)up_small(p + "ssm_a");
            lw.dn.ssm_norm = (const float*)up_small(p + "ssm_norm.weight");
            lw.dn.conv_state = (float*)(d_misc_ + off_conv[il]);
            lw.dn.ssm_state = (float*)(d_misc_ + off_ssm[il]);
        }
    }
    output_norm_ = (float*)up_small("output_norm.weight");
    if (cfg.verbose) {
        LOG_INFO("weights uploaded in %.1f s: VRAM %.0f MiB used, host weights %.0f MiB", now_s() - t0,
                 vram_used_ / 1048576.0, host_bytes / 1048576.0);
        // CPU-resident bytes per format and the planner's predicted CPU time for them
        const DeviceRates rates = DeviceRates::measured_defaults();
        std::map<int, double> cpu_by_type, gpu_by_type;
        auto acc = [&](const SplitMat& m) {
            if (m.type == QuantFormat::INVALID) return;
            cpu_by_type[(int)m.type] += (double)m.cpu_n() * m.cpu_rb;
            gpu_by_type[(int)m.type] += (double)m.gpu_n() * m.gpu_rb;
        };
        for (const LayerWeights& lw : L_)
            for (const SplitMat* m : {&lw.qkv, &lw.z, &lw.ssm_out, &lw.q, &lw.k, &lw.v, &lw.o, &lw.gate, &lw.up, &lw.down}) acc(*m);
        acc(lm_);
        double tot_ms = 0;
        for (auto& kv : cpu_by_type) {
            const double ms = kv.second / (rates.cpu_gbs[kv.first] * 1e9) * 1e3;
            tot_ms += ms;
            fprintf(stderr, "  %-8s CPU %7.1f MiB (%5.1f ms predicted)  GPU %7.1f MiB\n", quant_name((QuantFormat)kv.first),
                    kv.second / 1048576.0, ms, gpu_by_type[kv.first] / 1048576.0);
        }
        fprintf(stderr, "  CPU total predicted %.1f ms/token\n", tot_ms);
    }

    // ---- activation pointers ----
    auto F = [&](size_t off) { return (float*)(d_misc_ + off); };
    d_x = F(o_x); d_xn = F(o_xn); d_qkv = F(o_qkv); d_z = F(o_z); d_beta = F(o_beta); d_alpha = F(o_alpha);
    d_o = F(o_o); d_y = F(o_y); d_y2 = F(o_y2); d_g = F(o_g); d_u = F(o_u); d_qg = F(o_qg); d_k = F(o_k);
    d_v = F(o_v); d_q = F(o_q); d_gate = F(o_gate); d_logits = F(o_logits); d_tkv = F(o_tkv);
    d_tki = (int*)(d_misc_ + o_tki);
    d_dnsave_ = F(o_dnsave);
    d_hring_ = F(o_hring);
    d_hmtp_ = F(o_hmtp);
    d_attn_work = d_misc_ + o_attw;
    d_topk_work = d_misc_ + o_tkw;
    auto act = [&](size_t off, int K) {   // MAX_BATCH rows: q [MB][K], d [MB][K/32], s [MB][K/32]
        cuda::ActQ8 a;
        a.q = (int8_t*)(d_misc_ + off);
        a.d = (float*)(d_misc_ + off + MB * K);
        a.s = a.d + MB * (K / 32);
        return a;
    };
    act_x_ = act(o_ax, D);
    act_o_ = act(o_ao, A_O);
    act_h_ = act(o_ah, FF);

    // ---- pinned host buffers (one allocation) ----
    size_t hoff = 0;
    auto hres = [&](size_t bytes) { size_t o = hoff; hoff += align_up(bytes, 64); return o; };
    const size_t hx = hres(MB * D * 4), ho = hres(MB * A_O * 4), hrows = hres(MB * A_Q * 4),
                 hrows2 = hres(MB * DN_V * 4), hg = hres(MB * FF * 4), hu = hres(MB * FF * 4), hy2 = hres(MB * D * 4),
                 hlog = hres(MB * VOCAB * 4), htkv = hres(MB * 64 * 4), htki = hres(MB * 64 * 4),
                 hxq = hres(MB * FF / 256 * sizeof(block_q8_K)), hhq = hres(MB * FF / 256 * sizeof(block_q8_K));
    uint8_t* hbase;
    CHECK_CUDA(cudaHostAlloc((void**)&hbase, hoff, cudaHostAllocMapped));   // mapped: kernels access it directly
    h_base_ = hbase;
    auto HF = [&](size_t o) { return (float*)(hbase + o); };
    h_x = HF(hx); h_o = HF(ho); h_rows = HF(hrows); h_rows2 = HF(hrows2); h_g = HF(hg); h_u = HF(hu);
    h_y2 = HF(hy2); h_logits = HF(hlog); h_tkv = HF(htkv); h_tki = (int*)(hbase + htki);
    h_xq = (block_q8_K*)(hbase + hxq);
    h_hq = (block_q8_K*)(hbase + hhq);
    reset();
    return true;
}

bool Engine::save_snapshot() {
    if (spec_open_) { LOG_ERROR("save_snapshot: speculative pass not committed"); return false; }
    discard_predraft();
    flush_pending();
    const size_t conv = (size_t)2 * 3 * DN_CH * 4, ssm = (size_t)DN_VH * 128 * 128 * 4, ring = (size_t)HRING * D * 4;
    size_t need = ring;
    // the recurrent state of the 48 DeltaNet layers (~150 MB); the KV cache needs no copy
    for (const LayerWeights& w : L_) if (!w.full) need += conv + ssm;
    if (h_snap_size_ < need) {
        if (h_snap_) cudaFreeHost(h_snap_);
        h_snap_ = nullptr;
        if (cudaHostAlloc((void**)&h_snap_, need, cudaHostAllocDefault) != cudaSuccess) {
            (void)cudaGetLastError();
            h_snap_ = nullptr;
            h_snap_size_ = 0;
            snap_valid_ = false;
            LOG_WARN("save_snapshot: cannot allocate %.0f MiB of pinned memory", need / 1048576.0);
            return false;
        }
        h_snap_size_ = need;
    }
    uint8_t* cur = h_snap_;
    for (const LayerWeights& w : L_) {
        if (w.full) continue;
        // both conv buffers: conv_cur_ says which is current
        CHECK_CUDA(cudaMemcpyAsync(cur, w.dn.conv_state, conv, cudaMemcpyDeviceToHost, st_)); cur += conv;
        CHECK_CUDA(cudaMemcpyAsync(cur, w.dn.ssm_state, ssm, cudaMemcpyDeviceToHost, st_)); cur += ssm;
    }
    if (has_mtp_) CHECK_CUDA(cudaMemcpyAsync(cur, d_hring_, ring, cudaMemcpyDeviceToHost, st_));
    CHECK_CUDA(cudaStreamSynchronize(st_));
    snap_pos_ = pos_;
    snap_conv_cur_ = conv_cur_;
    snap_mtp_pos_ = mtp_pos_;
    snap_images_ = images_;
    snap_rope_delta_ = rope_delta_;
    memcpy(snap_tok_ring_, tok_ring_, sizeof tok_ring_);
    snap_valid_ = true;
    return true;
}

bool Engine::restore_snapshot() {
    // only backwards: KV rows below the snapshot are still intact
    if (!snap_valid_ || spec_open_ || pos_ < snap_pos_ || snap_pos_ > cfg_.max_context) return false;
    pre_.valid = false;
    const size_t conv = (size_t)2 * 3 * DN_CH * 4, ssm = (size_t)DN_VH * 128 * 128 * 4, ring = (size_t)HRING * D * 4;
    CHECK_CUDA(cudaStreamSynchronize(st_));
    const uint8_t* cur = h_snap_;
    for (const LayerWeights& w : L_) {
        if (w.full) continue;
        CHECK_CUDA(cudaMemcpyAsync(w.dn.conv_state, cur, conv, cudaMemcpyHostToDevice, st_)); cur += conv;
        CHECK_CUDA(cudaMemcpyAsync(w.dn.ssm_state, cur, ssm, cudaMemcpyHostToDevice, st_)); cur += ssm;
    }
    if (has_mtp_) CHECK_CUDA(cudaMemcpyAsync(d_hring_, cur, ring, cudaMemcpyHostToDevice, st_));
    CHECK_CUDA(cudaStreamSynchronize(st_));
    pos_ = snap_pos_;   // moving the position back is all the KV cache needs
    conv_cur_ = snap_conv_cur_;
    mtp_pos_ = snap_mtp_pos_;
    images_ = snap_images_;
    rope_delta_ = snap_rope_delta_;
    memcpy(tok_ring_, snap_tok_ring_, sizeof tok_ring_);
    dn_pending_ = 0;   // the snapshot was taken with no pending lazy commit
    spec_pass_ = spec_open_ = false;
    return true;
}

void Engine::reset() {
    pre_.valid = false;
    snap_valid_ = false;   // the KV rows below the snapshot will be rewritten
    pos_ = 0;
    conv_cur_ = dn_pending_ = dn_sb_ = spec_n_ = 0;
    mtp_pos_ = 0;
    images_.clear();
    rope_delta_ = 0;
    spec_pass_ = spec_open_ = false;
    const ModelConfig& c = gguf_.get_config();
    for (int il = 0; il < c.num_layers; il++) {
        if (!L_[il].full) {
            CHECK_CUDA(cudaMemsetAsync(L_[il].dn.conv_state, 0, 2 * 3 * DN_CH * 4, st_));
            CHECK_CUDA(cudaMemsetAsync(L_[il].dn.ssm_state, 0, (size_t)DN_VH * 128 * 128 * 4, st_));
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(st_));
}

// ---------------------------------------------------------------------------
// Decode-time PCIe streaming
// ---------------------------------------------------------------------------

// Choose which part of the host share is streamed to the GPU each token (at most `budget` bytes).
// Streaming moves work from a group's CPU part to its GPU part, so it only pays in groups whose CPU part is
// the bottleneck, and only as far as the copy stream can deliver the pieces in time through the ring of
// NSLOT slots. Greedy: propose the increment that reduces the group's max(CPU, GPU) the most per streamed
// byte, keep it if a timeline simulation of a steady-state token (groups in order, in-order copy stream at
// the PCIe rate, slot reuse, GPU waiting for arrivals) gets faster, else block that candidate.
void Engine::plan_streaming(double budget) {
    const DeviceRates rates = DeviceRates::measured_defaults();
    const size_t ng = plan_.groups.size();
    std::vector<double> C(ng, 0.0), G(ng, 0.0), chain(ng, 0.0);
    for (size_t g = 0; g < ng; g++) chain[g] = plan_.groups[g].gpu_fixed_s + kGroupHostOverhead;
    for (const PlanItem& it : plan_.items) {
        C[it.group] += (it.units - it.gpu_units) * it.cpu_s_per_unit;   // per group: CPU time of its host share ...
        // ... and GPU time of its resident share; a group takes max(C, G)
        G[it.group] += it.gpu_units * it.gpu_s_per_unit;
    }
    auto cpu_s = [&](QuantFormat t, double b) { return b / (rates.cpu_gbs[(int)t] * 1e9); };
    // streamed pieces run below the resident GEMV rate (smaller launches)
    auto gpu_s = [&](QuantFormat t, double b) { return b / (rates.gpu_gbs[(int)t] * 1e9 * kStreamGpuEff); };
    struct Part { QuantFormat type; double bytes_per_col; int per; };   // one streamed matrix of a candidate
    struct Cand {
        int group = 0;
        LayerWeights* w = nullptr;   // FFN triple of w when m == nullptr
        SplitMat* m = nullptr;
        int unit = 0, max_units = 0, units = 0;
        bool blocked = false;
        std::vector<Part> parts;     // bytes per streamed row/column, rows/columns per slot-sized piece
        double bytes_u = 0, cpu_u = 0, gpu_u = 0;
        int slices(int u) const {
            int n = 0;
            for (const Part& p : parts) n += (u * unit + p.per - 1) / p.per;
            return n;
        }
    };
    std::vector<Cand> cands;
    auto finish = [&](Cand& c) {
        for (const Part& p : c.parts) {
            const double b = p.bytes_per_col * c.unit;
            c.bytes_u += b;
            c.cpu_u += cpu_s(p.type, b);
            c.gpu_u += gpu_s(p.type, b);
        }
        if (c.max_units > 0) cands.push_back(c);
    };
    auto row_cand = [&](int group, SplitMat& m, int unit) {
        m.stream_part = 0;
        Cand c;
        c.group = group;
        c.m = &m;
        c.unit = unit;
        c.max_units = m.cpu_n() / unit;
        // rows per 8 MB slot, a multiple of the unit
        c.parts.push_back({m.type, (double)m.cpu_rb, (int)(SLOT_BYTES / m.cpu_rb) / unit * unit});
        finish(c);
    };
    for (int il = 0; il < n_trunk_; il++) {   // the MTP block is not streamed (it runs a variable number of times)
        LayerWeights& w = L_[il];
        const int gA = 3 * il, gB = gA + 1, gC = gA + 2;
        if (w.full) {
            row_cand(gA, w.q, 32); row_cand(gA, w.k, 32); row_cand(gA, w.v, 32); row_cand(gB, w.o, 32);
        } else {
            row_cand(gA, w.qkv, 32); row_cand(gA, w.z, 32); row_cand(gB, w.ssm_out, 32);
        }
        // FFN: 256-wide units of the intermediate dimension (gate/up rows + down columns)
        w.gate.stream_part = w.up.stream_part = w.down.stream_part = 0;
        Cand c;
        c.group = gC;
        c.w = &w;
        c.unit = 256;
        c.max_units = w.gate.cpu_n() / 256;
        const double down_col = quant_row_bytes(w.down.type, 256) / 256.0 * w.down.N;
        c.parts.push_back({w.gate.type, (double)w.gate.cpu_rb, (int)(SLOT_BYTES / w.gate.cpu_rb) / 256 * 256});
        c.parts.push_back({w.up.type, (double)w.up.cpu_rb, (int)(SLOT_BYTES / w.up.cpu_rb) / 256 * 256});
        // down streams column blocks (all N rows each)
        c.parts.push_back({w.down.type, down_col, (int)(SLOT_BYTES / (down_col * 256)) * 256});
        finish(c);
    }
    row_cand(3 * n_trunk_, lm_, 64);   // LM head (the group after the trunk)

    // ---- steady-state token simulation ----
    struct SimPiece { double bytes, gpu; };
    std::vector<std::vector<SimPiece>> gp(ng);
    std::vector<double> Cs(ng), Gs(ng);
    auto simulate = [&]() -> double {
        for (size_t g = 0; g < ng; g++) { gp[g].clear(); Cs[g] = C[g]; Gs[g] = G[g]; }
        size_t P = 0;
        for (const Cand& c : cands) {
            if (c.units == 0) continue;
            const int cols = c.units * c.unit;
            for (const Part& p : c.parts)
                for (int off = 0; off < cols; off += p.per) {
                    const double b = p.bytes_per_col * std::min(p.per, cols - off);
                    gp[c.group].push_back({b, gpu_s(p.type, b) + kSliceOverhead});
                    P++;
                }
            Cs[c.group] -= c.units * c.cpu_u;   // streamed rows leave the CPU's share
        }
        std::vector<double> release(2 * P + 1, 0.0);
        double t = 0, dma = 0, tok = 0;
        size_t k = 0;
        // two tokens: the second shows the steady state (prefetch across tokens)
        for (int pass = 0; pass < 2; pass++) {
            const double t0 = t;
            for (size_t g = 0; g < ng; g++) {
                double gpu = t + Gs[g];
                for (const SimPiece& sp : gp[g]) {
                    // a copy starts when the copy stream is free and its slot was released
                    const double cs = std::max(dma, k >= (size_t)NSLOT ? release[k - NSLOT] : 0.0);
                    dma = cs + sp.bytes / kPcieBytesPerS;   // in-order copy stream at the measured PCIe rate
                    gpu = std::max(gpu, dma) + sp.gpu;   // the GPU computes a piece after it has arrived
                    release[k++] = gpu;
                }
                const double w = plan_.groups[g].weight;   // MTP groups: k executions per cycle, no pieces
                // the group ends when both the GPU and the CPU are done, plus its serial chain
                t += w * (std::max(gpu, t + std::max(Cs[g], 0.0)) + chain[g] - t);
            }
            tok = t - t0;
        }
        return tok;
    };
    auto gain = [&](const Cand& c, int n) {
        const double over = (c.slices(c.units + n) - c.slices(c.units)) * kSliceOverhead;
        // group time saved by moving n units from the CPU to the GPU (+ piece overheads)
        return std::max(C[c.group], G[c.group]) - std::max(C[c.group] - n * c.cpu_u, G[c.group] + n * c.gpu_u + over);
    };
    const double t_before = simulate();
    double t_cur = t_before, used = 0;
    while (true) {
        int best = -1, best_n = 0;
        double best_score = 0;
        for (size_t i = 0; i < cands.size(); i++) {
            const Cand& c = cands[i];
            const int left = c.max_units - c.units;
            if (left <= 0 || c.blocked) continue;
            // the first step of a matrix takes one piece worth so the per-piece overhead is amortized
            const int n = c.units == 0 ? std::min(left, std::max(1, c.parts[0].per / c.unit)) : 1;
            if (used + n * c.bytes_u > budget) continue;
            const double sc = gain(c, n) / (n * c.bytes_u);   // seconds saved per streamed byte
            if (sc > best_score) { best_score = sc; best = (int)i; best_n = n; }
        }
        if (best < 0) break;
        Cand& c = cands[best];
        const double over = (c.slices(c.units + best_n) - c.slices(c.units)) * kSliceOverhead;
        c.units += best_n;
        G[c.group] += best_n * c.gpu_u + over;
        C[c.group] -= best_n * c.cpu_u;
        const double t_new = simulate();
        if (t_new < t_cur - 1e-7) {   // keep the step only if the full-token simulation improves
            t_cur = t_new;
            used += best_n * c.bytes_u;
        } else {   // the copy stream cannot deliver it in time (or no gain): undo, never retry this candidate
            c.units -= best_n;
            G[c.group] -= best_n * c.gpu_u + over;
            C[c.group] += best_n * c.cpu_u;
            c.blocked = true;
        }
    }
    for (const Cand& c : cands) {
        if (c.m) c.m->stream_part = c.units * c.unit;
        else c.w->gate.stream_part = c.w->up.stream_part = c.w->down.stream_part = c.units * c.unit;
    }
    if (cfg_.verbose)
        LOG_INFO("streaming plan: simulated token %.1f -> %.1f ms (%.0f MiB streamed)", t_before * 1e3, t_cur * 1e3,
                 used / 1048576.0);
    streamed_bytes_per_token_ = used;
    // consumption order of the decode path, cut into slot-sized pieces
    spieces_.clear();
    auto add_pieces = [&](const SplitMat& m) {
        if (m.stream_part <= 0) return;
        if (!m.col_split) {
            const int per = (int)(SLOT_BYTES / m.cpu_rb);
            for (int off = 0; off < m.stream_part; off += per) {
                const int n = std::min(per, m.stream_part - off);
                spieces_.push_back({&m, off, n, m.cpu + (size_t)off * m.cpu_rb, (size_t)n * m.cpu_rb, m.cpu_rb});
            }
        } else {
            const int per = (int)(SLOT_BYTES / (quant_row_bytes(m.type, 256) * (size_t)m.N)) * 256;
            for (int off = 0; off < m.stream_part; off += per) {
                const int n = std::min(per, m.stream_part - off);
                const size_t rb = quant_row_bytes(m.type, n);
                spieces_.push_back({&m, off, n, nullptr, rb * (size_t)m.N, rb});
            }
        }
    };
    for (int il = 0; il < n_trunk_; il++) {
        LayerWeights& w = L_[il];
        const std::vector<SplitMat*> order = w.full ? std::vector<SplitMat*>{&w.q, &w.k, &w.v, &w.o, &w.gate, &w.up, &w.down}
                                                    : std::vector<SplitMat*>{&w.qkv, &w.z, &w.ssm_out, &w.gate, &w.up, &w.down};
        for (SplitMat* m : order) add_pieces(*m);
    }
    add_pieces(lm_);
    if (prof_waits_) {
        sprof_ev_.resize(2 * spieces_.size());
        for (cudaEvent_t& e : sprof_ev_) CHECK_CUDA(cudaEventCreate(&e));
        sprof_stall_.assign(spieces_.size(), 0.0);
    }
}

void Engine::issue_spiece(uint64_t seq) {
    const SPiece& p = spieces_[seq % spieces_.size()];
    const int slot = (int)(seq % NSLOT);
    uint8_t* dst = d_slots_ + (size_t)slot * SLOT_BYTES;
    // the copy waits until the GEMV that used this slot has finished
    if (sslot_used_[slot]) CHECK_CUDA(cudaStreamWaitEvent(cst_, ev_sused_[slot], 0));
    CHECK_CUDA(cudaMemcpyAsync(dst, p.src, p.bytes, cudaMemcpyHostToDevice, cst_));
    CHECK_CUDA(cudaEventRecord(ev_scopied_[slot], cst_));   // signals the compute stream that the piece is in VRAM
}

void Engine::stream_gemv(const SplitMat& m, const cuda::ActQ8& act, float* y, bool accumulate, int nb, int act_ld,
                         int y_ld) {
    for (int done = 0; done < m.stream_part;) {
        const uint64_t q = sseq_++;
        const size_t si = q % spieces_.size();
        const SPiece& p = spieces_[si];
        if (p.m != &m || p.off != done) { LOG_ERROR("stream order mismatch at %s", m.name.c_str()); std::abort(); }
        const int slot = (int)(q % NSLOT);
        if (prof_waits_) CHECK_CUDA(cudaEventRecord(sprof_ev_[2 * si], st_));
        CHECK_CUDA(cudaStreamWaitEvent(st_, ev_scopied_[slot], 0));   // GPU-side wait: the host thread does not block
        if (prof_waits_) CHECK_CUDA(cudaEventRecord(sprof_ev_[2 * si + 1], st_));
        const uint8_t* W = d_slots_ + (size_t)slot * SLOT_BYTES;
        if (!m.col_split) {
            cuda::gemv_q8n(m.type, W, p.rb, m.K, p.n, act, nb, act_ld, y + p.off, y_ld, st_, accumulate);
        } else {
            cuda::ActQ8 a2 = act;
            a2.q += p.off; a2.d += p.off / 32; a2.s += p.off / 32;
            // column pieces add partial sums into the same outputs
            cuda::gemv_q8n(m.type, W, p.rb, p.n, m.N, a2, nb, act_ld, y, y_ld, st_, accumulate || p.off > 0);
        }
        CHECK_CUDA(cudaEventRecord(ev_sused_[slot], st_));
        sslot_used_[slot] = true;
        // keep NSLOT copies in flight, wrapping into the next token's pieces
        while (sissued_ < sseq_ + NSLOT) issue_spiece(sissued_++);
        done += p.n;
    }
}

// Advance the ring over m's pieces without computing them (their copies were already issued).
void Engine::skip_stream(const SplitMat& m) {
    for (int done = 0; done < m.stream_part;) {
        const SPiece& p = spieces_[sseq_ % spieces_.size()];
        if (p.m != &m || p.off != done) { LOG_ERROR("stream order mismatch at %s", m.name.c_str()); std::abort(); }
        sseq_++;
        while (sissued_ < sseq_ + NSLOT) issue_spiece(sissued_++);
        done += p.n;
    }
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

void Engine::wait_event(int site) {
    const double t0 = now_s();
    if (prof_waits_) {
        CHECK_CUDA(cudaEventSynchronize(ev_g_[gi_ ^ 1]));   // GEMVs of the group before the one just launched
        const double t1 = now_s();
        CHECK_CUDA(cudaEventSynchronize(ev_));
        const double t2 = now_s();
        gprof_site_[gprof_n_ - 1] = site;
        cur_site_ = site;
        wprof_[site].behind += t1 - t0;
        wprof_[site].chain += t2 - t1;
        wprof_[site].n++;
    } else {
        CHECK_CUDA(cudaEventSynchronize(ev_));   // the host blocks until the GPU reaches the last recorded point
    }
    stats_.t_wait_gpu += now_s() - t0;
}

// Called right after a group's GPU GEMVs are enqueued (profiling only).
void Engine::mark_gpu_group() {
    if (!prof_waits_) return;
    gi_ ^= 1;
    CHECK_CUDA(cudaEventRecord(ev_g_[gi_], st_));
    CHECK_CUDA(cudaEventRecord(gprof_ev_[2 * gprof_n_ + 1], st_));
    gprof_n_++;
}

// Called right before a group's GPU GEMVs are enqueued (profiling only).
void Engine::begin_gpu_group() {
    if (!prof_waits_) return;
    if ((size_t)(2 * gprof_n_ + 2) > gprof_ev_.size()) {
        gprof_ev_.resize(2 * gprof_n_ + 2);
        gprof_site_.resize(gprof_n_ + 1);
        for (size_t i = 2 * gprof_n_; i < gprof_ev_.size(); i++) CHECK_CUDA(cudaEventCreate(&gprof_ev_[i]));
        gcpu_last_.resize(gprof_n_ + 1, 0.0);
        ggpu_last_.resize(gprof_n_ + 1, 0.0);
    }
    gcpu_last_[gprof_n_] = 0;
    CHECK_CUDA(cudaEventRecord(gprof_ev_[2 * gprof_n_], st_));
}

void Engine::print_wait_profile() const {
    if (!prof_waits_) return;
    static const char* names[W_NSITES] = {"deltanet in", "deltanet out", "attn in", "attn out", "ffn in", "lm head", "mtp eh_proj"};
    printf("CPU job imbalance (wall - mean thread busy): %.2f ms/token\n", job_idle_ / std::max(stats_.tokens, 1) * 1e3);
    {
        static const char* sn[8] = {"deltanet in", "deltanet out", "attn in", "attn out", "ffn in", "lm head", "mtp eh_proj", "?"};
        printf("  per site: jobs/token, imbalance us/job, wall - longest thread us/job, mean start delay us, "
               "last end - mean end us, chunk count spread\n");
        for (int i = 0; i < 8; i++)
            if (site_jobs_[i])
                printf("    %-13s %6.1f %8.1f %8.1f %8.1f %8.1f %6.2f\n", sn[i], site_jobs_[i] / (double)std::max(stats_.tokens, 1),
                       site_idle_[i] / site_jobs_[i] * 1e6, site_maxgap_[i] / site_jobs_[i] * 1e6,
                       site_skew_[i] / site_jobs_[i] * 1e6, site_endspread_[i] / site_jobs_[i] * 1e6,
                       site_chunkspread_[i] / site_jobs_[i]);
        printf("  imbalance histogram (<5, <10, <20, <50, <100, <500, <2000, >=2000 us):");
        for (int i = 0; i < 8; i++) printf(" %ld", idle_hist_[i]);
        printf("\n");
    }
    printf("wait profile per token (ms): site  behind  chain  count/token  gpu-group  cpu-job(after this wait)\n");
    const double tok = std::max(stats_.tokens, 1);
    const double stok = std::max<long>(sprof_tokens_, 1);
    for (int i = 0; i < W_NSITES; i++)
        printf("  %-13s %7.2f %7.2f %6.1f %8.2f %8.2f\n", names[i], wprof_[i].behind / tok * 1e3, wprof_[i].chain / tok * 1e3,
               wprof_[i].n / tok, gprof_gpu_[i] / stok * 1e3, gprof_cpu_[i] / tok * 1e3);
    const char* lv = getenv("Q38_PROFILE_WAITS");
    if (lv && atoi(lv) >= 2) {
        // planned per-group times (resident split only, without streaming)
        std::vector<double> pc(plan_.groups.size(), 0.0), pg(plan_.groups.size(), 0.0);
        std::vector<double> gb(plan_.groups.size(), 0.0), cb(plan_.groups.size(), 0.0);
        for (const PlanItem& it : plan_.items) {
            pc[it.group] += (it.units - it.gpu_units) * it.cpu_s_per_unit;
            pg[it.group] += it.gpu_units * it.gpu_s_per_unit;
            gb[it.group] += it.gpu_units * it.bytes_per_unit;
            cb[it.group] += (it.units - it.gpu_units) * it.bytes_per_unit;
        }
        printf("last token per group: idx site gpu_ms cpu_ms | plan gpu_ms cpu_ms | gpu MiB cpu MiB\n");
        for (size_t g = 0; g < ggpu_last_.size() && g < gprof_site_.size(); g++)
            printf("  %3zu %-13s %6.3f %6.3f | %6.3f %6.3f | %6.1f %6.1f%s\n", g, names[gprof_site_[g]], ggpu_last_[g] * 1e3,
                   gcpu_last_[g] * 1e3, g < pg.size() ? pg[g] * 1e3 : 0, g < pc.size() ? pc[g] * 1e3 : 0,
                   g < gb.size() ? gb[g] / 1048576.0 : 0, g < cb.size() ? cb[g] / 1048576.0 : 0,
                   ggpu_last_[g] > gcpu_last_[g] ? "  GPU>CPU" : "");
    }
    if (sprof_tokens_ > 0) {
        // GPU stream stall waiting for streamed slices, aggregated per matrix kind and per layer range
        std::map<std::string, std::pair<double, double>> by_kind;   // stall, bytes
        double tot = 0;
        std::vector<double> by_layer(L_.size(), 0.0);
        for (size_t i = 0; i < spieces_.size(); i++) {
            const SplitMat& m = *spieces_[i].m;
            const double st = sprof_stall_[i] / sprof_tokens_;
            const std::string kind = m.name.substr(m.name.find('.', 4) + 1);
            by_kind[kind].first += st;
            by_kind[kind].second += spieces_[i].bytes;
            tot += st;
            if (m.name.rfind("blk.", 0) == 0) by_layer[atoi(m.name.c_str() + 4)] += st;
        }
        printf("GPU stall waiting for streamed slices: %.2f ms/token\n", tot * 1e3);
        for (auto& [k, v] : by_kind) printf("  %-20s stall %6.2f ms  streamed %6.1f MiB\n", k.c_str(), v.first * 1e3, v.second / 1048576.0);
        printf("  per 8 layers:");
        for (size_t l = 0; l < by_layer.size(); l += 8) {
            double a = 0;
            for (size_t j = l; j < std::min(l + 8, by_layer.size()); j++) a += by_layer[j];
            printf(" %.2f", a * 1e3);
        }
        printf("\n");
    }
}

void Engine::run_pool(cpu::ThreadPool::Fn fn, void* ctx) {
    if (pool_async_) fn(ctx, 0, 1);   // the pool is busy with the LM-head job: run on this thread
    else pool_->run(fn, ctx);
}

void Engine::gemv_gpu(const SplitMat& m, const cuda::ActQ8& act, float* y, int nb, int act_ld, int y_ld) {
    if (m.gpu_n() <= 0 || m.gpu_k() <= 0) return;
    if (!cuda::gemv_q8n(m.type, m.gpu, m.gpu_rb, m.gpu_k(), m.gpu_n(), act, nb, act_ld, y, y_ld, st_))
        LOG_ERROR("no GPU GEMV for %s", quant_name(m.type));
}

// Run a CPU job on the thread pool (the caller is worker 0), with optional profiling of thread imbalance.
#define RUN_CPU_JOB(job)                                    \
    do {                                                    \
        if ((job).n == 0) break;                            \
        if (pool_async_) { cpu_job_task(&(job), 0, 1); break; } \
        const double _t = now_s();                          \
        pool_->run(cpu_job_task, &(job));                   \
        stats_.t_cpu_work += now_s() - _t;                  \
        if (prof_waits_) {                                  \
            double _mb = 0;                                 \
            for (int _i = 0; _i < pool_->size(); _i++) _mb += g_thread_busy[_i]; \
            const double _w = now_s() - _t, _id = _w - _mb / pool_->size(); \
            double _mx = 0;                                 \
            for (int _i = 0; _i < pool_->size(); _i++) _mx = std::max(_mx, g_thread_busy[_i]); \
            job_idle_ += _id;                               \
            double _s0 = 0, _e0 = 0, _emx = 0;              \
            int _cmin = 1 << 30, _cmax = 0;                 \
            for (int _i = 0; _i < pool_->size(); _i++) {    \
                _s0 += g_thread_t0[_i] - _t; _e0 += g_thread_t1[_i] - _t; _emx = std::max(_emx, g_thread_t1[_i] - _t); \
                _cmin = std::min(_cmin, g_thread_chunks[_i]); _cmax = std::max(_cmax, g_thread_chunks[_i]); \
            }                                               \
            site_skew_[cur_site_] += _s0 / pool_->size();   \
            site_endspread_[cur_site_] += _emx - _e0 / pool_->size(); \
            site_chunkspread_[cur_site_] += _cmax - _cmin;  \
            site_idle_[cur_site_] += _id;                   \
            site_maxgap_[cur_site_] += _w - _mx;            \
            site_jobs_[cur_site_]++;                        \
            const double _us = _id * 1e6;                   \
            idle_hist_[_us < 5 ? 0 : _us < 10 ? 1 : _us < 20 ? 2 : _us < 50 ? 3 : _us < 100 ? 4 : _us < 500 ? 5 : _us < 2000 ? 6 : 7]++; \
        }                                                   \
        gprof_cpu_[cur_site_] += now_s() - _t;              \
        if (prof_waits_) gcpu_last_[gprof_n_ - 1] += now_s() - _t; \
    } while (0)

namespace {
// Row t of an ActQ8 holding rows of K elements.
cuda::ActQ8 act_row(const cuda::ActQ8& a, int K, int t) {
    return cuda::ActQ8{a.q + (size_t)t * K, a.d + (size_t)t * (K / 32), a.s + (size_t)t * (K / 32)};
}
void quantize_rows_q8_K(const float* x, block_q8_K* y, int K, int nb) {
    for (int t = 0; t < nb; t++) cpu::quantize_q8_K(x + (size_t)t * K, y + (size_t)t * (K / QK_K), K);
}
} // namespace

// Buffers hold nb token rows: d_qkv [nb][DN_CH], d_z / d_o [nb][DN_V], d_y [nb][D]; the host buffers receive the
// CPU rows of token t at + t * (CPU row count).
void Engine::layer_deltanet(int il, int nb) {
    LayerWeights& w = L_[il];
    // GPU share of the input projections
    begin_gpu_group();
    // enqueued without waiting: the GPU computes while the CPU starts its rows
    gemv_gpu(w.qkv, act_x_, d_qkv, nb, D, DN_CH);
    gemv_gpu(w.z, act_x_, d_z, nb, D, DN_V);
    for (int t = 0; t < nb; t++) {
        // the tiny BF16 beta / alpha projections stay on the GPU
        cuda::gemv_bf16(w.beta, D, DN_VH, d_xn + (size_t)t * D, d_beta + t * 64, st_);
        cuda::gemv_bf16(w.alpha, D, DN_VH, d_xn + (size_t)t * D, d_alpha + t * 64, st_);
    }
    // streamed rows continue right after the resident ones
    stream_gemv(w.qkv, act_x_, d_qkv + w.qkv.gpu_part, false, nb, D, DN_CH);
    stream_gemv(w.z, act_x_, d_z + w.z.gpu_part, false, nb, D, DN_V);
    mark_gpu_group();
    // CPU share; h_xq: Q8_K rows written by the norm kernel (add_rmsnorm_rows2 xq_host)
    CpuJob job;
    job.add(w.qkv, h_xq, h_rows, true, nb, D / QK_K, w.qkv.cpu2_n());
    job.add(w.z, h_xq, h_rows2, true, nb, D / QK_K, w.z.cpu2_n());
    wait_event(W_DN_IN);   // the Q8_K activations for the CPU are written by the previous GPU norm
    RUN_CPU_JOB(job);   // the recurrence kernel reads the CPU rows from h_rows / h_rows2 directly
    // recurrence on the GPU: replay the pending committed tokens, then the nb new ones; a speculative pass leaves
    // the state at the committed position and saves the new tokens' inputs for commit()
    {
        cuda::DNTokens in;
        const int m = dn_pending_;
        if (m > 0) {
            const float* sv = dn_save(dn_sb_, il);
            in.sqkv = sv;
            in.sbeta = sv + (size_t)MAX_BATCH * DN_CH;
            in.salpha = in.sbeta + MAX_BATCH * 64;
        }
        in.qkv = d_qkv; in.z = d_z; in.beta = d_beta; in.alpha = d_alpha;
        // the kernel reads the CPU's rows straight from mapped host memory
        if (w.qkv.cpu2_n() > 0) { in.qkv_host = h_rows; in.qkv_split = w.qkv.gpu_part + w.qkv.stream_part; in.qkv_ld = w.qkv.cpu2_n(); }
        if (w.z.cpu2_n() > 0) { in.z_host = h_rows2; in.z_split = w.z.gpu_part + w.z.stream_part; in.z_ld = w.z.cpu2_n(); }
        in.o_host = h_o;
        if (spec_pass_) {
            float* so = dn_save(dn_sb_ ^ 1, il);   // save this pass's inputs: rejected drafts must not touch the state
            in.oqkv = so;
            in.obeta = so + (size_t)MAX_BATCH * DN_CH;
            in.oalpha = in.obeta + MAX_BATCH * 64;
        }
        // state written back after the committed tokens only in a speculative pass
        cuda::deltanet_multi(w.dn, conv_cur_, in, m, nb, spec_pass_ ? m : m + nb, config().rms_eps, d_o, &act_o_, st_);
    }
    CHECK_CUDA(cudaEventRecord(ev_, st_));
    begin_gpu_group();
    gemv_gpu(w.ssm_out, act_o_, d_y, nb, DN_V, D);
    stream_gemv(w.ssm_out, act_o_, d_y + w.ssm_out.gpu_part, false, nb, DN_V, D);
    mark_gpu_group();
    CpuJob job2;
    job2.add(w.ssm_out, h_xq, h_rows, true, nb, DN_V / QK_K, w.ssm_out.cpu2_n());
    wait_event(W_DN_OUT);
    quantize_rows_q8_K(h_o, h_xq, DN_V, nb);   // the core output (in mapped memory) quantized for the CPU's ssm_out rows
    RUN_CPU_JOB(job2);
    ytail_split_ = w.ssm_out.gpu_part + w.ssm_out.stream_part;
    ytail_ld_ = w.ssm_out.cpu2_n();   // rows read from h_rows by the next norm
}

void Engine::layer_attention(int il, int nb, int pos0) {
    LayerWeights& w = L_[il];
    begin_gpu_group();
    gemv_gpu(w.q, act_x_, d_qg, nb, D, A_Q);
    gemv_gpu(w.k, act_x_, d_k, nb, D, A_KV);
    gemv_gpu(w.v, act_x_, d_v, nb, D, A_KV);
    stream_gemv(w.q, act_x_, d_qg + w.q.gpu_part, false, nb, D, A_Q);
    stream_gemv(w.k, act_x_, d_k + w.k.gpu_part, false, nb, D, A_KV);
    stream_gemv(w.v, act_x_, d_v + w.v.gpu_part, false, nb, D, A_KV);
    mark_gpu_group();
    // h_xq: Q8_K rows written by the norm kernel (add_rmsnorm_rows2 xq_host)
    // CPU rows of token t: q at h_rows + t*nq, k at h_rows2 + t*(nk+nv), v right after k
    const int nq = w.q.cpu2_n(), nk = w.k.cpu2_n(), nv = w.v.cpu2_n();
    CpuJob job;
    job.add(w.q, h_xq, h_rows, true, nb, D / QK_K, nq);
    job.add(w.k, h_xq, h_rows2, true, nb, D / QK_K, nk + nv);
    job.add(w.v, h_xq, h_rows2 + nk, true, nb, D / QK_K, nk + nv);
    wait_event(W_AT_IN);
    RUN_CPU_JOB(job);
    // token t: K/V appended at pos0+t (CPU rows read from h_rows / h_rows2), attention over positions [0, pos0+t]
    auto src = [](const float* dev, int n, const SplitMat& m, const float* host, int ld) {
        // GPU rows first, CPU rows from mapped host memory
        return m.cpu2_n() > 0 ? cuda::RowSrc::split_at(dev, m.gpu_part + m.stream_part, host, ld) : cuda::RowSrc::device(dev, n);
    };
    cuda::attn_prep_multi(src(d_qg, A_Q, w.q, h_rows, nq), src(d_k, A_KV, w.k, h_rows2, nk + nv),
                          // rotary position = cache row + offset after images (M-RoPE)
                          src(d_v, A_KV, w.v, h_rows2 + nk, nk + nv), w.q_norm, w.k_norm, pos0, pos0 + rope_delta_, nb,
                          config().rope_theta, config().rms_eps, cfg_.kv_type, w.kv, d_q, d_gate, st_);
    // output also to h_o: the CPU's share of the output projection reads it
    cuda::attn_decode_multi(d_q, d_gate, w.kv, cfg_.kv_type, pos0, nb, d_attn_work, d_o, &act_o_, st_, h_o);
    CHECK_CUDA(cudaEventRecord(ev_, st_));
    begin_gpu_group();
    gemv_gpu(w.o, act_o_, d_y, nb, A_O, D);
    stream_gemv(w.o, act_o_, d_y + w.o.gpu_part, false, nb, A_O, D);
    mark_gpu_group();
    CpuJob job2;
    job2.add(w.o, h_xq, h_rows, true, nb, A_O / QK_K, w.o.cpu2_n());
    wait_event(W_AT_OUT);
    quantize_rows_q8_K(h_o, h_xq, A_O, nb);
    RUN_CPU_JOB(job2);
    ytail_split_ = w.o.gpu_part + w.o.stream_part;
    ytail_ld_ = w.o.cpu2_n();
}

void Engine::layer_ffn(int il, int nb) {
    LayerWeights& w = L_[il];
    const int r = w.gate.gpu_part;       // resident GPU share of the intermediate dimension
    const int sp = w.gate.stream_part;   // streamed share (computed by the GPU from a ring slot)
    begin_gpu_group();
    if (r > 0) {
        gemv_gpu(w.gate, act_x_, d_g, nb, D, FF);
        gemv_gpu(w.up, act_x_, d_u, nb, D, FF);
    }
    if (sp > 0) {
        stream_gemv(w.gate, act_x_, d_g + r, false, nb, D, FF);
        stream_gemv(w.up, act_x_, d_u + r, false, nb, D, FF);
    }
    if (r + sp > 0) {
        // SwiGLU and int8 quantization fused for the GPU's down GEMV
        cuda::swiglu_rows(d_g, d_u, r + sp, FF, nb, nullptr, &act_h_, st_);
        // down is split by columns: the GPU's partial sums over its intermediate blocks
        if (r > 0) cuda::gemv_q8n(w.down.type, w.down.gpu, w.down.gpu_rb, r, D, act_h_, nb, FF, d_y, D, st_);
        if (sp > 0) {
            cuda::ActQ8 a2 = act_h_;
            a2.q += r; a2.d += r / 32; a2.s += r / 32;
            stream_gemv(w.down, a2, d_y, r > 0, nb, FF, D);
        }
    } else {
        // no GPU share: the CPU partial sums are the whole output
        CHECK_CUDA(cudaMemsetAsync(d_y, 0, (size_t)nb * D * 4, st_));
    }
    mark_gpu_group();
    const int rc = FF - r - sp;
    CpuJob job;   // token t: gate rows at h_g + t*rc, up rows at h_u + t*rc
    if (rc > 0) {
        job.add(w.gate, h_xq, h_g, true, nb, D / QK_K, rc);
        job.add(w.up, h_xq, h_u, true, nb, D / QK_K, rc);
    }
    wait_event(W_FFN);
    if (rc > 0) {
        RUN_CPU_JOB(job);
        SwigluCtx sc{h_g, h_u, h_g, nb * rc};
        run_pool(swiglu_task, &sc);
        quantize_rows_q8_K(h_g, h_hq, rc, nb);   // the CPU's intermediate rows quantized for its own down columns
        CpuJob job2;
        // CPU partial sums of down, added by the next norm kernel
        job2.add(w.down, h_hq, h_y2, true, nb, rc / QK_K, D);
        RUN_CPU_JOB(job2);
        ffn_cpu_ = true;   // h_y2 read by the next norm
    } else {
        ffn_cpu_ = false;
    }
}

cuda::RowSrc Engine::y_src() const {
    return ytail_ld_ > 0 ? cuda::RowSrc::split_at(d_y, ytail_split_, h_rows, ytail_ld_) : cuda::RowSrc::device(d_y, D);
}

cuda::RowSrc Engine::y2_src() const {
    return ffn_cpu_ ? cuda::RowSrc::split_at(nullptr, 0, h_y2, D) : cuda::RowSrc{};
}

void Engine::lm_head(int topk, bool want_logits, DecodeResult* out, int nb, bool streamed, const SpecGuessFn* guess) {
    const double t0 = now_s();
    // rows [0, rg) resident on the GPU, [rg, ng) streamed and computed by the GPU, [ng, VOCAB) on the CPU.
    // Outside the decode sequence (streamed == false, e.g. prefill) the ring is not at the LM-head pieces:
    // the CPU then computes the whole host share. Token t's logits: d_logits / h_logits + t*VOCAB.
    // vocabulary rows: resident | streamed | CPU
    const int rg = lm_.gpu_part, ng = rg + (streamed ? lm_.stream_part : 0), rc = VOCAB - ng;
    topk = std::min(std::max(topk, 1), 32);
    begin_gpu_group();
    gemv_gpu(lm_, act_x_, d_logits, nb, D, VOCAB);
    if (streamed) stream_gemv(lm_, act_x_, d_logits + rg, false, nb, D, VOCAB);
    const bool async = guess && has_mtp_ && rc > 0 && ng > 0 && !want_logits;
    if (ng > 0 && !want_logits) {
        for (int t = 0; t < nb; t++)
            // top-k of the GPU's rows on the GPU: no full-vocabulary copy
            cuda::topk(d_logits + (size_t)t * VOCAB, ng, topk, 0, d_tkv + t * 64, d_tki + t * 64, d_topk_work, st_);
        cuda::copy_floats(h_tkv, d_tkv, nb * 64, st_);
        cuda::copy_floats((float*)h_tki, (const float*)d_tki, nb * 64, st_);   // bitwise 4-byte copy
        if (async) CHECK_CUDA(cudaEventRecord(ev_lmtopk_, st_));
    }
    if (ng > 0 && want_logits)
        CHECK_CUDA(cudaMemcpy2DAsync(h_logits, (size_t)VOCAB * 4, d_logits, (size_t)VOCAB * 4, (size_t)ng * 4, nb,
                                     cudaMemcpyDeviceToHost, st_));
    mark_gpu_group();
    CpuJob job;
    if (rc > 0) job.add(lm_, async ? h_xq_lm_.data() : h_xq, h_logits + ng, streamed, nb, D / QK_K, VOCAB);
    wait_event(W_LM);
    if (async) {
        // The workers compute the CPU rows while this thread takes the GPU's top-k, asks the caller for the cycle's
        // likely outcome and drafts the next cycle on the GPU; then it joins the job. The job reads a private copy of
        // its activations (the MTP layer rewrites h_xq).
        const double ta = now_s();
        std::copy(h_xq, h_xq + (size_t)nb * (D / QK_K), h_xq_lm_.begin());
        pool_async_ = true;
        pool_->start(cpu_job_task, &job);   // workers start the CPU rows; this thread continues with the drafts
        CHECK_CUDA(cudaEventSynchronize(ev_lmtopk_));
        float gv[MAX_BATCH * 64];
        int32_t gi[MAX_BATCH * 64];
        std::copy(h_tkv, h_tkv + nb * 64, gv);
        std::copy(h_tki, h_tki + nb * 64, gi);
        DecodeResult part[MAX_BATCH];
        for (int t = 0; t < nb; t++) {
            part[t].topk_vals.assign(gv + t * 64, gv + t * 64 + topk);
            part[t].topk_ids.assign(gi + t * 64, gi + t * 64 + topk);
        }
        const SpecGuess g = (*guess)(part, nb);   // guess the cycle's outcome from the GPU's vocabulary share
        if (g.k > 0 && g.accepted >= 0 && g.accepted < nb) {
            discard_predraft();
            PreDraft pd;
            pd.pos = pos_ + g.accepted + 1;   // where the next cycle would start if the guess holds
            pd.next = g.next;
            pd.k = std::min(g.k, MAX_BATCH - 1);
            pd.old_mtp_pos = mtp_pos_;
            // draft the next cycle on the GPU while the CPU finishes the LM head
            mtp_draft_at(pd.pos, pd.next, pd.k, pd.d);
            pd.valid = true;
            pre_ = pd;
            stats_.predrafts++;
        }
        cpu_job_task(&job, 0, pool_->size());   // then help with the remaining CPU rows
        pool_->join();
        pool_async_ = false;
        std::copy(gv, gv + nb * 64, h_tkv);
        std::copy(gi, gi + nb * 64, h_tki);
        stats_.t_cpu_work += now_s() - ta;
    } else if (rc > 0) {
        RUN_CPU_JOB(job);
    }
    // per token: top-k of the CPU share in parallel (k <= 32 per thread), merged with the GPU share's top-k
    std::vector<std::vector<std::pair<float, int>>> cand(nb);
    if (!want_logits && rc > 0) {
        static TopkCtx tc;
        for (int t = 0; t < nb; t++) {
            tc.x = h_logits + (size_t)t * VOCAB + ng; tc.n = rc; tc.k = topk; tc.offset = ng;
            pool_->run(topk_task, &tc);
            for (int th = 0; th < pool_->size(); th++)
                // each thread's local top-k becomes a candidate
                for (int j = 0; j < topk; j++) if (tc.ids[th][j] >= 0) cand[t].push_back({tc.vals[th][j], tc.ids[th][j]});
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(st_));
    for (int t = 0; t < nb; t++) {
        DecodeResult& o = out[t];
        std::vector<std::pair<float, int>>& c = cand[t];
        if (want_logits) {
            const float* lg = h_logits + (size_t)t * VOCAB;
            o.logits.assign(lg, lg + VOCAB);
            for (int i = 0; i < VOCAB; i++) c.push_back({lg[i], i});
        } else {
            // plus the GPU share's top-k
            for (int i = 0; i < (ng > 0 ? topk : 0); i++) c.push_back({h_tkv[t * 64 + i], h_tki[t * 64 + i]});
        }
        std::partial_sort(c.begin(), c.begin() + std::min<size_t>(topk, c.size()), c.end(),
                          // ties by token id: deterministic order
                          [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        o.topk_vals.clear();
        o.topk_ids.clear();
        for (int i = 0; i < topk && i < (int)c.size(); i++) {
            o.topk_vals.push_back(c[i].first);
            o.topk_ids.push_back(c[i].second);
        }
    }
    stats_.t_lm += now_s() - t0;
}

void Engine::decode_batch(const int32_t* tokens, int nb, int topk, bool want_logits, DecodeResult* out, bool speculative,
                          const SpecGuessFn* guess) {
    const double t0 = now_s();
    const ModelConfig& c = config();
    const float eps = c.rms_eps;
    if (nb < 1 || nb > MAX_BATCH) { LOG_ERROR("decode_batch: n = %d out of range", nb); return; }
    if (spec_open_) { LOG_ERROR("decode_batch: previous speculative pass not committed"); std::abort(); }
    spec_pass_ = speculative;
    if (pos_ + nb > cfg_.max_context) { LOG_ERROR("context full (%d)", cfg_.max_context); return; }
    // embedding rows (CPU, IQ2_S) -> x
    const size_t erb = quant_row_bytes(tok_embd_->quant, D);
    for (int t = 0; t < nb; t++)
        // embedding rows dequantized on the CPU (the table stays in host memory)
        dequantize_row_ref(tok_embd_->quant, gguf_.tensor_data(*tok_embd_) + (size_t)tokens[t] * erb, h_x + (size_t)t * D, D);
    cuda::copy_floats(d_x, h_x, nb * D, st_);
    if (!spieces_.empty()) {
        if (sseq_ % spieces_.size() != 0) {   // a previous token was interrupted: drain and restart the ring
            CHECK_CUDA(cudaStreamSynchronize(cst_));
            CHECK_CUDA(cudaStreamSynchronize(st_));
            sseq_ = sissued_ = 0;
        }
        while (sissued_ < sseq_ + NSLOT) issue_spiece(sissued_++);   // prefetch the first streamed pieces of this token
    }
    // residual stream: x += previous FFN output (GPU partial sums in d_y + CPU partial sums read from h_y2)
    for (int il = 0; il < c.num_layers; il++) {
        LayerWeights& w = L_[il];
        // x += previous FFN output, then the normalized input of this layer (GPU int8 + CPU Q8_K)
        cuda::add_rmsnorm_rows2(d_x, il ? cuda::RowSrc::device(d_y, D) : cuda::RowSrc{}, il ? y2_src() : cuda::RowSrc{},
                                w.attn_norm, D, nb, eps, d_xn, nullptr, &act_x_, st_, h_xq);
        CHECK_CUDA(cudaEventRecord(ev_, st_));   // the CPU waits on this event before reading h_xq
        if (w.full) layer_attention(il, nb, pos_); else layer_deltanet(il, nb);
        cuda::add_rmsnorm_rows2(d_x, y_src(), cuda::RowSrc{}, w.post_norm, D, nb, eps, d_xn, nullptr, &act_x_, st_, h_xq);
        CHECK_CUDA(cudaEventRecord(ev_, st_));
        layer_ffn(il, nb);
    }
    cuda::add_rmsnorm_rows2(d_x, cuda::RowSrc::device(d_y, D), y2_src(), output_norm_, D, nb, eps, d_xn, nullptr, &act_x_, st_,
                            h_xq);
    CHECK_CUDA(cudaEventRecord(ev_, st_));
    if (has_mtp_)   // MTP inputs: final hidden state and token per position (rows of rejected tokens get overwritten)
        for (int t = 0; t < nb; t++) {
            const int q = pos_ + t;
            // final hidden state per position: the MTP block's input h_{q}
            CHECK_CUDA(cudaMemcpyAsync(d_hring_ + (size_t)(q % HRING) * D, d_xn + (size_t)t * D, D * 4,
                                       cudaMemcpyDeviceToDevice, st_));
            tok_ring_[q % HRING] = tokens[t];
        }
    if (topk > 0 || want_logits) {
        lm_head(topk, want_logits, out, nb, true, speculative && cfg_.mtp_overlap ? guess : nullptr);
    } else {
        skip_stream(lm_);
        CHECK_CUDA(cudaStreamSynchronize(st_));
    }
    if (prof_waits_) {
        CHECK_CUDA(cudaStreamSynchronize(st_));
        for (size_t i = 0; i < spieces_.size(); i++) {
            float ms = 0;
            if (cudaEventElapsedTime(&ms, sprof_ev_[2 * i], sprof_ev_[2 * i + 1]) == cudaSuccess) sprof_stall_[i] += ms * 1e-3;
        }
        sprof_tokens_++;
        for (int g = 0; g < gprof_n_; g++) {
            float ms = 0;
            if (cudaEventElapsedTime(&ms, gprof_ev_[2 * g], gprof_ev_[2 * g + 1]) == cudaSuccess)
                gprof_gpu_[gprof_site_[g]] += ms * 1e-3;
            ggpu_last_[g] = ms * 1e-3;
        }
    }
    gprof_n_ = 0;
    // DeltaNet bookkeeping: the conv window moved to the other buffer iff the state was written back
    if ((speculative ? dn_pending_ : dn_pending_ + nb) > 0) conv_cur_ ^= 1;
    if (speculative) {
        dn_sb_ ^= 1;   // the new tokens' inputs are the next pending candidates
        spec_open_ = true;
        spec_n_ = nb;
    } else {
        dn_pending_ = 0;
        pos_ += nb;   // non-speculative: every token is committed now
    }
    stats_.t_total += now_s() - t0;
    stats_.tokens += nb;
}

// ---------------------------------------------------------------------------
// MTP block (next-token prediction head) for speculative drafting
// ---------------------------------------------------------------------------

// MTP layer on m rows at positions q0..q0+m-1 with tokens toks[i]: x = eh_proj([enorm(embed(tok)) | hnorm(h)]),
// gated attention over the MTP's own KV cache, FFN, shared_head_norm. h of row i = target h_{q-1} from the ring
// (zero for q = 0) or, for a chained step (m = 1), h_first. If draft: argmax of the draft head (resident LM-head
// prefix) for the last row, whose MTP output state is kept in d_hmtp_ for the next chained step.
void Engine::mtp_rows(int q0, int m, const int32_t* toks, const float* h_first, bool h_from_ring, int32_t* draft) {
    LayerWeights& w = L_[n_trunk_];
    const float eps = config().rms_eps;
    const size_t erb = quant_row_bytes(tok_embd_->quant, D);
    for (int i = 0; i < m; i++)
        dequantize_row_ref(tok_embd_->quant, gguf_.tensor_data(*tok_embd_) + (size_t)toks[i] * erb, h_x + (size_t)i * D, D);
    cuda::copy_floats(d_x, h_x, m * D, st_);
    for (int i = 0; i < m; i++) {
        const int q = q0 + i;
        float* dst = d_y + (size_t)i * D;
        // chained draft step: the MTP's own previous output stands in for h
        if (!h_from_ring) CHECK_CUDA(cudaMemcpyAsync(dst, h_first, D * 4, cudaMemcpyDeviceToDevice, st_));
        else if (q == 0) CHECK_CUDA(cudaMemsetAsync(dst, 0, D * 4, st_));
        // the trunk's hidden state of the previous position
        else CHECK_CUDA(cudaMemcpyAsync(dst, d_hring_ + (size_t)((q - 1) % HRING) * D, D * 4, cudaMemcpyDeviceToDevice, st_));
    }
    // [enorm(e) | hnorm(h)] -> d_g rows of 2*D, quantized for the GPU (act_h_ rows of 2*D) and copied for the CPU
    cuda::add_rmsnorm_rows(d_x, nullptr, nullptr, w.enorm, D, m, eps, d_xn, nullptr, st_);
    cuda::copy_rows(d_g, 2 * D, d_xn, D, D, m, st_);
    cuda::add_rmsnorm_rows(d_y, nullptr, nullptr, w.hnorm, D, m, eps, d_xn, nullptr, st_);
    cuda::copy_rows(d_g + D, 2 * D, d_xn, D, D, m, st_);
    cuda::quantize_act_q8(d_g, 2 * D, m, act_h_, st_);
    cuda::copy_floats(h_g, d_g, m * 2 * D, st_);
    CHECK_CUDA(cudaEventRecord(ev_, st_));
    begin_gpu_group();
    // projects the concatenated [embedding | hidden] back to model width
    gemv_gpu(w.eh_proj, act_h_, d_x, m, 2 * D, D);
    mark_gpu_group();
    wait_event(W_MTP);
    quantize_rows_q8_K(h_g, h_xq, 2 * D, m);
    CpuJob job;
    job.add(w.eh_proj, h_xq, h_rows, true, m, 2 * D / QK_K, w.eh_proj.cpu2_n());
    RUN_CPU_JOB(job);
    if (w.eh_proj.cpu2_n() > 0)
        cuda::copy_rows(d_x + w.eh_proj.gpu_part, D, h_rows, w.eh_proj.cpu2_n(), w.eh_proj.cpu2_n(), m, st_);
    // decoder block on the MTP KV cache
    cuda::add_rmsnorm_rows2(d_x, cuda::RowSrc{}, cuda::RowSrc{}, w.attn_norm, D, m, eps, d_xn, nullptr, &act_x_, st_, h_xq);
    CHECK_CUDA(cudaEventRecord(ev_, st_));
    layer_attention(n_trunk_, m, q0);   // the MTP layer's attention over its own KV cache of earlier pairs
    cuda::add_rmsnorm_rows2(d_x, y_src(), cuda::RowSrc{}, w.post_norm, D, m, eps, d_xn, nullptr, &act_x_, st_, h_xq);
    CHECK_CUDA(cudaEventRecord(ev_, st_));
    layer_ffn(n_trunk_, m);
    cuda::add_rmsnorm_rows2(d_x, cuda::RowSrc::device(d_y, D), y2_src(), w.head_norm, D, m, eps, d_xn, nullptr, &act_x_, st_);
    if (draft) {
        const int last = m - 1;
        // kept for the next chained step
        CHECK_CUDA(cudaMemcpyAsync(d_hmtp_, d_xn + (size_t)last * D, D * 4, cudaMemcpyDeviceToDevice, st_));
        // drafts come from the resident vocabulary prefix only (no CPU rows: fast)
        const int nd = std::min(cfg_.draft_rows, lm_.gpu_part);
        const cuda::ActQ8 a = act_row(act_x_, D, last);
        cuda::gemv_q8n(lm_.type, lm_.gpu, lm_.gpu_rb, D, nd, a, 1, D, d_logits, VOCAB, st_);
        cuda::topk(d_logits, nd, 1, 0, d_tkv, d_tki, d_topk_work, st_);   // greedy draft: argmax
        cuda::copy_floats((float*)h_tki, (const float*)d_tki, 1, st_);
        CHECK_CUDA(cudaStreamSynchronize(st_));
        *draft = h_tki[0];
    }
}

void Engine::mtp_catchup() {
    if (!has_mtp_) return;
    discard_predraft();
    // the hidden-state ring holds only the last HRING positions
    if (pos_ - mtp_pos_ + 1 > HRING) { LOG_ERROR("MTP fell %d positions behind (ring %d)", pos_ - mtp_pos_, HRING); std::abort(); }
    while (mtp_pos_ < pos_) {
        const int m = std::min(MAX_BATCH, pos_ - mtp_pos_);
        int32_t toks[MAX_BATCH];
        for (int i = 0; i < m; i++) toks[i] = tok_ring_[(mtp_pos_ + i) % HRING];
        mtp_rows(mtp_pos_, m, toks, nullptr, true, nullptr);
        mtp_pos_ += m;
    }
}

void Engine::mtp_draft(int32_t next, int k, int32_t* drafts) {
    if (!has_mtp_ || k < 1) return;
    if (spec_open_) { LOG_ERROR("mtp_draft: speculative pass not committed"); std::abort(); }
    discard_predraft();
    mtp_draft_at(pos_, next, k, drafts);
}

void Engine::discard_predraft() {
    if (!pre_.valid) return;
    pre_.valid = false;
    mtp_pos_ = std::min(mtp_pos_, pre_.old_mtp_pos);   // rows from there on are recomputed by the next catch-up
}

bool Engine::take_drafts(int32_t next, int k, int32_t* drafts) {
    if (!pre_.valid) return false;
    // the guess was wrong (or not enough drafts): drafts unusable
    if (spec_open_ || pre_.pos != pos_ || pre_.next != next || k > pre_.k) {
        // The catch-up rows below both the committed position and the guessed draft row hold committed tokens and
        // stay valid; the draft row and the chained rows are recomputed by the next drafting.
        pre_.valid = false;
        if (!spec_open_) mtp_pos_ = std::min(mtp_pos_, std::max(pre_.old_mtp_pos, std::min(pos_, pre_.pos)));
        else mtp_pos_ = std::min(mtp_pos_, pre_.old_mtp_pos);
        stats_.predraft_misses++;
        return false;
    }
    for (int i = 0; i < k; i++) drafts[i] = pre_.d[i];
    pre_.valid = false;   // mtp_pos_ already includes the draft row (pos_ + 1)
    return true;
}

// mtp_draft for the token `next` at position pos (>= position(): the overlapped drafting runs before the commit)
void Engine::mtp_draft_at(int pos, int32_t next, int k, int32_t* drafts) {
    const int pos_saved = pos_;
    pos_ = pos;
    // pos_ is borrowed for the draft position, restored on return
    struct Restore { int& p; int v; ~Restore() { p = v; } } restore{pos_, pos_saved};
    if (pos_ - mtp_pos_ + 1 > HRING) { LOG_ERROR("MTP fell %d positions behind (ring %d)", pos_ - mtp_pos_, HRING); std::abort(); }
    // known pairs first if they do not fit one pass with the draft row
    while (pos_ - mtp_pos_ + 1 > MAX_BATCH) {
        int32_t toks[MAX_BATCH];
        for (int i = 0; i < MAX_BATCH; i++) toks[i] = tok_ring_[(mtp_pos_ + i) % HRING];
        mtp_rows(mtp_pos_, MAX_BATCH, toks, nullptr, true, nullptr);
        mtp_pos_ += MAX_BATCH;
    }
    // catch-up rows + the draft row q = pos_ (token `next`) in one pass
    const int m = pos_ - mtp_pos_ + 1;
    int32_t toks[MAX_BATCH];
    for (int i = 0; i < m - 1; i++) toks[i] = tok_ring_[(mtp_pos_ + i) % HRING];
    toks[m - 1] = next;
    mtp_rows(mtp_pos_, m, toks, nullptr, true, &drafts[0]);
    mtp_pos_ = pos_ + 1;
    // chained steps on the MTP's own state; their KV rows are tentative (rewritten by the next catch-up)
    // each further draft feeds on the previous draft token
    for (int i = 1; i < k; i++) mtp_rows(pos_ + i, 1, &drafts[i - 1], d_hmtp_, false, &drafts[i]);
}

void Engine::commit(int m) {
    if (!spec_open_ || m < 1 || m > spec_n_) { LOG_ERROR("commit(%d) without a matching speculative pass", m); std::abort(); }
    spec_open_ = false;
    dn_pending_ = m;   // the DeltaNet state catches up on these m tokens at the start of the next pass
    pos_ += m;
}

void Engine::flush_pending() {
    if (spec_open_) { LOG_ERROR("flush_pending: speculative pass not committed"); std::abort(); }
    if (dn_pending_ == 0) return;
    const int m = dn_pending_;
    for (size_t il = 0; il < L_.size(); il++) {
        if (L_[il].full) continue;
        cuda::DNTokens in;
        const float* sv = dn_save(dn_sb_, (int)il);
        in.sqkv = sv;
        in.sbeta = sv + (size_t)MAX_BATCH * DN_CH;
        in.salpha = in.sbeta + MAX_BATCH * 64;
        // replay the m committed tokens from their saved inputs, no outputs
        cuda::deltanet_multi(L_[il].dn, conv_cur_, in, m, 0, m, config().rms_eps, nullptr, nullptr, st_);
    }
    conv_cur_ ^= 1;
    dn_pending_ = 0;
}

// ---------------------------------------------------------------------------
// Batched prefill
// ---------------------------------------------------------------------------

void Engine::prefill_free() {
    if (pb_.base) cudaFree(pb_.base);
    for (float* h : pb_.h_emb) if (h) cudaFreeHost(h);
    if (pb_.borrowed)   // the resident LM-head rows held prefill buffers: restore them (stream-ordered after prefill)
        CHECK_CUDA(cudaMemcpyAsync(lm_.gpu, h_lm_gpu_copy_, (size_t)lm_.gpu_part * lm_.gpu_rb, cudaMemcpyHostToDevice, st_));
    pb_ = PB();
    emb_pending_[0] = emb_pending_[1] = false;
}

// cudaMemGetInfo overstates what can be allocated (measured: 116 MiB reported free, largest allocation 76 MiB) and cuBLAS
// loads its kernels lazily into device memory (a prefill GEMM failed with CUBLAS_STATUS_INTERNAL_ERROR when its buffers
// left the reported 16 MiB margin but not the real one): check that `bytes` more can really be allocated.
static bool vram_headroom(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { (void)cudaGetLastError(); return false; }
    CHECK_CUDA(cudaFree(p));
    return true;
}

bool Engine::scratch_acquire(const std::vector<size_t>& sizes, std::vector<void*>& out) {
    scratch_release();
    out.assign(sizes.size(), nullptr);
    size_t total = 0;
    for (size_t x : sizes) total += align_up(x, 256);
    const size_t lm_bytes = (h_lm_gpu_copy_ && !lm_.col_split) ? (size_t)lm_.gpu_part * lm_.gpu_rb : 0;
    const size_t margin = 16u << 20;
    size_t free_b, total_b;
    CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
    // largest buffers into the LM-head rows when free VRAM alone does not hold everything (as prefill_alloc); second
    // attempt with them when the cudaMalloc or the headroom check fails
    for (int borrow = 0; borrow < 2; borrow++) {
        std::vector<bool> in_lm(sizes.size(), false);
        size_t malloc_bytes = total;
        // short of free VRAM: the largest buffers go into the resident LM-head rows
        if ((borrow || total + margin > free_b) && lm_bytes > 0) {
            std::vector<size_t> order(sizes.size());
            for (size_t i = 0; i < order.size(); i++) order[i] = i;
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return sizes[a] > sizes[b]; });
            size_t left = lm_bytes;
            for (size_t i : order) {
                const size_t a = align_up(sizes[i], 256);
                if (sizes[i] && a <= left) { in_lm[i] = true; left -= a; malloc_bytes -= a; }
            }
        }
        if (malloc_bytes + margin > free_b) continue;
        if (malloc_bytes && cudaMalloc(&scratch_base_, malloc_bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            scratch_base_ = nullptr;
            continue;
        }
        if (!vram_headroom(margin)) {   // allocated, but cuBLAS would have no room left for its kernels: retry borrowing
            if (scratch_base_) CHECK_CUDA(cudaFree(scratch_base_));
            scratch_base_ = nullptr;
            continue;
        }
        uint8_t* lc = (uint8_t*)lm_.gpu;
        uint8_t* mc = scratch_base_;
        for (size_t i = 0; i < sizes.size(); i++) {
            if (!sizes[i]) continue;
            uint8_t*& c = in_lm[i] ? lc : mc;
            out[i] = c;
            c += align_up(sizes[i], 256);
        }
        scratch_borrowed_ = malloc_bytes < total;
        return true;
    }
    return false;
}

void Engine::scratch_release() {
    if (scratch_base_) {
        CHECK_CUDA(cudaStreamSynchronize(st_));
        cudaFree(scratch_base_);
    }
    scratch_base_ = nullptr;
    if (scratch_borrowed_)   // stream-ordered after the work that used the rows
        CHECK_CUDA(cudaMemcpyAsync(lm_.gpu, h_lm_gpu_copy_, (size_t)lm_.gpu_part * lm_.gpu_rb, cudaMemcpyHostToDevice, st_));
    scratch_borrowed_ = false;
}

bool Engine::prefill_alloc(int B, size_t stage, bool borrow) {
    if (pb_.B == B && pb_.stage_bytes == stage) return true;
    prefill_free();
    const size_t Bz = (size_t)B;
    // X, Y, Xn16 live through the whole chunk. One region reused by the phases of each layer (attention or DeltaNet, then FFN) and by the MTP block at the
    // chunk end: 136 KB per token instead of 330 KB as separate buffers, so 1024-token chunks fit (dequant and
    // host-weight streaming are paid once per chunk). Offsets are within the region.
    struct Sub { size_t off = 0; size_t take(size_t bytes) { size_t o = off; off += align_up(bytes, 256); return o; } };
    Sub at, dn, ff, mt;
    // attention: q|gate projection (dead after attn_prep: its slot then holds O16), K, V, roped q, gate
    const size_t aQKV = at.take(Bz * A_Q * 4), aK = at.take(Bz * A_KV * 4), aV = at.take(Bz * A_KV * 4),
                 aQ = at.take(Bz * A_O * 4), aGate = at.take(Bz * A_O * 4);
    // DeltaNet: qkv (dead after the conv kernels: its slot then holds the core output and O16), z, beta, alpha, work
    const size_t dQKV = dn.take(Bz * A_Q * 4), dZ = dn.take(Bz * DN_V * 4), dBeta = dn.take(Bz * 64 * 4),
                 dAlpha = dn.take(Bz * 64 * 4), dWork = dn.take(cuda::dn_work_bytes(B));
    static_assert(DN_CH <= A_Q && DN_V * 4 + A_O * 2 <= A_Q * 4, "qkv slot holds DeltaNet core output + O16");
    const size_t sO16 = Bz * DN_V * 4;   // O16 after the core output inside the qkv slot
    // FFN: gate, up (SwiGLU writes the FP16 rows in place over gate)
    const size_t fG = ff.take(Bz * FF * 4), fU = ff.take(Bz * FF * 4);
    // MTP block: output_norm(x), h_{q-1} rows, [enorm(e)|hnorm(h)] FP16, embeddings, K, V, q, gate
    const size_t mHn = mt.take(Bz * D * 4), mHs = mt.take(Bz * D * 4), mH16 = mt.take(Bz * 2 * D * 2),
                 mE = mt.take(Bz * D * 4), mK = mt.take(Bz * A_KV * 4), mV = mt.take(Bz * A_KV * 4),
                 mQ = mt.take(Bz * A_O * 4), mGate = mt.take(Bz * A_O * 4);
    // attn_prep_rows reads a (stale, unused) q|gate input of A_Q floats per row from mHs
    const size_t region = std::max({at.off, dn.off, ff.off, has_mtp_ ? std::max(mt.off, mHs + Bz * A_Q * 4) : 0});
    // Placement: everything in one cudaMalloc when the free VRAM fits it (+ margin for lazily loaded cuBLAS kernels);
    // otherwise the largest buffers go into the resident LM-head rows (unused during the prefill layers; restored from
    // the pinned host copy by prefill_free, ~30 ms) and the rest into the cudaMalloc. This lets decode keep that VRAM
    // for weights instead of a reserve that only prefill used.
    const bool ring = d_slots_ != nullptr;   // FP16 dequant scratch = the decode streaming ring (64 MB, idle now)
    const size_t wf16_elems = ring ? (size_t)NSLOT * SLOT_BYTES / 2 : (16u << 20);
    const size_t ws = 4u << 20;              // cuBLAS workspace
    enum { BR, BX, BY, BXN, BS0, BS1, BW, BWS, BROPE, NBUF };
    const size_t sz[NBUF] = {region, Bz * D * 4, Bz * D * 4, Bz * D * 2, stage, stage, ring ? 0 : wf16_elems * 2, ws,
                             Bz * 64 * 4 + Bz * 3 * 4};
    const size_t margin = 16u << 20;
    size_t free_b, total_b;
    CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
    size_t total = 0;
    for (size_t x : sz) total += align_up(x, 256);
    const size_t lm_bytes = (h_lm_gpu_copy_ && !lm_.col_split) ? (size_t)lm_.gpu_part * lm_.gpu_rb : 0;
    bool in_lm[NBUF] = {};
    size_t malloc_bytes = total;
    static const bool force_borrow = getenv("Q38_PREFILL_BORROW") != nullptr;   // test knob (integration tests)
    if ((total + margin > free_b || borrow || force_borrow) && lm_bytes > 0) {
        size_t left = lm_bytes;
        for (int i : {BR, BX, BY, BS0, BS1, BW, BXN, BWS, BROPE}) {
            const size_t a = align_up(sz[i], 256);
            if (sz[i] && a <= left) { in_lm[i] = true; left -= a; malloc_bytes -= a; }
        }
    }
    if (cfg_.verbose)
        LOG_INFO("prefill buffers B=%d: %.0f MiB (%.0f MiB in the LM-head rows), %.0f MiB free", B, total / 1048576.0,
                 (total - malloc_bytes) / 1048576.0, free_b / 1048576.0);
    if (malloc_bytes + margin > free_b) return false;
    if (malloc_bytes) {
        const cudaError_t e = cudaMalloc(&pb_.base, malloc_bytes);
        if (e != cudaSuccess) {
            (void)cudaGetLastError();
            if (cfg_.verbose) LOG_INFO("prefill buffers B=%d: cudaMalloc of %.0f MiB failed (%s)", B, malloc_bytes / 1048576.0, cudaGetErrorString(e));
            pb_.base = nullptr;
            return false;
        }
        if (!vram_headroom(margin)) {   // the reported free VRAM was not all allocatable
            if (cfg_.verbose) LOG_INFO("prefill buffers B=%d: less than %.0f MiB left after them", B, margin / 1048576.0);
            CHECK_CUDA(cudaFree(pb_.base));
            pb_.base = nullptr;
            return false;
        }
    }
    uint8_t* ptr[NBUF] = {};
    {
        uint8_t* lc = (uint8_t*)lm_.gpu;
        uint8_t* mc = pb_.base;
        for (int i = 0; i < NBUF; i++) {
            if (!sz[i]) continue;
            uint8_t*& c = in_lm[i] ? lc : mc;
            ptr[i] = c;
            c += align_up(sz[i], 256);
        }
    }
    pb_.borrowed = malloc_bytes < total;
    uint8_t* r = ptr[BR];
    pb_.B = B;
    pb_.X = (float*)ptr[BX]; pb_.Y = (float*)ptr[BY]; pb_.Xn16 = (__half*)ptr[BXN];
    pb_.qkv = (float*)(r + aQKV); pb_.k = (float*)(r + aK); pb_.v = (float*)(r + aV); pb_.q = (float*)(r + aQ);
    pb_.gate = (float*)(r + aGate);
    pb_.z = (float*)(r + dZ); pb_.beta = (float*)(r + dBeta); pb_.alpha = (float*)(r + dAlpha); pb_.dn_work = r + dWork;
    pb_.dn_out = (float*)(r + dQKV);
    pb_.O16 = (__half*)(r + dQKV + sO16);
    pb_.G = (float*)(r + fG); pb_.U = (float*)(r + fU); pb_.H16 = (__half*)pb_.G;
    pb_.mHn = (float*)(r + mHn); pb_.mHs = (float*)(r + mHs); pb_.mH16 = (__half*)(r + mH16); pb_.mE = (float*)(r + mE);
    pb_.mk = (float*)(r + mK); pb_.mv = (float*)(r + mV); pb_.mq = (float*)(r + mQ); pb_.mgate = (float*)(r + mGate);
    pb_.Wf16 = ring ? (__half*)d_slots_ : (__half*)ptr[BW]; pb_.stage[0] = ptr[BS0]; pb_.stage[1] = ptr[BS1];
    pb_.cublas_ws = ptr[BWS];
    pb_.rope = (float*)ptr[BROPE];
    pb_.rope_pos = (int*)(pb_.rope + Bz * 64);
    pb_.wf16_elems = wf16_elems; pb_.stage_bytes = stage;
    for (float*& h : pb_.h_emb) CHECK_CUDA(cudaHostAlloc((void**)&h, Bz * D * 4, cudaHostAllocDefault));
    if (!cuda::cublas_bind(cublas_, st_, pb_.cublas_ws, ws)) { LOG_ERROR("cuBLAS setup failed"); exit(1); }
    if (!ev_copied_[0]) {
        for (int i = 0; i < 2; i++) {
            CHECK_CUDA(cudaEventCreateWithFlags(&ev_copied_[i], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&ev_used_[i], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&ev_emb_[i], cudaEventDisableTiming));
        }
    }
    slot_used_[0] = slot_used_[1] = false;
    build_pieces();
    return true;
}

// Row-block size for the dequant + cuBLAS GEMM loop: multiples of 512 rows run at 30-45 TFLOPS on this GPU, odd
// sizes (e.g. 963, 3276) at 11-18 (tile quantization; measured).
static int gemm_rows(int rows, size_t fit) {
    int r = (int)std::min<size_t>((size_t)rows, fit);
    if (r >= 512) r -= r % 512;
    return std::max(r, 1);
}

// Order must match prefill_batched / gemm_split exactly.
void Engine::build_pieces() {
    pieces_.clear();
    auto add = [&](const SplitMat& m) {
        if (m.cpu_n() <= 0 || m.cpu_k() <= 0) return;
        int per = (int)std::max<size_t>(1, std::min<size_t>(m.cpu_n(), pb_.stage_bytes / m.cpu_rb));
        const int R = gemm_rows(m.cpu_n(), pb_.wf16_elems / m.cpu_k());
        if (per >= R && per < m.cpu_n()) per -= per % R;   // whole GEMM blocks per staged piece
        for (int r0 = 0; r0 < m.cpu_n(); r0 += per) pieces_.push_back({&m, r0, std::min(per, m.cpu_n() - r0)});
    };
    for (int il = 0; il < n_trunk_; il++) {
        const LayerWeights& w = L_[il];
        if (w.full) { add(w.q); add(w.k); add(w.v); add(w.o); }
        else { add(w.qkv); add(w.z); add(w.ssm_out); }
        add(w.gate); add(w.up); add(w.down);
    }
    if (has_mtp_) {   // MTP prefill only fills the MTP KV cache: eh_proj, then the K and V projections
        const LayerWeights& m = L_[n_trunk_];
        add(m.eh_proj); add(m.k); add(m.v);
    }
}

// MTP block over a prefill chunk (positions p0..p0+nb-1): only what its KV cache needs — pairs (h_{q-1}, x_q) ->
// eh_proj -> attn_norm -> K/V (RoPE, norms) appended at q. Also refreshes the h/token rings for the decode-time
// MTP passes that follow.
void Engine::prefill_mtp(const std::vector<int32_t>& tokens, int c0, int nb, int p0, const float* h_emb) {
    const LayerWeights& m = L_[n_trunk_];
    const ModelConfig& c = config();
    const float eps = c.rms_eps;
    float* Hn = pb_.mHn;   // output_norm(x) of the chunk rows
    float* Hs = pb_.mHs;   // row t = h_{p0+t-1}
    float* E = pb_.mE;     // the chunk's token embeddings (host copy still holds them)
    __half* H16 = pb_.mH16;
    CHECK_CUDA(cudaMemcpyAsync(E, h_emb, (size_t)nb * D * 4, cudaMemcpyHostToDevice, st_));
    cuda::rmsnorm_rows(pb_.X, nullptr, nullptr, output_norm_, nb, D, eps, nullptr, Hn, st_);
    // pair (h_{q-1}, x_q): the first row takes the last hidden state before the chunk
    if (p0 > 0) CHECK_CUDA(cudaMemcpyAsync(Hs, d_hring_ + (size_t)((p0 - 1) % HRING) * D, D * 4, cudaMemcpyDeviceToDevice, st_));
    else CHECK_CUDA(cudaMemsetAsync(Hs, 0, D * 4, st_));
    // the other rows take the chunk's own hidden states, shifted by one
    if (nb > 1) CHECK_CUDA(cudaMemcpyAsync(Hs + D, Hn, (size_t)(nb - 1) * D * 4, cudaMemcpyDeviceToDevice, st_));
    // [enorm(e) | hnorm(h)] as fp16 rows of 2*D
    cuda::rmsnorm_rows(E, nullptr, nullptr, m.enorm, nb, D, eps, pb_.Xn16, nullptr, st_);
    CHECK_CUDA(cudaMemcpy2DAsync(H16, 2 * D * 2, pb_.Xn16, D * 2, D * 2, nb, cudaMemcpyDeviceToDevice, st_));
    cuda::rmsnorm_rows(Hs, nullptr, nullptr, m.hnorm, nb, D, eps, pb_.Xn16, nullptr, st_);
    CHECK_CUDA(cudaMemcpy2DAsync(H16 + D, 2 * D * 2, pb_.Xn16, D * 2, D * 2, nb, cudaMemcpyDeviceToDevice, st_));
    gemm_split(m.eh_proj, H16, 2 * D, nb, pb_.Y, D);   // the MTP block's input projection for all chunk rows at once
    cuda::rmsnorm_rows(pb_.Y, nullptr, nullptr, m.attn_norm, nb, D, eps, pb_.Xn16, nullptr, st_);
    gemm_split(m.k, pb_.Xn16, D, nb, pb_.mk, A_KV);
    gemm_split(m.v, pb_.Xn16, D, nb, pb_.mv, A_KV);
    // Q is not needed for the cache: attn_prep_rows gets a stale Q buffer (its q/gate outputs are unused)
    cuda::attn_prep_rows(Hs, pb_.mk, pb_.mv, m.q_norm, m.k_norm, nb, p0, pb_.rope, eps, cfg_.kv_type,
                         const_cast<cuda::KVLayer&>(m.kv), pb_.mq, pb_.mgate, st_);
    const int keep = std::min(nb, HRING);
    for (int t = nb - keep; t < nb; t++) {   // the decode-time MTP passes continue from the chunk's last positions
        const int q = p0 + t;
        CHECK_CUDA(cudaMemcpyAsync(d_hring_ + (size_t)(q % HRING) * D, Hn + (size_t)t * D, D * 4, cudaMemcpyDeviceToDevice, st_));
        tok_ring_[q % HRING] = tokens[c0 + t];
    }
    mtp_pos_ = p0 + nb;
}

void Engine::issue_piece(size_t i) {
    const Piece& pc = pieces_[i % pieces_.size()];
    const int slot = (int)(i & 1);
    if (slot_used_[slot]) CHECK_CUDA(cudaStreamWaitEvent(cst_, ev_used_[slot], 0));   // slot free?
    CHECK_CUDA(cudaMemcpyAsync(pb_.stage[slot], pc.m->cpu + (size_t)pc.r0 * pc.m->cpu_rb, (size_t)pc.n * pc.m->cpu_rb,
                               cudaMemcpyHostToDevice, cst_));
    CHECK_CUDA(cudaEventRecord(ev_copied_[slot], cst_));
    pstats_.bytes_streamed += (double)pc.n * pc.m->cpu_rb;
}

// Y[:, cols] (+)= X . W^T for a split matrix: resident share dequantized from VRAM, host share streamed
// through the staging buffer. Row split -> output columns [0,gpu) and [gpu,N); column split (FFN down)
// -> two partial products over K accumulated into all N columns.
void Engine::pmark(int phase) {
    if (!pprof_) return;
    if (pp_n_ >= pp_ev_.size()) { pp_ev_.push_back(nullptr); CHECK_CUDA(cudaEventCreate(&pp_ev_.back())); pp_phase_.push_back(0); }
    CHECK_CUDA(cudaEventRecord(pp_ev_[pp_n_], st_));
    pp_phase_[pp_n_++] = phase;
}

void Engine::pflush() {
    if (!pprof_ || pp_n_ < 2) { pp_n_ = 0; return; }
    CHECK_CUDA(cudaEventSynchronize(pp_ev_[pp_n_ - 1]));
    for (size_t i = 0; i + 1 < pp_n_; i++) {
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, pp_ev_[i], pp_ev_[i + 1]));
        pp_ms_[pp_phase_[i]] += ms;
    }
    pp_n_ = 0;
}

void Engine::gemm_split(const SplitMat& m, const __half* X, int ldx, int B, float* Y, int ldy) {
    const size_t scratch = pb_.wf16_elems;
    if (m.gpu_n() > 0 && m.gpu_k() > 0) {
        const int K = m.gpu_k(), rows = m.gpu_n();
        // rows per dequant + GEMM block: as many as the FP16 scratch holds, multiple of 512
        const int R = gemm_rows(rows, scratch / K);
        for (int r0 = 0; r0 < rows; r0 += R) {
            const int n = std::min(R, rows - r0);
            pmark(PP_DEQ);
            // resident rows -> FP16 scratch
            cuda::dequant_rows_f16(m.type, (const uint8_t*)m.gpu + (size_t)r0 * m.gpu_rb, m.gpu_rb, K, n, pb_.Wf16, st_);
            pmark(PP_GEMM);
            // tensor cores: B tokens x n rows x K
            cuda::gemm_f16(cublas_, pb_.Wf16, n, K, X, ldx, B, Y + r0, ldy, false, st_);
            pmark(PP_OTHER);
        }
    }
    if (m.cpu_n() > 0 && m.cpu_k() > 0) {
        const int K = m.cpu_k();
        // consume this matrix's pieces from the staging ring (issued ahead on the copy stream)
        while (piece_next_ < pieces_.size() && pieces_[piece_next_].m == &m) {
            const size_t i = piece_next_++;
            const Piece& pc = pieces_[i];
            const int slot = (int)(i & 1);
            pmark(PP_WAIT);
            CHECK_CUDA(cudaStreamWaitEvent(st_, ev_copied_[slot], 0));   // the host piece has landed in its staging slot
            pmark(PP_OTHER);
            const int R = gemm_rows(pc.n, scratch / K);
            for (int s0 = 0; s0 < pc.n; s0 += R) {
                const int ns = std::min(R, pc.n - s0);
                pmark(PP_DEQ);
                cuda::dequant_rows_f16(m.type, pb_.stage[slot] + (size_t)s0 * m.cpu_rb, m.cpu_rb, K, ns, pb_.Wf16, st_);
                pmark(PP_GEMM);
                const int r = pc.r0 + s0;
                if (!m.col_split)
                    cuda::gemm_f16(cublas_, pb_.Wf16, ns, K, X, ldx, B, Y + m.gpu_part + r, ldy, false, st_);
                else
                    // column split: the host columns' partial products add to the GPU's
                    cuda::gemm_f16(cublas_, pb_.Wf16, ns, K, X + m.gpu_part, ldx, B, Y + r, ldy, m.gpu_part > 0, st_);
                pmark(PP_OTHER);
            }
            CHECK_CUDA(cudaEventRecord(ev_used_[slot], st_));
            slot_used_[slot] = true;
            if (i + 2 < piece_issue_limit_) issue_piece(i + 2);   // keep two pieces in flight
        }
    }
}

void Engine::gemm_bf16(const uint16_t* W, int N, const __half* X, int B, float* Y) {
    cuda::dequant_rows_f16(QuantFormat::BF16, W, (size_t)D * 2, D, N, pb_.Wf16, st_);
    cuda::gemm_f16(cublas_, pb_.Wf16, N, D, X, D, B, Y, N, false, st_);
}

int Engine::rope_delta_at(int pos) const {
    int d = 0;
    for (const ImageSpan& s : images_)
        // an image advances positions by max(nx, ny) but fills nx * ny rows
        if (s.start + s.nx * s.ny <= pos) d += std::max(s.nx, s.ny) - s.nx * s.ny;
    return d;
}

bool Engine::rope_positions(int p0, int n, std::vector<int>& pos3) const {
    bool any = false;
    for (const ImageSpan& s : images_) any |= s.start < p0 + n && s.start + s.nx * s.ny > p0;
    if (!any) return false;
    pos3.resize((size_t)3 * n);
    for (int i = 0; i < n; i++) {
        const int q = p0 + i;
        int d = 0, t = -1, h = 0, w = 0;
        for (const ImageSpan& s : images_) {
            const int len = s.nx * s.ny;
            // image entirely before this row: accumulate its offset
            if (q >= s.start + len) { d += std::max(s.nx, s.ny) - len; continue; }
            if (q >= s.start) {
                const int p = s.start + d, k = q - s.start;
                // image token: (t, row, column) relative to the image's start position
                t = p; h = p + k / s.nx; w = p + k % s.nx;
            }
            break;
        }
        if (t < 0) t = h = w = q + d;   // text token: all three components equal
        pos3[3 * i] = t; pos3[3 * i + 1] = h; pos3[3 * i + 2] = w;
    }
    return true;
}

void Engine::prefill_batched(const std::vector<int32_t>& tokens, DecodeResult* last, int topk,
                             const std::vector<int>& logit_rows, std::vector<std::vector<float>>* rows_out, int chunk,
                             const std::function<void(int)>& progress, const std::vector<PrefillImage>& images) {
    const double t_start = now_s();
    discard_predraft();
    const ModelConfig& c = config();
    const float eps = c.rms_eps;
    const int T = (int)tokens.size();
    if (T == 0) return;
    flush_pending();   // the batched DeltaNet kernels start from a fully applied state
    CHECK_CUDA(cudaStreamSynchronize(cst_));   // decode-ring prefetches land in d_slots_ = the prefill dequant scratch
    pprof_ = getenv("Q38_PROFILE_PREFILL") != nullptr;
    for (double& v : pp_ms_) v = 0;
    if (has_mtp_) mtp_catchup();   // MTP prefill continues from position()
    if (pos_ + T > cfg_.max_context) { LOG_ERROR("prefill exceeds context (%d + %d > %d)", pos_, T, cfg_.max_context); return; }
    for (size_t i = 0; i < images.size(); i++) {
        const PrefillImage& im = images[i];
        const int prev_end = i ? images[i - 1].offset + images[i - 1].nx * images[i - 1].ny : 0;
        if (im.nx < 1 || im.ny < 1 || !im.emb || im.offset < prev_end || im.offset + im.nx * im.ny > T) {
            LOG_ERROR("prefill: invalid image span %zu (offset %d, %d x %d, %d tokens)", i, im.offset, im.nx, im.ny, T);
            return;
        }
    }
    for (const PrefillImage& im : images) {
        // spans in cache positions, kept for later decode and snapshots
        images_.push_back({pos_ + im.offset, im.nx, im.ny});
        rope_delta_ += std::max(im.nx, im.ny) - im.nx * im.ny;
    }
    // 2 x 24 MB staging ring for the streamed host weights. When the buffers do not fit (free VRAM moves with the
    // desktop's use, and a cudaMalloc can fail below the reported free size), first place them in the borrowed LM-head
    // rows: that keeps the piece size, hence the GEMM shapes, so the results do not depend on free VRAM (12 MB pieces
    // changed cuBLAS's kernels and the logits). Then halve the pieces, then the chunk.
    int B = std::min(chunk > 0 ? chunk : 1024, T);
    while (!prefill_alloc(B, 24u << 20) && !prefill_alloc(B, 24u << 20, true) && !prefill_alloc(B, 12u << 20, true)) {
        B /= 2;
        if (B < 8) { LOG_ERROR("not enough VRAM for prefill buffers"); return; }
    }
    if (rows_out) rows_out->clear();
    std::vector<std::pair<int, std::vector<float>>> saved;   // (prompt index, final hidden state) for logits
    const size_t erb = quant_row_bytes(tok_embd_->quant, D);
    size_t img = 0;          // first image not entirely before the current row
    std::vector<int> pos3;   // (t, h, w) of a chunk's rows when it holds image tokens
    for (int c0 = 0; c0 < T; c0 += B) {
        const int nb = std::min(B, T - c0);
        const int p0 = pos_;
        const int par = (c0 / B) & 1;
        float* h_emb = pb_.h_emb[par];
        if (emb_pending_[par]) CHECK_CUDA(cudaEventSynchronize(ev_emb_[par]));   // chunk c-2's copies are done
        if (progress && c0 >= 2 * B) progress(c0 - B);   // ... and so is chunk c-2 (ev_emb_ is its last operation)
        for (int t = 0; t < nb; t++) {
            const int gi = c0 + t;
            while (img < images.size() && images[img].offset + images[img].nx * images[img].ny <= gi) img++;
            if (img < images.size() && gi >= images[img].offset)
                // image token: the vision encoder's embedding replaces the token embedding
                memcpy(h_emb + (size_t)t * D, images[img].emb + (size_t)(gi - images[img].offset) * D, D * 4);
            else
                dequantize_row_ref(tok_embd_->quant, gguf_.tensor_data(*tok_embd_) + (size_t)tokens[gi] * erb,
                                   h_emb + (size_t)t * D, D);
        }
        CHECK_CUDA(cudaMemcpyAsync(pb_.X, h_emb, (size_t)nb * D * 4, cudaMemcpyHostToDevice, st_));
        // shared by the attention layers and the MTP block
        if (rope_positions(p0, nb, pos3)) {   // pageable copy: pos3 is staged before the call returns
            CHECK_CUDA(cudaMemcpyAsync(pb_.rope_pos, pos3.data(), pos3.size() * 4, cudaMemcpyHostToDevice, st_));
            cuda::rope_table_pos(pb_.rope_pos, nb, c.rope_theta, pb_.rope, st_);
        } else {
            // text-only chunk: consecutive positions
            cuda::rope_table(p0 + rope_delta_at(p0), nb, c.rope_theta, pb_.rope, st_);
        }
        // start streaming this chunk's host-weight pieces (2 in flight)
        piece_next_ = 0;
        piece_issue_limit_ = pieces_.size();
        for (size_t i = 0; i < std::min<size_t>(2, pieces_.size()); i++) issue_piece(i);
        const float* pending = nullptr;
        for (int il = 0; il < c.num_layers; il++) {
            LayerWeights& w = L_[il];
            // x += previous FFN output, FP16 normalized input for the GEMMs
            cuda::rmsnorm_rows(pb_.X, pending, nullptr, w.attn_norm, nb, D, eps, pb_.Xn16, nullptr, st_);
            if (w.full) {
                gemm_split(w.q, pb_.Xn16, D, nb, pb_.qkv, A_Q);
                gemm_split(w.k, pb_.Xn16, D, nb, pb_.k, A_KV);
                gemm_split(w.v, pb_.Xn16, D, nb, pb_.v, A_KV);
                cuda::attn_prep_rows(pb_.qkv, pb_.k, pb_.v, w.q_norm, w.k_norm, nb, p0, pb_.rope, eps, cfg_.kv_type,
                                     w.kv, pb_.q, pb_.gate, st_);
                pmark(PP_ATTN);
                // causal attention over the cache, the chunk's own keys included
                cuda::attn_prefill(pb_.q, pb_.gate, w.kv, cfg_.kv_type, nb, p0, pb_.O16, st_);
                pmark(PP_OTHER);
                gemm_split(w.o, pb_.O16, A_O, nb, pb_.Y, D);
            } else {
                gemm_split(w.qkv, pb_.Xn16, D, nb, pb_.qkv, DN_CH);
                gemm_split(w.z, pb_.Xn16, D, nb, pb_.z, DN_V);
                gemm_bf16(w.beta, DN_VH, pb_.Xn16, nb, pb_.beta);
                gemm_bf16(w.alpha, DN_VH, pb_.Xn16, nb, pb_.alpha);
                pmark(PP_DN);
                cuda::deltanet_rows(w.dn, pb_.qkv, pb_.z, pb_.beta, pb_.alpha, nb, conv_cur_, eps, pb_.dn_work, pb_.dn_out,
                                    pb_.O16, st_);
                pmark(PP_OTHER);
                gemm_split(w.ssm_out, pb_.O16, DN_V, nb, pb_.Y, D);
            }
            cuda::rmsnorm_rows(pb_.X, pb_.Y, nullptr, w.post_norm, nb, D, eps, pb_.Xn16, nullptr, st_);
            gemm_split(w.gate, pb_.Xn16, D, nb, pb_.G, FF);
            gemm_split(w.up, pb_.Xn16, D, nb, pb_.U, FF);
            cuda::swiglu_rows_inplace(pb_.G, pb_.U, nb, FF, FF, st_);
            gemm_split(w.down, pb_.H16, 2 * FF, nb, pb_.Y, D);
            pending = pb_.Y;
        }
        // residual of the last FFN; keep pre-norm hidden states in X
        cuda::rmsnorm_rows(pb_.X, pb_.Y, nullptr, output_norm_, nb, D, eps, nullptr, nullptr, st_);
        conv_cur_ ^= 1;   // every DeltaNet layer wrote its conv window to the other buffer
        if (has_mtp_) prefill_mtp(tokens, c0, nb, p0, h_emb);
        CHECK_CUDA(cudaEventRecord(ev_emb_[par], st_));
        emb_pending_[par] = true;
        pmark(PP_OTHER);
        pflush();
        pos_ += nb;
        if (progress && c0 + nb == T && c0 >= B) {   // last chunk queued: chunk c-1 is the only one not reported yet
            CHECK_CUDA(cudaEventSynchronize(ev_emb_[par ^ 1]));
            progress(c0);
        }
        // final hidden states of the requested rows (their logits are computed after the prefill: the LM-head rows
        // may hold prefill buffers until then)
        for (int t = 0; t < nb; t++) {
            const int gi = c0 + t;
            const bool want = rows_out && std::find(logit_rows.begin(), logit_rows.end(), gi) != logit_rows.end();
            if (!want && !(gi == T - 1 && last)) continue;
            saved.push_back({gi, std::vector<float>(D)});
            CHECK_CUDA(cudaMemcpyAsync(saved.back().second.data(), pb_.X + (size_t)t * D, D * 4, cudaMemcpyDeviceToHost, st_));
            CHECK_CUDA(cudaStreamSynchronize(st_));
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(st_));
    if (progress) progress(T);
    sseq_ = sissued_ = 0;   // the decode ring held prefill scratch: the next pass re-issues its first pieces
    // Release the prefill scratch (and restore borrowed LM-head rows): keeping ~300 MB of VRAM pinned down left the
    // device nearly full, and decode afterwards ran ~35% slower (CPU work 129 -> 175 ms/token, measured; cause
    // inside the driver / display stack not identified). Re-allocation per prefill costs ~10 ms.
    prefill_free();
    for (const auto& [gi, h] : saved) {
        const bool want = rows_out && std::find(logit_rows.begin(), logit_rows.end(), gi) != logit_rows.end();
        const bool is_last = gi == T - 1;
        CHECK_CUDA(cudaMemcpyAsync(d_x, h.data(), D * 4, cudaMemcpyHostToDevice, st_));
        cuda::add_rmsnorm_rows2(d_x, cuda::RowSrc{}, cuda::RowSrc{}, output_norm_, D, 1, eps, d_xn, nullptr, &act_x_, st_, h_xq);
        CHECK_CUDA(cudaEventRecord(ev_, st_));
        DecodeResult r;
        lm_head(want ? 1 : topk, want, &r, 1, /*streamed=*/false);
        if (want) rows_out->push_back(r.logits);
        if (is_last && last) {
            if (want) {   // derive top-k from the full logits
                std::vector<int> order(r.logits.size());
                for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
                const int k = std::min(std::max(topk, 1), 32);
                std::partial_sort(order.begin(), order.begin() + k, order.end(),
                                  [&](int a, int b2) { return r.logits[a] > r.logits[b2]; });
                last->topk_ids.assign(order.begin(), order.begin() + k);
                last->topk_vals.clear();
                for (int i = 0; i < k; i++) last->topk_vals.push_back(r.logits[order[i]]);
                last->logits = r.logits;
            } else {
                *last = r;
            }
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(st_));
    pstats_.t_total += now_s() - t_start;
    pstats_.tokens += T;
    if (pprof_) {
        static const char* names[PP_N] = {"dequant", "gemm", "wait-copy", "attention", "deltanet", "other"};
        double tot = 0;
        for (double v : pp_ms_) tot += v;
        fprintf(stderr, "prefill GPU time by phase (%d tokens, %.0f ms):", T, tot);
        for (int i = 0; i < PP_N; i++) fprintf(stderr, " %s %.0f (%.0f%%)", names[i], pp_ms_[i], 100.0 * pp_ms_[i] / std::max(tot, 1e-9));
        fprintf(stderr, "\n");
    }
}

} // namespace q38rt
