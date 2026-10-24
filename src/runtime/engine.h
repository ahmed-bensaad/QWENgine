#pragma once
/// @file engine.h
/// Inference engine for Qwen3.8-27B (qwen35: 48 Gated DeltaNet + 16 gated-attention layers, optional MTP block)
/// on one 8 GB Turing GPU plus the host CPU. See docs/ARCHITECTURE.md for the full description.
///
/// Decode is hybrid: the residency planner splits every large matrix into a VRAM-resident prefix (rows; for the
/// FFN, 256-wide blocks of the intermediate dimension) computed by the GPU and a host-resident remainder computed
/// by the CPU thread pool at the same time; part of the host remainder is also streamed to the GPU over PCIe each
/// pass. Recurrent state, KV cache, norms, attention and all other non-GEMV work run on the GPU. decode_batch()
/// processes up to MAX_BATCH tokens per pass over the weights (speculative verification); the MTP block drafts.
/// Prefill runs entirely on the GPU (cuBLAS FP16 GEMMs on dequantized weight blocks, host shares streamed).

#include <cstdint>
#include <map>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cpu/cpu_kernels.h"
#include "cpu/thread_pool.h"
#include "cuda/gemv.cuh"
#include "cuda/ops.cuh"
#include "runtime/gguf_parser.h"
#include "runtime/planner.h"

namespace q38rt {

class VisionModel;
struct ImagePatches;

struct EngineConfig {
    std::string model_path;
    int max_context = 8192;
    cuda::KVType kv_type = cuda::KVType::Q8_0;
    int cpu_threads = 8;
    double vram_reserve_mib = 60;       // left free for the desktop / driver (prefill borrows the LM-head rows)
    double gpu_weight_cap_mib = -1;     // optional cap on the weight budget (testing)
    double stream_mib = 2000;           // cap on decode-time PCIe streaming per token (0 = off); the planner picks the amount
    bool mtp = true;                    // load the MTP (nextn) block if the GGUF has one (speculative drafting)
    int draft_rows = 65536;             // LM-head rows kept in VRAM for the MTP draft head (vocabulary prefix)
    bool mtp_q4k = true;                // requantize the MTP block Q6_K -> Q4_K at load (-31% bytes per draft step)
    bool cpu_r4 = true;                 // CPU kernels on R4 re-encoded IQ3_S / IQ3_XXS host rows (exact)
    bool cpu_r2 = true;                 // CPU kernels on R2 re-encoded IQ2_XXS / IQ2_XS / IQ2_S host rows (exact)
    bool mtp_overlap = true;            // speculative passes: draft the next cycle on the GPU (MTP block forced into
                                        // VRAM) while the CPU computes its LM-head rows (decode_batch guess)
    int spec_k = 2;                     // plan for speculative cycles with k drafts (trunk passes of k+1 tokens,
                                        // MTP block k times per cycle; only with the MTP block); 0 = plan for plain
                                        // decode. 2 beat 0 and 1 on all prompt types with adaptive drafts
    bool verbose = true;
};

/// A weight matrix split between GPU (prefix) and CPU (remainder).
struct SplitMat {
    std::string name;
    QuantFormat type = QuantFormat::INVALID;
    int K = 0, N = 0;            // N rows of K elements (GGUF ne0 = K, ne1 = N)
    bool col_split = false;      // FFN down: split along K
    int gpu_part = 0;            // rows (row split) or columns (col split) on the GPU
    void* gpu = nullptr;
    size_t gpu_rb = 0;           // bytes per GPU row
    uint8_t* cpu = nullptr;
    size_t cpu_rb = 0;           // bytes per CPU row
    int gpu_n() const { return col_split ? N : gpu_part; }
    int gpu_k() const { return col_split ? gpu_part : K; }
    int cpu_n() const { return col_split ? N : N - gpu_part; }
    int cpu_k() const { return col_split ? K - gpu_part : K; }
    // Decode-time PCIe streaming: the first `stream_part` rows (row split) or columns (col split) of the
    // host share are copied to VRAM ring slots every token (in slot-sized pieces) and computed by the GPU;
    // the CPU does the rest.
    int stream_part = 0;
    // CPU-only exact re-encoding of the host share (R4 for IQ3, R2 for IQ2; cpu::re_*): same layout as `cpu` (row
    // split: cpu_n rows; col split: N rows of cpu_k columns), used by the CPU kernels; `cpu` stays for streaming and
    // prefill
    uint8_t* cpu_re = nullptr;
    size_t cpu_re_rb = 0;
    const uint8_t* cpu2_ptr_re() const {
        return col_split ? cpu_re + (size_t)(stream_part / 256) * (cpu_re_rb / (cpu_k() / 256)) : cpu_re + (size_t)stream_part * cpu_re_rb;
    }
    size_t str_rb() const { return col_split ? quant_row_bytes(type, stream_part) : cpu_rb; }
    int cpu2_n() const { return col_split ? N : cpu_n() - stream_part; }       // CPU rows after streaming
    int cpu2_k() const { return col_split ? cpu_k() - stream_part : K; }       // CPU columns after streaming
    const uint8_t* cpu2_ptr() const { return col_split ? cpu + str_rb() : cpu + (size_t)stream_part * cpu_rb; }
};

struct LayerWeights {
    bool full = false;
    bool mtp = false;            // the MTP (nextn) block: full attention + FFN, eh_proj in front
    SplitMat eh_proj;            // MTP: [enorm(embed) | hnorm(h)] (10240) -> 5120
    float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
    float* attn_norm = nullptr;
    float* post_norm = nullptr;
    // DeltaNet
    SplitMat qkv, z, ssm_out;
    uint16_t* alpha = nullptr;   // bf16 [48][5120] (GPU)
    uint16_t* beta = nullptr;
    cuda::DeltaNetParams dn;
    // attention
    SplitMat q, k, v, o;
    float* q_norm = nullptr;
    float* k_norm = nullptr;
    cuda::KVLayer kv;
    // FFN: gate/up row split and down column split share gpu_part
    SplitMat gate, up, down;
};

struct DecodeResult {
    std::vector<float> topk_vals;    // descending
    std::vector<int32_t> topk_ids;
    std::vector<float> logits;       // full vocabulary, only if requested
};

/// Adaptive draft length for speculative decoding: tracks the acceptance rate of draft depth d (given that drafts
/// 1..d-1 were accepted) and the measured cycle time per k with exponential decay, and picks the k (1..max_k)
/// maximizing expected tokens per cycle / cycle time.
struct DraftController {
    static constexpr int K_MAX = 6;            // Engine::MAX_BATCH - 1
    int max_k = K_MAX;
    double decay = 0.92;                       // per cycle: memory of ~12 cycles
    // Acceptance at depth 1: prior 0.9 (weight 4). At depth d > 1 the evidence is blended with the estimate of depth
    // d-1 (weight 2): conditional acceptance is nearly constant over depths for a given text (q38_eval draftstats:
    // 0.97 ... 0.90 on code edits, 0.75 ... 0.70 on prose), so a depth not drafted yet inherits the last one's rate
    // and long drafts get tried exactly where they can pay.
    double hits[K_MAX + 1] = {0, 3.6};
    double tries[K_MAX + 1] = {0, 4};
    // relative cycle time per k: verify pass of k+1 tokens + k MTP steps (overlapped with the previous pass), replaced
    // by measurements when adapt_cost (off for reproducible seeded sampling: k must not depend on timing); priors:
    // measured k+1-token pass times relative to 2 tokens (batchtest, 4K context, 2026-10-22)
    double cost[K_MAX + 1] = {0, 1.00, 1.12, 1.27, 1.43, 1.63, 1.83};
    bool cost_measured[K_MAX + 1] = {};
    bool adapt_cost = true;
    int choose() const {
        int best = 1;
        double best_rate = 0, tpc = 1, pr = 1, p = 0;
        for (int k = 1; k <= max_k; k++) {
            p = k == 1 ? hits[1] / tries[1] : (hits[k] + 2.0 * p) / (tries[k] + 2.0);
            pr *= p;
            tpc += pr;
            const double rate = tpc / cost[k];
            if (rate > best_rate) { best_rate = rate; best = k; }
        }
        return best;
    }
    /// after a cycle with k drafts of which `accepted` were kept, taking `seconds`
    void update(int k, int accepted, double seconds) {
        for (int d = 1; d <= max_k; d++) { hits[d] *= decay; tries[d] *= decay; }
        for (int d = 1; d <= k && d <= accepted + 1; d++) {   // depth d evaluated iff drafts 1..d-1 accepted
            tries[d] += 1;
            if (d <= accepted) hits[d] += 1;
        }
        if (!adapt_cost) return;
        if (!cost_measured[k]) {   // first measurement of k: rescale the relative priors of the others
            const double scale = seconds / cost[k];
            for (int j = 1; j <= max_k; j++) if (!cost_measured[j]) cost[j] *= scale;
            cost_measured[k] = true;
        }
        cost[k] = 0.8 * cost[k] + 0.2 * seconds;
    }
};

/// Per-token timing breakdown (host wall clock, accumulated).
struct DecodeStats {
    double t_total = 0, t_cpu_work = 0, t_wait_gpu = 0, t_lm = 0;
    int tokens = 0;
    long predrafts = 0, predraft_misses = 0;   // overlapped drafting: drafts made during a verify pass / discarded
};

class Engine {
public:
    Engine();
    ~Engine();

    bool init(const EngineConfig& cfg);

    /// Process one token at position(); returns the top-k (k <= 32) of the next-token logits,
    /// plus the full logits if want_logits.
    void decode(int32_t token, int topk, bool want_logits, DecodeResult& out) { decode_batch(&token, 1, topk, want_logits, &out); }

    /// Process n <= MAX_BATCH consecutive tokens at positions position() .. position()+n-1 in one pass over the
    /// weights (multi-column GEMVs); out[i] receives the top-k / logits after token i. Results are bit-identical
    /// to n successive decode() calls. 7 = the token to verify + up to 6 drafts (code-like text accepts long drafts).
    static constexpr int MAX_BATCH = 7;
    /// Overlapped drafting (speculative passes with the MTP block): once the GPU's share of the LM head (the resident
    /// and streamed vocabulary rows) has its top-k, `guess(partial, n)` is called with results restricted to those
    /// rows while the CPU computes the rest. It returns the expected outcome of the cycle (accepted drafts, next
    /// token, drafts wanted for the next cycle; k = 0: none) and the engine drafts the next cycle on the GPU as if
    /// that outcome had been committed. take_drafts() hands those drafts out once the real outcome is known.
    struct SpecGuess { int accepted = -1; int32_t next = -1; int k = 0; };
    using SpecGuessFn = std::function<SpecGuess(const DecodeResult* partial, int n)>;
    void decode_batch(const int32_t* tokens, int n, int topk, bool want_logits, DecodeResult* out, bool speculative = false,
                      const SpecGuessFn* guess = nullptr);
    /// After commit(): the drafts made during the last verify pass if its guess matched the real outcome (same
    /// position, same next token, at least k drafts); otherwise the MTP state is rolled back and false is returned
    /// (draft with mtp_draft()).
    bool take_drafts(int32_t next, int k, int32_t* drafts);

    /// After a speculative decode_batch (tokens at position() .. position()+n-1): keep the first m (1 <= m <= n) of
    /// them. position() advances by m; the DeltaNet state is committed lazily (replayed at the start of the next
    /// pass), the KV rows of rejected tokens are simply overwritten later.
    void commit(int m);
    /// Apply pending (committed, not yet replayed) DeltaNet tokens to the state now.
    void flush_pending();

    /// MTP drafting (requires the MTP block, has_mtp()). The MTP layer keeps its own KV cache of pairs
    /// (target h_{q-1}, token x_q) at position q. mtp_catchup() adds the pairs whose token is already known
    /// (q < position()); mtp_draft(next, k) adds the pair for q = position() with the token `next` (sampled from
    /// the last committed logits, not yet decoded) and chains k-1 more steps on the MTP's own hidden state;
    /// writes k draft tokens (argmax of the draft head = resident LM-head rows [0, draft_rows)).
    bool has_mtp() const { return has_mtp_; }
    void mtp_catchup();
    void mtp_draft(int32_t next, int k, int32_t* drafts);

    /// An image in a prefill: tokens [offset, offset + nx * ny) of the prompt are its merged patches in raster order
    /// (their ids are ignored; the chat template uses <|image_pad|>) and take the embeddings `emb` (encode_image,
    /// nx * ny rows of 5120) instead of token embeddings. Positions follow Qwen3-VL's M-RoPE: with p the position the
    /// first image token would have as text, the token in row r, column c gets (t, h, w) = (p, p + r, p + c), and the
    /// text after the image continues at p + max(nx, ny) although the image fills nx * ny cache rows (rope_delta()).
    struct PrefillImage { int offset = 0, nx = 0, ny = 0; const float* emb = nullptr; };
    /// Batched prefill on the GPU (host-resident weight shares streamed over PCIe). Logits are produced for
    /// the prompt positions listed in `logit_rows` (indices into tokens) into `rows_out` (full logits) and the
    /// top-k of the last token into `last_result` (if non-null). chunk = tokens per batch (0 = auto). progress(n) is
    /// called once per chunk when the GPU has finished it, with n = tokens of `tokens` processed so far. `images`:
    /// image token spans, in order, inside `tokens`; the prompt must not end inside one.
    void prefill_batched(const std::vector<int32_t>& tokens, DecodeResult* last_result, int topk,
                         const std::vector<int>& logit_rows = {}, std::vector<std::vector<float>>* rows_out = nullptr,
                         int chunk = 0, const std::function<void(int)>& progress = {},
                         const std::vector<PrefillImage>& images = {});
    /// Rotary position of the next token minus position(): the sum of max(nx, ny) - nx * ny over the images in the
    /// cache (0 for text only).
    int rope_delta() const { return rope_delta_; }
    struct PrefillStats { double t_total = 0, t_copy_wait = 0; double bytes_streamed = 0; int tokens = 0; };
    PrefillStats& prefill_stats() { return pstats_; }

    void reset();

    /// Run the vision encoder on a preprocessed image (runtime/vision.h): FP16 GEMMs on the GPU with the weights
    /// streamed from pinned host memory through the decode streaming ring, activations in temporary VRAM. Writes
    /// n_tokens rows of 5120 floats (the image's embeddings for the language model). Between decode passes only.
    bool encode_image(const VisionModel& vm, const ImagePatches& ip, std::vector<float>& out, std::string& err);
    /// Temporary VRAM outside the decode plan: one buffer per size, in free VRAM or, when that is short, in the
    /// resident LM-head rows (restored by scratch_release). Between decode passes only.
    bool scratch_acquire(const std::vector<size_t>& sizes, std::vector<void*>& out);
    void scratch_release();
    int position() const { return pos_; }
    int max_context() const { return cfg_.max_context; }   // current KV capacity
    /// Largest supported context: the model's training context and the decode attention kernels' limit (262144).
    int max_context_limit() const;
    /// Engine log on stderr (plans, buffer sizes, resizes): EngineConfig::verbose, changeable after init().
    void set_verbose(bool v) { cfg_.verbose = v; }

    /// Grow the KV capacity to new_ctx positions keeping the conversation: the live state (KV rows, DeltaNet
    /// states, MTP rings) is parked in host memory, the engine is re-planned for the smaller weight budget and
    /// reloaded, and the state restored. A few seconds; position() and outputs are unchanged.
    bool resize(int new_ctx);
    /// State snapshot for conversation reuse (agent loops resend the whole history; the re-rendered previous answer
    /// rarely matches the generated tokens exactly): save_snapshot() records the recurrent state at position() in
    /// pinned host memory (DeltaNet states and conv windows, MTP rings; ~160 MB, ~25 ms). KV rows need no copy:
    /// positions below the snapshot are never rewritten unless reset() runs, which invalidates it.
    /// restore_snapshot() returns to that position, after which decoding continues from there.
    bool save_snapshot();
    bool restore_snapshot();
    int snapshot_position() const { return snap_valid_ ? snap_pos_ : -1; }
    /// Benchmark-only: jump to position `pos` over the current (zero-initialized) KV cache and
    /// recurrent state. Decode cost at depth depends only on the KV length, not its contents.
    void skip_to(int pos) { flush_pending(); pos_ = pos; }
    const ModelConfig& config() const { return gguf_.get_config(); }
    const GGUFParser& gguf() const { return gguf_; }
    const Plan& plan() const { return plan_; }
    DecodeStats& stats() { return stats_; }
    size_t vram_used_bytes() const { return vram_used_; }
    /// Wait profile (enabled by env Q38_PROFILE_WAITS=1): per wait site, host time spent while the GPU was
    /// still computing the previous group's GEMVs ("behind") vs the serial chain after them ("chain").
    void print_wait_profile() const;
    size_t host_weight_bytes() const { return host_bytes_; }

private:
    EngineConfig cfg_;
    GGUFParser gguf_;
    Plan plan_;
    std::unique_ptr<cpu::ThreadPool> pool_;
    cudaStream_t st_ = nullptr;
    cudaEvent_t ev_ = nullptr;
    int pos_ = 0;
    int n_trunk_ = 64;           // trunk layers; L_[n_trunk_] is the MTP block when has_mtp_
    bool has_mtp_ = false;
    // lazy DeltaNet commit: conv window buffer holding the current state, tokens committed but not yet applied
    // to the state (inputs in save buffer dn_sb_), an open speculative pass awaiting commit()
    int conv_cur_ = 0, dn_pending_ = 0, dn_sb_ = 0, spec_n_ = 0;
    bool spec_pass_ = false, spec_open_ = false;
    // images in the cache (M-RoPE): cache rows [start, start + nx * ny) per image, in order; text at cache row i >= the
    // last image's end turns at rotary position i + rope_delta_
    struct ImageSpan { int start, nx, ny; };
    std::vector<ImageSpan> images_;
    int rope_delta_ = 0;
    int rope_delta_at(int pos) const;                          // rotary offset of text at cache row pos
    bool rope_positions(int p0, int n, std::vector<int>& pos3) const;   // (t, h, w) of rows p0.. if an image is among them
    float* d_dnsave_ = nullptr;   // [2 buffers][num_layers][MAX_BATCH] x (qkv 10240 | beta 64 | alpha 64)
    // MTP inputs: target final hidden states (output_norm(x)) and tokens per position (ring), MTP chain state
    static constexpr int HRING = 16;
    float* d_hring_ = nullptr;    // [HRING][5120], row = position % HRING
    float* d_hmtp_ = nullptr;     // [5120] MTP output state of the last draft row (chained drafting)
    int32_t tok_ring_[HRING] = {};
    int mtp_pos_ = 0;             // MTP KV holds the true pairs for positions < mtp_pos_
    // snapshot (save_snapshot / restore_snapshot); host buffer kept across resize()
    uint8_t* h_snap_ = nullptr;
    size_t h_snap_size_ = 0;
    bool snap_valid_ = false;
    int snap_pos_ = 0, snap_conv_cur_ = 0, snap_mtp_pos_ = 0, snap_rope_delta_ = 0;
    std::vector<ImageSpan> snap_images_;
    int32_t snap_tok_ring_[HRING] = {};
    void mtp_rows(int q0, int m, const int32_t* toks, const float* h_first, bool h_from_ring, int32_t* draft);
    static constexpr int DNS_TOK = 10240 + 64 + 64;
    // lazy-commit save buffers of the DeltaNet layers only (layer il is attention when (il + 1) % 4 == 0)
    static constexpr int DN_LAYERS = 48;
    float* dn_save(int buf, int il) const {
        return d_dnsave_ + ((size_t)buf * DN_LAYERS + (il - (il + 1) / 4)) * MAX_BATCH * DNS_TOK;
    }
    DecodeStats stats_;
    size_t vram_used_ = 0, host_bytes_ = 0;

    std::vector<LayerWeights> L_;
    float* output_norm_ = nullptr;
    SplitMat lm_;
    const TensorInfo* tok_embd_ = nullptr;

    // device memory
    uint8_t* d_arena_ = nullptr;      // all GPU weights
    uint8_t* d_misc_ = nullptr;       // states, KV, activations
    uint8_t* h_weights_ = nullptr;    // all CPU weights (huge-page backed)
    size_t h_weights_size_ = 0;

    // activations (device)
    float *d_x = nullptr, *d_xn = nullptr, *d_qkv = nullptr, *d_z = nullptr, *d_beta = nullptr, *d_alpha = nullptr,
          *d_o = nullptr, *d_y = nullptr, *d_y2 = nullptr, *d_g = nullptr, *d_u = nullptr, *d_qg = nullptr,
          *d_k = nullptr, *d_v = nullptr, *d_q = nullptr, *d_gate = nullptr, *d_logits = nullptr, *d_tkv = nullptr;
    int* d_tki = nullptr;
    void* d_attn_work = nullptr;
    void* d_topk_work = nullptr;
    cuda::ActQ8 act_x_, act_o_, act_h_;

    // pinned host buffers
    float *h_x = nullptr, *h_o = nullptr, *h_rows = nullptr, *h_rows2 = nullptr, *h_g = nullptr,
          *h_u = nullptr, *h_y2 = nullptr, *h_logits = nullptr, *h_tkv = nullptr;
    int* h_tki = nullptr;
    block_q8_K *h_xq = nullptr, *h_hq = nullptr;

    bool load_weights();
    bool check_formats() const;   // every tensor in a format the kernels handle (other quantizations of the model)
    void teardown();   // free everything init() allocated except the parsed GGUF and the thread pool
    // weight tensors with substituted data (the requantized MTP block); kept across resize()
    struct WOverride { TensorInfo info; std::vector<uint8_t> data; };
    std::map<std::string, WOverride> woverride_;
    const TensorInfo& wt(const std::string& name) const;
    const uint8_t* wdata(const TensorInfo& t) const;
    void requantize_mtp();
    bool parsed_ = false;
    uint8_t* h_base_ = nullptr;   // pinned activation buffers (one allocation)
    bool build_plan(double budget_bytes);
    bool alloc_runtime();
    void upload_split(SplitMat& m, const TensorInfo& t, uint8_t*& dcur, uint8_t*& hcur);

    enum WaitSite { W_DN_IN, W_DN_OUT, W_AT_IN, W_AT_OUT, W_FFN, W_LM, W_MTP, W_NSITES };
    void wait_event(int site);
    void mark_gpu_group();
    void begin_gpu_group();
    std::vector<cudaEvent_t> gprof_ev_;   // per group of a token: GPU begin/end timing events
    std::vector<int> gprof_site_;
    int gprof_n_ = 0;
    double gprof_gpu_[W_NSITES] = {}, gprof_cpu_[W_NSITES] = {};
    int cur_site_ = 0;
    std::vector<double> gcpu_last_, ggpu_last_;   // per group of the last token (profile level 2 dump)
    bool prof_waits_ = false;
    double job_idle_ = 0;   // profiling: CPU job wall time minus mean thread busy time
    double site_idle_[8] = {}, site_maxgap_[8] = {};   // per wait site: imbalance, wall minus the longest thread
    long site_jobs_[8] = {}, idle_hist_[8] = {};       // jobs per site; histogram of per-job imbalance
    double site_skew_[8] = {}, site_endspread_[8] = {}, site_chunkspread_[8] = {};
    cudaEvent_t ev_g_[2] = {nullptr, nullptr};
    int gi_ = 0;
    struct WaitProf { double behind = 0, chain = 0; long n = 0; } wprof_[W_NSITES];
    std::vector<cudaEvent_t> sprof_ev_;   // per slice: GPU stream time before/after waiting for its copy
    std::vector<double> sprof_stall_;     // accumulated GPU stall per slice (s)
    long sprof_tokens_ = 0;
    // nb = tokens in the pass; activations of token t at act + t*act_ld, outputs at y + t*y_ld
    void gemv_gpu(const SplitMat& m, const cuda::ActQ8& act, float* y, int nb = 1, int act_ld = 0, int y_ld = 0);
    void layer_deltanet(int il, int nb);
    void layer_attention(int il, int nb, int pos0);
    void layer_ffn(int il, int nb);
    // CPU-share tails read directly from mapped host memory by the next GPU kernel (no copy kernels):
    // the attention/ssm_out output rows [ytail_split_, D) live in h_rows (stride ytail_ld_, 0 = none),
    // the FFN down CPU partial sums in h_y2 (stride D) when ffn_cpu_ is set.
    int ytail_split_ = 0, ytail_ld_ = 0;
    bool ffn_cpu_ = false;
    cuda::RowSrc y_src() const;      // attention/ssm_out output (device head + host tail)
    cuda::RowSrc y2_src() const;     // FFN CPU partial sums (host) or absent
    void lm_head(int topk, bool want_logits, DecodeResult* out, int nb = 1, bool streamed = true,
                 const SpecGuessFn* guess = nullptr);
    // overlapped drafting (decode_batch guess): drafts of the next cycle made during the LM head of a verify pass
    struct PreDraft { bool valid = false; int pos = 0; int32_t next = -1; int k = 0; int32_t d[MAX_BATCH] = {}; int old_mtp_pos = 0; };
    PreDraft pre_;
    bool pool_async_ = false;            // the pool runs the LM-head job asynchronously: CPU jobs run on the caller
    std::vector<block_q8_K> h_xq_lm_;    // private copy of the LM-head activations for that job
    cudaEvent_t ev_lmtopk_ = nullptr;    // GPU top-k of the LM head copied to the host
    void mtp_draft_at(int pos, int32_t next, int k, int32_t* drafts);
    void discard_predraft();             // roll the MTP state back past unused drafts
    void run_pool(cpu::ThreadPool::Fn fn, void* ctx);   // pool run, or inline on the caller while pool_async_

    // ---- batched prefill ----
    void* cublas_ = nullptr;
    PrefillStats pstats_;
    struct PB {
        int B = 0;
        uint8_t* base = nullptr;
        // live through the whole chunk: residual stream, layer output, normed FP16 input
        float *X, *Y;
        __half* Xn16;
        // one region shared by the phases of a layer and by the MTP block (see prefill_alloc)
        float *qkv, *k, *v, *q, *gate;                 // attention (O16 inside qkv's slot)
        float *z, *beta, *alpha, *dn_out;              // DeltaNet (dn_out and O16 inside qkv's slot)
        void* dn_work;
        __half* O16;
        float *G, *U;                                  // FFN; H16 = G's rows in place (row stride 2*FF halves)
        __half* H16;
        float *mHn, *mHs, *mE, *mk, *mv, *mq, *mgate;  // MTP block
        __half* mH16;
        __half* Wf16;
        uint8_t* stage[2];
        size_t wf16_elems = 0, stage_bytes = 0;   // stage_bytes per slot
        float* h_emb[2] = {nullptr, nullptr};     // pinned [B][5120], per chunk parity (read by async copies)
        void* cublas_ws = nullptr;
        float* rope = nullptr;                    // [B][64]: cos / sin of the chunk's positions (cuda::rope_table)
        int* rope_pos = nullptr;                  // [B][3]: (t, h, w) of chunks with image tokens (cuda::rope_table_pos)
        bool borrowed = false;                    // part of the buffers sits in the resident LM-head rows
    } pb_;
    uint8_t* scratch_base_ = nullptr;   // scratch_acquire's cudaMalloc
    bool scratch_borrowed_ = false;     // ... and whether it borrowed the LM-head rows
    uint8_t* h_lm_gpu_copy_ = nullptr;   // pinned host copy of the resident LM-head rows (restored after prefill)
    uint8_t* h_re_ = nullptr;            // re-encoded copies of the IQ2 / IQ3 host shares (CPU kernels only)
    size_t h_re_size_ = 0;
    void build_re();
    cudaEvent_t ev_emb_[2] = {nullptr, nullptr};  // last async read of h_emb[i] done
    bool emb_pending_[2] = {false, false};
    // streamed host-weight pieces, in the exact order gemm_split consumes them (2-slot staging ring)
    struct Piece { const SplitMat* m; int r0, n; };
    std::vector<Piece> pieces_;
    size_t piece_issue_limit_ = 0, piece_next_ = 0;
    cudaStream_t cst_ = nullptr;                  // copy stream
    cudaEvent_t ev_copied_[2] = {nullptr, nullptr}, ev_used_[2] = {nullptr, nullptr};
    bool slot_used_[2] = {false, false};
    bool host_registered_ = false;
    // decode-time streaming ring
    static constexpr int NSLOT = 8;
    // streaming planner model constants (measured on the target machine)
    static constexpr double kStreamGpuEff = 0.75;        // streamed pieces run at ~75% of the full-matrix GPU rate
    static constexpr double kSliceOverhead = 10e-6;      // per streamed piece: launch + copy-event wait (s)
    static constexpr double kGroupHostOverhead = 15e-6;  // per sync group: quantize, launches, event latency (s)
    static constexpr double kPcieBytesPerS = 6.0e9;      // pinned H2D under decode load
    static constexpr size_t SLOT_BYTES = 8u << 20;
    uint8_t* d_slots_ = nullptr;
    uint8_t* h_strdup_ = nullptr;   // pinned contiguous duplicates of streamed FFN-down column slices
    cudaEvent_t ev_scopied_[NSLOT] = {}, ev_sused_[NSLOT] = {};
    bool sslot_used_[NSLOT] = {};
    // one streamed piece (fits a ring slot): rows [off, off+n) of a row-split stream part, or columns
    // [off, off+n) of a column-split one (host source then is a contiguous pinned duplicate)
    struct SPiece { const SplitMat* m; int off, n; const uint8_t* src; size_t bytes, rb; };
    std::vector<SPiece> spieces_;            // in decode consumption order
    // Streaming sequence numbers run across tokens (slice = seq % count, slot = seq % NSLOT), so the ring keeps
    // prefetching the next token's first slices while the LM head runs (weights are constant).
    uint64_t sseq_ = 0, sissued_ = 0;
    double streamed_bytes_per_token_ = 0;
    void plan_streaming(double budget_bytes);
    void issue_spiece(uint64_t seq);
    void stream_gemv(const SplitMat& m, const cuda::ActQ8& act, float* y, bool accumulate, int nb = 1, int act_ld = 0,
                     int y_ld = 0);
    void skip_stream(const SplitMat& m);
    void build_pieces();
    void issue_piece(size_t i);
    bool prefill_alloc(int B, size_t stage_bytes = 24u << 20, bool borrow = false);
    void prefill_free();
    void prefill_mtp(const std::vector<int32_t>& tokens, int c0, int nb, int p0, const float* h_emb);
    // prefill phase profiling (env Q38_PROFILE_PREFILL=1): GPU time between consecutive marks, per phase
    enum PPhase { PP_DEQ, PP_GEMM, PP_WAIT, PP_ATTN, PP_DN, PP_OTHER, PP_N };
    bool pprof_ = false;
    std::vector<cudaEvent_t> pp_ev_;
    std::vector<int> pp_phase_;
    size_t pp_n_ = 0;
    double pp_ms_[PP_N] = {};
    void pmark(int phase);
    void pflush();
    void gemm_split(const SplitMat& m, const __half* X, int ldx, int B, float* Y, int ldy);
    void gemm_bf16(const uint16_t* W, int N, const __half* X, int B, float* Y);
};

} // namespace q38rt
