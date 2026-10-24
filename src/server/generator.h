#pragma once
/// @file generator.h
/// Text generation on top of Engine + Tokenizer: prompt reuse (strict continuation of the cached conversation, or
/// the engine snapshot taken at the end of the previous request's conversation), MTP speculative decoding with an
/// adaptive draft length, greedy (exact) or sampled output (temperature, top-k, top-p and min-p over the engine's top-32 logits), stop
/// handling and UTF-8-safe streaming deltas. Prompts may hold images (GenImage): they are encoded by the vision model
/// when their tokens must be processed, and prompt reuse compares them by content hash.
///
/// Sampling with drafts uses speculative sampling with deterministic (argmax) drafts: draft d is accepted with
/// probability p(d) under the target distribution; on rejection the token is drawn from p with d removed. This is
/// exact for the target distribution (temperature, then top-k, top-p and min-p over the engine's top-32 logits).

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "runtime/engine.h"
#include "runtime/tokenizer.h"
#include "runtime/vision.h"

namespace q38rt {

struct GenParams {
    int max_tokens = 512;
    float temperature = 0.0f;     // 0 = greedy
    float top_p = 1.0f;
    int top_k = 0;                // 0 = all of the engine's top-32 candidates
    float min_p = 0.0f;           // drop candidates below min_p x the most likely one's probability (0 = off)
    uint64_t seed = 0;            // 0 = nondeterministic
    std::vector<std::string> stop;
    bool speculative = true;      // use MTP drafts when the engine has the MTP block
};

/// An image in a prompt: tokens [offset, offset + patches->n_tokens()) are its placeholders (<|image_pad|>).
struct GenImage {
    int offset = 0;
    uint64_t hash = 0;                              // identity for prompt reuse: equal hashes = the same image
    std::shared_ptr<const ImagePatches> patches;    // VisionModel::preprocess
    int n_tokens() const { return patches->n_tokens(); }
};

struct GenResult {
    std::string text;
    std::string finish_reason = "length";   // "stop" (EOS or stop string) | "length" | "abort" | "error"
    int prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;
    double prefill_s = 0, decode_s = 0;     // prompt processing and generation time, without KV-cache growth
    double resize_s = 0;                    // time spent growing the KV cache
    int cycles = 0, drafted = 0, accepted = 0;   // speculative verify passes, drafts, accepted drafts
    int passes = 0;                         // engine forward passes (prefill chunks, decode and verify passes)
    int n_past = 0;                         // tokens in the engine's KV cache
    int kv_capacity = 0;                    // the engine's KV capacity (grows on demand)
    bool restored = false;   // the prompt continued from the snapshot of the previous request
    int images = 0;          // images encoded for this prompt (others were cached)
    double encode_s = 0;     // ... and the time spent encoding them (part of prefill_s)
    std::string error;       // finish_reason "error": what failed
};

/// Progress of a generation, reported to an observer on the generating thread with the GenResult so far (for logs
/// and metrics; prefill_s and decode_s are kept current at each event).
enum class GenEvent {
    PromptStart,   // prompt reuse decided: prompt_tokens, cached_tokens, restored, n_past
    Prompt,        // a prefill chunk or decode batch of the prompt is done: n_past
    PromptDone,    // the whole prompt is in the KV cache
    Decode,        // a decode or verify pass is done: completion_tokens, drafted, accepted
    Resized,       // the KV cache grew: kv_capacity, resize_s
};
using GenObserver = std::function<void(GenEvent, const GenResult&)>;

class Generator {
public:
    /// ctx_limit: largest context the engine may grow to (Engine::resize doubles its capacity on demand).
    Generator(Engine& eng, const Tokenizer& tk, int ctx_limit = 0)
        : eng_(eng), tk_(tk), ctx_limit_(ctx_limit > 0 ? ctx_limit : eng.max_context()) {}

    /// Generate a continuation of `prompt`. on_delta(text) receives newly completed text (whole UTF-8 sequences,
    /// stop strings withheld until resolved); returning false aborts generation. `keep`: number of leading prompt
    /// tokens the next request is expected to start with (the conversation, without the assistant's generation
    /// prompt); the engine state there is snapshotted for reuse. 0 = the whole prompt. `observe` receives progress
    /// events. `images`: the prompt's images in order (requires set_vision).
    GenResult generate(const std::vector<int32_t>& prompt, const GenParams& p,
                       const std::function<bool(const std::string&)>& on_delta, size_t keep = 0,
                       const GenObserver& observe = {}, const std::vector<GenImage>& images = {});

    /// The vision encoder for prompts with images.
    void set_vision(const VisionModel* vm) { vision_ = vm; }

    /// Forget the cached conversation (next request re-processes its whole prompt).
    void reset() { history_.clear(); history_images_.clear(); eng_.reset(); }

private:
    Engine& eng_;
    const Tokenizer& tk_;
    std::vector<int32_t> history_;   // tokens committed in the engine (positions 0 .. position()-1)
    struct HistImage { int offset, n; uint64_t hash; };
    std::vector<HistImage> history_images_;   // images among them, in order
    const VisionModel* vision_ = nullptr;
    std::mt19937_64 rng_{std::random_device{}()};
    DraftController dc_;             // draft-length statistics, kept across requests (they decay within ~12 cycles)
    int ctx_limit_;
    bool ensure_capacity(int positions);   // grow the engine (doubling) up to ctx_limit_; false if impossible

    bool is_eos(int32_t t) const;
    // target distribution over a result's candidates (temperature, top-k, top-p, min-p); returns (ids, probs)
    void distribution(const DecodeResult& r, const GenParams& p, std::vector<int32_t>& ids, std::vector<double>& pr) const;
    int32_t sample(const DecodeResult& r, const GenParams& p, int32_t exclude = -1) { return sample(r, p, exclude, rng_); }
    int32_t sample(const DecodeResult& r, const GenParams& p, int32_t exclude, std::mt19937_64& rng) const;
    // acceptance of a verify pass over batch = [next, d_1..d_k]: number of accepted drafts and the token after them
    void accept(const DecodeResult* out, const int32_t* batch, int k, const GenParams& p, std::mt19937_64& rng,
                int& accepted, int32_t& after) const;
};

} // namespace q38rt
