/// @file generator.cpp
/// See generator.h.

#include "server/generator.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace q38rt {

namespace {
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Length of the longest prefix of s made of complete UTF-8 sequences.
size_t utf8_complete(const std::string& s) {
    size_t i = 0, ok = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        const size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        if (i + n > s.size()) break;
        i += n;
        ok = i;
    }
    return ok;
}
} // namespace

bool Generator::ensure_capacity(int positions) {
    if (positions <= eng_.max_context()) return true;
    if (positions > ctx_limit_) return false;
    int cap = eng_.max_context();
    while (cap < positions) cap *= 2;
    return eng_.resize(std::min(cap, ctx_limit_));
}

bool Generator::is_eos(int32_t t) const {
    return t == tk_.eos() || t == 248046 /* <|im_end|> */ || t == 248044 /* <|endoftext|> */;
}

void Generator::distribution(const DecodeResult& r, const GenParams& gp, std::vector<int32_t>& ids,
                             std::vector<double>& pr) const {
    ids.clear();
    pr.clear();
    size_t n = r.topk_ids.size();
    if (gp.top_k > 0) n = std::min(n, (size_t)gp.top_k);
    if (n == 0) return;
    const double l0 = r.topk_vals[0];
    const double temp = gp.temperature, top_p = gp.top_p;
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        const double p = std::exp(((double)r.topk_vals[i] - l0) / temp);
        ids.push_back(r.topk_ids[i]);
        pr.push_back(p);
        sum += p;
    }
    // top-p over the (descending) candidates, then min-p relative to the first (Hugging Face's order)
    double cum = 0;
    size_t keep = n;
    for (size_t i = 0; i < n; i++) {
        cum += pr[i] / sum;
        if (cum >= top_p) { keep = i + 1; break; }
    }
    for (size_t i = 1; i < keep; i++)
        if (pr[i] < gp.min_p * pr[0]) { keep = i; break; }
    ids.resize(keep);
    pr.resize(keep);
    double s2 = 0;
    for (double p : pr) s2 += p;
    for (double& p : pr) p /= s2;
}

int32_t Generator::sample(const DecodeResult& r, const GenParams& p, int32_t exclude, std::mt19937_64& rng) const {
    if (p.temperature <= 0.0f) {
        for (size_t i = 0; i < r.topk_ids.size(); i++) if (r.topk_ids[i] != exclude) return r.topk_ids[i];
        return r.topk_ids.empty() ? 0 : r.topk_ids[0];
    }
    std::vector<int32_t> ids;
    std::vector<double> pr;
    distribution(r, p, ids, pr);
    double tot = 0;
    for (size_t i = 0; i < ids.size(); i++) if (ids[i] != exclude) tot += pr[i];
    if (tot <= 0) return ids.empty() ? r.topk_ids[0] : ids[0];
    double u = std::uniform_real_distribution<double>(0.0, tot)(rng);
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] == exclude) continue;
        if ((u -= pr[i]) <= 0) return ids[i];
    }
    for (size_t i = ids.size(); i-- > 0;) if (ids[i] != exclude) return ids[i];
    return ids[0];
}

// Greedy: a draft is accepted iff it equals the target's argmax. Sampled: accepted with probability p(draft); on
// rejection the token is drawn from p without the draft (speculative sampling with deterministic drafts).
void Generator::accept(const DecodeResult* out, const int32_t* batch, int k, const GenParams& p, std::mt19937_64& rng,
                       int& accepted, int32_t& after) const {
    int j = 0;
    after = -1;
    for (; j < k; j++) {
        const int32_t d = batch[1 + j];
        bool ok;
        if (p.temperature <= 0.0f) {
            ok = !out[j].topk_ids.empty() && d == out[j].topk_ids[0];
        } else {
            std::vector<int32_t> ids;
            std::vector<double> pr;
            distribution(out[j], p, ids, pr);
            double pd = 0;
            for (size_t i = 0; i < ids.size(); i++) if (ids[i] == d) pd = pr[i];
            ok = std::uniform_real_distribution<double>(0.0, 1.0)(rng) < pd;
            if (!ok) after = sample(out[j], p, d, rng);
        }
        if (!ok) break;
    }
    if (after < 0) after = sample(out[j], p, -1, rng);   // all accepted (or greedy): the target's own next token
    accepted = j;
}

GenResult Generator::generate(const std::vector<int32_t>& prompt, const GenParams& p,
                              const std::function<bool(const std::string&)>& on_delta, size_t keep,
                              const GenObserver& observe, const std::vector<GenImage>& images) {
    GenResult res;
    res.prompt_tokens = (int)prompt.size();
    res.n_past = eng_.position();
    res.kv_capacity = eng_.max_context();
    auto event = [&](GenEvent e) { if (observe) observe(e, res); };
    // grow the KV cache to hold `positions` (timed apart: a resize takes seconds and is not prompt or decode work)
    auto grow = [&](int positions) {
        const int before = eng_.max_context();
        const double t = now_s();
        const bool ok = ensure_capacity(positions);
        if (eng_.max_context() != before) {
            res.resize_s += now_s() - t;
            res.kv_capacity = eng_.max_context();
            event(GenEvent::Resized);
        }
        return ok;
    };
    if (prompt.empty()) { res.finish_reason = "stop"; return res; }
    if (p.seed) rng_.seed(p.seed);
    const int topk = p.temperature > 0.0f ? 32 : 1;

    // ---- prompt: reuse the engine state ----
    // 1. the cached conversation (prompt + generated tokens) is a strict prefix: continue from its end;
    // 2. else the new prompt still starts with the previous request's conversation (typical agent loop: the client
    //    re-renders our last answer differently, e.g. without its reasoning): restore the snapshot taken there and
    //    process the rest;
    // 3. else start over.
    const double t0 = now_s();
    size_t common = 0;
    while (common < history_.size() && common < prompt.size() && history_[common] == prompt[common]) common++;
    // images: equal placeholder tokens are not enough, the image must be the same; and reuse never ends inside one
    auto inside = [](size_t at, int offset, int n) { return at > (size_t)offset && at < (size_t)offset + n; };
    for (const HistImage& h : history_images_) {
        if ((size_t)h.offset >= common) break;
        const auto it = std::find_if(images.begin(), images.end(), [&](const GenImage& g) { return g.offset == h.offset; });
        if (it == images.end() || it->hash != h.hash || it->n_tokens() != h.n || inside(common, h.offset, h.n)) {
            common = h.offset;
            break;
        }
    }
    for (const GenImage& g : images)
        if (inside(common, g.offset, g.n_tokens())) common = g.offset;
    size_t start = 0;
    const int snap = eng_.snapshot_position();
    if (common > 0 && common == history_.size() && common < prompt.size()) {
        start = common;
    } else if (snap > 0 && (size_t)snap <= common && (size_t)snap < prompt.size() && eng_.restore_snapshot()) {
        start = (size_t)snap;
        history_.resize(start);
        while (!history_images_.empty() && (size_t)history_images_.back().offset >= start) history_images_.pop_back();
        res.restored = true;
    }
    if (start == 0) { eng_.reset(); history_.clear(); history_images_.clear(); }
    res.cached_tokens = (int)start;
    res.n_past = (int)start;
    event(GenEvent::PromptStart);
    // capacity for the prompt and the requested tokens (as far as the limit allows)
    const int want = (int)prompt.size() + std::min(p.max_tokens, 1024) + 8;
    if ((int)prompt.size() + 8 > ctx_limit_ || !grow(std::min(want, ctx_limit_))) {
        res.finish_reason = "length";
        return res;
    }
    DecodeResult last;
    auto processed = [&](int n_past) {   // a prompt chunk is in the KV cache
        res.passes++;
        res.n_past = n_past;
        res.prefill_s = now_s() - t0 - res.resize_s;
        event(GenEvent::Prompt);
    };
    // prompt[a, b): batched prefill (with the images starting in it, encoded first), or decode batches when short
    auto process = [&](size_t a, size_t b) {
        std::vector<std::vector<float>> emb;
        std::vector<Engine::PrefillImage> spans;
        for (const GenImage& g : images) {
            if ((size_t)g.offset < a || (size_t)g.offset >= b) continue;
            const double te = now_s();
            emb.emplace_back();
            std::string err = vision_ ? "" : "no vision encoder loaded";
            if (!vision_ || !eng_.encode_image(*vision_, *g.patches, emb.back(), err)) {
                res.finish_reason = "error";
                res.error = "image encoding failed: " + err;
                return false;
            }
            res.encode_s += now_s() - te;
            res.images++;
            spans.push_back({g.offset - (int)a, g.patches->nx, g.patches->ny, nullptr});
        }
        for (size_t i = 0; i < spans.size(); i++) spans[i].emb = emb[i].data();
        if (b - a > 16 || !spans.empty()) {
            eng_.prefill_batched(std::vector<int32_t>(prompt.begin() + a, prompt.begin() + b), &last, topk, {}, nullptr,
                                 0, [&](int done) { processed((int)a + done); }, spans);
            return true;
        }
        DecodeResult out[Engine::MAX_BATCH];
        for (size_t i = a; i < b; i += Engine::MAX_BATCH) {
            const int n = (int)std::min<size_t>(Engine::MAX_BATCH, b - i);
            eng_.decode_batch(prompt.data() + i, n, topk, false, out);
            if (eng_.has_mtp()) eng_.mtp_catchup();
            last = out[n - 1];
            processed((int)(i + n));
        }
        return true;
    };
    // snapshot (~25 ms) where the next request is expected to diverge from this one; the generation prompt after
    // it costs one or two decode batches
    size_t snap_at = keep > start && keep < prompt.size() ? keep : prompt.size();
    for (const GenImage& g : images)
        if (inside(snap_at, g.offset, g.n_tokens())) snap_at = prompt.size();
    // an image could not be encoded (e.g. no VRAM for its activations): the request fails, the next one starts over
    auto fail = [&]() {
        eng_.reset();
        history_.clear();
        history_images_.clear();
        return res;
    };
    if (!process(start, snap_at)) return fail();
    eng_.save_snapshot();
    if (snap_at < prompt.size() && !process(snap_at, prompt.size())) return fail();
    history_ = prompt;
    history_images_.clear();
    for (const GenImage& g : images) history_images_.push_back({g.offset, g.n_tokens(), g.hash});
    res.prefill_s = now_s() - t0 - res.resize_s;
    res.n_past = eng_.position();
    event(GenEvent::PromptDone);

    // ---- generation ----
    const double t1 = now_s(), resize0 = res.resize_s;
    auto decoded = [&]() {   // a decode or verify pass is committed
        res.passes++;
        res.n_past = eng_.position();
        res.decode_s = now_s() - t1 - (res.resize_s - resize0);
        event(GenEvent::Decode);
    };
    std::string pending;   // decoded bytes not yet delivered (incomplete UTF-8 or possible stop-string prefix)
    size_t max_stop = 0;
    for (const std::string& s : p.stop) max_stop = std::max(max_stop, s.size());
    bool finished = false;
    // deliver text; returns false when generation must end (stop string or client abort)
    auto emit = [&](int32_t tok) -> bool {
        res.completion_tokens++;
        pending += tk_.token_to_piece(tok);
        for (const std::string& s : p.stop) {
            const size_t at = s.empty() ? std::string::npos : pending.find(s);
            if (at != std::string::npos) {
                const std::string out = pending.substr(0, at);
                res.text += out;
                if (!out.empty() && !on_delta(out)) res.finish_reason = "abort";
                else res.finish_reason = "stop";
                pending.clear();
                return false;
            }
        }
        const size_t complete = utf8_complete(pending);
        const size_t hold = max_stop > 0 ? std::min(complete, max_stop - 1) : 0;   // could start a stop string
        const size_t n_out = complete - hold;
        if (n_out > 0) {
            const std::string out = pending.substr(0, n_out);
            pending.erase(0, n_out);
            res.text += out;
            if (!on_delta(out)) { res.finish_reason = "abort"; return false; }
        }
        return true;
    };

    int32_t next = sample(last, p);
    // Seeded sampling must be reproducible: its draft lengths (which set how the RNG is consumed) may depend neither on
    // timing nor on earlier requests, so it uses a fresh controller without measured costs.
    DraftController seeded;
    seeded.adapt_cost = false;
    DraftController& dc = p.seed ? seeded : dc_;
    const bool spec = p.speculative && eng_.has_mtp();
    int k_next = 0;   // draft count chosen for the next cycle during the previous verify pass (0: choose now)
    while (!finished) {
        if (is_eos(next)) { res.finish_reason = "stop"; break; }
        if (res.completion_tokens >= p.max_tokens) { res.finish_reason = "length"; break; }
        if (eng_.position() + Engine::MAX_BATCH + 1 >= eng_.max_context() &&
            !grow(eng_.position() + Engine::MAX_BATCH + 1 + 1024)) {
            res.finish_reason = "length";
            break;
        }
        if (!spec) {
            if (!emit(next)) break;
            DecodeResult r;
            eng_.decode(next, topk, false, r);
            history_.push_back(next);
            next = sample(r, p);
            decoded();
            continue;
        }
        const int k = std::min(k_next > 0 ? k_next : dc.choose(), p.max_tokens - res.completion_tokens);
        if (k < 1) {   // last token of the budget: no point drafting
            if (!emit(next)) break;
            res.finish_reason = "length";
            break;
        }
        int32_t batch[Engine::MAX_BATCH];
        batch[0] = next;
        const double c0 = now_s();
        if (!eng_.take_drafts(next, k, batch + 1)) eng_.mtp_draft(next, k, batch + 1);
        // While the CPU computes its LM-head rows, guess the outcome from the GPU's rows (same rule, a copy of the
        // RNG) so the engine can draft the next cycle; take_drafts() uses those drafts only if the guess was right.
        k_next = 0;
        const int done = res.completion_tokens;
        const Engine::SpecGuessFn guess = [&](const DecodeResult* part, int) {
            Engine::SpecGuess g;
            std::mt19937_64 rng = rng_;
            int jj;
            int32_t aft;
            accept(part, batch, k, p, rng, jj, aft);
            for (int t = 1; t <= jj; t++) if (is_eos(batch[t])) return g;
            const int left = p.max_tokens - (done + jj + 1);
            if (is_eos(aft) || left < 1) return g;
            g.accepted = jj;
            g.next = aft;
            g.k = k_next = std::min(dc.choose(), left);
            return g;
        };
        DecodeResult out[Engine::MAX_BATCH];
        eng_.decode_batch(batch, k + 1, topk, false, out, true, &guess);
        int j;
        int32_t after;
        accept(out, batch, k, p, rng_, j, after);
        eng_.commit(j + 1);
        dc.update(k, j, now_s() - c0);
        res.cycles++;
        res.drafted += k;
        res.accepted += j;
        for (int t = 0; t <= j; t++) history_.push_back(batch[t]);   // all committed in the engine
        for (int t = 0; t <= j; t++) {
            if (t > 0 && is_eos(batch[t])) { res.finish_reason = "stop"; finished = true; break; }
            if (!emit(batch[t])) { finished = true; break; }
            if (res.completion_tokens >= p.max_tokens) { res.finish_reason = "length"; finished = true; break; }
        }
        next = after;
        decoded();
    }
    if (!pending.empty() && res.finish_reason != "abort") {   // flush what is left (not a stop string)
        res.text += pending;
        on_delta(pending);
    }
    res.decode_s = now_s() - t1 - (res.resize_s - resize0);
    res.n_past = eng_.position();
    return res;
}

} // namespace q38rt
