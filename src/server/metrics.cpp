/// @file metrics.cpp
/// See metrics.h.

#include "server/metrics.h"
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>

namespace q38rt {
namespace metrics {

namespace {
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    const int n = vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    if (n < (int)sizeof buf) return std::string(buf, n > 0 ? n : 0);
    std::string s(n, '\0');
    va_start(ap, f);
    vsnprintf(&s[0], n + 1, f, ap);
    va_end(ap);
    return s;
}

double per(double a, double b) { return b > 0 ? a / b : 0.0; }
} // namespace

void Monitor::log(const std::string& text) {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    tm t;
    localtime_r(&ts.tv_sec, &t);
    const std::string line = fmt("%02d:%02d:%02d.%03ld ", t.tm_hour, t.tm_min, t.tm_sec, ts.tv_nsec / 1000000) + text + "\n";
    fwrite(line.data(), 1, line.size(), stderr);   // one call: lines of concurrent requests do not interleave
}

Monitor::Monitor(int ctx_limit, int kv_capacity, double log_interval_s)
    : ctx_limit_(ctx_limit), log_interval_(log_interval_s), start_time_((double)time(nullptr)), kv_capacity_(kv_capacity) {}

// ---- requests ----

Monitor::Request::Request(Monitor& m, const RequestInfo& info) : m_(m), info_(info), t_arrive_(now_s()) {
    int ahead;
    {
        std::lock_guard<std::mutex> l(m_.mu_);
        task_ = m_.next_task_++;
        ahead = m_.deferred_ + m_.processing_;
        m_.deferred_++;
    }
    queued_ = ahead > 0;
    std::string s = tag() + "new request " + info.id + ": " + info.endpoint + fmt(", %d prompt tokens", info.prompt_tokens);
    if (info.tools) s += fmt(", %d tool%s", info.tools, info.tools > 1 ? "s" : "");
    if (info.images) s += fmt(", %d image%s", info.images, info.images > 1 ? "s" : "");
    if (!info.effort.empty()) s += ", reasoning " + info.effort;
    s += fmt(", max %d tokens, temperature %.2f", info.max_tokens, info.temperature);
    if (info.stream) s += ", streamed";
    if (queued_) s += fmt(" (queued behind %d request%s)", ahead, ahead > 1 ? "s" : "");
    log(s);
}

Monitor::Request::~Request() {
    if (state_ == State::Done) return;
    {
        std::lock_guard<std::mutex> l(m_.mu_);
        if (state_ == State::Queued) {
            m_.deferred_--;
        } else {
            m_.processing_ = 0;
            m_.progress_ = 0;
            m_.predicted_now_ = 0;
        }
        m_.requests_[{info_.endpoint, state_ == State::Queued ? "abort" : "error"}]++;
    }
    log(tag() + (state_ == State::Queued ? "client disconnected while queued" : "ended without a result"));
}

std::string Monitor::Request::tag() const { return fmt("task %d | ", task_); }

void Monitor::Request::start() {
    {
        std::lock_guard<std::mutex> l(m_.mu_);
        m_.deferred_--;
        m_.processing_ = 1;
        m_.progress_ = 0;
        m_.predicted_now_ = 0;
    }
    state_ = State::Running;
    t_log_ = now_s();
    if (queued_) log(tag() + fmt("started after waiting %.1f s", t_log_ - t_arrive_));
}

Monitor::Request::Totals Monitor::Request::totals(const GenResult& g) {
    Totals t;
    t.prompt_tokens = std::max(0, std::min(g.n_past, g.prompt_tokens) - g.cached_tokens);
    t.prompt_s = g.prefill_s;
    t.predicted = g.completion_tokens;
    t.predicted_s = g.decode_s;
    t.passes = g.passes;
    t.drafted = g.drafted;
    t.accepted = g.accepted;
    t.cycles = g.cycles;
    t.resize_s = g.resize_s;
    t.images = g.images;
    t.encode_s = g.encode_s;
    return t;
}

void Monitor::Request::account(const GenResult& g) {
    const Totals t = totals(g);
    Monitor& m = m_;
    m.prompt_tokens_ += t.prompt_tokens - seen_.prompt_tokens;
    m.prompt_s_ += t.prompt_s - seen_.prompt_s;
    m.predicted_ += t.predicted - seen_.predicted;
    m.predicted_s_ += t.predicted_s - seen_.predicted_s;
    m.b_prompt_tokens_ += t.prompt_tokens - seen_.prompt_tokens;
    m.b_prompt_s_ += t.prompt_s - seen_.prompt_s;
    m.b_predicted_ += t.predicted - seen_.predicted;
    m.b_predicted_s_ += t.predicted_s - seen_.predicted_s;
    m.passes_ += t.passes - seen_.passes;
    m.drafted_ += t.drafted - seen_.drafted;
    m.accepted_ += t.accepted - seen_.accepted;
    m.cycles_ += t.cycles - seen_.cycles;
    m.resize_s_ += t.resize_s - seen_.resize_s;
    m.images_ += t.images - seen_.images;
    m.encode_s_ += t.encode_s - seen_.encode_s;
    m.n_past_ = g.n_past;
    m.kv_capacity_ = g.kv_capacity;
    seen_ = t;
}

void Monitor::Request::event(GenEvent e, const GenResult& g) {
    const double now = now_s();
    const double resize_s = g.resize_s - seen_.resize_s;   // this resize (before account() moves seen_)
    {
        std::lock_guard<std::mutex> l(m_.mu_);
        account(g);
        m_.progress_ = g.prompt_tokens > 0 ? std::min(1.0, (double)g.n_past / g.prompt_tokens) : 1.0;
        m_.predicted_now_ = g.completion_tokens;
        if (e == GenEvent::PromptStart) {
            m_.cached_ += g.cached_tokens;
            if (g.restored) m_.restores_++;
        }
        if (e == GenEvent::Resized) m_.resizes_++;
    }
    switch (e) {
    case GenEvent::PromptStart: {
        std::string s = tag() + fmt("prompt: %d tokens", g.prompt_tokens);
        if (g.cached_tokens)
            s += fmt(", %d reused from the %s", g.cached_tokens,
                     g.restored ? "snapshot of the previous request" : "cached conversation");
        log(s + fmt(", %d to process", g.prompt_tokens - g.cached_tokens));
        t_log_ = now;
        break;
    }
    case GenEvent::Prompt:   // every prefill chunk (~2 s), at most one line per second
        if (m_.log_interval_ > 0 && g.n_past < g.prompt_tokens && now - t_log_ >= std::min(1.0, m_.log_interval_)) {
            log(tag() + fmt("prompt processing progress: n_past = %d / %d (%.1f%%), %.1f tok/s", g.n_past, g.prompt_tokens,
                            100.0 * g.n_past / g.prompt_tokens, per(g.n_past - g.cached_tokens, g.prefill_s)));
            t_log_ = now;
        }
        break;
    case GenEvent::PromptDone: {
        const int n = g.prompt_tokens - g.cached_tokens;
        std::string s = tag() + fmt("prompt done: %d tokens in %.0f ms (%.1f tok/s), n_past = %d", n, g.prefill_s * 1e3,
                                    per(n, g.prefill_s), g.n_past);
        if (g.images) s += fmt(", %d image%s encoded in %.0f ms", g.images, g.images > 1 ? "s" : "", g.encode_s * 1e3);
        log(s);
        t_log_ = now;
        break;
    }
    case GenEvent::Decode:
        if (m_.log_interval_ > 0 && now - t_log_ >= m_.log_interval_) {
            std::string s = tag() + fmt("decoding: n_decoded = %d, %.2f tok/s", g.completion_tokens,
                                        per(g.completion_tokens, g.decode_s));
            if (logged_tokens_ > 0)
                s += fmt(" (last %.0f s: %.2f tok/s)", g.decode_s - logged_s_,
                         per(g.completion_tokens - logged_tokens_, g.decode_s - logged_s_));
            if (g.drafted) s += fmt(", draft acceptance %.0f%%", 100.0 * g.accepted / g.drafted);
            log(s);
            t_log_ = now;
            logged_tokens_ = g.completion_tokens;
            logged_s_ = g.decode_s;
        }
        break;
    case GenEvent::Resized:
        log(tag() + fmt("KV cache grown to %d tokens in %.1f s (n_past = %d)", g.kv_capacity, resize_s, g.n_past));
        break;
    }
}

void Monitor::Request::finish(const std::string& finish, const std::vector<std::string>& tool_calls, const GenResult& g) {
    const int n = std::max(0, g.prompt_tokens - g.cached_tokens);
    {
        std::lock_guard<std::mutex> l(m_.mu_);
        account(g);
        m_.requests_[{info_.endpoint, finish}]++;
        m_.processing_ = 0;
        m_.progress_ = 0;
        m_.predicted_now_ = 0;
        if (n > 0 && g.prefill_s > 0) m_.last_prompt_tps_ = n / g.prefill_s;
        if (g.completion_tokens > 0 && g.decode_s > 0) m_.last_predicted_tps_ = g.completion_tokens / g.decode_s;
    }
    state_ = State::Done;
    std::string what = finish;
    if (finish == "tool_calls") {
        what = "tool calls (";
        for (size_t i = 0; i < tool_calls.size(); i++) what += (i ? ", " : "") + tool_calls[i];
        what += ")";
    } else if (finish == "length") {
        what = "length limit";
    } else if (finish == "abort") {
        what = "aborted (client disconnected)";
    } else if (finish == "error") {
        what = "error: " + g.error;
    }
    const double pt = g.prefill_s * 1e3, et = g.decode_s * 1e3;
    std::string s = tag() + fmt("done: %s, n_past = %d (%.1f%% of the %d-token context)\n", what.c_str(), g.n_past,
                                100.0 * g.n_past / m_.ctx_limit_, m_.ctx_limit_);
    s += fmt("prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n", pt, n,
             per(pt, n), per(n, g.prefill_s));
    if (g.images) s += fmt("image encode time = %9.2f ms / %5d images (included in prompt eval)\n", g.encode_s * 1e3, g.images);
    s += fmt("       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n", et,
             g.completion_tokens, per(et, g.completion_tokens), per(g.completion_tokens, g.decode_s));
    if (g.resize_s > 0) s += fmt("     resize time = %10.2f ms (KV cache grown to %d tokens)\n", g.resize_s * 1e3, g.kv_capacity);
    s += fmt("      total time = %10.2f ms / %5d tokens", pt + et + g.resize_s * 1e3, n + g.completion_tokens);
    if (g.drafted)
        s += fmt("\ndraft acceptance rate = %0.5f (%5d accepted / %5d generated), %d verify passes, %.2f tokens per pass",
                 per(g.accepted, g.drafted), g.accepted, g.drafted, g.cycles, per(g.accepted + g.cycles, g.cycles));
    log(s);
}

// ---- exposition ----

std::string Monitor::prometheus() {
    std::lock_guard<std::mutex> l(mu_);
    std::string out;
    auto head = [&](const char* name, const char* type, const char* help) {
        out += fmt("# HELP %s %s\n# TYPE %s %s\n", name, help, name, type);
    };
    auto metric = [&](const char* name, const char* type, const char* help, double v) {
        head(name, type, help);
        out += fmt("%s %.10g\n", name, v);
    };
    // llama.cpp server names
    metric("llamacpp:prompt_tokens_total", "counter", "Number of prompt tokens processed.", prompt_tokens_);
    metric("llamacpp:prompt_seconds_total", "counter", "Prompt process time", prompt_s_);
    metric("llamacpp:tokens_predicted_total", "counter", "Number of generation tokens processed.", predicted_);
    metric("llamacpp:tokens_predicted_seconds_total", "counter", "Predict process time", predicted_s_);
    metric("llamacpp:n_decode_total", "counter",
           "Total number of engine forward passes (prefill chunks, decode and verify passes).", passes_);
    metric("llamacpp:n_busy_slots_per_decode", "counter", "Average number of busy slots per forward pass (one slot).",
           passes_ > 0 ? 1.0 : 0.0);
    metric("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt throughput in tokens/s (since the previous query).",
           per(b_prompt_tokens_, b_prompt_s_));
    metric("llamacpp:predicted_tokens_seconds", "gauge",
           "Average generation throughput in tokens/s (since the previous query).", per(b_predicted_, b_predicted_s_));
    metric("llamacpp:kv_cache_usage_ratio", "gauge", "KV-cache usage (of the context limit). 1 = 100 percent usage.",
           per(n_past_, ctx_limit_));
    metric("llamacpp:kv_cache_tokens", "gauge", "KV-cache tokens.", n_past_);
    metric("llamacpp:requests_processing", "gauge", "Number of requests processing.", processing_);
    metric("llamacpp:requests_deferred", "gauge", "Number of requests deferred.", deferred_);
    // engine-specific
    metric("q38:prompt_tokens_cached_total", "counter",
           "Prompt tokens reused from the KV cache (cached conversation or snapshot) instead of processed.", cached_);
    metric("q38:snapshot_restores_total", "counter",
           "Requests whose prompt continued from the engine snapshot of the previous request.", restores_);
    metric("q38:draft_tokens_total", "counter", "MTP draft tokens verified.", drafted_);
    metric("q38:draft_tokens_accepted_total", "counter", "MTP draft tokens accepted.", accepted_);
    metric("q38:verify_passes_total", "counter", "Speculative verify passes.", cycles_);
    metric("q38:draft_acceptance_ratio", "gauge", "Accepted / verified MTP draft tokens since start.",
           per(accepted_, drafted_));
    metric("q38:images_encoded_total", "counter", "Images run through the vision encoder (cached ones are not).", images_);
    metric("q38:image_encode_seconds_total", "counter", "Time spent in the vision encoder (part of prompt processing).",
           encode_s_);
    metric("q38:kv_cache_resizes_total", "counter", "KV-cache growths.", resizes_);
    metric("q38:kv_cache_resize_seconds_total", "counter", "Time spent growing the KV cache.", resize_s_);
    metric("q38:kv_cache_capacity_tokens", "gauge", "Allocated KV-cache capacity (grows on demand up to the context limit).",
           kv_capacity_);
    metric("q38:context_limit_tokens", "gauge", "Largest context (prompt + generated tokens).", ctx_limit_);
    metric("q38:request_prompt_progress", "gauge", "Running request: share of its prompt in the KV cache (0 when idle).",
           processing_ ? progress_ : 0.0);
    metric("q38:request_tokens_predicted", "gauge", "Running request: tokens generated so far (0 when idle).",
           processing_ ? predicted_now_ : 0);
    metric("q38:last_prompt_tokens_seconds", "gauge", "Prompt throughput of the last finished request.", last_prompt_tps_);
    metric("q38:last_predicted_tokens_seconds", "gauge", "Generation throughput of the last finished request.",
           last_predicted_tps_);
    head("q38:requests_total", "counter", "Finished requests by endpoint and finish reason.");
    for (const auto& [key, n] : requests_)
        out += fmt("q38:requests_total{endpoint=\"%s\",finish=\"%s\"} %llu\n", key.first.c_str(), key.second.c_str(),
                   (unsigned long long)n);
    metric("process_start_time_seconds", "gauge", "Start time of the process since unix epoch in seconds.", start_time_);
    b_prompt_tokens_ = b_prompt_s_ = b_predicted_ = b_predicted_s_ = 0;
    return out;
}

} // namespace metrics
} // namespace q38rt
