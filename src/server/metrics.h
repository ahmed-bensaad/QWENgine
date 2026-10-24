#pragma once
/// @file metrics.h
/// Server observability: the metrics behind GET /metrics and the request log on stderr.
///
/// Metrics use the Prometheus text format. The llama.cpp server's names (llamacpp:*) keep existing dashboards and
/// scrapers working; engine-specific series are under q38:*. Counters advance while a request runs (after every
/// prefill chunk and decode pass), so rate() stays smooth through long generations. As in llama.cpp, the gauges
/// llamacpp:prompt_tokens_seconds and llamacpp:predicted_tokens_seconds average over the interval since the
/// previous /metrics query; q38:last_*_tokens_seconds give the most recent request's speeds.
///
/// The log has one line per request event, tagged with a task number (llama.cpp style): arrival, prompt reuse,
/// prompt processing progress (share of the prompt in the KV cache), periodic decode speed, KV-cache growth, and a
/// timing summary per request (prompt eval / eval / total time, draft acceptance).

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "server/generator.h"

namespace q38rt {
namespace metrics {

/// What the log reports about a new request.
struct RequestInfo {
    std::string id, endpoint;   // response id; chat.completions | responses | completions
    int prompt_tokens = 0, tools = 0, max_tokens = 0, images = 0;
    std::string effort;         // reasoning effort ("" for raw completions)
    float temperature = 0;
    bool stream = false;
};

class Monitor {
public:
    /// ctx_limit: largest context (KV-cache usage ratio); kv_capacity: initial KV capacity; log_interval_s: seconds
    /// between the decode progress lines of a running request (0: no progress lines, only arrival, prompt and
    /// summary).
    Monitor(int ctx_limit, int kv_capacity, double log_interval_s);

    /// One request from arrival to completion: deferred until start(), running until finish(). The destructor
    /// settles the counters when a request ends otherwise (client gone while queued, exception).
    class Request {
    public:
        Request(Monitor& m, const RequestInfo& info);
        ~Request();
        Request(const Request&) = delete;
        Request& operator=(const Request&) = delete;
        void start();                                 // the request has the engine
        void event(GenEvent e, const GenResult& g);   // Generator observer
        /// finish: stop | length | tool_calls | abort; tool_calls: names of the called tools (logged)
        void finish(const std::string& finish, const std::vector<std::string>& tool_calls, const GenResult& g);

    private:
        struct Totals {   // a request's contribution to the counters
            double prompt_tokens = 0, prompt_s = 0, predicted = 0, predicted_s = 0, passes = 0, drafted = 0,
                   accepted = 0, cycles = 0, resize_s = 0, images = 0, encode_s = 0;
        };
        static Totals totals(const GenResult& g);
        void account(const GenResult& g);   // add the counters' progress since the last event (holds m_.mu_)
        std::string tag() const;
        Monitor& m_;
        RequestInfo info_;
        double t_arrive_;
        bool queued_ = false;       // another request had the engine on arrival
        int task_;
        enum class State { Queued, Running, Done } state_ = State::Queued;
        Totals seen_;
        double t_log_ = 0;          // last progress line
        int logged_tokens_ = 0;     // completion tokens at the last decode line
        double logged_s_ = 0;       // decode time at the last decode line
    };

    /// Prometheus text exposition; starts a new interval for the two llama.cpp throughput gauges.
    std::string prometheus();

    /// Write "HH:MM:SS.mmm " + text (+ newline) to stderr in one piece.
    static void log(const std::string& text);

private:
    std::mutex mu_;
    const int ctx_limit_;
    const double log_interval_;
    const double start_time_;
    int next_task_ = 0;
    int deferred_ = 0, processing_ = 0;
    // counters
    double prompt_tokens_ = 0, prompt_s_ = 0, predicted_ = 0, predicted_s_ = 0, passes_ = 0, cached_ = 0;
    double drafted_ = 0, accepted_ = 0, cycles_ = 0, restores_ = 0, resizes_ = 0, resize_s_ = 0;
    double images_ = 0, encode_s_ = 0;
    std::map<std::pair<std::string, std::string>, uint64_t> requests_;   // (endpoint, finish) -> requests
    // throughput since the previous /metrics query
    double b_prompt_tokens_ = 0, b_prompt_s_ = 0, b_predicted_ = 0, b_predicted_s_ = 0;
    // state
    int n_past_ = 0, kv_capacity_ = 0;
    double progress_ = 0;   // running request: share of its prompt in the KV cache
    int predicted_now_ = 0; // running request: tokens generated so far
    double last_prompt_tps_ = 0, last_predicted_tps_ = 0;
};

} // namespace metrics
} // namespace q38rt
