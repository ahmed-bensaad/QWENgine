/// @file test_metrics.cpp
/// Server metrics: a request's progress events (as Generator reports them) advance the Prometheus counters
/// incrementally and exactly once; queue / processing gauges follow the request's lifetime (also when it ends
/// early); the llama.cpp throughput gauges reset at each query; the exposition is well-formed (HELP and TYPE before
/// each family). The request log goes to stderr.
/// Usage: test_metrics

#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include "server/metrics.h"

using namespace q38rt;

static int g_fail = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); g_fail++; } \
    } while (0)

// sample name (with labels) -> value; checks that every family is declared before its samples
static std::map<std::string, double> parse(const std::string& text) {
    std::map<std::string, double> v;
    std::map<std::string, int> declared;   // family -> HELP / TYPE lines seen
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("# HELP ", 0) == 0 || line.rfind("# TYPE ", 0) == 0) {
            declared[line.substr(7, line.find(' ', 7) - 7)]++;
            continue;
        }
        const size_t sp = line.rfind(' ');
        const std::string name = line.substr(0, sp);
        CHECK(declared[name.substr(0, name.find('{'))] == 2);
        v[name] = std::stod(line.substr(sp + 1));
    }
    return v;
}

static bool near(double a, double b) { return a > b - 1e-6 && a < b + 1e-6; }

int main() {
    metrics::Monitor mon(4096, 512, 0.0);
    metrics::RequestInfo info;
    info.id = "chatcmpl-test";
    info.endpoint = "chat.completions";
    info.prompt_tokens = 3000;
    info.tools = 1;
    info.max_tokens = 64;
    info.effort = "low";
    {
        metrics::Monitor::Request r(mon, info);
        auto m = parse(mon.prometheus());
        CHECK(m["llamacpp:requests_deferred"] == 1 && m["llamacpp:requests_processing"] == 0);
        CHECK(m["q38:kv_cache_capacity_tokens"] == 512 && m["q38:context_limit_tokens"] == 4096);
        r.start();
        m = parse(mon.prometheus());
        CHECK(m["llamacpp:requests_deferred"] == 0 && m["llamacpp:requests_processing"] == 1);

        GenResult g;
        g.prompt_tokens = 3000;
        g.cached_tokens = g.n_past = 1000;
        g.restored = true;
        g.kv_capacity = 512;
        r.event(GenEvent::PromptStart, g);
        g.kv_capacity = 4096;
        g.resize_s = 1.5;
        r.event(GenEvent::Resized, g);
        g.n_past = 2024;
        g.prefill_s = 2.0;
        g.passes = 1;
        r.event(GenEvent::Prompt, g);
        m = parse(mon.prometheus());
        CHECK(m["llamacpp:prompt_tokens_total"] == 1024 && near(m["llamacpp:prompt_seconds_total"], 2.0));
        CHECK(near(m["llamacpp:prompt_tokens_seconds"], 512.0));
        CHECK(near(m["q38:request_prompt_progress"], 2024.0 / 3000) && m["llamacpp:kv_cache_tokens"] == 2024);
        CHECK(m["q38:kv_cache_resizes_total"] == 1 && near(m["q38:kv_cache_resize_seconds_total"], 1.5));
        g.n_past = 3000;
        g.prefill_s = 4.0;
        g.passes = 2;
        r.event(GenEvent::Prompt, g);
        r.event(GenEvent::PromptDone, g);
        m = parse(mon.prometheus());
        CHECK(m["llamacpp:prompt_tokens_total"] == 2000 && near(m["llamacpp:prompt_seconds_total"], 4.0));
        CHECK(near(m["llamacpp:prompt_tokens_seconds"], 976.0 / 2.0));   // since the previous query only
        g.completion_tokens = 4;
        g.decode_s = 0.2;
        g.cycles = 1;
        g.drafted = 3;
        g.accepted = 2;
        g.passes = 3;
        g.n_past = 3004;
        r.event(GenEvent::Decode, g);
        m = parse(mon.prometheus());
        CHECK(m["q38:request_tokens_predicted"] == 4 && m["llamacpp:tokens_predicted_total"] == 4);
        g.completion_tokens = 10;
        g.decode_s = 0.5;
        g.cycles = 3;
        g.drafted = 7;
        g.accepted = 6;
        g.passes = 5;
        g.n_past = 3010;
        r.finish("tool_calls", {"get_weather"}, g);
        m = parse(mon.prometheus());
        CHECK(m["llamacpp:prompt_tokens_total"] == 2000 && m["llamacpp:tokens_predicted_total"] == 10);
        CHECK(near(m["llamacpp:tokens_predicted_seconds_total"], 0.5) && near(m["llamacpp:predicted_tokens_seconds"], 6 / 0.3));
        CHECK(m["llamacpp:n_decode_total"] == 5 && m["llamacpp:n_busy_slots_per_decode"] == 1);
        CHECK(m["llamacpp:kv_cache_tokens"] == 3010 && near(m["llamacpp:kv_cache_usage_ratio"], 3010.0 / 4096));
        CHECK(m["q38:draft_tokens_total"] == 7 && m["q38:draft_tokens_accepted_total"] == 6 && m["q38:verify_passes_total"] == 3);
        CHECK(near(m["q38:draft_acceptance_ratio"], 6.0 / 7));
        CHECK(m["q38:prompt_tokens_cached_total"] == 1000 && m["q38:snapshot_restores_total"] == 1);
        CHECK(m["llamacpp:requests_processing"] == 0 && m["q38:request_tokens_predicted"] == 0);
        CHECK(near(m["q38:last_prompt_tokens_seconds"], 500) && near(m["q38:last_predicted_tokens_seconds"], 20));
        CHECK((m["q38:requests_total{endpoint=\"chat.completions\",finish=\"tool_calls\"}"] == 1));
        m = parse(mon.prometheus());
        CHECK(m["llamacpp:predicted_tokens_seconds"] == 0 && m["llamacpp:prompt_tokens_seconds"] == 0);   // idle interval
    }
    {   // the client leaves while another request has the engine
        metrics::Monitor::Request busy(mon, info);
        busy.start();
        {
            metrics::Monitor::Request r(mon, info);
            CHECK(parse(mon.prometheus())["llamacpp:requests_deferred"] == 1);
        }
        auto m = parse(mon.prometheus());
        CHECK(m["llamacpp:requests_deferred"] == 0 && m["llamacpp:requests_processing"] == 1);
        CHECK((m["q38:requests_total{endpoint=\"chat.completions\",finish=\"abort\"}"] == 1));
    }   // ... and the running one ends without a result
    auto m = parse(mon.prometheus());
    CHECK(m["llamacpp:requests_processing"] == 0);
    CHECK((m["q38:requests_total{endpoint=\"chat.completions\",finish=\"error\"}"] == 1));
    printf("%s\n", g_fail ? "FAILED" : "all metrics checks passed");
    return g_fail ? 1 : 0;
}
