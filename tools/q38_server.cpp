/// @file q38_server.cpp
/// OpenAI-compatible HTTP server for QWENgine (see src/server/api.h for the endpoints, docs/SERVER.md for usage).
///
///   q38_server <model.gguf> [--host 127.0.0.1] [--port 8080] [--ctx 32768] [--ctx-init 4096] [--kv q8_0|q4_0|f16]
///              [--name qwen3.8-27b] [--api-key KEY] [--no-think] [--effort xhigh|medium|low]
///              [--temp T] [--top-p P] [--top-k K] [--min-p P] [--max-tokens N] [--no-spec] [--threads N]
///              [--log-interval S] [--mmproj mmproj.gguf] [--image-min-tokens N] [--image-max-tokens N]
///
/// Sampling defaults (used when a request does not set them) come from the GGUF's general.sampling.* keys, else
/// greedy. The API key can also be given in the Q38_API_KEY environment variable.

#include <cstdio>
#include <cstdlib>
#include <string>
#include "runtime/engine.h"
#include "runtime/tokenizer.h"
#include "runtime/vision.h"
#include "server/api.h"
#include "server/generator.h"
#include "server/http.h"

using namespace q38rt;

static void usage(const char* argv0) {
    fprintf(stderr,
            "usage: %s <model.gguf> [options]\n"
            "  --host H           listen address (default 127.0.0.1; 0.0.0.0 for all interfaces)\n"
            "  --port P           port (default 8080)\n"
            "  --ctx N            largest context (prompt + output tokens; default 32768, at most 262144)\n"
            "  --ctx-init N       initial KV-cache size, doubled on demand up to --ctx (default 4096)\n"
            "  --kv T             KV-cache type: q8_0 (default), q4_0, f16\n"
            "  --name NAME        model id reported by /v1/models (default qwen3.8-27b; requests may use any name)\n"
            "  --api-key KEY      require 'Authorization: Bearer KEY' (or set Q38_API_KEY)\n"
            "  --no-think         answer without reasoning unless a request asks for it\n"
            "  --effort E         default reasoning effort: xhigh (default), medium, low\n"
            "  --temp T, --top-p P, --top-k K, --min-p P\n"
            "                     sampling defaults (default: the GGUF's recommended values)\n"
            "  --max-tokens N     cap on generated tokens per request (default: up to the context limit)\n"
            "  --no-spec          disable MTP speculative decoding\n"
            "  --threads N        CPU threads for the CPU-resident weights\n"
            "  --log-interval S   seconds between decode progress lines of a running request (default 5; 0: none)\n"
            "  --mmproj FILE      vision encoder (mmproj GGUF) for image input\n"
            "  --image-min-tokens N, --image-max-tokens N\n"
            "                     image size range in tokens of 32x32 pixels (default 8 and 1024): images are scaled\n"
            "                     to fit, keeping their aspect ratio\n",
            argv0);
}

int main(int argc, char** argv) {
    if (argc < 2 || argv[1][0] == '-') { usage(argv[0]); return 2; }
    EngineConfig cfg;
    cfg.model_path = argv[1];
    api::Config ac;
    int ctx_init = 4096;
    std::string host = "127.0.0.1";
    int port = 8080;
    float temp = -1, top_p = -1, min_p = -1;
    int top_k = -1;
    std::string mmproj;
    int img_min = 8, img_max = 1024;
    if (const char* k = getenv("Q38_API_KEY")) ac.api_key = k;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) { fprintf(stderr, "option %s needs a value\n", a.c_str()); exit(2); }
            return std::string(argv[++i]);
        };
        if (a == "--host") host = next();
        else if (a == "--port") port = atoi(next().c_str());
        else if (a == "--ctx") ac.ctx_limit = atoi(next().c_str());
        else if (a == "--ctx-init") ctx_init = atoi(next().c_str());
        else if (a == "--threads") cfg.cpu_threads = atoi(next().c_str());
        else if (a == "--name") ac.model_name = next();
        else if (a == "--api-key") ac.api_key = next();
        else if (a == "--no-think") ac.thinking = false;
        else if (a == "--effort") ac.effort = next();
        else if (a == "--temp") temp = (float)atof(next().c_str());
        else if (a == "--top-p") top_p = (float)atof(next().c_str());
        else if (a == "--top-k") top_k = atoi(next().c_str());
        else if (a == "--min-p") min_p = (float)atof(next().c_str());
        else if (a == "--max-tokens") ac.max_tokens = atoi(next().c_str());
        else if (a == "--no-spec") ac.speculative = false;
        else if (a == "--log-interval") ac.log_interval = atof(next().c_str());
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--image-min-tokens") img_min = atoi(next().c_str());
        else if (a == "--image-max-tokens") img_max = atoi(next().c_str());
        else if (a == "--kv") {
            const std::string k = next();
            if (k != "q8_0" && k != "q4_0" && k != "f16") { fprintf(stderr, "--kv: q8_0, q4_0 or f16\n"); return 2; }
            cfg.kv_type = k == "f16" ? cuda::KVType::F16 : k == "q4_0" ? cuda::KVType::Q4_0 : cuda::KVType::Q8_0;
        } else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (ac.effort != "xhigh" && ac.effort != "medium" && ac.effort != "low") {
        fprintf(stderr, "--effort: xhigh, medium or low\n");
        return 2;
    }
    if (ac.ctx_limit < 256) { fprintf(stderr, "--ctx must be at least 256\n"); return 2; }
    if (img_min < 1 || img_max < img_min) { fprintf(stderr, "--image-min-tokens / --image-max-tokens: 1 <= min <= max\n"); return 2; }
    VisionModel vision;   // loaded first: a bad file fails before the long engine start
    if (!mmproj.empty() && !vision.load(mmproj, img_min, img_max)) return 1;
    // the KV cache starts small (more VRAM for weights = faster decode) and doubles on demand up to --ctx
    cfg.max_context = std::min(ctx_init, ac.ctx_limit);
    if (!ac.speculative) cfg.spec_k = 0;   // weight plan for plain decode
    ac.kv_init = cfg.max_context;
    Engine eng;
    if (!eng.init(cfg)) return 1;
    if (ac.ctx_limit > eng.max_context_limit()) {
        fprintf(stderr, "--ctx %d exceeds the supported context of %d tokens\n", ac.ctx_limit, eng.max_context_limit());
        return 2;
    }
    Tokenizer tk;
    if (!tk.load(eng.gguf())) return 1;
    eng.set_verbose(false);   // the plan was shown; later re-plans (KV-cache growth) are logged per request

    const GGUFParser& g = eng.gguf();
    ac.temperature = temp >= 0 ? temp : g.find_kv("general.sampling.temp") ? g.get_kv_float("general.sampling.temp") : 0.0f;
    ac.top_p = top_p >= 0 ? top_p : g.find_kv("general.sampling.top_p") ? g.get_kv_float("general.sampling.top_p") : 1.0f;
    ac.top_k = top_k >= 0 ? top_k : (int)g.get_kv_int("general.sampling.top_k", 0);
    ac.min_p = min_p >= 0 ? min_p : g.find_kv("general.sampling.min_p") ? g.get_kv_float("general.sampling.min_p") : 0.0f;

    Generator gen(eng, tk, ac.ctx_limit);
    api::Server server(gen, tk, ac, vision.loaded() ? &vision : nullptr);
    fprintf(stderr,
            "q38_server: %s on http://%s:%d/v1 (context up to %d tokens, KV cache growing from %d; MTP %s; thinking %s "
            "by default, effort %s; sampling temp %.2f top-p %.2f top-k %d min-p %.2f%s%s); metrics at http://%s:%d/metrics\n",
            ac.model_name.c_str(), host.c_str(), port, ac.ctx_limit, cfg.max_context,
            eng.has_mtp() && ac.speculative ? "speculative" : "off", ac.thinking ? "on" : "off", ac.effort.c_str(),
            ac.temperature, ac.top_p, ac.top_k, ac.min_p, vision.loaded() ? "; image input" : "",
            ac.api_key.empty() ? "" : "; API key required", host.c_str(), port);
    return http::serve(host, port, [&](const http::Request& req, http::Responder& res) { server.handle(req, res); }) ? 0 : 1;
}
