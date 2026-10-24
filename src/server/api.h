#pragma once
/// @file api.h
/// OpenAI-compatible HTTP API on top of Generator, for agent harnesses and OpenAI SDK clients:
///
///   POST /v1/chat/completions  Chat Completions: tools, tool_choice, parallel_tool_calls, reasoning control
///                              (reasoning_effort, chat_template_kwargs), streaming with tool-call deltas
///   POST /v1/responses         Responses API subset: message / function_call / function_call_output / reasoning
///                              input items, instructions, function tools, previous_response_id (in-memory),
///                              streaming events
///   POST /v1/completions       raw text completion (no chat template)
///   GET  /v1/models, /v1/models/{id}, /health
///   GET  /metrics              Prometheus metrics (llama.cpp names + engine-specific ones; see metrics.h)
///
/// The /v1 prefix is optional. Prompts are rendered with chat::render (the GGUF's chat template); the model's
/// output is split into reasoning (inside <think>), content and tool calls (the template's XML format, returned
/// as OpenAI tool calls with JSON arguments). With a vision encoder, messages may carry images (Chat image_url parts,
/// Responses input_image parts) as data: URLs (base64 PNG, JPEG, BMP, GIF, ...); remote URLs are not fetched. One
/// generation runs at a time: concurrent requests wait their turn. Each request is logged on stderr (arrival, prompt
/// reuse and processing progress, decode speed, timings).

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "runtime/tokenizer.h"
#include "server/generator.h"
#include "server/http.h"
#include "server/json.h"
#include "server/metrics.h"

namespace q38rt {
namespace api {

struct Config {
    std::string model_name = "qwen3.8-27b";
    int ctx_limit = 32768;                     // prompt + generated tokens
    int max_tokens = 0;                        // cap on generated tokens per request (0: up to the context limit)
    bool thinking = true;                      // default when a request does not choose
    std::string effort = "xhigh";              // default reasoning effort (template values: xhigh | medium | low)
    float temperature = 0.0f, top_p = 1.0f;    // sampling defaults when a request does not choose
    int top_k = 0;
    float min_p = 0.0f;
    bool speculative = true;
    std::string api_key;                       // if set, /v1/* and /metrics require "Authorization: Bearer <key>"
    int kv_init = 0;                           // initial KV capacity (reported by /metrics)
    double log_interval = 5;                   // seconds between decode progress lines (0: none)
};

/// Responses API `input` (a string or a list of items) as Chat Completions messages: message items keep their
/// role, function_call items become assistant tool calls (consecutive ones grouped into one assistant turn),
/// function_call_output items become tool messages, reasoning items become the next assistant turn's
/// reasoning_content. False with a client-facing message on unsupported items.
bool responses_input_to_messages(const json::Value& input, json::Value& messages, std::string& err);

/// Tools of either API in the Chat Completions form the template renders ({"type": "function", "function":
/// {"name", "description", "parameters"}}). Chat-form tools are kept as sent. Tools that are not functions
/// (built-in tools of the Responses API) are left out and their types listed in `skipped`.
bool normalize_tools(const json::Value& tools, json::Value& out, std::vector<std::string>& skipped, std::string& err);

class Server {
public:
    /// vision: the mmproj encoder for image input (null: text only).
    Server(Generator& gen, const Tokenizer& tk, const Config& cfg, const VisionModel* vision = nullptr);
    ~Server();
    /// http::Handler. Thread-safe.
    void handle(const http::Request& req, http::Responder& res);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace api
} // namespace q38rt
