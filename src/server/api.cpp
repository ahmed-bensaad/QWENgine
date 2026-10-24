/// @file api.cpp
/// See api.h.

#include "server/api.h"
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <exception>
#include <functional>
#include <random>
#include "server/chat.h"

namespace q38rt {
namespace api {

using json::Value;

namespace {

std::string random_id(const std::string& prefix) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char* hex = "0123456789abcdef";
    std::string id = prefix;
    for (int i = 0; i < 24; i++) id += hex[rng() % 16];
    return id;
}

bool is_blank(const std::string& s) {
    return std::all_of(s.begin(), s.end(), [](char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; });
}

std::string strip(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\n' || s[a] == '\t' || s[a] == '\r')) a++;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\n' || s[b - 1] == '\t' || s[b - 1] == '\r')) b--;
    return s.substr(a, b - a);
}

/// Request error: HTTP status, message, OpenAI error code.
struct Error {
    int status = 400;
    std::string message, code;
    bool set(const std::string& m, const std::string& c = "", int s = 400) {
        message = m;
        code = c;
        status = s;
        return false;
    }
};

// The request being handled on this thread (one thread per connection), for the error log.
thread_local const http::Request* t_request = nullptr;

void send_error(http::Responder& res, const Error& e) {
    if (t_request)
        metrics::Monitor::log(t_request->method + " " + t_request->path + " -> " + std::to_string(e.status) + " " + e.message);
    Value err = Value::object(), inner = Value::object();
    inner.set("message", e.message);
    inner.set("type", e.status == 401 ? "authentication_error" : e.status >= 500 ? "server_error" : "invalid_request_error");
    inner.set("param", Value());
    inner.set("code", e.code.empty() ? Value() : Value(e.code));
    err.set("error", inner);
    res.send(e.status, "application/json", err.dump());
}

// A Responses content value as Chat Completions content: its text (a string, or parts carrying "text": input_text,
// output_text, ...; refusals) or, when it has input_image parts, an array of text and image_url parts in order.
bool parts_content(const Value& c, Value& out, std::string& err) {
    if (c.is_string() || c.is_null()) { out = Value(c.as_string()); return true; }
    if (!c.is_array()) { err = "unexpected content type"; return false; }
    Value parts = Value::array();
    std::string text, all;
    bool images = false;
    auto flush = [&]() {
        if (text.empty()) return;
        Value p = Value::object();
        p.set("type", "text");
        p.set("text", text);
        parts.push(p);
        text.clear();
    };
    for (const Value& p : c.items()) {
        const std::string type = p["type"].as_string();
        std::string t;
        if (p["text"].is_string()) t = p["text"].as_string();
        else if (type == "refusal") t = p["refusal"].as_string();
        else if (type == "input_image") {
            if (!p["image_url"].is_string()) { err = "input_image parts need an image_url (file ids are not supported)"; return false; }
            flush();
            Value im = Value::object(), u = Value::object();
            u.set("url", p["image_url"].as_string());
            im.set("type", "image_url");
            im.set("image_url", u);
            parts.push(im);
            images = true;
            continue;
        } else {
            err = "unsupported content part type '" + type + "'";
            return false;
        }
        text += t;
        all += t;
    }
    flush();
    out = images ? parts : Value(all);
    return true;
}

// Bytes of a data: URL with a base64 payload ("data:image/png;base64,...").
bool decode_data_url(const std::string& url, std::vector<uint8_t>& out, std::string& err) {
    if (url.compare(0, 5, "data:") != 0) {
        err = url.compare(0, 4, "http") == 0 ? "remote image URLs are not fetched: send the image as a data: URL (base64)"
                                             : "image URLs must be data: URLs (base64)";
        return false;
    }
    const size_t comma = url.find(',');
    if (comma == std::string::npos || comma < 12 || url.compare(comma - 7, 7, ";base64") != 0) {
        err = "image data: URLs must be base64-encoded";
        return false;
    }
    out.clear();
    out.reserve((url.size() - comma) * 3 / 4);
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = comma + 1; i < url.size(); i++) {
        const char c = url[i];
        int v;
        if (c >= 'A' && c <= 'Z') v = c - 'A';
        else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
        else if (c >= '0' && c <= '9') v = c - '0' + 52;
        else if (c == '+' || c == '-') v = 62;
        else if (c == '/' || c == '_') v = 63;
        else if (c == '=') break;
        else if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        else { err = "invalid base64 in image data: URL"; return false; }
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t)(acc >> bits));
        }
    }
    if (out.empty()) { err = "empty image data: URL"; return false; }
    return true;
}

uint64_t fnv1a(const std::vector<uint8_t>& b) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t c : b) h = (h ^ c) * 1099511628211ull;
    return h;
}

/// A prepared generation request.
struct Prepared {
    std::vector<int32_t> prompt;
    Value tools = Value::array();   // tools in the prompt (types the arguments of parsed tool calls)
    bool chat = true;               // false: raw completion (no reasoning / tool-call splitting)
    bool thinking = false;          // the output starts inside <think>
    std::string effort = "none";    // effective reasoning effort (reported)
    size_t keep = 0;                // prompt tokens the next request of the conversation will start with
    std::string prefill;            // assistant output already at the end of the prompt (forced tool call)
    std::vector<GenImage> images;   // images in the prompt (their tokens are <|image_pad|>)
    bool parallel = true;           // more than one tool call per turn
    bool stream = false;
    GenParams gp;
};

/// One assistant turn.
struct Turn {
    std::string reasoning, content;
    std::vector<chat::ToolCall> calls;
    std::string finish = "stop";   // stop | length | tool_calls | abort
    GenResult g;
};

/// Receives a turn's output as it is produced; returning false means the client is gone.
class Sink {
public:
    virtual ~Sink() = default;
    virtual bool reasoning(const std::string&) { return true; }
    virtual bool content(const std::string&) { return true; }
    virtual bool tool_call(const chat::ToolCall&, size_t /*index*/) { return true; }
};

Value tool_call_json(const chat::ToolCall& c) {
    Value fn = Value::object(), tc = Value::object();
    fn.set("name", c.name);
    fn.set("arguments", c.arguments.dump());
    tc.set("id", c.id);
    tc.set("type", "function");
    tc.set("function", fn);
    return tc;
}

/// Chat Completions streaming: one chat.completion.chunk per delta.
class ChatSink : public Sink {
public:
    ChatSink(http::Responder* res, std::function<Value(Value, Value)> chunk) : res_(res), chunk_(std::move(chunk)) {}
    bool reasoning(const std::string& s) override { return send("reasoning_content", Value(s)); }
    bool content(const std::string& s) override { return send("content", Value(s)); }
    bool tool_call(const chat::ToolCall& c, size_t index) override {
        Value tc = tool_call_json(c), item = Value::object(), arr = Value::array();
        item.set("index", (int)index);
        for (const auto& kv : tc.members()) item.set(kv.first, kv.second);
        arr.push(item);
        return send("tool_calls", arr);
    }

private:
    http::Responder* res_;
    std::function<Value(Value, Value)> chunk_;
    bool send(const char* key, Value v) {
        if (!res_) return true;
        Value delta = Value::object();
        delta.set(key, std::move(v));
        return res_->event(chunk_(delta, Value()).dump());
    }
};

/// Responses API: builds the output items and, when streaming, sends the corresponding events.
class ResponsesSink : public Sink {
public:
    explicit ResponsesSink(http::Responder* res) : res_(res) {}
    std::vector<Value> items;   // completed output items

    bool event(const std::string& type, Value e) {
        if (!res_) return true;
        e.set("sequence_number", seq_++);
        return res_->event(type, e.dump());
    }
    static Value ev(const std::string& type) {
        Value e = Value::object();
        e.set("type", type);
        return e;
    }
    bool reasoning(const std::string& s) override {
        bool ok = kind_ == "reasoning" || open("reasoning");
        text_ += s;
        Value e = ev("response.reasoning_text.delta");
        e.set("item_id", id_);
        e.set("output_index", (int)items.size());
        e.set("content_index", 0);
        e.set("delta", s);
        return event("response.reasoning_text.delta", e) && ok;
    }
    bool content(const std::string& s) override {
        bool ok = kind_ == "message" || open("message");
        text_ += s;
        Value e = ev("response.output_text.delta");
        e.set("item_id", id_);
        e.set("output_index", (int)items.size());
        e.set("content_index", 0);
        e.set("delta", s);
        e.set("logprobs", Value::array());
        return event("response.output_text.delta", e) && ok;
    }
    bool tool_call(const chat::ToolCall& c, size_t) override {
        bool ok = close();
        const std::string args = c.arguments.dump();
        Value it = Value::object();
        it.set("id", random_id("fc_"));
        it.set("type", "function_call");
        it.set("status", "in_progress");
        it.set("arguments", "");
        it.set("call_id", c.id);
        it.set("name", c.name);
        const int idx = (int)items.size();
        Value e = ev("response.output_item.added");
        e.set("output_index", idx);
        e.set("item", it);
        ok = event("response.output_item.added", e) && ok;
        e = ev("response.function_call_arguments.delta");
        e.set("item_id", it["id"]);
        e.set("output_index", idx);
        e.set("delta", args);
        ok = event("response.function_call_arguments.delta", e) && ok;
        e = ev("response.function_call_arguments.done");
        e.set("item_id", it["id"]);
        e.set("output_index", idx);
        e.set("arguments", args);
        ok = event("response.function_call_arguments.done", e) && ok;
        it.set("status", "completed");
        it.set("arguments", args);
        e = ev("response.output_item.done");
        e.set("output_index", idx);
        e.set("item", it);
        ok = event("response.output_item.done", e) && ok;
        items.push_back(it);
        return ok;
    }
    /// Close the open reasoning / message item.
    bool close() {
        if (kind_.empty()) return true;
        const int idx = (int)items.size();
        bool ok = true;
        if (kind_ == "reasoning") {
            Value e = ev("response.reasoning_text.done");
            e.set("item_id", id_);
            e.set("output_index", idx);
            e.set("content_index", 0);
            e.set("text", text_);
            ok = event("response.reasoning_text.done", e);
        } else {
            Value e = ev("response.output_text.done");
            e.set("item_id", id_);
            e.set("output_index", idx);
            e.set("content_index", 0);
            e.set("text", text_);
            e.set("logprobs", Value::array());
            ok = event("response.output_text.done", e);
            e = ev("response.content_part.done");
            e.set("item_id", id_);
            e.set("output_index", idx);
            e.set("content_index", 0);
            e.set("part", text_part());
            ok = event("response.content_part.done", e) && ok;
        }
        Value e = ev("response.output_item.done");
        e.set("output_index", idx);
        e.set("item", item(true));
        ok = event("response.output_item.done", e) && ok;
        items.push_back(item(true));
        kind_.clear();
        return ok;
    }

private:
    http::Responder* res_;
    int seq_ = 0;
    std::string kind_, id_, text_;   // open item: "reasoning" | "message" | ""

    Value text_part() const {
        Value p = Value::object();
        p.set("type", "output_text");
        p.set("text", text_);
        p.set("annotations", Value::array());
        return p;
    }
    Value item(bool done) const {
        Value it = Value::object();
        it.set("id", id_);
        if (kind_ == "reasoning") {
            it.set("type", "reasoning");
            it.set("summary", Value::array());
            Value c = Value::array();
            if (done) {
                Value part = Value::object();
                part.set("type", "reasoning_text");
                part.set("text", text_);
                c.push(part);
            }
            it.set("content", c);
        } else {
            it.set("type", "message");
            it.set("status", done ? "completed" : "in_progress");
            it.set("role", "assistant");
            Value c = Value::array();
            if (done) c.push(text_part());
            it.set("content", c);
        }
        return it;
    }
    bool open(const std::string& kind) {
        bool ok = close();
        kind_ = kind;
        text_.clear();
        id_ = random_id(kind == "reasoning" ? "rs_" : "msg_");
        const int idx = (int)items.size();
        Value e = ev("response.output_item.added");
        e.set("output_index", idx);
        e.set("item", item(false));
        ok = event("response.output_item.added", e) && ok;
        if (kind == "message") {
            e = ev("response.content_part.added");
            e.set("item_id", id_);
            e.set("output_index", idx);
            e.set("content_index", 0);
            Value part = Value::object();
            part.set("type", "output_text");
            part.set("text", "");
            part.set("annotations", Value::array());
            e.set("part", part);
            ok = event("response.content_part.added", e) && ok;
        }
        return ok;
    }
};

} // namespace

bool responses_input_to_messages(const Value& input, Value& messages, std::string& err) {
    messages = Value::array();
    if (input.is_string()) {
        Value m = Value::object();
        m.set("role", "user");
        m.set("content", input.as_string());
        messages.push(m);
        return true;
    }
    if (!input.is_array()) { err = "'input' must be a string or an array of items"; return false; }
    std::vector<Value> out;
    std::string reasoning;   // from reasoning items, for the next assistant turn
    auto take_reasoning = [&](Value& m) {
        if (reasoning.empty()) return;
        const std::string prev = m["reasoning_content"].as_string();
        m.set("reasoning_content", prev.empty() ? reasoning : prev + "\n\n" + reasoning);
        reasoning.clear();
    };
    for (const Value& it : input.items()) {
        if (!it.is_object()) { err = "every input item must be an object"; return false; }
        const std::string type = it["type"].as_string();
        if (type.empty() || type == "message") {
            const std::string role = it["role"].as_string();
            if (role != "user" && role != "system" && role != "developer" && role != "assistant") {
                err = "unsupported input message role '" + role + "'";
                return false;
            }
            Value content;
            if (!parts_content(it["content"], content, err)) return false;
            Value m = Value::object();
            m.set("role", role);
            m.set("content", content);
            if (role == "assistant") take_reasoning(m);
            out.push_back(m);
        } else if (type == "function_call") {
            if (!it["arguments"].is_string() && !it["arguments"].is_null()) {
                err = "function_call arguments must be a JSON string";
                return false;
            }
            Value fn = Value::object(), tc = Value::object();
            fn.set("name", it["name"].as_string());
            fn.set("arguments", it["arguments"].as_string());
            tc.set("id", it["call_id"].as_string());
            tc.set("type", "function");
            tc.set("function", fn);
            if (out.empty() || out.back()["role"].as_string() != "assistant") {   // a new assistant turn
                Value m = Value::object();
                m.set("role", "assistant");
                m.set("content", "");
                out.push_back(m);
            }
            Value& m = out.back();   // the assistant turn this call belongs to
            take_reasoning(m);
            Value calls = m["tool_calls"].is_array() ? m["tool_calls"] : Value::array();
            calls.push(tc);
            m.set("tool_calls", calls);
        } else if (type == "function_call_output") {
            Value content;
            if (!parts_content(it["output"], content, err)) return false;
            Value m = Value::object();
            m.set("role", "tool");
            m.set("tool_call_id", it["call_id"].as_string());
            m.set("content", content);
            out.push_back(m);
        } else if (type == "reasoning") {
            std::string text;
            for (const Value& p : it["content"].items()) text += p["text"].as_string();
            if (text.empty())
                for (const Value& p : it["summary"].items())
                    if (!p["text"].as_string().empty()) text += (text.empty() ? "" : "\n\n") + p["text"].as_string();
            reasoning += text;
        } else if (type == "item_reference") {
            err = "item_reference input items are not supported (this server keeps no item store)";
            return false;
        } else {
            err = "unsupported input item type '" + type + "'";
            return false;
        }
    }
    for (Value& m : out) messages.push(std::move(m));
    return true;
}

bool normalize_tools(const Value& tools, Value& out, std::vector<std::string>& skipped, std::string& err) {
    out = Value::array();
    skipped.clear();
    if (tools.is_null()) return true;
    if (!tools.is_array()) { err = "'tools' must be an array"; return false; }
    for (const Value& t : tools.items()) {
        const std::string type = t["type"].as_string();
        if (type != "function") { skipped.push_back(type.empty() ? "(none)" : type); continue; }
        if (t["function"].is_object()) {   // Chat Completions form: rendered as sent
            if (t["function"]["name"].as_string().empty()) { err = "a function tool has no name"; return false; }
            out.push(t);
            continue;
        }
        if (t["name"].as_string().empty()) { err = "a function tool has no name"; return false; }
        Value f = Value::object(), w = Value::object();   // Responses form {type, name, description, parameters}
        f.set("name", t["name"]);
        if (t.has("description")) f.set("description", t["description"]);
        if (t.has("parameters")) f.set("parameters", t["parameters"]);
        w.set("type", "function");
        w.set("function", f);
        out.push(w);
    }
    return true;
}

struct Server::Impl {
    Generator& gen;
    const Tokenizer& tk;
    Config cfg;
    const int64_t started = (int64_t)time(nullptr);
    std::mutex gen_mu;   // one generation at a time
    std::mutex store_mu;
    std::deque<std::pair<std::string, Value>> store;   // recent Responses: id -> conversation (without instructions)
    static constexpr size_t kStoreSize = 32;
    metrics::Monitor monitor;
    const VisionModel* vision = nullptr;   // image input when set
    int32_t image_pad = -1;                // <|image_pad|>

    Impl(Generator& g, const Tokenizer& t, const Config& c, const VisionModel* vm)
        : gen(g), tk(t), cfg(c), monitor(c.ctx_limit, c.kv_init, c.log_interval) {
        if (!vm) return;
        const std::vector<int32_t> pad = tk.encode("<|image_pad|>");
        if (pad.size() != 1) { metrics::Monitor::log("the vocabulary has no <|image_pad|> token: image input disabled"); return; }
        image_pad = pad[0];
        vision = vm;
        gen.set_vision(vm);
    }

    // Images of a rendered conversation (URLs in prompt order): decoded and preprocessed for the vision encoder, and
    // each <|image_pad|> of the conversation's tokens expanded to the image's nx * ny tokens.
    bool add_images(const std::vector<std::string>& urls, Prepared& pr, Error& e) const {
        if ((size_t)std::count(pr.prompt.begin(), pr.prompt.end(), image_pad) != urls.size())
            return e.set("the messages contain the text '<|image_pad|>' (reserved for images)");
        if (urls.empty()) return true;
        std::vector<int32_t> out;
        size_t k = 0;
        for (const int32_t t : pr.prompt) {
            if (t != image_pad) { out.push_back(t); continue; }
            std::vector<uint8_t> bytes;
            std::string err;
            Image img;
            const std::string which = "image " + std::to_string(k + 1) + ": ";
            if (!decode_data_url(urls[k], bytes, err) || !decode_image(bytes.data(), bytes.size(), img, err))
                return e.set(which + err);
            GenImage g;
            g.offset = (int)out.size();
            g.hash = fnv1a(bytes);
            g.patches = std::make_shared<const ImagePatches>(vision->preprocess(img));
            out.insert(out.end(), g.patches->n_tokens(), image_pad);
            pr.images.push_back(std::move(g));
            k++;
        }
        pr.prompt = std::move(out);
        return true;
    }

    // ---- request options ----

    bool sampling(const Value& body, size_t prompt_tokens, std::initializer_list<const char*> max_keys, GenParams& gp,
                  Error& e) const {
        gp.speculative = cfg.speculative;
        gp.temperature = (float)body["temperature"].as_number(cfg.temperature);
        gp.top_p = (float)body["top_p"].as_number(cfg.top_p);
        gp.top_k = (int)body["top_k"].as_number(cfg.top_k);
        gp.min_p = (float)body["min_p"].as_number(cfg.min_p);
        if (gp.temperature < 0 || !(gp.top_p > 0 && gp.top_p <= 1) || gp.top_k < 0 || !(gp.min_p >= 0 && gp.min_p <= 1))
            return e.set("invalid sampling parameters (temperature >= 0, 0 < top_p <= 1, top_k >= 0, 0 <= min_p <= 1)");
        gp.seed = (uint64_t)body["seed"].as_number(0);
        const Value& st = body["stop"];
        if (st.is_string()) {
            if (!st.as_string().empty()) gp.stop.push_back(st.as_string());
        } else if (st.is_array()) {
            for (const Value& s : st.items())
                if (s.is_string() && !s.as_string().empty()) gp.stop.push_back(s.as_string());
        } else if (!st.is_null()) {
            return e.set("'stop' must be a string or an array of strings");
        }
        const int room = cfg.ctx_limit - (int)prompt_tokens - 8;
        if (room < 1)
            return e.set("the prompt (" + std::to_string(prompt_tokens) + " tokens) does not fit the context limit (" +
                         std::to_string(cfg.ctx_limit) + " tokens)", "context_length_exceeded");
        const int cap = cfg.max_tokens > 0 ? std::min(room, cfg.max_tokens) : room;
        gp.max_tokens = cap;
        for (const char* k : max_keys) {
            if (!body[k].is_number()) continue;
            const double v = body[k].as_number();
            if (v < 1) return e.set(std::string("'") + k + "' must be at least 1");
            gp.max_tokens = (int)std::min<double>(v, cap);
            break;
        }
        return true;
    }

    // thinking on/off and effort: reasoning_effort (Chat), reasoning.effort (Responses), chat_template_kwargs
    bool reasoning_options(const Value& body, chat::Options& opt, Error& e) const {
        opt.enable_thinking = cfg.thinking;
        opt.reasoning_effort = cfg.effort;
        const Value& kw = body["chat_template_kwargs"];
        std::string eff;
        if (body["reasoning_effort"].is_string()) eff = body["reasoning_effort"].as_string();
        if (body["reasoning"]["effort"].is_string()) eff = body["reasoning"]["effort"].as_string();
        if (kw["reasoning_effort"].is_string()) eff = kw["reasoning_effort"].as_string();
        if (!eff.empty()) {
            if (eff == "none" || eff == "minimal") {
                opt.enable_thinking = false;
            } else if (eff == "low" || eff == "medium") {
                opt.enable_thinking = true;
                opt.reasoning_effort = eff;
            } else if (eff == "high" || eff == "xhigh") {
                opt.enable_thinking = true;
                opt.reasoning_effort = "xhigh";
            } else {
                return e.set("unsupported reasoning effort '" + eff + "' (none, minimal, low, medium, high, xhigh)");
            }
        }
        if (body["enable_thinking"].is_bool()) opt.enable_thinking = body["enable_thinking"].as_bool();
        if (kw["enable_thinking"].is_bool()) opt.enable_thinking = kw["enable_thinking"].as_bool();
        if (kw["preserve_thinking"].is_bool()) opt.preserve_thinking = kw["preserve_thinking"].as_bool();
        return true;
    }

    // Chat prompt from messages + tools + tool_choice (shared by Chat Completions and Responses).
    bool prepare_chat(const Value& messages, const Value& body, std::initializer_list<const char*> max_keys,
                      Prepared& pr, Error& e) const {
        if (!messages.is_array() || messages.items().empty()) return e.set("'messages' must be a non-empty array");
        chat::Options opt;
        if (!reasoning_options(body, opt, e)) return false;
        std::string err;
        std::vector<std::string> skipped;
        Value tools;
        if (!normalize_tools(body["tools"], tools, skipped, err)) return e.set(err);
        for (const std::string& s : skipped) metrics::Monitor::log("tool type '" + s + "' is not supported, left out");
        // tool_choice: auto | none | required | {"type": "function", "function": {"name": X}} (or "name": X)
        std::string forced;
        bool required = false;
        const Value& tc = body["tool_choice"];
        if (tc.is_string()) {
            const std::string c = tc.as_string();
            if (c == "none") tools = Value::array();
            else if (c == "required") required = true;
            else if (c != "auto") return e.set("unsupported tool_choice '" + c + "' (auto, none, required or a function)");
        } else if (tc.is_object()) {
            if (tc["type"].as_string() != "function") return e.set("unsupported tool_choice type '" + tc["type"].as_string() + "'");
            forced = tc["function"]["name"].is_string() ? tc["function"]["name"].as_string() : tc["name"].as_string();
            bool known = false;
            for (const Value& t : tools.items()) known = known || t["function"]["name"].as_string() == forced;
            if (!known) return e.set("tool_choice names an unknown function '" + forced + "'");
        } else if (!tc.is_null()) {
            return e.set("invalid tool_choice");
        }
        const bool force = required || !forced.empty();
        if (force && tools.items().empty()) return e.set("tool_choice requires tools");
        // the conversation, then the assistant's generation prompt (as the template renders it)
        opt.add_generation_prompt = false;
        std::string conv;
        std::vector<std::string> images;
        if (!chat::render(messages, tools, opt, conv, err, vision ? &images : nullptr)) return e.set(err);
        pr.thinking = opt.enable_thinking;
        std::string gen = opt.enable_thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        if (force) {
            // The call is prefilled after an empty reasoning block. The system prompt stays as rendered for
            // tool_choice "auto", so the conversation prefix (and the engine's cached state) is unchanged.
            pr.prefill = forced.empty() ? "<tool_call>\n<function=" : chat::tool_call_opening(forced);
            gen = "<|im_start|>assistant\n<think>\n\n</think>\n\n" + pr.prefill;
            pr.thinking = false;
        }
        pr.effort = pr.thinking ? opt.reasoning_effort : "none";
        pr.tools = tools;
        pr.parallel = body["parallel_tool_calls"].as_bool(true);
        // Special tokens delimit tokenization, so encoding the two parts separately gives the same tokens as
        // encoding the whole text. The next request's conversation continues this one with an assistant turn that
        // starts with "<|im_start|>": the engine snapshot goes right after that token.
        pr.prompt = tk.encode(conv);
        if (!add_images(images, pr, e)) return false;
        pr.keep = pr.prompt.size() + 1;
        const std::vector<int32_t> g = tk.encode(gen);
        pr.prompt.insert(pr.prompt.end(), g.begin(), g.end());
        return sampling(body, pr.prompt.size(), max_keys, pr.gp, e);
    }

    // ---- generation ----

    metrics::RequestInfo request_info(const std::string& id, const char* endpoint, const Prepared& pr) const {
        metrics::RequestInfo ri;
        ri.id = id;
        ri.endpoint = endpoint;
        ri.prompt_tokens = (int)pr.prompt.size();
        ri.tools = (int)pr.tools.items().size();
        ri.images = (int)pr.images.size();
        ri.max_tokens = pr.gp.max_tokens;
        ri.effort = pr.chat ? pr.effort : "";
        ri.temperature = pr.gp.temperature;
        ri.stream = pr.stream;
        return ri;
    }

    // Run one turn (the caller holds gen_mu and has started `mr`): reasoning / content / tool calls go to `sink` as
    // they complete; progress and the result go to the monitor.
    Turn run(const Prepared& pr, Sink& sink, http::Responder& res, metrics::Monitor::Request& mr) {
        Turn t;
        chat::ThinkSplitter split(pr.thinking);
        chat::ToolCallStreamer tcs;
        const bool tools_on = pr.chat && !pr.tools.items().empty();
        size_t call_done = 0;      // tcs.call_text() consumed up to here
        bool gone = false;         // client disconnected
        bool enough = false;       // parallel_tool_calls = false and one call is complete
        auto add_content = [&](const std::string& s) {
            if (s.empty()) return;
            t.content += s;
            if (!sink.content(s)) gone = true;
        };
        // tool calls completed in the call text (final: also an unterminated last one, kept if `keep_open`)
        auto take_calls = [&](bool final, bool keep_open) {
            const std::string& ct = tcs.call_text();
            while (call_done < ct.size() && !enough) {
                const size_t e = ct.find("</tool_call>", call_done);
                if (e == std::string::npos && !final) break;
                const size_t end = e == std::string::npos ? ct.size() : e + 12;
                const std::string seg = ct.substr(call_done, end - call_done);
                call_done = end;
                std::string between;
                std::vector<chat::ToolCall> cs;
                chat::parse_tool_calls(seg, pr.tools, between, cs);
                if (!is_blank(between)) add_content((t.content.empty() ? "" : "\n\n") + strip(between));
                if (e == std::string::npos && !keep_open) cs.clear();
                for (chat::ToolCall& c : cs) {
                    t.calls.push_back(std::move(c));
                    if (!sink.tool_call(t.calls.back(), t.calls.size() - 1)) gone = true;
                    if (!pr.parallel) { enough = true; break; }
                }
            }
        };
        auto feed_content = [&](const std::string& c) {
            if (!tools_on) { add_content(c); return; }
            add_content(tcs.feed(c));
            if (tcs.in_call()) take_calls(false, false);
        };
        auto feed = [&](const std::string& text) {
            if (!pr.chat || !pr.prefill.empty()) { feed_content(text); return; }
            std::string r, c;
            split.feed(text, r, c);
            if (!r.empty()) {
                t.reasoning += r;
                if (!sink.reasoning(r)) gone = true;
            }
            if (!c.empty()) feed_content(c);
        };
        if (!pr.prefill.empty()) feed_content(pr.prefill);
        t.g = gen.generate(pr.prompt, pr.gp, [&](const std::string& text) {
            feed(text);
            if (!pr.stream && res.client_closed()) gone = true;
            return !gone && !enough;
        }, pr.keep, [&](GenEvent e, const GenResult& g) { mr.event(e, g); }, pr.images);
        if (pr.chat && pr.prefill.empty()) {
            std::string r, c;
            split.finish(r, c);
            if (!r.empty()) {
                t.reasoning += r;
                sink.reasoning(r);
            }
            if (!c.empty()) feed_content(c);
        }
        if (tools_on && !enough) {
            add_content(tcs.finish());
            if (tcs.in_call()) take_calls(true, t.g.finish_reason == "stop");   // no truncated call at the length limit
        }
        if (t.g.finish_reason == "error") t.finish = "error";
        else if (!t.calls.empty()) t.finish = "tool_calls";
        else if (gone) t.finish = "abort";
        else t.finish = t.g.finish_reason == "length" ? "length" : "stop";
        std::vector<std::string> names;
        for (const chat::ToolCall& c : t.calls) names.push_back(c.name);
        mr.finish(t.finish, names, t.g);
        return t;
    }

    Value timings(const GenResult& g) const {
        Value t = Value::object();
        t.set("cached_tokens", g.cached_tokens);
        t.set("snapshot_restored", g.restored);
        t.set("prefill_ms", g.prefill_s * 1e3);
        t.set("prefill_tok_s", (g.prompt_tokens - g.cached_tokens) / std::max(g.prefill_s, 1e-9));
        t.set("decode_tok_s", g.completion_tokens / std::max(g.decode_s, 1e-9));
        t.set("draft_acceptance", g.drafted ? (double)g.accepted / g.drafted : 0.0);
        if (g.resize_s > 0) t.set("kv_resize_ms", g.resize_s * 1e3);
        return t;
    }

    int reasoning_tokens(const Turn& t) const { return t.reasoning.empty() ? 0 : (int)tk.encode(t.reasoning, false).size(); }

    // ---- endpoints ----

    Value model_json() const {
        Value m = Value::object();
        m.set("id", cfg.model_name);
        m.set("object", "model");
        m.set("created", started);
        m.set("owned_by", "qwengine");
        m.set("context_length", cfg.ctx_limit);
        return m;
    }

    void chat_completions(const Value& body, http::Responder& res) {
        Error e;
        if (body["n"].is_number() && body["n"].as_number() != 1) return send_error(res, Error{400, "n > 1 is not supported", ""});
        if (body["logprobs"].as_bool(false)) return send_error(res, Error{400, "logprobs are not supported", ""});
        Prepared pr;
        if (!prepare_chat(body["messages"], body, {"max_completion_tokens", "max_tokens"}, pr, e)) return send_error(res, e);
        pr.stream = body["stream"].as_bool(false);
        const bool usage_chunk = body["stream_options"]["include_usage"].as_bool(false);
        const std::string id = random_id("chatcmpl-");
        const int64_t created = (int64_t)time(nullptr);
        auto chunk = [&](Value delta, Value finish) {
            Value ch = Value::object(), choices = Value::array(), c = Value::object();
            ch.set("id", id);
            ch.set("object", "chat.completion.chunk");
            ch.set("created", created);
            ch.set("model", cfg.model_name);
            c.set("index", 0);
            c.set("delta", std::move(delta));
            c.set("logprobs", Value());
            c.set("finish_reason", std::move(finish));
            choices.push(c);
            ch.set("choices", choices);
            return ch;
        };
        metrics::Monitor::Request mr(monitor, request_info(id, "chat.completions", pr));
        std::lock_guard<std::mutex> lock(gen_mu);
        if (res.client_closed()) return;
        mr.start();
        ChatSink sink(pr.stream ? &res : nullptr, chunk);
        if (pr.stream) {
            res.start_events();
            Value delta = Value::object();
            delta.set("role", "assistant");
            delta.set("content", "");
            res.event(chunk(delta, Value()).dump());
        }
        const Turn t = run(pr, sink, res, mr);
        if (t.finish == "error") {   // before any output (image encoding)
            if (!pr.stream) return send_error(res, Error{500, t.g.error, "server_error"});
            Value err = Value::object(), inner = Value::object();
            inner.set("message", t.g.error);
            inner.set("type", "server_error");
            inner.set("code", Value());
            err.set("error", inner);
            res.event(err.dump());
            res.event("[DONE]");
            return;
        }
        Value usage = Value::object(), ptd = Value::object(), ctd = Value::object();
        usage.set("prompt_tokens", t.g.prompt_tokens);
        usage.set("completion_tokens", t.g.completion_tokens);
        usage.set("total_tokens", t.g.prompt_tokens + t.g.completion_tokens);
        ptd.set("cached_tokens", t.g.cached_tokens);
        usage.set("prompt_tokens_details", ptd);
        ctd.set("reasoning_tokens", reasoning_tokens(t));
        usage.set("completion_tokens_details", ctd);
        const std::string finish = t.finish == "abort" ? "stop" : t.finish;
        if (pr.stream) {
            Value last = chunk(Value::object(), Value(finish));
            last.set("usage", usage);
            last.set("timings", timings(t.g));
            res.event(last.dump());
            if (usage_chunk) {
                Value u = chunk(Value::object(), Value());
                u.set("choices", Value::array());
                u.set("usage", usage);
                res.event(u.dump());
            }
            res.event("[DONE]");
            return;
        }
        Value out = Value::object(), choices = Value::array(), c = Value::object(), msg = Value::object();
        out.set("id", id);
        out.set("object", "chat.completion");
        out.set("created", created);
        out.set("model", cfg.model_name);
        msg.set("role", "assistant");
        msg.set("content", t.content.empty() && !t.calls.empty() ? Value() : Value(t.content));
        if (!strip(t.reasoning).empty()) msg.set("reasoning_content", strip(t.reasoning));
        if (!t.calls.empty()) {
            Value calls = Value::array();
            for (const chat::ToolCall& tc : t.calls) calls.push(tool_call_json(tc));
            msg.set("tool_calls", calls);
        }
        c.set("index", 0);
        c.set("message", msg);
        c.set("logprobs", Value());
        c.set("finish_reason", finish);
        choices.push(c);
        out.set("choices", choices);
        out.set("usage", usage);
        out.set("timings", timings(t.g));
        res.send(200, "application/json", out.dump());
    }

    void responses(const Value& body, http::Responder& res) {
        Error e;
        std::string err;
        Value conv;   // the conversation without the instructions
        if (!responses_input_to_messages(body["input"], conv, err)) return send_error(res, Error{400, err, ""});
        if (body["previous_response_id"].is_string()) {
            const std::string prev = body["previous_response_id"].as_string();
            Value stored;
            bool found = false;
            {
                std::lock_guard<std::mutex> lock(store_mu);
                for (const auto& kv : store)
                    if (kv.first == prev) { stored = kv.second; found = true; }
            }
            if (!found)
                return send_error(res, Error{400, "previous response '" + prev + "' not found (this server keeps the last " +
                                                      std::to_string(kStoreSize) + " responses in memory)",
                                             "previous_response_not_found"});
            for (const Value& m : conv.items()) stored.push(m);
            conv = stored;
        }
        const Value& instr = body["instructions"];
        if (!instr.is_null() && !instr.is_string()) return send_error(res, Error{400, "'instructions' must be a string", ""});
        Value messages = Value::array();
        if (!instr.as_string().empty()) {
            Value m = Value::object();
            m.set("role", "system");
            m.set("content", instr.as_string());
            messages.push(m);
        }
        for (const Value& m : conv.items()) messages.push(m);
        Prepared pr;
        if (!prepare_chat(messages, body, {"max_output_tokens"}, pr, e)) return send_error(res, e);
        pr.stream = body["stream"].as_bool(false);
        const bool store_it = body["store"].as_bool(true);
        const std::string id = random_id("resp_");
        const int64_t created = (int64_t)time(nullptr);
        auto response = [&](const std::string& status, const Value& output, const Value& usage, const Value& incomplete) {
            Value r = Value::object(), reasoning = Value::object(), text = Value::object(), fmt = Value::object();
            r.set("id", id);
            r.set("object", "response");
            r.set("created_at", created);
            r.set("status", status);
            r.set("background", false);
            r.set("error", Value());
            r.set("incomplete_details", incomplete);
            r.set("instructions", instr.is_string() ? instr : Value());
            r.set("max_output_tokens", body["max_output_tokens"].is_number() ? body["max_output_tokens"] : Value());
            r.set("model", cfg.model_name);
            r.set("output", output);
            r.set("parallel_tool_calls", pr.parallel);
            r.set("previous_response_id", body["previous_response_id"].is_string() ? body["previous_response_id"] : Value());
            reasoning.set("effort", pr.effort);
            reasoning.set("summary", Value());
            r.set("reasoning", reasoning);
            r.set("store", store_it);
            r.set("temperature", (double)pr.gp.temperature);
            fmt.set("type", "text");
            text.set("format", fmt);
            r.set("text", text);
            r.set("tool_choice", body["tool_choice"].is_null() ? Value("auto") : body["tool_choice"]);
            r.set("tools", body["tools"].is_array() ? body["tools"] : Value::array());
            r.set("top_p", (double)pr.gp.top_p);
            r.set("truncation", "disabled");
            r.set("usage", usage);
            r.set("metadata", body["metadata"].is_object() ? body["metadata"] : Value::object());
            return r;
        };
        metrics::Monitor::Request mr(monitor, request_info(id, "responses", pr));
        std::lock_guard<std::mutex> lock(gen_mu);
        if (res.client_closed()) return;
        mr.start();
        ResponsesSink sink(pr.stream ? &res : nullptr);
        if (pr.stream) {
            res.start_events();
            Value ev = ResponsesSink::ev("response.created");
            ev.set("response", response("in_progress", Value::array(), Value(), Value()));
            sink.event("response.created", ev);
            ev = ResponsesSink::ev("response.in_progress");
            ev.set("response", response("in_progress", Value::array(), Value(), Value()));
            sink.event("response.in_progress", ev);
        }
        const Turn t = run(pr, sink, res, mr);
        if (t.finish == "error") {   // before any output (image encoding)
            if (!pr.stream) return send_error(res, Error{500, t.g.error, "server_error"});
            Value ev = ResponsesSink::ev("error");
            ev.set("code", "server_error");
            ev.set("message", t.g.error);
            ev.set("param", Value());
            sink.event("error", ev);
            return;
        }
        sink.close();
        Value usage = Value::object(), itd = Value::object(), otd = Value::object(), output = Value::array();
        usage.set("input_tokens", t.g.prompt_tokens);
        itd.set("cached_tokens", t.g.cached_tokens);
        usage.set("input_tokens_details", itd);
        usage.set("output_tokens", t.g.completion_tokens);
        otd.set("reasoning_tokens", reasoning_tokens(t));
        usage.set("output_tokens_details", otd);
        usage.set("total_tokens", t.g.prompt_tokens + t.g.completion_tokens);
        for (const Value& it : sink.items) output.push(it);
        Value incomplete;
        if (t.finish == "length") {
            incomplete = Value::object();
            incomplete.set("reason", "max_output_tokens");
        }
        const std::string status = t.finish == "length" ? "incomplete" : "completed";
        Value r = response(status, output, usage, incomplete);
        r.set("timings", timings(t.g));
        if (store_it && t.finish != "abort") {   // conversation + this answer, for previous_response_id
            Value msg = Value::object();
            msg.set("role", "assistant");
            msg.set("content", t.content);
            if (!strip(t.reasoning).empty()) msg.set("reasoning_content", strip(t.reasoning));
            if (!t.calls.empty()) {
                Value calls = Value::array();
                for (const chat::ToolCall& tc : t.calls) calls.push(tool_call_json(tc));
                msg.set("tool_calls", calls);
            }
            conv.push(msg);
            std::lock_guard<std::mutex> l2(store_mu);
            store.emplace_back(id, conv);
            if (store.size() > kStoreSize) store.pop_front();
        }
        if (pr.stream) {
            const std::string type = status == "completed" ? "response.completed" : "response.incomplete";
            Value ev = ResponsesSink::ev(type);
            ev.set("response", r);
            sink.event(type, ev);
            return;
        }
        res.send(200, "application/json", r.dump());
    }

    void completions(const Value& body, http::Responder& res) {
        Error e;
        const Value& p = body["prompt"];
        std::string text;
        if (p.is_string()) text = p.as_string();
        else if (p.is_array() && p.items().size() == 1 && p.items()[0].is_string()) text = p.items()[0].as_string();
        if (text.empty()) return send_error(res, Error{400, "'prompt' must be a non-empty string", ""});
        if (body["n"].is_number() && body["n"].as_number() != 1) return send_error(res, Error{400, "n > 1 is not supported", ""});
        Prepared pr;
        pr.chat = false;
        pr.prompt = tk.encode(text);
        if (!sampling(body, pr.prompt.size(), {"max_tokens"}, pr.gp, e)) return send_error(res, e);
        pr.stream = body["stream"].as_bool(false);
        const std::string id = random_id("cmpl-");
        const int64_t created = (int64_t)time(nullptr);
        auto chunk = [&](const std::string& s, Value finish) {
            Value ch = Value::object(), choices = Value::array(), c = Value::object();
            ch.set("id", id);
            ch.set("object", "text_completion");
            ch.set("created", created);
            ch.set("model", cfg.model_name);
            c.set("index", 0);
            c.set("text", s);
            c.set("logprobs", Value());
            c.set("finish_reason", std::move(finish));
            choices.push(c);
            ch.set("choices", choices);
            return ch;
        };
        struct TextSink : Sink {
            http::Responder* res;
            std::function<Value(const std::string&, Value)> chunk;
            bool content(const std::string& s) override { return !res || res->event(chunk(s, Value()).dump()); }
        } sink;
        sink.res = pr.stream ? &res : nullptr;
        sink.chunk = chunk;
        metrics::Monitor::Request mr(monitor, request_info(id, "completions", pr));
        std::lock_guard<std::mutex> lock(gen_mu);
        if (res.client_closed()) return;
        mr.start();
        if (pr.stream) res.start_events();
        const Turn t = run(pr, sink, res, mr);
        Value usage = Value::object();
        usage.set("prompt_tokens", t.g.prompt_tokens);
        usage.set("completion_tokens", t.g.completion_tokens);
        usage.set("total_tokens", t.g.prompt_tokens + t.g.completion_tokens);
        const std::string finish = t.finish == "length" ? "length" : "stop";
        if (pr.stream) {
            Value last = chunk("", Value(finish));
            last.set("usage", usage);
            last.set("timings", timings(t.g));
            res.event(last.dump());
            res.event("[DONE]");
            return;
        }
        Value out = chunk(t.content, Value(finish));
        out.set("usage", usage);
        out.set("timings", timings(t.g));
        res.send(200, "application/json", out.dump());
    }

    void handle(const http::Request& req, http::Responder& res) {
        std::string path = req.path;
        if (path.size() > 1 && path.back() == '/') path.pop_back();
        if (req.method == "GET" && (path == "/health" || path == "/")) {
            res.send(200, "application/json", "{\"status\":\"ok\"}");
            return;
        }
        if (path.compare(0, 4, "/v1/") == 0 || path == "/v1") path = path.substr(3);
        if (!cfg.api_key.empty()) {
            auto it = req.headers.find("authorization");
            if (it == req.headers.end() || it->second != "Bearer " + cfg.api_key)
                return send_error(res, Error{401, "invalid or missing API key (Authorization: Bearer <key>)", "invalid_api_key"});
        }
        if (req.method == "GET" && path == "/metrics") {
            res.send(200, "text/plain; version=0.0.4; charset=utf-8", monitor.prometheus());
            return;
        }
        if (req.method == "GET" && path == "/models") {
            Value list = Value::object(), data = Value::array();
            data.push(model_json());
            list.set("object", "list");
            list.set("data", data);
            res.send(200, "application/json", list.dump());
            return;
        }
        if (req.method == "GET" && path.compare(0, 8, "/models/") == 0) {
            if (path.substr(8) != cfg.model_name)
                return send_error(res, Error{404, "model '" + path.substr(8) + "' not found (this server serves '" + cfg.model_name + "')", "model_not_found"});
            res.send(200, "application/json", model_json().dump());
            return;
        }
        const bool post = req.method == "POST";
        if (!post || (path != "/chat/completions" && path != "/responses" && path != "/completions"))
            return send_error(res, Error{404, "unknown endpoint " + req.method + " " + req.path, ""});
        Value body;
        std::string err;
        if (!Value::parse(req.body, body, &err) || !body.is_object())
            return send_error(res, Error{400, "invalid JSON body: " + (err.empty() ? std::string("not an object") : err), ""});
        if (path == "/chat/completions") chat_completions(body, res);
        else if (path == "/responses") responses(body, res);
        else completions(body, res);
    }
};

Server::Server(Generator& gen, const Tokenizer& tk, const Config& cfg, const VisionModel* vision)
    : impl_(new Impl(gen, tk, cfg, vision)) {}
Server::~Server() = default;

void Server::handle(const http::Request& req, http::Responder& res) {
    t_request = &req;
    try {
        impl_->handle(req, res);
    } catch (const std::exception& ex) {
        if (!res.sent()) send_error(res, Error{500, std::string("internal error: ") + ex.what(), ""});
    }
    t_request = nullptr;
}

} // namespace api
} // namespace q38rt
