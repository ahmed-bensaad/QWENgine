/// @file chat.cpp
/// See chat.h. The rendering follows the Jinja template in the GGUF statement by statement; the comments quote the
/// template's conditions.

#include "server/chat.h"
#include <cstring>
#include <random>

namespace q38rt {
namespace chat {

namespace {

const char* kReasoningXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
    "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const char* kReasoningLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
    "unnecessary elaboration.";
const char* kToolsHeader = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const char* kToolsFooter =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

// Length of the whitespace character (Python str.isspace) starting at s[i], 0 if none.
size_t space_len(const std::string& s, size_t i) {
    const unsigned char c = (unsigned char)s[i];
    if (c == ' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f)) return 1;
    if (c == 0xC2 && i + 1 < s.size() && ((unsigned char)s[i + 1] == 0x85 || (unsigned char)s[i + 1] == 0xA0)) return 2;
    if (c == 0xE1 && i + 2 < s.size() && (unsigned char)s[i + 1] == 0x9A && (unsigned char)s[i + 2] == 0x80) return 3;
    if (c == 0xE2 && i + 2 < s.size()) {
        const unsigned char b = (unsigned char)s[i + 1], d = (unsigned char)s[i + 2];
        if (b == 0x80 && ((d >= 0x80 && d <= 0x8A) || d == 0xA8 || d == 0xA9 || d == 0xAF)) return 3;
        if (b == 0x81 && d == 0x9F) return 3;
    }
    if (c == 0xE3 && i + 2 < s.size() && (unsigned char)s[i + 1] == 0x80 && (unsigned char)s[i + 2] == 0x80) return 3;
    return 0;
}

// Python str.strip() (Jinja |trim)
std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size()) { const size_t n = space_len(s, a); if (!n) break; a += n; }
    size_t b = s.size();
    while (b > a) {
        size_t n = 0;
        for (size_t k = 1; k <= 3 && k <= b - a; k++) if (space_len(s, b - k) == k) { n = k; break; }
        if (!n) break;
        b -= n;
    }
    return s.substr(a, b - a);
}

std::string rstrip(const std::string& s) {
    size_t b = s.size();
    while (b > 0 && (s[b - 1] == ' ' || s[b - 1] == '\n' || s[b - 1] == '\t' || s[b - 1] == '\r')) b--;
    return s.substr(0, b);
}

bool starts_with(const std::string& s, const char* p) { return s.compare(0, strlen(p), p) == 0; }
bool ends_with(const std::string& s, const char* p) {
    const size_t n = strlen(p);
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

// render_content(content): string, array of parts with "text" or an image (URLs appended to `images`), or none
bool render_content(const json::Value& c, std::string& out, std::string& err, std::vector<std::string>* images,
                    bool is_system) {
    out.clear();
    if (c.is_string()) { out = c.as_string(); return true; }
    if (c.is_null()) return true;
    if (c.is_array()) {
        for (const json::Value& item : c.items()) {
            const std::string type = item["type"].as_string();
            if (item.has("image") || item.has("image_url") || type == "image" || type == "image_url") {
                if (is_system) { err = "System message cannot contain images."; return false; }
                if (!images) { err = "image input needs the vision encoder (start the server with --mmproj)"; return false; }
                const json::Value& u = item.has("image_url") ? item["image_url"] : item["image"];
                const std::string url = u.is_object() ? u["url"].as_string() : u.as_string();
                if (url.empty()) { err = "image content part without a URL"; return false; }
                images->push_back(url);
                out += "<|vision_start|><|image_pad|><|vision_end|>";
                continue;
            }
            if (item.has("video") || type == "video" || type == "input_audio" || type == "file") {
                err = "video, audio and file content parts are not supported";
                return false;
            }
            if (item.has("text")) { out += item["text"].as_string(); continue; }
            err = "unexpected item type in message content";
            return false;
        }
        return true;
    }
    err = "unexpected message content type";
    return false;
}

std::string role_of(const json::Value& m) {
    std::string r = m["role"].as_string();
    if (r == "developer") r = "system";
    if (r == "function") r = "tool";
    return r;
}

std::string random_call_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char* hex = "0123456789abcdef";
    std::string id = "call_";
    for (int i = 0; i < 24; i++) id += hex[rng() % 16];
    return id;
}

// JSON schema of tool parameter `param` of function `name` (null if unknown)
const json::Value& param_schema(const json::Value& tools, const std::string& name, const std::string& param) {
    static const json::Value none;
    for (const json::Value& t : tools.items()) {
        const json::Value& f = t.has("function") ? t["function"] : t;   // chat format or flat (Responses API)
        if (f["name"].as_string() == name) return f["parameters"]["properties"][param];
    }
    return none;
}

bool schema_is_string(const json::Value& schema) {
    const json::Value& t = schema["type"];
    if (t.is_string()) return t.as_string() == "string";
    if (t.is_array()) {
        bool str = false, other = false;
        for (const json::Value& x : t.items()) {
            if (x.as_string() == "string") str = true;
            else if (x.as_string() != "null") other = true;
        }
        return str && !other;
    }
    return false;
}

json::Value typed_value(const std::string& raw, const json::Value& schema) {
    if (schema_is_string(schema)) return json::Value(raw);
    json::Value v;
    const std::string t = trim(raw);
    if (!t.empty() && json::Value::parse(t, v)) return v;
    if (t == "True" || t == "TRUE") return json::Value(true);
    if (t == "False" || t == "FALSE") return json::Value(false);
    return json::Value(raw);
}

// one "<tool_call>" body (without the tags): XML <function=...> form, or a JSON {"name", "arguments"} object
bool parse_one(const std::string& body_in, const json::Value& tools, ToolCall& call) {
    const std::string body = trim(body_in);
    call.id = random_call_id();
    call.arguments = json::Value::object();
    if (starts_with(body, "{")) {
        json::Value j;
        if (!json::Value::parse(body, j) || !j["name"].is_string()) return false;
        call.name = j["name"].as_string();
        const json::Value& a = j.has("arguments") ? j["arguments"] : j["parameters"];
        if (a.is_object()) call.arguments = a;
        else if (a.is_string()) { json::Value p; if (json::Value::parse(a.as_string(), p) && p.is_object()) call.arguments = p; }
        return true;
    }
    const size_t f = body.find("<function=");
    if (f == std::string::npos) return false;
    const size_t ne = body.find('>', f);
    if (ne == std::string::npos) return false;
    call.name = trim(body.substr(f + 10, ne - f - 10));
    size_t cur = ne + 1;
    while (true) {
        const size_t ps = body.find("<parameter=", cur);
        if (ps == std::string::npos) break;
        const size_t ke = body.find('>', ps);
        if (ke == std::string::npos) break;
        const std::string key = trim(body.substr(ps + 11, ke - ps - 11));
        size_t vs = ke + 1, ve = body.find("</parameter>", vs);
        size_t next = ve == std::string::npos ? std::string::npos : ve + 12;
        if (ve == std::string::npos) {   // unterminated (truncated output): up to the next tag or the end
            ve = std::min(body.find("<parameter=", vs), body.find("</function>", vs));
            if (ve == std::string::npos) ve = body.size();
            next = ve;
        }
        std::string val = body.substr(vs, ve - vs);
        if (!val.empty() && val.front() == '\n') val.erase(0, 1);   // template: "<parameter=k>\n" value "\n</parameter>"
        if (!val.empty() && val.back() == '\n') val.pop_back();
        call.arguments.set(key, typed_value(val, param_schema(tools, call.name, key)));
        if (next == std::string::npos || next >= body.size()) break;
        cur = next;
    }
    return !call.name.empty();
}

} // namespace

bool render(const json::Value& messages, const json::Value& tools, const Options& opt, std::string& out, std::string& err,
            std::vector<std::string>* images) {
    out.clear();
    if (!messages.is_array() || messages.items().empty()) { err = "'messages' must be a non-empty array"; return false; }
    const std::vector<json::Value>& msgs = messages.items();
    const size_t n = msgs.size();
    for (const json::Value& m : msgs) {
        if (!m.is_object()) { err = "every message must be an object"; return false; }
        const std::string r = role_of(m);
        if (r != "system" && r != "user" && r != "assistant" && r != "tool") {
            err = "unsupported message role '" + m["role"].as_string() + "'";
            return false;
        }
    }
    // contents in message order = the order they appear in the prompt (each is rendered once), so images are listed in
    // prompt order
    std::vector<std::string> content(n);
    if (images) images->clear();
    for (size_t i = 0; i < n; i++)
        if (!render_content(msgs[i]["content"], content[i], err, images, role_of(msgs[i]) == "system")) return false;

    // {%- if enable_thinking is undefined or enable_thinking is true %} reasoning instructions
    std::string ri;
    if (opt.enable_thinking) {
        if (opt.reasoning_effort == "xhigh") ri = kReasoningXhigh;
        else if (opt.reasoning_effort == "low") ri = kReasoningLow;
        else if (opt.reasoning_effort != "medium") {
            err = "unsupported reasoning effort '" + opt.reasoning_effort + "' (xhigh, medium, low)";
            return false;
        }
    }
    const bool first_system = role_of(msgs[0]) == "system";
    const bool have_tools = tools.is_array() && !tools.items().empty();
    if (have_tools) {
        out += "<|im_start|>system\n";
        if (!ri.empty()) out += ri + "\n\n";
        out += kToolsHeader;
        for (const json::Value& t : tools.items()) out += "\n" + t.dump_py();
        out += kToolsFooter;
        if (first_system) {
            const std::string c = trim(content[0]);
            if (!c.empty()) out += "\n\n" + c;
        }
        out += "<|im_end|>\n";
    } else if (first_system) {
        const std::string c = trim(content[0]);
        if (!c.empty()) out += "<|im_start|>system\n" + (ri.empty() ? std::string() : ri + "\n\n") + c + "<|im_end|>\n";
        else if (!ri.empty()) out += "<|im_start|>system\n" + ri + "<|im_end|>\n";
    } else if (!ri.empty()) {
        out += "<|im_start|>system\n" + ri + "<|im_end|>\n";
    }

    // last real user query (user messages that only carry tool responses do not count)
    long last_query = -1;
    for (size_t k = n; k-- > 0;) {
        if (role_of(msgs[k]) != "user") continue;
        const std::string c = trim(content[k]);
        if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) { last_query = (long)k; break; }
    }
    if (last_query < 0) { err = "no user query found in messages"; return false; }

    for (size_t i = 0; i < n; i++) {
        const json::Value& m = msgs[i];
        const std::string role = role_of(m);
        const std::string c = trim(content[i]);
        if (role == "system") {
            if (i == 0) continue;
            out += "<|im_start|>system\n" + c + "<|im_end|>\n";   // relaxation: the template raises here
        } else if (role == "user") {
            out += "<|im_start|>user\n" + c + "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string rc = m["reasoning_content"].is_string() ? m["reasoning_content"].as_string()
                                                                : m["reasoning"].as_string();
            rc = trim(rc);
            if (opt.preserve_thinking || (long)i > last_query) out += "<|im_start|>assistant\n<think>\n" + rc + "\n</think>\n\n" + c;
            else out += "<|im_start|>assistant\n" + c;
            const json::Value& tcs = m["tool_calls"];
            if (tcs.is_array()) {
                for (size_t j = 0; j < tcs.items().size(); j++) {
                    const json::Value& tc0 = tcs.items()[j];
                    const json::Value& tc = tc0.has("function") ? tc0["function"] : tc0;
                    const std::string name = tc["name"].as_string();
                    if (j == 0) out += (trim(c).empty() ? "" : "\n\n") + std::string("<tool_call>\n<function=") + name + ">\n";
                    else out += "\n<tool_call>\n<function=" + name + ">\n";
                    json::Value args = tc["arguments"];
                    if (args.is_string()) {
                        if (args.as_string().empty()) args = json::Value();
                        else {
                            json::Value parsed;
                            if (!json::Value::parse(args.as_string(), parsed) || !parsed.is_object()) {
                                err = "tool call arguments of '" + name + "' are not a JSON object";
                                return false;
                            }
                            args = parsed;
                        }
                    }
                    if (args.is_object())
                        for (const auto& kv : args.members())
                            out += "<parameter=" + kv.first + ">\n" +
                                   (kv.second.is_string() ? kv.second.as_string() : kv.second.dump_py()) + "\n</parameter>\n";
                    out += "</function>\n</tool_call>";
                }
            }
            out += "<|im_end|>\n";
        } else {   // tool
            if (i > 0 && role_of(msgs[i - 1]) != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + c + "\n</tool_response>";
            if (i + 1 == n || role_of(msgs[i + 1]) != "tool") out += "<|im_end|>\n";
        }
    }
    if (opt.add_generation_prompt)
        out += opt.enable_thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return true;
}

std::string tool_call_opening(const std::string& name) { return "<tool_call>\n<function=" + name + ">\n"; }

void parse_tool_calls(const std::string& text, const json::Value& tools, std::string& content, std::vector<ToolCall>& calls) {
    calls.clear();
    size_t p = text.find("<tool_call>");
    if (p == std::string::npos) { content = text; return; }
    content = rstrip(text.substr(0, p));
    std::string tail;
    while (p != std::string::npos) {
        const size_t q = p + 11;
        const size_t end = text.find("</tool_call>", q);
        ToolCall call;
        if (parse_one(text.substr(q, end == std::string::npos ? std::string::npos : end - q), tools, call))
            calls.push_back(std::move(call));
        if (end == std::string::npos) break;
        const size_t nx = text.find("<tool_call>", end + 12);
        if (nx == std::string::npos) tail = text.substr(end + 12);
        p = nx;
    }
    const std::string t = trim(tail);
    if (!t.empty()) content += (content.empty() ? "" : "\n\n") + t;   // text after the calls (the format forbids it)
}

void ThinkSplitter::feed(const std::string& text, std::string& reasoning, std::string& content) {
    static const std::string marker = "</think>";
    held_ += text;
    if (thinking_) {
        const size_t at = held_.find(marker);
        if (at == std::string::npos) {
            size_t cut = held_.size();   // hold back a suffix that is a prefix of the marker
            for (size_t k = std::min(held_.size(), marker.size() - 1); k > 0; k--)
                if (marker.compare(0, k, held_, held_.size() - k, k) == 0) { cut = held_.size() - k; break; }
            reasoning += held_.substr(0, cut);
            held_.erase(0, cut);
            return;
        }
        reasoning += held_.substr(0, at);
        held_.erase(0, at + marker.size());
        thinking_ = false;
    }
    if (!content_started_) {
        size_t i = 0;
        while (i < held_.size() && held_[i] == '\n') i++;
        held_.erase(0, i);
        if (held_.empty()) return;
        content_started_ = true;
    }
    content += held_;
    held_.clear();
}

void ThinkSplitter::finish(std::string& reasoning, std::string& content) {
    if (thinking_) reasoning += held_; else content += held_;
    held_.clear();
}

std::string ToolCallStreamer::feed(const std::string& text) {
    static const std::string marker = "<tool_call>";
    if (in_call_) { call_ += text; return ""; }
    held_ += text;
    const size_t at = held_.find(marker);
    if (at != std::string::npos) {
        const std::string out = rstrip(held_.substr(0, at));   // the separator before the call is not content
        call_ = held_.substr(at);
        held_.clear();
        in_call_ = true;
        return out;
    }
    size_t cut = held_.size();
    for (size_t k = std::min(held_.size(), marker.size() - 1); k > 0; k--)
        if (marker.compare(0, k, held_, held_.size() - k, k) == 0) { cut = held_.size() - k; break; }
    while (cut > 0 && (held_[cut - 1] == ' ' || held_[cut - 1] == '\n' || held_[cut - 1] == '\t' || held_[cut - 1] == '\r'))
        cut--;
    const std::string out = held_.substr(0, cut);
    held_.erase(0, cut);
    return out;
}

std::string ToolCallStreamer::finish() {
    if (in_call_) return "";
    std::string out = held_;
    held_.clear();
    return out;
}

} // namespace chat
} // namespace q38rt
