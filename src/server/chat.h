#pragma once
/// @file chat.h
/// The model's chat format, ported from the chat template stored in the GGUF (tokenizer.chat_template, a Jinja
/// template of the Qwen3.5 family): rendering of OpenAI-style message lists (system / user / assistant with
/// reasoning and tool calls / tool results) and tool definitions into the prompt text, and parsing of the model's
/// output (reasoning in <think>, content, tool calls in the XML format the template teaches).
///
/// render_chat() reproduces the template byte for byte (tests/unit/test_chat checks it against the Jinja template
/// rendered by jinja2), with two documented relaxations: system/developer messages after the first are rendered in
/// place as system turns (the template raises), and video parts are rejected (no video input). Image parts render as
/// the template's placeholder <|vision_start|><|image_pad|><|vision_end|>; the caller expands each <|image_pad|> to
/// the image's tokens.

#include <string>
#include <vector>
#include "server/json.h"

namespace q38rt {
namespace chat {

struct Options {
    bool enable_thinking = true;            // false: empty <think></think> block, no reasoning instructions
    std::string reasoning_effort = "xhigh"; // xhigh | medium | low (template values; used when thinking is on)
    bool preserve_thinking = true;          // false: drop reasoning of assistant turns before the last user query
    bool add_generation_prompt = true;
};

/// Render `messages` (+ `tools`, an array of OpenAI tool objects or null) into the prompt text. Returns false
/// with a client-facing message on invalid input. `images` receives the URL of each image part in prompt order
/// ({"type": "image_url", "image_url": {"url": U}} or "image_url": U, "image": U); null: images are refused.
bool render(const json::Value& messages, const json::Value& tools, const Options& opt, std::string& out, std::string& err,
            std::vector<std::string>* images = nullptr);

/// Tool call parsed from model output. `arguments` is a JSON object.
struct ToolCall {
    std::string id, name;
    json::Value arguments;
};

/// Split assistant output (the part after </think>) into content and tool calls. Parameter values are typed with
/// the tool's JSON schema (strings stay strings; other types are parsed as JSON, falling back to the raw string).
void parse_tool_calls(const std::string& text, const json::Value& tools, std::string& content,
                      std::vector<ToolCall>& calls);

/// Render one tool call the way the template renders an assistant's tool call (used to prefill forced calls).
std::string tool_call_opening(const std::string& name);

/// Streaming split of model text into reasoning (before </think>) and content (after it, leading newlines removed).
/// Text that may be the start of the marker is held back until resolved.
class ThinkSplitter {
public:
    explicit ThinkSplitter(bool thinking) : thinking_(thinking) {}
    void feed(const std::string& text, std::string& reasoning, std::string& content);
    void finish(std::string& reasoning, std::string& content);
    bool thinking() const { return thinking_; }

private:
    bool thinking_;
    bool content_started_ = false;
    std::string held_;
};

/// Streaming split of content into plain text (safe to emit) and tool-call text (from the first "<tool_call>" on).
/// Trailing whitespace and any suffix that may start "<tool_call>" are held back until resolved.
class ToolCallStreamer {
public:
    std::string feed(const std::string& text);   // returns plain text to emit now
    std::string finish();                        // remaining plain text (no tool call started)
    bool in_call() const { return in_call_; }
    const std::string& call_text() const { return call_; }   // "<tool_call>..." (complete after finish)

private:
    bool in_call_ = false;
    std::string held_, call_;
};

} // namespace chat
} // namespace q38rt
