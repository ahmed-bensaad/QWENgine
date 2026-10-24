/// @file test_chat.cpp
/// Chat format: render() must reproduce the GGUF's Jinja chat template byte for byte (reference renderings by
/// jinja2 in tests/unit/data/chat_cases.json, made by make_chat_cases.py); tool-call parsing of model output;
/// streaming splitters (reasoning / content / tool calls) with markers split across chunks; Responses API input
/// and tool conversion to the Chat Completions form. With the model: the server's prompt split (conversation +
/// generation prompt, tokenized separately for the engine snapshot) gives the same tokens as the whole prompt.
/// Usage: test_chat <chat_cases.json> [model.gguf]

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include "runtime/gguf_parser.h"
#include "runtime/tokenizer.h"
#include "server/api.h"
#include "server/chat.h"
#include "server/json.h"

using namespace q38rt;

static int g_fail = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); g_fail++; } \
    } while (0)

static void show_diff(const std::string& name, const std::string& a, const std::string& b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) i++;
    const size_t s = i > 60 ? i - 60 : 0;
    printf("  case %s differs at byte %zu:\n  got:      %s\n  expected: %s\n", name.c_str(), i,
           json::quote(a.substr(s, 120)).c_str(), json::quote(b.substr(s, 120)).c_str());
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <chat_cases.json> [model.gguf]\n", argv[0]); return 2; }
    std::ifstream f(argv[1]);
    std::stringstream ss;
    ss << f.rdbuf();
    json::Value cases;
    std::string err;
    if (!json::Value::parse(ss.str(), cases, &err)) { printf("cannot parse %s: %s\n", argv[1], err.c_str()); return 1; }

    // ---- template rendering vs jinja2 ----
    int n = 0;
    for (const json::Value& c : cases.items()) {
        chat::Options opt;
        const json::Value& o = c["options"];
        if (o.has("enable_thinking")) opt.enable_thinking = o["enable_thinking"].as_bool();
        if (o.has("reasoning_effort")) opt.reasoning_effort = o["reasoning_effort"].as_string();
        if (o.has("preserve_thinking")) opt.preserve_thinking = o["preserve_thinking"].as_bool();
        std::string out;
        std::vector<std::string> images;
        const bool ok = chat::render(c["messages"], c["tools"], opt, out, err, &images);
        if (!ok) printf("  case %s: render failed: %s\n", c["name"].as_string().c_str(), err.c_str());
        CHECK(ok);
        if (ok && out != c["expected"].as_string()) { show_diff(c["name"].as_string(), out, c["expected"].as_string()); g_fail++; }
        std::vector<std::string> want;
        for (const json::Value& u : c["images"].items()) want.push_back(u.as_string());
        CHECK(images == want);
        n++;
    }
    printf("template: %d cases compared\n", n);

    // ---- tokenization of the conversation and the generation prompt separately (api.cpp prepare_chat) ----
    if (argc > 2) {
        GGUFParser g;
        Tokenizer tk;
        if (!g.parse(argv[2]) || !tk.load(g)) { printf("cannot load %s\n", argv[2]); return 1; }
        int m = 0;
        for (const json::Value& c : cases.items()) {
            const std::string& full = c["expected"].as_string();
            const size_t at = full.rfind("<|im_start|>assistant\n<think>\n");
            CHECK(at != std::string::npos);
            if (at == std::string::npos) continue;
            for (const std::string& gen : {full.substr(at), std::string("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=")}) {
                std::vector<int32_t> split = tk.encode(full.substr(0, at));
                const std::vector<int32_t> tail = tk.encode(gen);
                split.insert(split.end(), tail.begin(), tail.end());
                CHECK(split == tk.encode(full.substr(0, at) + gen));
                m++;
            }
        }
        printf("tokenization split: %d prompts compared\n", m);
    }

    // invalid inputs
    json::Value bad;
    std::string out;
    json::Value::parse(R"([{"role":"assistant","content":"x"}])", bad);
    CHECK(!chat::render(bad, json::Value(), chat::Options(), out, err));   // no user query
    json::Value::parse(R"([{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}])", bad);
    CHECK(!chat::render(bad, json::Value(), chat::Options(), out, err));   // images without a vision encoder
    std::vector<std::string> images;
    json::Value::parse(R"([{"role":"system","content":[{"type":"image_url","image_url":{"url":"x"}}]},{"role":"user","content":"a"}])", bad);
    CHECK(!chat::render(bad, json::Value(), chat::Options(), out, err, &images));   // the template raises
    json::Value::parse(R"([{"role":"user","content":[{"type":"video","video":"x"}]}])", bad);
    CHECK(!chat::render(bad, json::Value(), chat::Options(), out, err, &images));

    // ---- tool-call parsing ----
    json::Value tools;
    json::Value::parse(R"([{"type":"function","function":{"name":"get_weather","parameters":{"type":"object",
        "properties":{"city":{"type":"string"},"days":{"type":"integer"},"zip":{"type":"string"},
        "opts":{"type":"object"}}}}}])", tools);
    std::string content;
    std::vector<chat::ToolCall> calls;
    chat::parse_tool_calls("Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n"
                           "<parameter=days>\n3\n</parameter>\n<parameter=zip>\n10001\n</parameter>\n<parameter=opts>\n"
                           "{\"metric\": true}\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n"
                           "<parameter=city>\nLine1\nLine2\n</parameter>\n</function>\n</tool_call>",
                           tools, content, calls);
    CHECK(content == "Let me check.");
    CHECK(calls.size() == 2);
    if (calls.size() == 2) {
        CHECK(calls[0].name == "get_weather");
        CHECK(calls[0].arguments.dump() == "{\"city\":\"New York\",\"days\":3,\"zip\":\"10001\",\"opts\":{\"metric\":true}}");
        CHECK(calls[1].arguments["city"].as_string() == "Line1\nLine2");
        CHECK(calls[0].id.rfind("call_", 0) == 0 && calls[0].id != calls[1].id);
    }
    chat::parse_tool_calls("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call>", tools,
                           content, calls);
    CHECK(content.empty() && calls.size() == 1 && calls[0].arguments["city"].as_string() == "Rome");
    chat::parse_tool_calls("<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo", tools, content, calls);   // truncated
    CHECK(calls.size() == 1 && calls[0].arguments["city"].as_string() == "Oslo");
    chat::parse_tool_calls("No tools needed.", tools, content, calls);
    CHECK(content == "No tools needed." && calls.empty());

    // ---- streaming splitters, fed one byte at a time ----
    {
        const std::string text = "Plan: call it.\n</think>\n\nSure.\n\n<tool_call>\n<function=f>\n</function>\n</tool_call>";
        chat::ThinkSplitter ts(true);
        chat::ToolCallStreamer tcs;
        std::string reasoning, emitted;
        for (char ch : text) {
            std::string r, c;
            ts.feed(std::string(1, ch), r, c);
            reasoning += r;
            emitted += tcs.feed(c);
        }
        std::string r, c;
        ts.finish(r, c);
        reasoning += r;
        emitted += tcs.feed(c);
        emitted += tcs.finish();
        CHECK(reasoning == "Plan: call it.\n");
        CHECK(emitted == "Sure.");
        CHECK(tcs.in_call() && tcs.call_text() == "<tool_call>\n<function=f>\n</function>\n</tool_call>");
    }
    {
        chat::ToolCallStreamer tcs;
        std::string emitted;
        for (char ch : std::string("a < b <tool and\n<tool_c ok\n")) emitted += tcs.feed(std::string(1, ch));
        emitted += tcs.finish();
        CHECK(emitted == "a < b <tool and\n<tool_c ok\n" && !tcs.in_call());
    }

    // ---- Responses API input -> chat messages ----
    {
        json::Value input, msgs;
        json::Value::parse(R"([
            {"role":"developer","content":"Be terse."},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"Weather "},{"type":"input_text","text":"in Rome?"}]},
            {"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"Need the tool."}]},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Checking."}]},
            {"type":"function_call","call_id":"call_a","name":"get_weather","arguments":"{\"city\":\"Rome\"}"},
            {"type":"function_call","call_id":"call_b","name":"get_weather","arguments":"{\"city\":\"Milan\"}"},
            {"type":"function_call_output","call_id":"call_a","output":"20 C"},
            {"type":"function_call_output","call_id":"call_b","output":[{"type":"input_text","text":"18 C"}]},
            {"type":"function_call","call_id":"call_c","name":"get_weather","arguments":"{}"}])", input);
        CHECK(api::responses_input_to_messages(input, msgs, err));
        const auto& m = msgs.items();
        CHECK(m.size() == 6);
        if (m.size() == 6) {
            CHECK(m[0]["role"].as_string() == "developer" && m[0]["content"].as_string() == "Be terse.");
            CHECK(m[1]["content"].as_string() == "Weather in Rome?");
            CHECK(m[2]["role"].as_string() == "assistant" && m[2]["content"].as_string() == "Checking." &&
                  m[2]["reasoning_content"].as_string() == "Need the tool." && m[2]["tool_calls"].items().size() == 2);
            CHECK(m[2]["tool_calls"].items()[1]["id"].as_string() == "call_b" &&
                  m[2]["tool_calls"].items()[1]["function"]["arguments"].as_string() == "{\"city\":\"Milan\"}");
            CHECK(m[3]["role"].as_string() == "tool" && m[3]["tool_call_id"].as_string() == "call_a");
            CHECK(m[4]["content"].as_string() == "18 C");
            CHECK(m[5]["role"].as_string() == "assistant" && m[5]["tool_calls"].items().size() == 1);
        }
        // the converted conversation renders (tool results grouped in one user turn)
        json::Value tools, norm;
        json::Value::parse(R"([{"type":"function","name":"get_weather","description":"d","parameters":{"type":"object",
            "properties":{"city":{"type":"string"}}},"strict":true},{"type":"web_search"}])", tools);
        std::vector<std::string> skipped;
        CHECK(api::normalize_tools(tools, norm, skipped, err));
        CHECK(norm.items().size() == 1 && skipped.size() == 1 && skipped[0] == "web_search");
        CHECK(norm.dump() == R"([{"type":"function","function":{"name":"get_weather","description":"d","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}])");
        json::Value::parse(R"("Hi")", input);
        CHECK(api::responses_input_to_messages(input, msgs, err) && msgs.items().size() == 1 &&
              msgs.items()[0]["role"].as_string() == "user");
        json::Value::parse(R"([{"type":"item_reference","id":"x"}])", input);
        CHECK(!api::responses_input_to_messages(input, msgs, err));
        // input_image parts become Chat Completions image parts (text parts around them kept in order)
        json::Value::parse(R"([{"role":"user","content":[{"type":"input_text","text":"What is "},{"type":"input_text","text":"this?"},
            {"type":"input_image","image_url":"data:image/png;base64,QQ==","detail":"auto"}]}])", input);
        CHECK(api::responses_input_to_messages(input, msgs, err));
        CHECK(msgs.dump() == R"([{"role":"user","content":[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,QQ=="}}]}])");
        json::Value::parse(R"([{"role":"user","content":[{"type":"input_image","file_id":"file-1"}]}])", input);
        CHECK(!api::responses_input_to_messages(input, msgs, err));
        json::Value::parse(R"([{"role":"user","content":[{"type":"input_file","file_data":"x"}]}])", input);
        CHECK(!api::responses_input_to_messages(input, msgs, err));
    }

    if (g_fail) { printf("test_chat: %d failures\n", g_fail); return 1; }
    printf("test_chat: all checks passed\n");
    return 0;
}
