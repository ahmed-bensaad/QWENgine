#!/usr/bin/env python3
"""Reference renderings of the model's chat template for tests/unit/test_chat.

Reads tokenizer.chat_template from the GGUF and renders a set of OpenAI-style conversations with jinja2 the way
Hugging Face transformers does (trim_blocks, lstrip_blocks, tojson = json.dumps(ensure_ascii=False),
raise_exception). Tool-call arguments are given to the C++ renderer as JSON strings (what clients send) and to the
template as objects (what llama.cpp / transformers pass after parsing them).

Usage: python3 tests/unit/make_chat_cases.py <model.gguf> > tests/unit/data/chat_cases.json
"""
import copy
import json
import struct
import sys

import jinja2


def chat_template(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        f.read(4)
        _, n_kv = struct.unpack("<QQ", f.read(16))
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def string():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8")

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                for _ in range(n):
                    string() if et == 8 else f.read(sizes[et])
                return None
            return f.read(sizes[t])

        for _ in range(n_kv):
            key = string()
            (t,) = struct.unpack("<I", f.read(4))
            v = value(t)
            if key == "tokenizer.chat_template":
                return v
    raise SystemExit("no tokenizer.chat_template in the GGUF")


def raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)


def render(tmpl, case):
    msgs = copy.deepcopy(case["messages"])
    for m in msgs:   # the template iterates arguments|items: give it objects
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", tc)
            if isinstance(f.get("arguments"), str) and f["arguments"]:
                f["arguments"] = json.loads(f["arguments"])
    kw = dict(messages=msgs, add_generation_prompt=True)
    if case.get("tools") is not None:
        kw["tools"] = case["tools"]
    opt = case.get("options", {})
    if "enable_thinking" in opt:
        kw["enable_thinking"] = opt["enable_thinking"]
    if "reasoning_effort" in opt:
        kw["reasoning_effort"] = opt["reasoning_effort"]
    if "preserve_thinking" in opt:
        kw["preserve_thinking"] = opt["preserve_thinking"]
    return tmpl.render(**kw)


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name, e.g. Paris"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "days": {"type": "integer", "minimum": 1, "maximum": 7.5, "default": 1.0}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search_files", "description": "Search files. Supports ünïcødé — and \"quotes\".",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}},
        "options": {"type": "object", "properties": {"regex": {"type": "boolean"}}}},
        "required": ["query"]}}}

CONV = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Find TODOs and check the weather in Paris."},
    {"role": "assistant", "content": "", "reasoning_content": "  I should search first.\n",
     "tool_calls": [{"id": "call_1", "type": "function", "function": {
         "name": "search_files",
         "arguments": json.dumps({"query": "TODO", "paths": ["src", "docs/é"], "options": {"regex": False}})}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "src/a.cpp:12: TODO fix\nsrc/b.cpp:3: TODO"},
    {"role": "assistant", "content": "Found two TODOs. Now the weather.",
     "tool_calls": [{"id": "call_2", "type": "function", "function": {
         "name": "get_weather", "arguments": json.dumps({"city": "Paris", "days": 2})}},
         {"id": "call_3", "type": "function", "function": {
             "name": "get_weather", "arguments": json.dumps({"city": "Lyon\nFrance", "unit": "celsius"})}}]},
    {"role": "tool", "tool_call_id": "call_2", "content": "{\"temp\": 18, \"sky\": \"clear\"}"},
    {"role": "tool", "tool_call_id": "call_3", "content": [{"type": "text", "text": "21 C, cloudy"}]},
]

CASES = [
    {"name": "user_only", "messages": [{"role": "user", "content": "Hello!"}]},
    {"name": "system_user_no_think", "messages": [{"role": "system", "content": "  Be brief. "},
                                                  {"role": "user", "content": "Hi\n"}],
     "options": {"enable_thinking": False}},
    {"name": "effort_low", "messages": [{"role": "user", "content": "2+2?"}], "options": {"reasoning_effort": "low"}},
    {"name": "effort_medium", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "2+2?"}],
     "options": {"reasoning_effort": "medium"}},
    {"name": "tools_system", "messages": CONV[:2], "tools": [WEATHER, SEARCH]},
    {"name": "tools_no_system_no_think", "messages": CONV[1:2], "tools": [WEATHER],
     "options": {"enable_thinking": False}},
    {"name": "agent_loop", "messages": CONV, "tools": [WEATHER, SEARCH]},
    {"name": "agent_loop_followup", "messages": CONV + [
        {"role": "assistant", "content": "Paris 18 C clear, Lyon 21 C cloudy.", "reasoning_content": "Summarize."},
        {"role": "user", "content": [{"type": "text", "text": "Thanks! "}, {"type": "text", "text": "And Rome?"}]}],
     "tools": [WEATHER, SEARCH]},
    {"name": "agent_loop_drop_thinking", "messages": CONV + [
        {"role": "assistant", "content": "Done.", "reasoning_content": "Summarize."},
        {"role": "user", "content": "Again?"}],
     "tools": [WEATHER, SEARCH], "options": {"preserve_thinking": False}},
    {"name": "assistant_history_no_tools", "messages": [
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the chicken cross the road?"},
        {"role": "user", "content": "Why?"}], "options": {"enable_thinking": False}},
    # images: the template's placeholder per image part; "images" lists the URLs render() must return, in order
    {"name": "images", "messages": [
        {"role": "system", "content": "Describe images."},
        {"role": "user", "content": [{"type": "text", "text": "Compare "},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}},
                                     {"type": "text", "text": " with "},
                                     {"type": "image", "image": "data:image/jpeg;base64,Qg=="}]},
        {"role": "assistant", "content": "The first is brighter."},
        {"role": "user", "content": [{"type": "image_url", "image_url": "data:image/png;base64,Qw=="},
                                     {"type": "text", "text": "And this one?"}]}],
     "images": ["data:image/png;base64,QQ==", "data:image/jpeg;base64,Qg==", "data:image/png;base64,Qw=="]},
    {"name": "tool_image", "messages": CONV[1:3] + [
        {"role": "tool", "tool_call_id": "call_1", "content": [
            {"type": "text", "text": "Screenshot:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,RA=="}}]}],
     "tools": [WEATHER, SEARCH], "images": ["data:image/png;base64,RA=="]},
]


def main():
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    tmpl = env.from_string(chat_template(sys.argv[1]))
    out = []
    for c in CASES:
        c = dict(c)
        c["expected"] = render(tmpl, c)
        out.append(c)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
