#!/usr/bin/env python3
"""End-to-end test of q38_server (standard library only; the OpenAI SDK checks run when `openai` is installed).

Chat Completions: plain, streamed with reasoning, conversation reuse, tool-call round trip (plain and streamed,
with engine-state reuse across the agent loop), forced tool choice, seeded sampling, min-p. Responses API: function call,
previous_response_id follow-up, streamed events. Completions with stop strings. API key and error handling.
Metrics endpoint and request log. With an mmproj: image questions (Chat and Responses), image reuse in a follow-up,
a different image of the same size at the same place (must not reuse), image errors.
Usage: test_server.py <q38_server binary> <model.gguf> [mmproj.gguf]"""
import base64, json, os, signal, socket, subprocess, sys, tempfile, time, urllib.request, urllib.error

server, model = sys.argv[1], sys.argv[2]
mmproj = sys.argv[3] if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else None
with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
base = f"http://127.0.0.1:{port}/v1"
KEY = "test-key"
log = tempfile.NamedTemporaryFile(prefix="q38_server_", suffix=".log", delete=False)   # the request log
proc = subprocess.Popen([server, model, "--port", str(port), "--ctx", "4096", "--ctx-init", "512", "--api-key", KEY] +
                        (["--mmproj", mmproj] if mmproj else []), stdout=subprocess.DEVNULL, stderr=log)
fails = 0
HDRS = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}


def check(cond, what):
    global fails
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    fails += 0 if cond else 1


def post(path, body, raw=False, headers=HDRS):
    req = urllib.request.Request(base + path, data=(body if raw else json.dumps(body)).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=900) as r: return r.status, r.read().decode()
    except urllib.error.HTTPError as e: return e.code, e.read().decode()


def stream(path, body):
    """POST with stream=true; returns the list of (event name, parsed data) and whether [DONE] arrived."""
    req = urllib.request.Request(base + path, data=json.dumps(dict(body, stream=True)).encode(), headers=HDRS)
    events, done, name = [], False, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r.read().decode().split("\n"):
            if line.startswith("event: "): name = line[7:]
            elif line == "data: [DONE]": done = True
            elif line.startswith("data: "): events.append((name, json.loads(line[6:]))); name = None
    return events, done


def chat(body):
    st, text = post("/chat/completions", body)
    return st, json.loads(text)


def squares_ppm(left, right, w=320, h=160):
    """data: URL of a binary PPM (stb_image reads PNM): white, a `left`-colored square left, a `right` one right."""
    px = bytearray(b"\xff" * (w * h * 3))
    for (x0, color) in ((40, left), (200, right)):
        for y in range(40, 120):
            for x in range(x0, x0 + 80):
                px[(y * w + x) * 3:(y * w + x) * 3 + 3] = bytes(color)
    return "data:image/x-portable-pixmap;base64," + base64.b64encode(b"P6\n%d %d\n255\n" % (w, h) + px).decode()


try:
    for _ in range(300):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200: break
        except Exception: pass
        if proc.poll() is not None: raise SystemExit("server died: " + open(log.name).read()[-2000:])
        time.sleep(1)

    # ---- models, auth ----
    req = urllib.request.Request(base + "/models", headers=HDRS)
    with urllib.request.urlopen(req) as r: models = json.load(r)
    name = models["data"][0]["id"]
    check(models["data"][0]["object"] == "model", f"GET /v1/models -> {name}")
    req = urllib.request.Request(base + "/models/" + name, headers=HDRS)
    with urllib.request.urlopen(req) as r: check(json.load(r)["id"] == name, "GET /v1/models/{id}")
    check(post("/chat/completions", {"messages": []}, headers={"Content-Type": "application/json"})[0] == 401,
          "missing API key -> 401")

    # ---- plain chat, thinking off ----
    msgs = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    st, o = chat({"model": "anything", "messages": msgs, "reasoning_effort": "none", "temperature": 0, "max_tokens": 20})
    answer = o["choices"][0]["message"]["content"]
    check(st == 200 and "4" in answer and o["choices"][0]["finish_reason"] == "stop", f"chat completion -> {answer!r}")

    # ---- streamed thinking chat: reasoning and content separated, usage chunk, [DONE] ----
    ev, done = stream("/chat/completions", {"messages": msgs, "temperature": 0, "max_tokens": 400,
                                            "reasoning_effort": "low", "stream_options": {"include_usage": True}})
    reasoning = content = ""; finish = None; usage = None
    for _, c in ev:
        if not c["choices"]:
            usage = c.get("usage"); continue
        d = c["choices"][0]["delta"]
        reasoning += d.get("reasoning_content", ""); content += d.get("content") or ""
        finish = c["choices"][0]["finish_reason"] or finish
    check(done and finish == "stop" and "4" in content and len(reasoning) > 0 and usage and usage["completion_tokens"] > 0,
          f"streamed thinking chat -> {content!r} ({len(reasoning)} reasoning chars)")

    # ---- the next turn reuses the cached conversation ----
    msgs2 = msgs + [{"role": "assistant", "content": content, "reasoning_content": reasoning},
                    {"role": "user", "content": "And 3+3? Just the number."}]
    st, o = chat({"messages": msgs2, "temperature": 0, "max_tokens": 400, "reasoning_effort": "low"})
    check(st == 200 and o["timings"]["cached_tokens"] > 0 and "6" in o["choices"][0]["message"]["content"],
          f"continuation (cached {o['timings']['cached_tokens']}) -> {o['choices'][0]['message']['content']!r}")

    # ---- tool-call round trip (agent loop) ----
    conv = [{"role": "system", "content": "You are a helpful assistant. Use tools when they help."},
            {"role": "user", "content": "What's the weather in Paris right now?"}]
    st, o = chat({"messages": conv, "tools": [WEATHER], "temperature": 0, "max_tokens": 200, "reasoning_effort": "none"})
    m = o["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    check(st == 200 and o["choices"][0]["finish_reason"] == "tool_calls" and len(calls) >= 1 and
          calls[0]["function"]["name"] == "get_weather" and "paris" in str(args.get("city", "")).lower() and
          calls[0]["id"].startswith("call_"), f"tool call -> {calls[0]['function'] if calls else m}")
    first_prompt = o["usage"]["prompt_tokens"]
    if calls:
        conv2 = conv + [m] + [{"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"temp_c": 18, "sky": "sunny"})}
                              for c in calls]
        st, o = chat({"messages": conv2, "tools": [WEATHER], "temperature": 0, "max_tokens": 120, "reasoning_effort": "none"})
        ans = o["choices"][0]["message"]["content"] or ""
        check(st == 200 and o["choices"][0]["finish_reason"] == "stop" and "18" in ans and
              o["timings"]["cached_tokens"] >= first_prompt,
              f"tool result -> {ans!r} (cached {o['timings']['cached_tokens']} of {o['usage']['prompt_tokens']})")

    # ---- thinking agent loop where the client drops the reasoning (most harnesses do): the next prompt diverges
    #      inside the re-rendered assistant turn, the engine snapshot at the end of the conversation is reused ----
    st, o = chat({"messages": conv, "tools": [WEATHER], "temperature": 0, "max_tokens": 800, "reasoning_effort": "low"})
    m = o["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    p1 = o["usage"]["prompt_tokens"]
    check(st == 200 and calls and m.get("reasoning_content"), f"thinking tool call -> {calls[0]['function'] if calls else m}")
    if calls:
        conv3 = conv + [{"role": "assistant", "content": m["content"], "tool_calls": calls}] + [
            {"role": "tool", "tool_call_id": c["id"], "content": "{\"temp_c\": 11}"} for c in calls]
        st, o = chat({"messages": conv3, "tools": [WEATHER], "temperature": 0, "max_tokens": 800, "reasoning_effort": "low"})
        t = o.get("timings", {})
        check(st == 200 and t.get("snapshot_restored") and t.get("cached_tokens", 0) >= p1 - 4,
              f"reasoning dropped by the client -> snapshot reused (cached {t.get('cached_tokens')} of {o['usage']['prompt_tokens']})")

    # ---- streamed tool call ----
    ev, done = stream("/chat/completions", {"messages": conv, "tools": [WEATHER], "temperature": 0, "max_tokens": 200,
                                            "reasoning_effort": "none"})
    acc = {}; finish = None
    for _, c in ev:
        for tc in c["choices"][0]["delta"].get("tool_calls", []) if c["choices"] else []:
            a = acc.setdefault(tc["index"], {"id": "", "name": "", "arguments": ""})
            a["id"] += tc.get("id") or ""; a["name"] += tc.get("function", {}).get("name") or ""
            a["arguments"] += tc.get("function", {}).get("arguments") or ""
        if c["choices"]: finish = c["choices"][0]["finish_reason"] or finish
    a = acc.get(0, {})
    check(done and finish == "tool_calls" and a.get("name") == "get_weather" and "city" in json.loads(a.get("arguments") or "{}"),
          f"streamed tool call -> {a}")

    # ---- forced tool choice ----
    st, o = chat({"messages": [{"role": "user", "content": "Hello there!"}], "tools": [WEATHER], "temperature": 0,
                  "max_tokens": 60, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}})
    calls = o["choices"][0]["message"].get("tool_calls") or []
    check(st == 200 and len(calls) == 1 and calls[0]["function"]["name"] == "get_weather", f"forced tool choice -> {calls}")

    # ---- seeded sampling is reproducible ----
    req = {"messages": [{"role": "user", "content": "Name a color."}], "reasoning_effort": "none",
           "temperature": 0.9, "seed": 7, "max_tokens": 16}
    a = chat(req)[1]["choices"][0]["message"]["content"]
    b = chat(req)[1]["choices"][0]["message"]["content"]
    check(a == b, f"seeded sampling reproducible ({a!r})")
    # min_p = 1 keeps only the most likely token: sampling at temperature 1 must give the greedy text
    req = {"messages": [{"role": "user", "content": "Write one sentence about the sea."}], "reasoning_effort": "none",
           "max_tokens": 24}
    g = chat(dict(req, temperature=0))[1]["choices"][0]["message"]["content"]
    m = chat(dict(req, temperature=1.0, top_p=1.0, top_k=0, min_p=1.0))[1]["choices"][0]["message"]["content"]
    check(g == m, f"min_p = 1 equals greedy ({m!r})")
    check(post("/chat/completions", dict(req, min_p=1.5))[0] == 400, "min_p > 1 -> 400")

    # ---- Responses API ----
    rtools = [{"type": "function", "name": "get_weather", "description": "Get the current weather for a city.",
               "parameters": WEATHER["function"]["parameters"]}]
    st, text = post("/responses", {"model": name, "instructions": "Use tools when they help.", "tools": rtools,
                                   "input": "What's the weather in Tokyo?", "reasoning": {"effort": "none"},
                                   "temperature": 0, "max_output_tokens": 200})
    r = json.loads(text)
    fcs = [it for it in r.get("output", []) if it["type"] == "function_call"]
    check(st == 200 and r["status"] == "completed" and fcs and fcs[0]["name"] == "get_weather" and
          "tokyo" in fcs[0]["arguments"].lower(), f"responses: function call -> {fcs[0] if fcs else text[:300]}")
    if fcs:
        # instructions are not inherited from the previous response (OpenAI semantics): resend them
        st, text = post("/responses", {"previous_response_id": r["id"], "instructions": "Use tools when they help.",
                                       "tools": rtools, "reasoning": {"effort": "none"},
                                       "input": [{"type": "function_call_output", "call_id": fcs[0]["call_id"],
                                                  "output": "{\"temp_c\": 25, \"sky\": \"rain\"}"}],
                                       "temperature": 0, "max_output_tokens": 120})
        r2 = json.loads(text)
        msg = "".join(p["text"] for it in r2.get("output", []) if it["type"] == "message" for p in it["content"])
        check(st == 200 and "25" in msg and r2["usage"]["input_tokens_details"]["cached_tokens"] > 0,
              f"responses: previous_response_id follow-up -> {msg!r}")
    ev, _ = stream("/responses", {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}],
                                  "reasoning": {"effort": "low"}, "temperature": 0, "max_output_tokens": 300})
    types = [e for e, _ in ev]
    deltas = "".join(d["delta"] for e, d in ev if e == "response.output_text.delta")
    final = ev[-1][1]["response"] if ev else {}
    check(types[:2] == ["response.created", "response.in_progress"] and "response.reasoning_text.delta" in types and
          types[-1] == "response.completed" and deltas and
          final["output"][-1]["content"][0]["text"] == deltas and
          [d["sequence_number"] for _, d in ev] == list(range(len(ev))), f"responses: streamed events -> {deltas!r}")

    # ---- completions ----
    st, text = post("/completions", {"prompt": "1, 2, 3, 4,", "max_tokens": 30, "temperature": 0, "stop": [", 8"]})
    o = json.loads(text)
    check(st == 200 and o["choices"][0]["text"].strip().startswith("5") and ", 8" not in o["choices"][0]["text"],
          f"completion with stop -> {o['choices'][0]['text']!r}")

    # ---- errors ----
    check(post("/chat/completions", "{bad", raw=True)[0] == 400, "malformed JSON -> 400")
    check(post("/chat/completions", {"messages": []})[0] == 400, "empty messages -> 400")
    st, text = post("/chat/completions", {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]})
    check(st == 400 and ("image 1:" in text if mmproj else "--mmproj" in text),
          "invalid image -> 400" if mmproj else "image without a vision encoder -> 400")
    st, text = post("/chat/completions", {"messages": [{"role": "user", "content": "word " * 5000}]})
    check(st == 400 and json.loads(text)["error"]["code"] == "context_length_exceeded", "prompt over the context -> 400")

    # ---- metrics, request log ----
    req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers=HDRS)
    with urllib.request.urlopen(req) as r: prom = r.read().decode()
    vals = {l.rsplit(" ", 1)[0]: float(l.rsplit(" ", 1)[1]) for l in prom.splitlines() if l and not l.startswith("#")}
    reqs = sum(v for k, v in vals.items() if k.startswith("q38:requests_total{"))
    check(vals.get("llamacpp:prompt_tokens_total", 0) > 0 and vals.get("llamacpp:tokens_predicted_total", 0) > 0
          and vals.get("llamacpp:requests_processing") == 0 and vals.get("q38:prompt_tokens_cached_total", 0) > 0
          and vals.get("q38:draft_tokens_total", 0) > 0 and reqs >= 10,
          f"GET /metrics -> {reqs:.0f} requests, {vals.get('llamacpp:tokens_predicted_total', 0):.0f} tokens predicted, "
          f"{vals.get('llamacpp:prompt_tokens_total', 0):.0f} prompt tokens processed")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics") as r: st = r.status
    except urllib.error.HTTPError as e: st = e.code
    check(st == 401, "GET /metrics without the key -> 401")
    text = open(log.name).read()
    check(text.count("prompt eval time =") >= 10 and "draft acceptance rate" in text and "-> 400" in text
          and "KV cache grown to" in text, "request log: timings per request, KV-cache growth, errors")

    # ---- images ----
    if mmproj:
        RED, BLUE = (220, 20, 20), (20, 40, 220)
        q = "What color is the square on the left? Answer with one word."
        def img_msgs(url):
            return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": q}]}]
        st, o = chat({"messages": img_msgs(squares_ppm(RED, BLUE)), "reasoning_effort": "none", "temperature": 0, "max_tokens": 10})
        a1 = o["choices"][0]["message"]["content"] if st == 200 else o
        check(st == 200 and "red" in a1.lower() and o["usage"]["prompt_tokens"] > 50, f"image question -> {a1!r}")
        follow = img_msgs(squares_ppm(RED, BLUE)) + [{"role": "assistant", "content": a1},
                                                       {"role": "user", "content": "And the one on the right? One word."}]
        st, o = chat({"messages": follow, "reasoning_effort": "none", "temperature": 0, "max_tokens": 10})
        a2 = o["choices"][0]["message"]["content"] if st == 200 else o
        check(st == 200 and "blue" in a2.lower() and o["usage"]["prompt_tokens_details"]["cached_tokens"] > 50,
              f"image follow-up reuses the encoded image -> {a2!r}, cached {o['usage']['prompt_tokens_details']['cached_tokens'] if st == 200 else '-'}")
        st, o = chat({"messages": img_msgs(squares_ppm(BLUE, RED)), "reasoning_effort": "none", "temperature": 0, "max_tokens": 10})
        a3 = o["choices"][0]["message"]["content"] if st == 200 else o
        check(st == 200 and "blue" in a3.lower(), f"another image, same size and place, is not taken from the cache -> {a3!r}")
        body = {"input": [{"role": "user", "content": [{"type": "input_text", "text": q},
                                                       {"type": "input_image", "image_url": squares_ppm(RED, BLUE)}]}],
                "reasoning": {"effort": "none"}, "temperature": 0, "max_output_tokens": 10}
        st, text = post("/responses", body)
        out = "".join(c.get("text", "") for it in json.loads(text).get("output", []) for c in it.get("content", []) or [])
        check(st == 200 and "red" in out.lower(), f"Responses input_image -> {out!r}")
        st, text = post("/chat/completions", {"messages": img_msgs("https://example.com/a.png")})
        check(st == 400 and "data: URL" in text, "remote image URL -> 400")
        st, text = post("/chat/completions", {"messages": img_msgs("data:image/png;base64,QUJD")})
        check(st == 400 and "decode" in text, "undecodable image -> 400")
        text = open(log.name).read()
        check("1 image encoded in" in text and "image encode time" in text, "request log: image encoding")

    # ---- OpenAI SDK (optional) ----
    try:
        import openai
    except ImportError:
        print("skip OpenAI SDK checks (pip install openai)")
    else:
        client = openai.OpenAI(base_url=base, api_key=KEY)
        r = client.chat.completions.create(model=name, messages=conv, tools=[WEATHER], temperature=0, max_tokens=200,
                                           reasoning_effort="none")
        tc = r.choices[0].message.tool_calls
        check(tc and tc[0].function.name == "get_weather", f"SDK chat tool call -> {tc[0].function if tc else r}")
        parts = []
        with client.chat.completions.stream(model=name, messages=msgs, temperature=0, max_tokens=20,
                                            reasoning_effort="none") as s:
            for e in s:
                if e.type == "content.delta": parts.append(e.delta)
        check("4" in "".join(parts), f"SDK streamed chat -> {''.join(parts)!r}")
        r = client.responses.create(model=name, input="What is 2+2? Just the number.", temperature=0,
                                    max_output_tokens=20, reasoning={"effort": "none"})
        check("4" in r.output_text, f"SDK responses -> {r.output_text!r}")
finally:
    proc.send_signal(signal.SIGTERM)
    try: proc.wait(timeout=10)
    except Exception: proc.kill()
if fails == 0: os.unlink(log.name)
else: print("server log:", log.name)
print("test_server:", "all checks passed" if fails == 0 else f"{fails} failures")
sys.exit(1 if fails else 0)
