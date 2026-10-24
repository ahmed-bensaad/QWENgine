# Serving: the OpenAI-compatible API

`q38_server` exposes the engine over HTTP with the OpenAI API shapes that agent harnesses and SDKs speak: Chat
Completions (with tool calling and streaming), a subset of the Responses API, and legacy Completions. Any client
that lets you set an OpenAI **base URL** can use it.

## Quick start

```bash
cmake -S . -B build -G Ninja && cmake --build build      # once
tools/serve.sh                                            # http://127.0.0.1:8080/v1, MTP model, 32K context
```

`tools/serve.sh` runs `build/q38_server` on the `-mtp.gguf` model (speculative decoding) and passes its arguments
through. Set `Q38_MTP_MODEL` to use another path. Loading takes about 7 s when the model file is in the page cache
(longer from a cold disk): the weights are planned, uploaded and re-encoded. The server is ready when it prints
`q38_server: qwen3.8-27b on http://127.0.0.1:8080/v1 ...`.

Smoke test:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "reasoning_effort": "low"
}' | python3 -m json.tool
```

Any client configuration needs three values:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8080/v1` |
| API key | anything (e.g. `none`), or the key given with `--api-key` / `Q38_API_KEY` |
| Model | `qwen3.8-27b`. Requests may name any model: there is only one. |

## Server options

```
q38_server <model.gguf> [options]          (tools/serve.sh [options] fills in the model)
```

| Option | Default | Meaning |
|---|---|---|
| `--host H` | `127.0.0.1` | listen address; `0.0.0.0` exposes the server to the network (set an API key) |
| `--port P` | `8080` | TCP port |
| `--ctx N` | `32768` | largest context, prompt plus output, in tokens; at most 262144 (the model's training context) |
| `--ctx-init N` | `4096` | initial KV-cache capacity. A short conversation thus leaves the VRAM to the weights, for faster decode. |
| `--kv T` | `q8_0` | KV-cache type: `q8_0`; `q4_0`, half the KV VRAM per token at some accuracy cost; `f16`. The tested 100K configuration is `--ctx 100352 --kv q4_0`. |
| `--api-key KEY` | none | require `Authorization: Bearer KEY` on `/v1/*` and `/metrics`; also read from `Q38_API_KEY` |
| `--name NAME` | `qwen3.8-27b` | model id reported by `/v1/models` |
| `--no-think` | thinking on | answer without reasoning unless a request asks for it |
| `--effort E` | `xhigh` | default reasoning effort: `xhigh`, `medium`, `low` |
| `--temp T`, `--top-p P`, `--top-k K`, `--min-p P` | from the GGUF | sampling defaults for requests that do not set them; the GGUF recommends temperature 1.0, top-p 0.95, top-k 20, min-p 0. Temperature 0.6 is ~4% faster (more drafts accepted; README, *Best settings*). |
| `--max-tokens N` | context limit | cap on generated tokens per request |
| `--no-spec` | speculative | disable MTP speculative decoding |
| `--threads N` | 8 | CPU threads for the CPU-resident weights |
| `--log-interval S` | `5` | seconds between the decode progress lines of a running request; `0` keeps only the per-request lines |
| `--mmproj FILE` | none | vision encoder GGUF: enables image input (see *Images*). `tools/serve.sh` adds `models/mmproj-Qwen3.8-27B-BF16.gguf` when it exists. |
| `--image-min-tokens N`, `--image-max-tokens N` | `8`, `1024` | size range of an image in tokens of 32x32 pixels; images are scaled into it |

The KV cache doubles when a request needs more room than it has (prompt + up to 1024 output tokens), up to
`--ctx`. Each doubling re-plans the weight placement and re-uploads the weights: about 6.5 s, once per size.
Agent harnesses with long prompts pass 4K, 8K and 16K early. Starting with `--ctx-init 16384` avoids these steps,
but it gives that VRAM to the KV cache from the start, which makes decode slower for short conversations.

## Request log

The server logs to stderr. At startup it prints the weight plan; after that, each line is about one request and
carries its task number. A request that runs a 6.6K-token prompt into a 2048-token KV cache, then a follow-up
turn of the same conversation:

```
22:01:40.050 task 0 | new request chatcmpl-eec9d176f368c5803fb49b1f: chat.completions, 6613 prompt tokens, reasoning none, max 250 tokens, temperature 0.00
22:01:40.051 task 0 | prompt: 6613 tokens, 6613 to process
22:01:46.920 task 0 | KV cache grown to 8192 tokens in 6.9 s (n_past = 0)
22:01:50.331 task 0 | prompt processing progress: n_past = 1024 / 6613 (15.5%), 300.1 tok/s
22:01:52.291 task 0 | prompt processing progress: n_past = 2048 / 6613 (31.0%), 381.2 tok/s
...
22:01:59.684 task 0 | prompt processing progress: n_past = 6144 / 6613 (92.9%), 481.3 tok/s
22:02:00.472 task 0 | prompt done: 6613 tokens in 13553 ms (487.9 tok/s), n_past = 6613
22:02:05.567 task 0 | decoding: n_decoded = 78, 15.31 tok/s, draft acceptance 48%
22:02:10.661 task 0 | decoding: n_decoded = 163, 16.00 tok/s (last 5 s: 16.69 tok/s), draft acceptance 56%
22:02:11.624 task 0 | done: stop, n_past = 6785 (41.4% of the 16384-token context)
prompt eval time =   13552.84 ms /  6613 tokens (    2.05 ms per token,   487.94 tokens per second)
       eval time =   11152.24 ms /   172 tokens (   64.84 ms per token,    15.42 tokens per second)
     resize time =    6869.09 ms (KV cache grown to 8192 tokens)
      total time =   31574.17 ms /  6785 tokens
draft acceptance rate = 0.52542 (   93 accepted /   177 generated), 79 verify passes, 2.18 tokens per pass
22:02:12.048 task 1 | new request chatcmpl-3fe9ac6b222a906ae135a84d: chat.completions, 6804 prompt tokens, reasoning none, max 60 tokens, temperature 1.00
22:02:12.048 task 1 | prompt: 6804 tokens, 6785 reused from the cached conversation, 19 to process
22:02:12.945 task 1 | prompt done: 19 tokens in 897 ms (21.2 tok/s), n_past = 6804
...
```

- **`new request`**: the response id, the endpoint and the request's settings. A request that arrives while
  another one is generating says `(queued behind N requests)`, and later `started after waiting X s`.
- **`prompt`**: how many prompt tokens were reused from the KV cache, either from the cached conversation or
  from the snapshot of the previous request (see *Conversation reuse*), and how many must be processed. Requests
  with images say how many on the `new request` line; `prompt done` and the timing block give the images encoded
  and the encoding time (part of the prompt time).
- **`prompt processing progress`**: printed after each 1024-token prefill chunk, at most once per second.
  `n_past` is the number of prompt tokens in the KV cache, reused ones included, and the percentage is its share
  of the prompt. The speed is the average since the start of the prompt.
- **`decoding`**: every `--log-interval` seconds while generating. It gives the tokens generated so far, the
  average speed, the speed over the last interval and the draft acceptance.
- **`done`**: the finish reason (`stop`, `length limit`, `tool calls (names)`, `aborted (client disconnected)`)
  and the KV-cache use, then the timing block in the llama.cpp server's format. Prompt and eval times exclude
  KV-cache growth, which gets its own line. *Tokens per pass* counts the accepted drafts plus the token every
  verify pass produces.
- Requests rejected with an HTTP error are logged as `POST /v1/chat/completions -> 400 <message>`.

## Endpoints

| Method and path | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat Completions, streamed or not |
| `POST /v1/responses` | Responses API subset, streamed or not |
| `POST /v1/completions` | raw text completion; the prompt is used as is, without the chat template |
| `GET /v1/models`, `GET /v1/models/{id}` | the served model (`context_length` = `--ctx`) |
| `GET /health` | `{"status":"ok"}` once loaded; no API key needed |
| `GET /metrics` | Prometheus metrics (see *Metrics*); needs the API key if one is set |

The `/v1` prefix is optional (`/chat/completions` works too). Errors use the OpenAI shape
`{"error": {"message", "type", "param", "code"}}`. Examples: 400 for invalid input, 400 with code
`context_length_exceeded` when the prompt does not fit `--ctx`, 401 for a wrong key.

## Chat Completions

### Request fields

| Field | Support |
|---|---|
| `messages` | roles `system`, `developer` (treated as system), `user`, `assistant`, `tool` (and legacy `function`); content as a string or parts: text, and `image_url` images when the server has a vision encoder (see *Images*). Audio, video and file parts are rejected (400). Assistant messages may carry `tool_calls` (arguments as a JSON string or object) and `reasoning_content` (or `reasoning`). |
| `tools` | function tools `{"type": "function", "function": {"name", "description", "parameters"}}`. Other tool types are left out (logged). |
| `tool_choice` | `auto` (default), `none` (tools left out of the prompt), `required`, or `{"type": "function", "function": {"name": ...}}` |
| `parallel_tool_calls` | default `true`; `false` stops generation after the first complete call |
| `reasoning_effort` | `none` / `minimal` = no reasoning; `low`, `medium`; `high` / `xhigh` = the template's `xhigh` |
| `chat_template_kwargs` | `enable_thinking`, `reasoning_effort`, `preserve_thinking` (the template's variables); top-level `enable_thinking` is also accepted |
| `max_completion_tokens`, `max_tokens` | default: the rest of the context (or `--max-tokens`) |
| `temperature`, `top_p`, `top_k`, `min_p`, `seed` | `temperature: 0` = greedy. Sampling is over the engine's top-32 logits: temperature, then top-k, top-p and min-p (relative to the most likely token). With a `seed`, output is reproducible. Repetition, presence and frequency penalties are not supported (ignored). |
| `stop` | string or list; applies to the generated text, reasoning included |
| `stream`, `stream_options.include_usage` | Server-Sent Events |
| `n` | only 1; `logprobs: true` is rejected |
| other fields | ignored. This includes `response_format`: JSON output is not enforced. |

### Response

- `message.content`: the answer. It is `null` when the turn only calls tools.
- `message.reasoning_content`: the model's reasoning (the `<think>` block).
- `message.tool_calls`: `[{"id": "call_...", "type": "function", "function": {"name", "arguments": "<JSON>"}}]`.
- `finish_reason`: `stop`, `length` or `tool_calls`.
- `usage`: OpenAI usage with `prompt_tokens_details.cached_tokens` (tokens whose engine state was reused) and
  `completion_tokens_details.reasoning_tokens`.
- `timings` (extension): `cached_tokens`, `snapshot_restored`, `prefill_ms`, `prefill_tok_s`, `decode_tok_s`,
  `draft_acceptance`, and `kv_resize_ms` when the KV cache grew during the request. Prefill and decode times
  exclude that growth.

Streaming sends `chat.completion.chunk` events: a first chunk with `role`, then `reasoning_content` and `content`
deltas as they are generated. Each tool call goes out as a `tool_calls` delta (`index`, `id`, `type`, `function`
with `name` and `arguments`) as soon as its closing tag is generated. The final chunk carries `finish_reason`,
`usage` and `timings`. With `stream_options.include_usage`, a usage chunk with empty `choices` follows, then
`data: [DONE]`.

## Reasoning ("thinking")

The model reasons inside `<think>` before answering. The chat template has three effort levels; each adds an
instruction to the system prompt:

| Effort | Effect |
|---|---|
| `xhigh` (default; requests saying `high` get it) | think carefully, validate assumptions, consider alternatives |
| `medium` | no effort instruction |
| `low` | keep thinking brief |
| off (`reasoning_effort: "none"`, `enable_thinking: false`, `--no-think`) | empty `<think></think>` block, direct answer |

Decode runs at roughly 10-20 tok/s, so long reasoning costs minutes per turn. For interactive agent work, start
the server with `--effort low` or `--effort medium`, or set `reasoning_effort` per request. A request's own
choice always wins over the server default.

Previous assistant turns are rendered with the reasoning the client sends back (`reasoning_content`). Most clients
drop it; the template then renders an empty `<think>` block for those turns, which is what the model expects.

## Tool calling

The model's chat template (stored in the GGUF and ported byte for byte in `src/server/chat.cpp`) lists the tools
in the system prompt as JSON. It also teaches the model this call format:

```
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

The server converts in both directions:

- Outgoing: it parses these blocks into OpenAI `tool_calls`. Parameter values are typed with the tool's JSON
  schema: a value declared `string` stays a string, and other values are parsed as JSON, falling back to the raw
  text.
- Incoming: it renders the assistant `tool_calls` and the `tool` results the client sends back exactly as the
  template does. Consecutive tool results are grouped into one user turn.

Other details:

- With `tool_choice: "required"` or a named function, the call is started in the prompt (prefilled), so the model
  must produce one. That turn runs without reasoning.
- A call cut off by `max_tokens` is dropped, and the response says `finish_reason: "length"`.
- Text the model writes before the calls is returned as `content`.

## Conversation reuse (agent loops)

The engine keeps the state of one conversation (KV cache and DeltaNet state). Agent harnesses resend the whole
conversation every turn, and the server avoids reprocessing it in two ways:

1. **Exact continuation.** If the new prompt starts with everything the engine has processed (the previous prompt
   plus the generated answer), only the new tokens are processed.
2. **Snapshot.** After each prompt, the server saves the recurrent state at the end of the conversation, before
   the assistant's generation prompt (about 25 ms, kept in pinned host memory). The KV cache needs no copy: it is
   simply truncated. If the next prompt starts with that conversation but differs afterwards, the state is
   restored and processing resumes from there. This happens when a client re-renders the previous answer
   differently, most often by dropping its reasoning.

`usage.prompt_tokens_details.cached_tokens` shows how many tokens were reused.

Measured in a five-turn coding-agent loop driven by the OpenAI SDK, at effort `low`. The client dropped the
reasoning, as most harnesses do. The model listed the files, ran the tests, read two files, fixed the bug and
re-ran the tests.

- Setup: a 3.2K-token prompt with 16 tools.
- Every follow-up turn restored the snapshot and reused about 3.2K tokens.
- Processing the 38-125 new tokens per turn took 0.85 s.
- Decode ran at 15.6-19.2 tok/s.
- Turns took 4-6 s each.
- The first turn took 16 s: 7 s of prefill plus one KV-cache growth from 4K to 8K.
 Reuse fails when a client edits
earlier messages, for example by summarizing or truncating history, or puts changing data such as a timestamp
near the top of the system prompt. The whole prompt is then processed again, at about 490 tok/s at 4K context and
375 tok/s at 32K.

There is one cached conversation. Requests are served one at a time; concurrent requests wait. Two clients
alternating evict each other's state, and every switch costs a full prefill.

## Images

With a vision encoder (`--mmproj`, or `tools/serve.sh` when `models/mmproj-Qwen3.8-27B-BF16.gguf` exists), messages
can include images. The startup line then ends with `; image input`.

```bash
img=$(base64 -w0 photo.jpg)
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is written on this sign?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$img"'"}}]}],
  "reasoning_effort": "none"
}' | python3 -m json.tool
```

With the OpenAI Python SDK:

```python
import base64, openai
client = openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
url = "data:image/png;base64," + base64.b64encode(open("chart.png", "rb").read()).decode()
r = client.chat.completions.create(model="qwen3.8-27b", messages=[{"role": "user", "content": [
    {"type": "text", "text": "Summarize this chart."}, {"type": "image_url", "image_url": {"url": url}}]}])
```

- **Format.** Images are sent inline as `data:` URLs with base64 data: PNG, JPEG, BMP, GIF (first frame), TGA, PSD,
  HDR or PNM. Remote `http(s)://` URLs are not fetched (400). The forms `"image_url": "data:..."` and
  `{"type": "image", "image": "data:..."}` are accepted too, and so is Responses' `input_image` (`image_url`;
  `file_id` is not supported).
- **Where.** In user, assistant and tool messages (a tool can return a screenshot), any number per message. System
  messages cannot contain images (400, as in the chat template).
- **Size.** Each image is scaled, keeping its aspect ratio, to sides that are multiples of 32 pixels and to between
  `--image-min-tokens` and `--image-max-tokens` tokens of 32x32 pixels: a 640x480 photo is 300 tokens, anything
  beyond about one megapixel is scaled down to 1024 tokens. Image tokens count as prompt tokens (`usage`,
  context limit).
- **Cost.** The encoder runs on the GPU when an image's tokens must be processed: about 150 ms for 300 tokens and
  0.6 s for 1024, on top of the prefill of those tokens. Its weights (870 MiB) stay in pinned host memory and are
  streamed layer by layer, and its activations use VRAM only while it runs, so decode speed does not change. The
  encoder costs about 1 GB of host RAM.
- **Reuse.** Images already in the cached conversation are not encoded again: a follow-up question about an image
  costs only the new text. Reuse compares images by content, so a different image in the same place is processed.

Images are resized the way the Hugging Face processor used to train the model does (stretched with Pillow's bicubic
filter). llama.cpp pads them to keep their aspect ratio instead, so answers can differ slightly from llama.cpp's
for the same image.

## Responses API (subset)

`POST /v1/responses` accepts:

- **Input:** `input` as a string, or as a list of items:
  - `message` items with roles `user`, `system`, `developer`, `assistant`, and `input_text` / `output_text` parts,
    plus `input_image` parts with a vision encoder (see *Images*);
  - `function_call` and `function_call_output` items;
  - `reasoning` items (their text becomes the reasoning of the following assistant turn).
- **Instructions:** `instructions`, rendered as the system prompt. As in the OpenAI API, they are not inherited
  through `previous_response_id`.
- **Tools:** `tools` with function tools `{"type": "function", "name", "description", "parameters"}`. Built-in
  tools are left out and logged.
- **Control:** `tool_choice`, `parallel_tool_calls`, `reasoning.effort`, `max_output_tokens`, `temperature`,
  `top_p`, `top_k`, `min_p`, `seed`.
- **Storage:** `store` (default true) and `previous_response_id`. The last 32 responses are kept in memory, not on
  disk, so clients that send the full input every time (`store: false`) are the robust choice.
- **Streaming:** `stream`.

Output items are `reasoning` (the text in `content[0].text`, type `reasoning_text`; `summary` is empty),
`message` (`output_text`) and `function_call` (`call_id`, `name`, `arguments`). The response `status` is
`completed`, or `incomplete` with `incomplete_details.reason: "max_output_tokens"`.

Streaming emits these events, each carrying a `sequence_number`:

1. `response.created`, then `response.in_progress`.
2. For reasoning: `response.output_item.added`, `response.reasoning_text.delta`, `response.reasoning_text.done`,
   `response.output_item.done`.
3. For the answer: `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`,
   `response.output_text.done`, `response.content_part.done`, `response.output_item.done`.
4. For each tool call: `response.output_item.added`, `response.function_call_arguments.delta`,
   `response.function_call_arguments.done`, `response.output_item.done`.
5. `response.completed`, or `response.incomplete`.

Not supported: `item_reference` items, built-in tools, `background`, conversations, `GET /v1/responses/{id}`, and
enforcement of `text.format`.

## Connecting clients

The test suite (`tests/integration/test_server.py`) runs the OpenAI Python SDK against the server when the SDK is
installed. It covers chat with tool calls, streamed chat and Responses. The other configurations below follow each
tool's documentation for OpenAI-compatible endpoints and have not been run here.

### Environment variables (OpenAI SDKs and many tools)

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=none            # or your --api-key
```

### OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
messages = [{"role": "user", "content": "Weather in Paris?"}]
r = client.chat.completions.create(model="qwen3.8-27b", messages=messages, tools=tools, reasoning_effort="low")
msg = r.choices[0].message
if msg.tool_calls:
    messages.append(msg)
    for call in msg.tool_calls:
        messages.append({"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 18}'})
    r = client.chat.completions.create(model="qwen3.8-27b", messages=messages, tools=tools, reasoning_effort="low")
print(r.choices[0].message.content)
```

### LangChain

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="qwen3.8-27b", base_url="http://127.0.0.1:8080/v1", api_key="none",
                 reasoning_effort="low")
llm_with_tools = llm.bind_tools([...])
```

### OpenAI Agents SDK

```python
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled

set_tracing_disabled(True)   # traces would otherwise be uploaded to OpenAI
model = OpenAIChatCompletionsModel(model="qwen3.8-27b",
                                   openai_client=AsyncOpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none"))
agent = Agent(name="assistant", instructions="You are a helpful assistant.", model=model, tools=[...])
print(Runner.run_sync(agent, "Hello").final_output)
```

### Aider

```bash
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=none
aider --model openai/qwen3.8-27b
```

### Continue (VS Code / JetBrains), `config.yaml`

```yaml
models:
  - name: QWENgine
    provider: openai
    model: qwen3.8-27b
    apiBase: http://127.0.0.1:8080/v1
    apiKey: none
    capabilities: [tool_use]
```

### Cline, Roo Code, Kilo Code, Open WebUI

Choose the provider "OpenAI Compatible" (Open WebUI: Settings, Connections, OpenAI API). Set the base URL to
`http://127.0.0.1:8080/v1`, any API key, and model `qwen3.8-27b`. These harnesses send large system prompts
(10K tokens or more). The first request processes it in about 25 s, plus the KV-cache growth steps; later turns
reuse it.

### opencode, `opencode.json`

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "qwengine": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "QWENgine",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": { "qwen3.8-27b": { "name": "Qwen3.8 27B" } }
    }
  }
}
```

### Codex CLI, `~/.codex/config.toml`

```toml
model = "qwen3.8-27b"
model_provider = "qwengine"

[model_providers.qwengine]
name = "QWENgine"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"      # or "chat"
```

## Metrics

`GET /metrics` returns the Prometheus text format, with no flag needed. The series that the llama.cpp server also
exports use its names, so dashboards and scrapers built for llama.cpp work unchanged. Series specific to this
engine are under `q38:`. Counters advance while a request runs, after every prefill chunk and decode pass, so
`rate()` stays smooth during long generations.

```bash
curl -s http://127.0.0.1:8080/metrics                                    # add -H "Authorization: Bearer $KEY" with --api-key
watch -n1 'curl -s localhost:8080/metrics | grep -E "^(llamacpp|q38):(requests_processing|request_|kv_cache_tokens)"'
```

| Metric | Type | Meaning |
|---|---|---|
| `llamacpp:prompt_tokens_total` | counter | prompt tokens processed (reused ones not included) |
| `llamacpp:prompt_seconds_total` | counter | time spent processing prompts |
| `llamacpp:tokens_predicted_total` | counter | tokens generated |
| `llamacpp:tokens_predicted_seconds_total` | counter | time spent generating |
| `llamacpp:n_decode_total` | counter | engine forward passes: prefill chunks, decode and verify passes |
| `llamacpp:n_busy_slots_per_decode` | counter | always 1: one request runs at a time |
| `llamacpp:prompt_tokens_seconds` | gauge | prompt throughput since the previous `/metrics` query, as in llama.cpp (0 if no prompt was processed in between) |
| `llamacpp:predicted_tokens_seconds` | gauge | generation throughput since the previous query |
| `llamacpp:kv_cache_tokens` | gauge | tokens in the KV cache; the cached conversation stays there between requests |
| `llamacpp:kv_cache_usage_ratio` | gauge | `kv_cache_tokens` / `--ctx` |
| `llamacpp:requests_processing` | gauge | 1 while a request is generating |
| `llamacpp:requests_deferred` | gauge | requests waiting for the running one |
| `q38:request_prompt_progress` | gauge | running request: share of its prompt in the KV cache (0 when idle) |
| `q38:request_tokens_predicted` | gauge | running request: tokens generated so far |
| `q38:last_prompt_tokens_seconds`, `q38:last_predicted_tokens_seconds` | gauge | prompt and generation speed of the last finished request |
| `q38:prompt_tokens_cached_total` | counter | prompt tokens reused instead of processed |
| `q38:snapshot_restores_total` | counter | requests that continued from the previous request's snapshot |
| `q38:draft_tokens_total`, `q38:draft_tokens_accepted_total` | counter | MTP drafts verified and accepted |
| `q38:verify_passes_total` | counter | speculative verify passes |
| `q38:draft_acceptance_ratio` | gauge | accepted / verified drafts since start |
| `q38:kv_cache_capacity_tokens` | gauge | allocated KV-cache capacity (grows up to `--ctx`) |
| `q38:context_limit_tokens` | gauge | `--ctx` |
| `q38:kv_cache_resizes_total`, `q38:kv_cache_resize_seconds_total` | counter | KV-cache growths and the time they took |
| `q38:images_encoded_total`, `q38:image_encode_seconds_total` | counter | images run through the vision encoder (reused ones not included) and the time it took |
| `q38:requests_total{endpoint, finish}` | counter | finished requests; `finish` is `stop`, `length`, `tool_calls`, `abort` (client gone) or `error` |
| `process_start_time_seconds` | gauge | server start time |

A Prometheus scrape job:

```yaml
scrape_configs:
  - job_name: qwengine
    static_configs: [{targets: ["127.0.0.1:8080"]}]
    # authorization: {credentials: "<key>"}     # with --api-key
```

Useful queries: `rate(llamacpp:tokens_predicted_total[5m]) / rate(llamacpp:tokens_predicted_seconds_total[5m])`
gives the decode speed, `rate(q38:draft_tokens_accepted_total[5m]) / rate(q38:draft_tokens_total[5m])` the draft
acceptance, and `rate(q38:prompt_tokens_cached_total[5m])` how much prompt work the conversation reuse saves.

## Remote access

The server speaks plain HTTP. To use it from another machine, prefer an SSH tunnel:

```bash
ssh -L 8080:127.0.0.1:8080 laptop
```

The other option is `--host 0.0.0.0 --api-key <secret>` on a trusted network. Put a TLS reverse proxy in front of
it for anything else.
