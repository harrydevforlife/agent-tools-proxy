# Roadmap — agent-tools-proxy

Current status: **v0.1 — core working, Ollama + OpenAI-compatible backends, integration tests, Dockerfile, and catch-all passthrough proxy.**

---

## Phase 2 — robustness & backend coverage

### 2.1 vLLM / OpenAI-compatible backend adapter

Status: **DONE**

Extracted into a pluggable `BackendAdapter` base class in `app/core/adapters.py`
and selected via `LLM_BACKEND`.

```python
class BackendAdapter:
    async def stream_tokens(...): ...
    async def complete(...): ...
```

Built-in adapters: `OllamaAdapter`, `OpenAIAdapter` (covers vLLM, LiteLLM, OpenAI, Groq, etc.).

### 2.2 Retry + fallback on malformed tool call JSON

Currently: malformed JSON falls back to emitting as content.
Planned: on parse failure, re-prompt the model once with:
> Your previous response was not valid JSON. Please respond only with the JSON object.

Configurable retry count via `TOOL_CALL_MAX_RETRIES` (default: 1).

### 2.3 Context window management

Long tool call chains accumulate history fast. Add a `ContextManager` that:
- Tracks approximate token count (via `len(text) / 4` heuristic or tiktoken)
- Summarises old tool results to one-line descriptions once count exceeds
  `CONTEXT_WARN_TOKENS` (default: 3000)
- Hard-truncates oldest non-system turns at `CONTEXT_MAX_TOKENS` (default: 6000)

Priority: **P2**

Implementation plan:
1. Add configurable history policies (env/config):
   - **Windowing**: keep last \(N\) messages (always keep system + injected tools block)
   - **Summarization**: replace older tool rounds with a compact summary message
2. Ensure correctness constraints:
   - Never drop the most recent user turn
   - Never drop the tool result that the assistant is about to use
   - Keep tool definitions consistent with any forced `tool_choice`
3. Optional debugging headers (opt-in) to report how much was dropped/summarized.
4. Add tests for policy behaviour (keeps newest rounds, deterministic trimming, etc.).

### 2.4 Streaming for non-Ollama backends

Status: **DONE**

SSE line parsing implemented for OpenAI-compatible streaming:
```
data: {"choices":[{"delta":{"content":"…"},"finish_reason":null}]}
```

---

## Phase 3 — auth & production hardening

### 3.1 Wrapper-level API key auth

Add a FastAPI `Security` dependency:
```python
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.wrapper_api_key:
        raise HTTPException(401)
```

Controlled by `WRAPPER_API_KEY` env var. Empty = auth disabled (dev default).

### 3.2 Request validation & limits

Status: **PARTIAL**

- `max_tokens` passthrough to backend (**DONE**)
- `temperature` passthrough (**DONE**)
- client `model` passthrough (**DONE**)
- Request size limit (prevent prompt injection via enormous `messages[]`)
- Rate limiting via `slowapi` or a reverse proxy (nginx/caddy)

### 3.3 Observability

- Structured JSON logging (replace basicConfig with `structlog`)
- Prometheus metrics endpoint `/metrics`:
  - `tool_calls_total` counter (by function name)
  - `llm_request_duration_seconds` histogram
  - `tool_call_parse_failures_total` counter
- Optional OpenTelemetry trace export

---

## Phase 4 — parallel tool calls

Currently one tool call per turn. OpenAI supports returning multiple tool calls
in a single response — some agent frameworks depend on this.

Priority: **P1**

Implementation plan:
1. Update prompt rules to allow **multiple tool calls** while keeping strict JSON-only responses:
   - Prefer a JSON array output: `[{"tool_call": ...}, {"tool_call": ...}]`
2. Extend `ToolCallDetector` / `detect_tool_calls` to accumulate either:
   - a JSON array of tool calls, or
   - multiple sequential `{"tool_call": ...}` objects in one output (optional compatibility mode)
3. Update `formatter.py` to emit tool-call chunks for indexes `[0..n-1]`, maintaining stable stream `id/created`.
4. Update history serializer (`serialize_history`) to preserve all tool calls in a single assistant turn (already done for OpenAI `tool_calls[]`; extend prompt-side expectations accordingly).
5. Add integration tests for “2 tool calls in one response” (streaming + non-streaming).

---

## Phase 5 — `tool_choice` enforcement

Status: **DONE**

Implemented enforcement:

| `tool_choice` value | Behaviour |
|---|---|
| `"auto"` (default) | Current behaviour — model decides |
| `"none"` | Strip `<tools>` block entirely — model cannot call |
| `"required"` | Add rule: "You MUST call one of the available functions. Do not respond in plain text." |
| `{"function": {"name": "X"}}` | Add rule: "You MUST call the function named X." |

---

## Phase 6 — structured output mode

Some use cases want the model to always respond in a JSON schema (not tool
calls, just structured output). Extend the wrapper to support OpenAI's
`response_format: {"type": "json_schema", "json_schema": {…}}` by injecting
a schema description into the system prompt and validating the response.

---

## Backlog (unscheduled)

- **MCP server integration** — expose registered tools as an MCP server so
  MCP-aware clients can discover tools dynamically (instead of sending `tools[]`
  in each request)
- **WebSocket transport** — alternative to SSE for clients that prefer WS
- **Docker image** — `Dockerfile` (**DONE**); `docker-compose.yml` with Ollama sidecar (TODO)
- **Helm chart** — for Kubernetes deployment alongside vLLM
- **Admin UI** — simple web page showing registered tools, recent requests,
  and parse failure rate
