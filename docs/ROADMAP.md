# Roadmap — agent-tools-proxy

Current status: **v0.1 — core working, Ollama backend, single tool call per turn.**

---

## Phase 2 — robustness & backend coverage

### 2.1 vLLM / OpenAI-compatible backend adapter

Isolated to two functions in `routers/chat.py`. Extract into a pluggable
`BackendAdapter` protocol so multiple backends can be registered and selected
via config.

```python
class BackendAdapter(Protocol):
    async def stream_tokens(self, messages: list[dict]) -> AsyncIterator[str]: ...
    async def complete(self, messages: list[dict]) -> str: ...
```

Built-in adapters: `OllamaAdapter`, `OpenAICompatAdapter` (covers vLLM, LiteLLM).

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

### 2.4 Streaming for non-Ollama backends

Add SSE line parsing for OpenAI-compatible streaming:
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

- `max_tokens` passthrough to backend
- `temperature` passthrough
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

Implementation plan:
1. Update prompt to allow comma-separated JSON array:
   `[{"tool_call": …}, {"tool_call": …}]`
2. Update `ToolCallDetector` to handle array depth (outer `[` + inner `{`)
3. Update `formatter.py` to emit multiple `tool_calls` entries with incrementing
   `index` values
4. Update history serializer to handle multiple `tool_calls` in one assistant turn

---

## Phase 5 — `tool_choice` enforcement

Currently `tool_choice: "required"` and `tool_choice: {"function": {"name": "…"}}`
are ignored — the model may or may not call a tool.

Planned enforcement:

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
- **Docker image** — `Dockerfile` + `docker-compose.yml` with Ollama sidecar
- **Helm chart** — for Kubernetes deployment alongside vLLM
- **Admin UI** — simple web page showing registered tools, recent requests,
  and parse failure rate
