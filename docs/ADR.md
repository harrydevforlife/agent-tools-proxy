# ADR-001 — Brace-depth counting over regex for tool call detection

**Date:** 2025  
**Status:** Accepted

## Context

We need to detect when the LLM is emitting a tool call JSON object in a live
token stream, with minimal added latency.

## Options considered

| Option | Pros | Cons |
|---|---|---|
| Buffer full response, then parse | Simple, reliable | Adds full-response latency — unacceptable for long prose replies |
| Regex on each chunk | Fast | Cannot handle nested JSON in arguments; backtracking risks |
| Brace-depth counting | O(n), handles nesting, low latency | Slightly more complex state machine |
| JSON streaming parser (ijson etc.) | Most correct | Heavy dependency; partial-token handling complex |

## Decision

Brace-depth counting. Watch for the sentinel prefix `{"tool_call"`, then count
`{` increments and `}` decrements. When depth returns to 0, the JSON object
is complete.

## Consequences

- Content tokens pass through immediately — no added latency for prose replies
- Nested argument objects (e.g. `"filters": {"color": "blue"}`) handled correctly
- Edge case: a `{` or `}` inside a JSON string value (e.g. `"text": "use {this}"`)
  would confuse the counter. Mitigated by: the sentinel prefix is distinctive
  enough that the model very rarely emits `{"tool_call"` in prose; if it does,
  the worst case is a parse failure caught by `parse_tool_call()`.

---

# ADR-002 — Custom `{"tool_call": …}` envelope over bare JSON

**Date:** 2025  
**Status:** Accepted

## Context

We need to decide what JSON shape the model should emit for tool calls.

## Options considered

1. **Bare:** `{"name": "fn", "arguments": {…}}`
2. **OpenAI-style:** `{"function": {"name": "fn", "arguments": "…"}}`
3. **Custom envelope:** `{"tool_call": {"name": "fn", "arguments": {…}}}`

## Decision

Custom envelope with `{"tool_call": …}`.

## Rationale

- `{"name":` and `{"function":` are common in normal JSON — high false-positive
  risk for the buffer detector
- `{"tool_call"` is a rare, distinctive prefix in prose — very low false-positive rate
- Arguments stored as a parsed object (not a JSON string like OpenAI's wire format)
  simplifies the prompt and reduces the model's serialization burden
- The token `TOOL_CALL_OPEN_TOKEN` is configurable in `.env` if needed

## Consequences

The formatter re-serializes arguments as a JSON string when building OpenAI
SSE chunks, since OpenAI's spec requires `arguments` to be a string.

---

# ADR-003 — Tool results as user turns (not a `tool` role)

**Date:** 2025  
**Status:** Accepted

## Context

LLMs without native tool support don't have a `tool` role. We need to inject
tool results back into the conversation in a way the model understands.

## Options considered

1. Inject as `assistant` turn — confuses model into thinking it generated the result
2. Inject as `user` turn with plain content — model may treat as user message
3. Inject as `user` turn with `[TOOL RESULT: name]` header — clear provenance

## Decision

Option 3: user turn with `[TOOL RESULT: function_name]` header.

## Consequences

- Model sees tool results as "coming from the environment" (via user turn)
  rather than from itself — preserves the correct mental model
- Function name in header makes multi-tool sessions unambiguous
- Few-shot example in system prompt shows the model this exact pattern so it
  knows how to interpret it

---

# ADR-004 — Single httpx.AsyncClient shared via app.state

**Date:** 2025  
**Status:** Accepted

## Context

Each request to the wrapper needs to make an HTTP call to the LLM backend.
How should the HTTP client be managed?

## Options considered

1. Create a new `httpx.AsyncClient` per request
2. Module-level global client
3. Lifespan-managed client stored in `app.state`

## Decision

Option 3: lifespan-managed, stored in `app.state`.

## Rationale

- Per-request clients don't pool connections — overhead on every call
- Module globals make testing harder (can't easily mock or reset)
- `app.state` is the FastAPI-idiomatic pattern; accessible from any route via
  `request.app.state.http_client`; cleaned up properly in lifespan `finally`

## Consequences

- Tests that exercise the route layer need the TestClient context manager to
  trigger lifespan (handled in `conftest.py`)
- Connection pool settings (`max_connections`, `max_keepalive_connections`)
  are set once at startup — tune via config if needed

---

# ADR-005 — BackendAdapter protocol for pluggable backends

**Date:** 2025
**Status:** Accepted

## Context

The initial implementation hardcoded Ollama's NDJSON streaming format in
`chat.py`. Supporting OpenAI-compatible backends (vLLM, LiteLLM, Groq, real
OpenAI) requires different URL paths and different line-parsing logic.

## Options considered

1. `if/else` on `LLM_BACKEND` setting inside `chat.py`
2. Separate router files per backend
3. `BackendAdapter` abstract class with concrete implementations + factory

## Decision

Option 3: `BackendAdapter` abstract class.

## Structure

```
BackendAdapter (ABC)
├── OllamaAdapter    — /api/chat, NDJSON lines
└── OpenAIAdapter    — /v1/chat/completions, SSE data: lines
```

Factory `get_adapter(settings)` instantiated once at lifespan startup,
stored in `app.state.adapter`, injected into the route via `request.app.state`.

## Rationale

- `chat.py` becomes zero backend-specific code — only consumes `stream_tokens()`
  and `complete()` from the adapter interface
- Adding a third backend (e.g. Anthropic, Cohere) = one new class + one dict entry
- Each adapter is independently unit-testable by calling `_parse_stream_line()`
  directly — no HTTP mocking needed for the parsing logic
- Adapter is selected once at startup — no per-request overhead

## Consequences

- `LLM_CHAT_PATH` setting removed — each adapter owns its URL path
- `LLM_BACKEND` is now the primary config knob (`"ollama"` | `"openai"`)
- `app.state.adapter` must be set in lifespan before any request is served
  (handled; `TestClient` context manager triggers lifespan in tests)