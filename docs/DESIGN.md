# Design document — agent-tools-proxy

## Problem statement

Many capable LLM backends (Ollama-hosted open models, self-hosted vLLM, older
APIs) do not natively implement the OpenAI function/tool calling protocol.
Clients and agent frameworks (LangChain, LlamaIndex, AutoGen, custom OpenAI-SDK
code) assume the OpenAI spec and cannot easily swap backends.

**Goal:** build a thin wrapper server that:

1. Exposes a fully OpenAI-compatible `/v1/chat/completions` endpoint
2. Translates `tools[]` schemas into plain-text system prompt instructions
3. Detects tool calls in raw LLM output via streaming token inspection
4. Re-formats everything as OpenAI SSE chunks — including `tool_calls` deltas
5. Supports multi-turn tool call loops transparently

---

## Why this is possible

The OpenAI tool-call spec is fundamentally a **structured-output contract**, not
a special model capability. The flow is:

```
1. Caller describes available functions (name, params, description)
2. Model emits a JSON object naming the function + arguments
3. Model does NOT execute — the host program does
4. Result is fed back into context
5. Model continues
```

Step 2 is just constrained text generation. Any model that can follow
instructions and produce JSON can do it — with the right prompt.

---

## Core design decisions

### Decision 1 — Prompt injection, not fine-tuning

We teach the model the tool-call format at inference time via the system prompt.
This works with any base model and requires no retraining.

**Trade-off:** reliability depends on model instruction-following quality.
Mitigated by: structured prompt format + mandatory few-shot example in every
system prompt.

### Decision 2 — Brace-depth counting for stream detection

We detect tool calls in the token stream by watching for the sentinel prefix
`{"tool_call"` then counting `{` / `}` until depth returns to 0.

**Why not regex?** Regex cannot handle nested JSON (e.g. `"filters": {"color":
"blue"}`). Brace-depth counting is O(n), handles arbitrary nesting, and
requires no backtracking.

**Why not buffer the whole response?** Buffering the full response before
deciding would add unacceptable latency for long plain-text replies. With
brace-depth we can pass content tokens through immediately and only buffer when
we detect we're inside a tool call.

### Decision 3 — Sentinel token `{"tool_call": ...}`

We chose a custom envelope `{"tool_call": {"name": ..., "arguments": ...}}`
rather than a bare `{"name": ..., "arguments": ...}` for two reasons:

1. The prefix `{"tool_call"` is highly distinctive — far less likely to appear
   in normal prose than a bare `{`.
2. It gives the brace-depth buffer an unambiguous trigger to start accumulating.

This sentinel is configurable via `TOOL_CALL_OPEN_TOKEN` env var.

### Decision 4 — Tool results re-injected as user turns

LLMs without native tool support have no concept of a `tool` role. We serialize
tool results as user turns with a clear header:

```
[TOOL RESULT: function_name]
{...result json...}
```

The header includes the function name so multi-tool conversations stay
unambiguous. Prior tool call turns from the assistant are re-serialized as
their raw JSON string — the model recognises its own prior output format.

### Decision 5 — One tool call per turn

Parallel tool calls (OpenAI allows requesting multiple in one response) are
intentionally not supported. Multi-step tool use is handled by the natural
request loop: each tool call closes the stream, the client executes and sends
the result, the wrapper makes a new LLM call.

This keeps the buffer and prompt logic simple and the failure modes predictable.

### Decision 6 — `httpx.AsyncClient` shared via app state

A single async HTTP client is created at startup (lifespan) and shared across
all requests via `app.state`. This avoids per-request connection overhead and
enables connection pooling to the LLM backend.

---

## Prompt template structure

Every system prompt sent to the LLM has this structure:

```
<tools>
  [format instructions]
  [one-line JSON format example]
  [function list with signatures + param descriptions]
</tools>

<tool_rules>
  [5 numbered behavioral rules]
</tool_rules>

[Few-shot example: full USER → ASSISTANT → TOOL RESULT → ASSISTANT turn]

[Original system message from caller, if any]
```

The few-shot example is the most important element — it shows the model the
exact output format it must produce, not just describes it.

---

## Limitations and known gaps

| Limitation | Impact | Mitigation |
|---|---|---|
| Parallel tool calls not supported | Agents requiring simultaneous calls need multi-turn | Document clearly; most agent frameworks fall back gracefully |
| Model may produce plain text instead of a tool call | Agent loop breaks | Caller can retry; `tool_choice: "required"` via prompt strengthening |
| Context grows with tool round-trips | Long sessions hit context window | Summarise old tool results after N rounds (future work) |
| Ollama streaming format assumed | Other backends need `_stream_tokens` adaptation | Isolated in one function; vLLM adaptation documented |
| No auth on the wrapper itself | Fine for local/internal use | Add FastAPI `Security` dependency for production |

---

## What's next

See `docs/ROADMAP.md` for planned work.
