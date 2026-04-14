# Architecture — tool-call-wrapper

## System context

```
┌─────────────────────────────────────────────────────────────────┐
│  Consumer                                                       │
│  (OpenAI SDK, LangChain, AutoGen, curl, any OAI-compat client)  │
└────────────────────────────┬────────────────────────────────────┘
                             │  POST /v1/chat/completions
                             │  { messages, tools[], stream: true }
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  tool-call-wrapper  (this project)                              │
│                                                                 │
│  FastAPI  ·  uvicorn  ·  sse-starlette  ·  httpx               │
└────────────────────────────┬────────────────────────────────────┘
                             │  POST /api/chat (or /v1/chat/completions)
                             │  { model, messages (prompt-injected), stream }
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM Backend                                                    │
│  Ollama · vLLM · LiteLLM proxy · any HTTP streaming endpoint   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Request lifecycle — streaming with tool call

```
Client                      Wrapper                         LLM Backend
  │                           │                                 │
  │  POST /v1/chat/…          │                                 │
  │  tools=[…], stream=true   │                                 │
  ├──────────────────────────►│                                 │
  │                           │  build_llm_messages()           │
  │                           │  ├─ build_tools_block()         │
  │                           │  └─ serialize_history()         │
  │                           │                                 │
  │                           │  POST /api/chat                 │
  │                           │  (prompt-injected messages)     │
  │                           ├────────────────────────────────►│
  │                           │                                 │
  │                           │  ◄── token stream ──────────────┤
  │                           │                                 │
  │                           │  detect_tool_calls()            │
  │                           │  ToolCallDetector.feed()        │
  │                           │  ├─ state: SCANNING             │
  │                           │  ├─ state: ACCUMULATING         │
  │                           │  └─ state: DONE → parse         │
  │                           │                                 │
  │  SSE: role chunk          │                                 │
  │◄──────────────────────────┤                                 │
  │  SSE: tool_call name      │                                 │
  │◄──────────────────────────┤                                 │
  │  SSE: tool_call arguments │                                 │
  │◄──────────────────────────┤                                 │
  │  SSE: finish tool_calls   │                                 │
  │◄──────────────────────────┤                                 │
  │  SSE: [DONE]              │                                 │
  │◄──────────────────────────┤                                 │
  │                           │                                 │
  │  [client executes tool]   │                                 │
  │                           │                                 │
  │  POST /v1/chat/…          │                                 │
  │  messages + tool_result   │                                 │
  ├──────────────────────────►│                                 │
  │                           │  serialize_history()            │
  │                           │  role=tool → [TOOL RESULT: fn]  │
  │                           ├────────────────────────────────►│
  │                           │  ◄── token stream ──────────────┤
  │  SSE: content deltas      │                                 │
  │◄──────────────────────────┤                                 │
  │  SSE: [DONE]              │                                 │
  │◄──────────────────────────┤                                 │
```

---

## Module map

```
app/
├── main.py
│   Responsibility: app factory, lifespan (shared httpx client),
│                   CORS middleware, router mounting, /health endpoint.
│   Key decisions: single AsyncClient shared via app.state avoids
│                  per-request connection overhead.
│
├── config.py
│   Responsibility: all env-var settings via pydantic-settings.
│                   Single source of truth — no magic strings elsewhere.
│   Key fields: llm_base_url, llm_model, llm_chat_path,
│               tool_call_open_token, log_level.
│
├── models/
│   ├── openai.py
│   │   Responsibility: Pydantic models for the OpenAI wire format.
│   │   Covers: ChatCompletionRequest, Message (all roles), ToolDef,
│   │           ChatCompletionChunk, Delta, StreamChoice, ToolCallMessage.
│   │   Note: only fields actually used are modelled — unknown fields ignored.
│   │
│   └── llm.py
│       Responsibility: internal models for the LLM backend (Ollama shape).
│       Swappable: change LLMRequest / LLMStreamChunk here when
│                  targeting a different backend format.
│
├── core/
│   ├── prompt.py
│   │   Responsibility: converts OpenAI tools[] + messages[] into plain-text
│   │                   LLM input. Two public functions:
│   │     build_tools_block(tools)       → <tools>...</tools> string
│   │     build_llm_messages(msgs, tools) → list[{role, content}]
│   │   Internals: _signature(), _param_lines() build human-readable
│   │              function signatures from JSON Schema properties.
│   │              serialize_history() handles all four message roles
│   │              including tool call/result round-trips.
│   │
│   ├── buffer.py
│   │   Responsibility: streaming brace-depth detection.
│   │   Key class: ToolCallDetector — character-level state machine.
│   │     States: SCANNING → ACCUMULATING → DONE
│   │     feed(chunk) → (content_deltas, tool_call_json | None)
│   │   Public helpers:
│   │     parse_tool_call(json_str) → (name, args) | None
│   │     detect_tool_calls(stream)  → AsyncIterator of tagged events
│   │
│   └── formatter.py
│       Responsibility: produce OpenAI-spec SSE data: frames.
│       Functions (one per chunk type):
│         role_chunk()           → first chunk, establishes role=assistant
│         content_chunk(text)    → plain text delta
│         content_stop_chunk()   → finish_reason=stop
│         tool_call_chunks(name, args) → 3-chunk sequence for tool calls
│         done_sentinel()        → data: [DONE]
│
└── routers/
    └── chat.py
        Responsibility: the single route POST /v1/chat/completions.
        Streaming path:  EventSourceResponse(_sse_generator())
        Sync path:       await _complete_sync() + post-hoc tool call check
        Backend glue:    _stream_tokens() (yields str tokens from backend)
                         _build_backend_payload() (constructs backend request)
        Isolation:       these two functions are the only backend-specific code.
                         Adapting for vLLM / LiteLLM = change these only.
```

---

## Data flow — prompt injection detail

```
OpenAI request                        LLM backend input
──────────────                        ─────────────────

messages: [                           messages: [
  { role: "system",          ──┐        { role: "system",
    content: "You are…" }     │          content: "<tools>
                               │                   …function list…
tools: [                       ├────►             </tools>
  { type: "function",          │
    function: {                │                   <tool_rules>
      name: "search",          │                   …rules…
      description: "…",        │                   </tool_rules>
      parameters: {…}          │
    }                          │                   [few-shot example]
  }                          ──┘
]                                                  You are…"  },

                                        { role: "user",
  { role: "user",            ──────►     content: "find a shirt" },
    content: "find a shirt" }
]                                     ]
```

## Data flow — tool result serialization

```
OpenAI request (round 2)              LLM backend input (round 2)
────────────────────────              ──────────────────────────

messages: [                           messages: [
  …prior turns…                         …prior turns…

  { role: "assistant",                  { role: "assistant",
    tool_calls: [{          ──────►       content: '{"tool_call":
      function: {                          {"name":"search",
        name: "search",                    "arguments":{"keywords":
        arguments: '{"keywords":           ["shirt"]}}}' },
          ["shirt"]}'
      }
    }]
  },

  { role: "tool",                       { role: "user",
    name: "search",         ──────►       content: "[TOOL RESULT: search]
    tool_call_id: "c_…",                  {\"results\":[…]}" },
    content: '{"results":[…]}'
  }
]                                     ]
```

---

## SSE chunk sequence — tool call response

```
data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{
         "index":0,"id":"call_abc123","type":"function",
         "function":{"name":"search_products","arguments":""}}]},
       "finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{
         "index":0,
         "function":{"arguments":"{\"keywords\":[\"blue\",\"shirt\"]}"}}]},
       "finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
```

---

## SSE chunk sequence — plain content response

```
data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"I"},"finish_reason":null}]}
data: {"choices":[{"delta":{"content":" found"},"finish_reason":null}]}
data: {"choices":[{"delta":{"content":" two"},"finish_reason":null}]}
…

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## Buffer state machine

```
          feed("{")                    feed(char), depth>0
              │                               │
              ▼                               ▼
  ┌─────────────────┐  prefix match  ┌──────────────────┐
  │    SCANNING     │───────────────►│  ACCUMULATING    │
  │                 │                │  depth counter   │
  │ lookahead buf   │                │  += { -= }       │
  └────────┬────────┘                └────────┬─────────┘
           │                                  │
           │ mismatch / no prefix             │ depth == 0
           │                                  ▼
           │ emit lookahead          ┌──────────────────┐
           │ as content delta        │      DONE        │
           │                         │  emit tool_call  │
           ▼                         │  json string     │
      (SCANNING)                     └──────────────────┘
```

Lookahead buffer handles partial prefix matches across chunk boundaries —
e.g. if one chunk ends with `{"tool_` and the next starts with `call"`,
the detector correctly accumulates both before deciding.
