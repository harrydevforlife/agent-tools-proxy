# tool-call-wrapper

A FastAPI server that gives **any non-function-calling LLM** an OpenAI-compatible
`/v1/chat/completions` endpoint — including full SSE streaming and multi-turn
tool call loops.

```
Client (OpenAI SDK)  →  Wrapper (this server)  →  Your LLM backend (Ollama, vLLM…)
```

---

## Project layout

```
tool-wrapper/
├── app/
│   ├── main.py              # FastAPI app, lifespan, CORS
│   ├── config.py            # Settings via pydantic-settings + .env
│   ├── models/
│   │   ├── openai.py        # OpenAI-spec request/response Pydantic models
│   │   └── llm.py           # Internal LLM backend models (Ollama-style)
│   ├── core/
│   │   ├── prompt.py        # Tool schema → system prompt injection
│   │   ├── buffer.py        # Streaming brace-depth buffer & detector
│   │   └── formatter.py     # LLM output → OpenAI SSE chunk formatter
│   └── routers/
│       └── chat.py          # POST /v1/chat/completions handler
├── tests/
│   ├── test_prompt.py
│   ├── test_buffer.py
│   └── test_formatter.py
├── conftest.py
├── pyproject.toml
└── .env.example
```

---

## Quick start

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. configure
cp .env.example .env   # set LLM_BASE_URL, LLM_MODEL, LLM_CHAT_PATH

# 3. start your backend (e.g. Ollama)
ollama serve && ollama pull llama3.1

# 4. run the wrapper
uvicorn app.main:app --reload --port 8080

# 5. run tests (no backend needed)
pytest tests/ -v
```

---

## Adapting for other backends

The backend integration is in two functions inside `routers/chat.py`:

```python
def _build_backend_payload(messages, stream) -> dict: ...
async def _stream_tokens(client, messages) -> AsyncIterator[str]: ...
```

### vLLM / OpenAI-compatible backends

Change `_stream_tokens` to parse OpenAI-style SSE lines:

```python
async for line in response.aiter_lines():
    if line.startswith("data: ") and "[DONE]" not in line:
        data = json.loads(line[6:])
        content = data["choices"][0]["delta"].get("content", "")
        if content:
            yield content
```

Set `.env`: `LLM_CHAT_PATH=/v1/chat/completions`

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434` | LLM backend base URL |
| `LLM_MODEL` | `llama3.1` | Model name passed to backend |
| `LLM_API_KEY` | `` | Bearer token if backend requires auth |
| `LLM_CHAT_PATH` | `/api/chat` | Path on backend for chat completions |
| `TOOL_CALL_OPEN_TOKEN` | `{"tool_call"` | Prefix that signals a tool call in the stream |
| `LOG_LEVEL` | `info` | Python logging level |

---

## Known limitations

- **One tool call per turn** — parallel calls not supported
- **Ollama/OpenAI streaming format assumed** — adjust `_stream_tokens` for other backends
- **Context grows with tool rounds** — consider summarizing old results after N turns
- **No `tool_choice: "required"` enforcement** — rely on few-shot prompt examples
