# agent-tools-proxy

A FastAPI server that gives **any non-function-calling LLM** an OpenAI-compatible
`/v1/chat/completions` endpoint — including full SSE streaming and multi-turn
tool call loops.


## Sequence diagram (Swimlanes.io)

- **Live diagram**: [`swimlanes.io` link](https://swimlanes.io/#nVRNj5tIEL3zK+qISbCVK9pMlM3MSkm09koeKYfRiClDYTrTdJP+GNsb5b+nGrCNVzhS9gZFfbx671FOOEkZ4JaUS53W0qat0fsDxKuW1PuPaaGbFp3YSIKdwbYlM4uiD1JwPqQ38KWPZfDPan0Pi5c3i6JGtwhVkpzQykZKO8qiFPqGsL79DOemHO+mPjzCq+4pL2otCuI36wxhwwkWuZlQW2jRYGMzaHRJEhbgqOHZ6Lwhfmtwnzv9TDwyGmBdINx4IctcyiZvyFre2MazyUShvlLh4I8O2A1spC6eIfZKctkY41ulFU23sGQESvEv5bWwTptDPIO4L0Up7bArGLJeOsuM9iStu5X7VV0NcU/BW2c8zxkP+hMLXrTMBpIgkA4xltg6MvM+OJAxi4ZkSNObEcT+K8Tr9R1oA8vbT+vVcnqbkhwTkp/x8zYbX1Vkzumhd2+LDEJLo2UQsdDKBauwYA6vCHPqOwyi8t31viMSi9qr58Dlw+1qefc4cuU9J2VAeyq8I6i8KoIVoxC+7DhS4bdM3Ys5VEJjt7NJdYYK+o8+x/A1aSqhULK791zzygerzqYJCZIBWsseQ4Y++PropqVWqZ12VIXSXrPU/wXt5VXMozT+hy2drcROEtXIAWGjX+zKhLdMP0F8NBZb92yJ04/0F79tGCaPs9bVRvttDRXnaldzY0bfaqGcnTxl79VhyAvSJ6dsiGm+nXdBajZUlkyrnRbe0AvxntDf0hGIS/KOu22G4HE911/lLySZdeIFwe5EI1GRnQsdRdFKUefznc7g70vVk2ScmyQgLCBYEdQDrfiQUm/7QEfBbnDBHbmlb54UH95S4Dac2XweRXelcMBM9LoyjPAsqeqefVsiuySEhhquSJIPuj10QW8kTw/Y8YWCTrZGc5H/GvKAmUPojiXBdYpvM+xqUoFFOGg/MSvAWxl4Wout4osN3ytMLT+nQqUo3Y+nMLlbjwAVYFFozxIis1xoKcOBD37TVehvTltz14hZPZF7z/BJVkMwaHZfG2Jt8qV2ohIFhnuQn4pYmAye9PPTIAesiSBJHrq/wx6Uwz1o3upF0O4xXmwZA5nDInxO+88zJi1I02gmiw8iCml/Ag==)
- **Preview image**: ![](https://static.swimlanes.io/31c36fc37debb851a1a529ccf50190f8.png)



## Supported endpoints

- **`POST /v1/chat/completions`**: OpenAI-compatible Chat Completions with tool-call wrapping (streaming + non-streaming).
- **`GET /v1/models`**, **`GET /v1/models/{model}`**: Minimal models endpoint for OpenAI SDK compatibility.
- **Catch-all passthrough**: Any other path is reverse-proxied to your backend (e.g. `/v1/embeddings`, `/v1/audio/*`, `/v1/fine_tuning/*`).
- **`GET /health`**: Wrapper health info.

## Project layout

```
agent-tools-proxy/
├── app/
│   ├── main.py              # FastAPI app, lifespan, CORS
│   ├── config.py            # Settings via pydantic-settings + .env
│   ├── models/
│   │   ├── openai.py        # OpenAI-spec request/response Pydantic models
│   ├── core/
│   │   ├── adapters.py      # Backend adapters (ollama vs openai-compatible)
│   │   ├── prompt.py        # Tool schema → system prompt injection
│   │   ├── buffer.py        # Streaming brace-depth buffer & detector
│   │   └── formatter.py     # LLM output → OpenAI SSE chunk formatter
│   └── routers/
│       ├── chat.py          # POST /v1/chat/completions handler
│       ├── models.py        # GET /v1/models
│       └── proxy.py         # Catch-all reverse proxy passthrough
├── tests/
│   ├── test_prompt.py
│   ├── test_buffer.py
│   └── test_formatter.py
├── conftest.py
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── .dockerignore
└── .env.example
```

---

## Quick start

```bash
# 1. install (uv)
uv sync --dev

# 2. configure
cp .env.example .env   # set LLM_BACKEND, LLM_BASE_URL, LLM_MODEL

# 3. start your backend (e.g. Ollama)
ollama serve && ollama pull llama3.1

# 4. run the wrapper
uv run uvicorn app.main:app --reload --port 8080

# 5. run tests (no backend needed)
uv run pytest tests/ -v
```

---

## Docker

```bash
docker build -t agent-tools-proxy:local .

# Example: connect to local Ollama from inside container (macOS)
docker run --rm -p 8080:8080 \
  -e LLM_BACKEND=ollama \
  -e LLM_BASE_URL=http://host.docker.internal:11434 \
  -e LLM_MODEL=llama3.1 \
  agent-tools-proxy:local

# with .env file
docker run --rm -p 8080:8080 \
  -v $(pwd)/.env:/app/.env \
  agent-tools-proxy:local

# Example: connect to local vLLM from inside container (macOS)
docker run --rm -p 8080:8080 \
  -e LLM_BACKEND=openai \
  -e LLM_BASE_URL=https://vllm.zalopay.vn \
  -e LLM_MODEL=gemma-3-27b \
  agent-tools-proxy:local
```

---

## Adapting for other backends

The wrapper uses a pluggable adapter layer in `app/core/adapters.py`:

- **`LLM_BACKEND=ollama`**: talks to Ollama `/api/chat` (NDJSON streaming)
- **`LLM_BACKEND=openai`**: talks to OpenAI-compatible `/v1/chat/completions` backends (SSE streaming)

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | Backend adapter: `ollama` or `openai` |
| `LLM_BASE_URL` | `http://localhost:11434` | LLM backend base URL |
| `LLM_MODEL` | `llama3.1` | Model name passed to backend |
| `LLM_API_KEY` | `` | Bearer token if backend requires auth |
| `TOOL_CALL_OPEN_TOKEN` | `{"tool_call"` | Prefix that signals a tool call in the stream |
| `LOG_LEVEL` | `info` | Python logging level |

---

## Known limitations

- **One tool call per turn** — parallel calls not supported
- **Context grows with tool rounds** — consider summarizing old results after N turns
- **Prompt-based tool calling** — some models may still produce malformed tool JSON (we fall back to treating it as plain content)
