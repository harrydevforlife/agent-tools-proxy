"""
tests/test_adapters.py

Unit tests for OllamaAdapter and OpenAIAdapter.
No real HTTP calls — we test payload building and line parsing directly.
"""
import json
import pytest

from app.config import Settings
from app.core.adapters import OllamaAdapter, OpenAIAdapter, SamplingParams, get_adapter


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _settings(**overrides) -> Settings:
    base = dict(
        llm_backend="ollama",
        llm_base_url="http://localhost:11434",
        llm_model="llama3.1",
        llm_api_key="",
    )
    base.update(overrides)
    return Settings(**base)


MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
]

DEFAULT_PARAMS = SamplingParams()


# ── Factory ───────────────────────────────────────────────────────────────────

class TestGetAdapter:
    def test_ollama(self):
        adapter = get_adapter(_settings(llm_backend="ollama"))
        assert isinstance(adapter, OllamaAdapter)

    def test_openai(self):
        adapter = get_adapter(_settings(llm_backend="openai"))
        assert isinstance(adapter, OpenAIAdapter)

# ── OllamaAdapter ─────────────────────────────────────────────────────────────

class TestOllamaAdapter:
    def setup_method(self):
        self.adapter = OllamaAdapter(_settings(llm_backend="ollama"))

    def test_stream_url(self):
        assert self.adapter._stream_url() == "http://localhost:11434/api/chat"

    def test_complete_url(self):
        assert self.adapter._complete_url() == "http://localhost:11434/api/chat"

    def test_stream_payload_sets_stream_true(self):
        payload = self.adapter._stream_payload(MESSAGES, DEFAULT_PARAMS)
        assert payload["stream"] is True
        assert payload["model"] == "llama3.1"
        assert payload["messages"] == MESSAGES

    def test_complete_payload_sets_stream_false(self):
        payload = self.adapter._complete_payload(MESSAGES, DEFAULT_PARAMS)
        assert payload["stream"] is False

    def test_payload_forwards_temperature(self):
        params = SamplingParams(temperature=0.7)
        payload = self.adapter._stream_payload(MESSAGES, params)
        assert payload["temperature"] == 0.7

    def test_payload_forwards_max_tokens_as_num_predict(self):
        params = SamplingParams(max_tokens=100)
        payload = self.adapter._complete_payload(MESSAGES, params)
        assert payload["num_predict"] == 100

    def test_payload_forwards_model_override(self):
        params = SamplingParams(model="custom-model")
        payload = self.adapter._stream_payload(MESSAGES, params)
        assert payload["model"] == "custom-model"

    def test_payload_omits_none_params(self):
        payload = self.adapter._stream_payload(MESSAGES, DEFAULT_PARAMS)
        assert "temperature" not in payload
        assert "num_predict" not in payload

    def test_parse_stream_line_content(self):
        line = json.dumps({
            "message": {"role": "assistant", "content": "Hello"},
            "done": False,
        })
        token, done = self.adapter._parse_stream_line(line)
        assert token == "Hello"
        assert done is False

    def test_parse_stream_line_done(self):
        line = json.dumps({
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        })
        token, done = self.adapter._parse_stream_line(line)
        assert done is True

    def test_parse_stream_line_empty_content(self):
        line = json.dumps({"message": {"content": ""}, "done": False})
        token, done = self.adapter._parse_stream_line(line)
        assert token == ""
        assert done is False

    def test_parse_complete_response(self):
        data = {"message": {"role": "assistant", "content": "Full response"}, "done": True}
        result = self.adapter._parse_complete_response(data)
        assert result.text == "Full response"
        assert result.usage == {}

    def test_parse_complete_response_with_eval_counts(self):
        data = {
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
            "eval_count": 5,
            "prompt_eval_count": 10,
        }
        result = self.adapter._parse_complete_response(data)
        assert result.text == "Hello"
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_headers_no_key(self):
        h = self.adapter._headers
        assert "Authorization" not in h

    def test_headers_with_key(self):
        adapter = OllamaAdapter(_settings(llm_backend="ollama", llm_api_key="sk-test"))
        assert adapter._headers["Authorization"] == "Bearer sk-test"


# ── OpenAIAdapter ─────────────────────────────────────────────────────────────

class TestOpenAIAdapter:
    def setup_method(self):
        self.adapter = OpenAIAdapter(_settings(
            llm_backend="openai",
            llm_base_url="http://localhost:8000",
            llm_model="mistral-7b",
        ))

    def test_stream_url(self):
        assert self.adapter._stream_url() == "http://localhost:8000/v1/chat/completions"

    def test_complete_url(self):
        assert self.adapter._complete_url() == "http://localhost:8000/v1/chat/completions"

    def test_stream_payload(self):
        payload = self.adapter._stream_payload(MESSAGES, DEFAULT_PARAMS)
        assert payload["stream"] is True
        assert payload["model"] == "mistral-7b"

    def test_complete_payload(self):
        payload = self.adapter._complete_payload(MESSAGES, DEFAULT_PARAMS)
        assert payload["stream"] is False

    def test_payload_forwards_max_tokens(self):
        params = SamplingParams(max_tokens=256)
        payload = self.adapter._stream_payload(MESSAGES, params)
        assert payload["max_tokens"] == 256

    def test_payload_forwards_temperature(self):
        params = SamplingParams(temperature=0.0)
        payload = self.adapter._complete_payload(MESSAGES, params)
        assert payload["temperature"] == 0.0

    def test_payload_omits_none_params(self):
        payload = self.adapter._complete_payload(MESSAGES, DEFAULT_PARAMS)
        assert "temperature" not in payload
        assert "max_tokens" not in payload

    # ── SSE line parsing ──────────────────────────────────────────────────────

    def test_parse_sse_content_line(self):
        data = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
        line = f"data: {json.dumps(data)}"
        token, done = self.adapter._parse_stream_line(line)
        assert token == "Hello"
        assert done is False

    def test_parse_sse_done_sentinel(self):
        token, done = self.adapter._parse_stream_line("data: [DONE]")
        assert token == ""
        assert done is True

    def test_parse_sse_finish_reason_stop(self):
        data = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        line = f"data: {json.dumps(data)}"
        token, done = self.adapter._parse_stream_line(line)
        assert done is True

    def test_parse_sse_finish_reason_tool_calls(self):
        data = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
        line = f"data: {json.dumps(data)}"
        token, done = self.adapter._parse_stream_line(line)
        assert done is True

    def test_parse_sse_null_content(self):
        data = {"choices": [{"delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
        line = f"data: {json.dumps(data)}"
        token, done = self.adapter._parse_stream_line(line)
        assert token == ""
        assert done is False

    def test_parse_sse_no_data_prefix(self):
        data = {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}
        line = json.dumps(data)
        token, done = self.adapter._parse_stream_line(line)
        assert token == "Hi"

    def test_parse_sse_empty_choices(self):
        data = {"choices": []}
        line = f"data: {json.dumps(data)}"
        token, done = self.adapter._parse_stream_line(line)
        assert token == ""
        assert done is False

    def test_parse_complete_response(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "Full response"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter._parse_complete_response(data)
        assert result.text == "Full response"
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_parse_complete_response_empty_choices(self):
        result = self.adapter._parse_complete_response({"choices": []})
        assert result.text == ""

    def test_parse_complete_response_no_usage(self):
        data = {"choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}]}
        result = self.adapter._parse_complete_response(data)
        assert result.text == "Hi"
        assert result.usage == {}

    def test_headers_with_api_key(self):
        adapter = OpenAIAdapter(_settings(
            llm_backend="openai",
            llm_api_key="sk-real-key",
        ))
        assert adapter._headers["Authorization"] == "Bearer sk-real-key"


# ── Real-world vLLM / LiteLLM SSE shapes ─────────────────────────────────────

class TestOpenAIAdapterRealWorldLines:
    def setup_method(self):
        self.adapter = OpenAIAdapter(_settings(llm_backend="openai"))

    def test_vllm_first_chunk(self):
        line = 'data: {"id":"cmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}'
        token, done = self.adapter._parse_stream_line(line)
        assert token == ""
        assert done is False

    def test_vllm_content_chunk(self):
        line = 'data: {"id":"cmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}'
        token, done = self.adapter._parse_stream_line(line)
        assert token == " world"
        assert done is False

    def test_vllm_stop_chunk(self):
        line = 'data: {"id":"cmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        token, done = self.adapter._parse_stream_line(line)
        assert done is True

    def test_litellm_done(self):
        token, done = self.adapter._parse_stream_line("data: [DONE]")
        assert done is True

    def test_groq_content(self):
        line = 'data: {"id":"chatcmpl-xyz","choices":[{"delta":{"content":"Hello"},"finish_reason":null,"index":0}]}'
        token, done = self.adapter._parse_stream_line(line)
        assert token == "Hello"
