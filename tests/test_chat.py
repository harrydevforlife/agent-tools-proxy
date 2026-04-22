"""
tests/test_chat.py — Integration tests for POST /v1/chat/completions

Uses FastAPI TestClient with mocked adapter to exercise the full HTTP path.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.adapters import CompleteResult, SamplingParams
from app.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_request(
    *,
    messages=None,
    stream=False,
    tools=None,
    tool_choice="auto",
    model="test-model",
    temperature=None,
    max_tokens=None,
):
    body = {
        "model": model,
        "messages": messages or [{"role": "user", "content": "Hello"}],
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]

TOOL_CALL_JSON = '{"tool_call": {"name": "get_weather", "arguments": {"city": "Hanoi"}}}'


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ── Non-streaming tests ──────────────────────────────────────────────────────

class TestNonStreaming:
    def test_plain_text_response(self, client):
        result = CompleteResult(text="Hello there!", usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8})
        with patch.object(app.state.adapter, "complete", new_callable=AsyncMock, return_value=result):
            resp = client.post("/v1/chat/completions", json=_make_request())

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello there!"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 5
        assert data["usage"]["total_tokens"] == 8

    def test_model_echoed_in_response(self, client):
        result = CompleteResult(text="Hi")
        with patch.object(app.state.adapter, "complete", new_callable=AsyncMock, return_value=result):
            resp = client.post("/v1/chat/completions", json=_make_request())

        data = resp.json()
        assert data["model"] == app.state.adapter.settings.llm_model

    def test_tool_call_response(self, client):
        result = CompleteResult(text=TOOL_CALL_JSON, usage={})
        with patch.object(app.state.adapter, "complete", new_callable=AsyncMock, return_value=result):
            resp = client.post(
                "/v1/chat/completions",
                json=_make_request(tools=SAMPLE_TOOLS),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        args = json.loads(tc["function"]["arguments"])
        assert args["city"] == "Hanoi"
        assert tc["id"].startswith("call_")

    def test_tool_choice_none_skips_tool_detection(self, client):
        result = CompleteResult(text="I cannot call tools right now.")
        with patch.object(app.state.adapter, "complete", new_callable=AsyncMock, return_value=result):
            resp = client.post(
                "/v1/chat/completions",
                json=_make_request(tools=SAMPLE_TOOLS, tool_choice="none"),
            )

        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["choices"][0]["message"]["content"] == "I cannot call tools right now."

    def test_no_usage_field_when_backend_omits(self, client):
        result = CompleteResult(text="Hi", usage={})
        with patch.object(app.state.adapter, "complete", new_callable=AsyncMock, return_value=result):
            resp = client.post("/v1/chat/completions", json=_make_request())

        data = resp.json()
        assert "usage" not in data

    def test_content_parts_array_accepted(self, client):
        """Agents SDK sends content as [{"type":"text","text":"..."}] — must not 422."""
        result = CompleteResult(text="Got it.")
        with patch.object(app.state.adapter, "complete", new_callable=AsyncMock, return_value=result):
            resp = client.post("/v1/chat/completions", json={
                "model": "test",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello from SDK"}]},
                ],
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Got it."

    def test_content_parts_tool_result(self, client):
        """Tool result sent as content-parts array should be accepted."""
        result = CompleteResult(text="Weather is nice.")
        with patch.object(app.state.adapter, "complete", new_callable=AsyncMock, return_value=result):
            resp = client.post("/v1/chat/completions", json={
                "model": "test",
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Hanoi"}'}
                    }]},
                    {"role": "tool", "tool_call_id": "call_1", "name": "get_weather",
                     "content": [{"type": "text", "text": '{"temp": 34}'}]},
                ],
            })

        assert resp.status_code == 200

    def test_sampling_params_forwarded(self, client):
        result = CompleteResult(text="warm")
        mock_complete = AsyncMock(return_value=result)
        with patch.object(app.state.adapter, "complete", mock_complete):
            client.post(
                "/v1/chat/completions",
                json=_make_request(temperature=0.7, max_tokens=50),
            )

        call_args = mock_complete.call_args
        params: SamplingParams = call_args[0][2]
        assert params.temperature == 0.7
        assert params.max_tokens == 50


# ── Streaming tests ───────────────────────────────────────────────────────────

def _parse_sse_events(raw: str) -> list[dict | str]:
    """Parse raw SSE text into a list of JSON dicts or the '[DONE]' string."""
    events = []
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                events.append("[DONE]")
            else:
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
    return events


class TestStreaming:
    def test_plain_text_stream(self, client):
        async def _fake_stream(*args, **kwargs):
            for token in ["Hello", " world"]:
                yield token

        with patch.object(app.state.adapter, "stream_tokens", _fake_stream):
            resp = client.post(
                "/v1/chat/completions",
                json=_make_request(stream=True),
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        json_events = [e for e in events if isinstance(e, dict)]
        assert len(json_events) >= 3  # role + content(s) + stop

        # All chunks share the same id
        ids = {e["id"] for e in json_events}
        assert len(ids) == 1

        # First chunk has role
        assert json_events[0]["choices"][0]["delta"]["role"] == "assistant"

        # Last JSON event before [DONE] has finish_reason=stop
        assert json_events[-1]["choices"][0]["finish_reason"] == "stop"

        # [DONE] is the final event
        assert events[-1] == "[DONE]"

    def test_tool_call_stream(self, client):
        async def _fake_stream(*args, **kwargs):
            for char in TOOL_CALL_JSON:
                yield char

        with patch.object(app.state.adapter, "stream_tokens", _fake_stream):
            resp = client.post(
                "/v1/chat/completions",
                json=_make_request(stream=True, tools=SAMPLE_TOOLS),
            )

        events = _parse_sse_events(resp.text)
        json_events = [e for e in events if isinstance(e, dict)]

        # Should have: role chunk, name chunk, args chunk, finish chunk
        finish_events = [e for e in json_events if e["choices"][0].get("finish_reason") == "tool_calls"]
        assert len(finish_events) == 1

        # Tool name should appear
        tool_chunks = [
            e for e in json_events
            if "tool_calls" in e["choices"][0].get("delta", {})
        ]
        assert len(tool_chunks) >= 1
        first_tc = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert first_tc["function"]["name"] == "get_weather"

        assert events[-1] == "[DONE]"

    def test_stream_parse_failure_fallback(self, client):
        """If tool call JSON is malformed, it should fall back to content."""
        bad_json = '{"tool_call": {"name": "f", "arguments": BROKEN}'

        async def _fake_stream(*args, **kwargs):
            for char in bad_json:
                yield char

        with patch.object(app.state.adapter, "stream_tokens", _fake_stream):
            resp = client.post(
                "/v1/chat/completions",
                json=_make_request(stream=True, tools=SAMPLE_TOOLS),
            )

        events = _parse_sse_events(resp.text)
        json_events = [e for e in events if isinstance(e, dict)]

        # Should still get content with fallback
        content_parts = []
        for e in json_events:
            c = e["choices"][0].get("delta", {}).get("content")
            if c:
                content_parts.append(c)

        assert len(content_parts) > 0
        assert events[-1] == "[DONE]"

    def test_model_echoed_in_stream_chunks(self, client):
        async def _fake_stream(*args, **kwargs):
            yield "hi"

        with patch.object(app.state.adapter, "stream_tokens", _fake_stream):
            resp = client.post(
                "/v1/chat/completions",
                json=_make_request(stream=True),
            )

        events = _parse_sse_events(resp.text)
        json_events = [e for e in events if isinstance(e, dict)]
        for e in json_events:
            assert e["model"] == app.state.adapter.settings.llm_model


# ── Error shape tests ─────────────────────────────────────────────────────────

class TestErrorShape:
    def test_validation_error_openai_shape(self, client):
        resp = client.post("/v1/chat/completions", json={"messages": "not a list"})
        assert resp.status_code == 422
        data = resp.json()
        assert "error" in data
        assert "message" in data["error"]
        assert data["error"]["type"] == "invalid_request_error"

    def test_404_openai_shape(self, client):
        resp = client.get("/v1/models/nonexistent-model")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == 404
