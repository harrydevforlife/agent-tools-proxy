"""Tests for app/core/formatter.py"""
import json
import pytest
from app.core.formatter import (
    role_chunk, content_chunk, content_stop_chunk,
    tool_call_chunks, done_sentinel,
)


def _parse(raw: str) -> dict:
    """Formatter now returns raw JSON strings (no SSE data: prefix)."""
    return json.loads(raw)


class TestDoneSentinel:
    def test_done_sentinel(self):
        assert done_sentinel() == "[DONE]"


class TestRoleChunk:
    def test_role_assistant(self):
        data = _parse(role_chunk())
        delta = data["choices"][0]["delta"]
        assert delta["role"] == "assistant"

    def test_no_finish_reason(self):
        data = _parse(role_chunk())
        assert data["choices"][0]["finish_reason"] is None


class TestContentChunk:
    def test_content_present(self):
        data = _parse(content_chunk("hello"))
        assert data["choices"][0]["delta"]["content"] == "hello"

    def test_stop_chunk_finish_reason(self):
        data = _parse(content_stop_chunk())
        assert data["choices"][0]["finish_reason"] == "stop"


class TestToolCallChunks:
    def setup_method(self):
        self.chunks = tool_call_chunks("search_products", {"keywords": ["shirt"]})

    def test_produces_three_chunks(self):
        assert len(self.chunks) == 3

    def test_first_chunk_has_name(self):
        data = _parse(self.chunks[0])
        tc = data["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "search_products"
        assert tc["type"] == "function"
        assert tc["id"].startswith("call_")

    def test_second_chunk_has_arguments(self):
        data = _parse(self.chunks[1])
        tc = data["choices"][0]["delta"]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        assert args == {"keywords": ["shirt"]}

    def test_third_chunk_finish_reason(self):
        data = _parse(self.chunks[2])
        assert data["choices"][0]["finish_reason"] == "tool_calls"

    def test_complex_arguments_serialized(self):
        chunks = tool_call_chunks("f", {"filters": {"color": "blue", "size": "M"}, "page": 1})
        data = _parse(chunks[1])
        tc = data["choices"][0]["delta"]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        assert args["filters"]["color"] == "blue"
        assert args["page"] == 1
