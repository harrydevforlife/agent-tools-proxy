"""Tests for app/core/formatter.py"""
import json
import pytest
from app.core.formatter import (
    StreamContext,
    role_chunk, content_chunk, content_stop_chunk,
    tool_call_chunks, done_sentinel,
)


def _parse(raw: str) -> dict:
    return json.loads(raw)


CTX = StreamContext(chunk_id="chatcmpl-test123", created=1700000000, model="test-model")


class TestDoneSentinel:
    def test_done_sentinel(self):
        assert done_sentinel() == "[DONE]"


class TestStreamContext:
    def test_stable_id_across_chunks(self):
        d1 = _parse(role_chunk(CTX))
        d2 = _parse(content_chunk("hi", CTX))
        d3 = _parse(content_stop_chunk(CTX))
        assert d1["id"] == d2["id"] == d3["id"] == "chatcmpl-test123"

    def test_stable_created_across_chunks(self):
        d1 = _parse(role_chunk(CTX))
        d2 = _parse(content_chunk("hi", CTX))
        assert d1["created"] == d2["created"] == 1700000000

    def test_model_echoed(self):
        data = _parse(role_chunk(CTX))
        assert data["model"] == "test-model"


class TestRoleChunk:
    def test_role_assistant(self):
        data = _parse(role_chunk(CTX))
        delta = data["choices"][0]["delta"]
        assert delta["role"] == "assistant"

    def test_no_finish_reason(self):
        data = _parse(role_chunk(CTX))
        assert data["choices"][0]["finish_reason"] is None


class TestContentChunk:
    def test_content_present(self):
        data = _parse(content_chunk("hello", CTX))
        assert data["choices"][0]["delta"]["content"] == "hello"

    def test_stop_chunk_finish_reason(self):
        data = _parse(content_stop_chunk(CTX))
        assert data["choices"][0]["finish_reason"] == "stop"


class TestToolCallChunks:
    def setup_method(self):
        self.chunks = tool_call_chunks("search_products", {"keywords": ["shirt"]}, CTX)

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
        chunks = tool_call_chunks("f", {"filters": {"color": "blue", "size": "M"}, "page": 1}, CTX)
        data = _parse(chunks[1])
        tc = data["choices"][0]["delta"]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        assert args["filters"]["color"] == "blue"
        assert args["page"] == 1

    def test_all_chunks_share_id_and_created(self):
        ids = {_parse(c)["id"] for c in self.chunks}
        assert ids == {"chatcmpl-test123"}
        timestamps = {_parse(c)["created"] for c in self.chunks}
        assert timestamps == {1700000000}
