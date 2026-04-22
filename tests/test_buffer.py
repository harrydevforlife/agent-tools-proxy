"""Tests for app/core/buffer.py"""
import pytest
from app.core.buffer import BufferState, ToolCallDetector, parse_tool_call, detect_tool_calls


class TestToolCallDetector:
    def _feed_all(self, text: str) -> tuple[list[str], str | None]:
        detector = ToolCallDetector()
        all_deltas: list[str] = []
        tool_call = None
        for char in text:
            deltas, tc = detector.feed(char)
            all_deltas.extend(deltas)
            if tc:
                tool_call = tc
        return all_deltas, tool_call

    def test_plain_text_passes_through(self):
        text = "Hello, world!"
        deltas, tc = self._feed_all(text)
        assert "".join(deltas) == text
        assert tc is None

    def test_detects_simple_tool_call(self):
        json_str = '{"tool_call": {"name": "search", "arguments": {"q": "shirt"}}}'
        deltas, tc = self._feed_all(json_str)
        assert tc == json_str
        assert deltas == []  # nothing passed through as content

    def test_content_before_tool_call(self):
        text = 'Sure! {"tool_call": {"name": "foo", "arguments": {}}}'
        deltas, tc = self._feed_all(text)
        assert "Sure! " in "".join(deltas)
        assert tc is not None
        assert '"name": "foo"' in tc

    def test_nested_arguments_handled(self):
        json_str = '{"tool_call": {"name": "f", "arguments": {"filters": {"color": "blue", "size": "M"}}}}'
        deltas, tc = self._feed_all(json_str)
        assert tc == json_str

    def test_chunked_feeding(self):
        """Detector must work when JSON is split across arbitrary chunk boundaries."""
        json_str = '{"tool_call": {"name": "x", "arguments": {"a": 1}}}'
        detector = ToolCallDetector()
        tc_result = None
        # Feed 3 chars at a time
        for i in range(0, len(json_str), 3):
            chunk = json_str[i:i+3]
            _, tc = detector.feed(chunk)
            if tc:
                tc_result = tc
        assert tc_result == json_str

    def test_plain_text_after_no_tool_call(self):
        """Stream ending without a tool call should emit all text."""
        text = "No tools needed here."
        deltas, tc = self._feed_all(text)
        assert "".join(deltas) == text
        assert tc is None

    def test_partial_match_then_mismatch(self):
        """A '{' that doesn't start a tool call should be emitted as content."""
        text = '{"other": 1} and some text'
        deltas, tc = self._feed_all(text)
        assert tc is None
        print(f"deltas: {deltas}")
        print(f"tc: {tc}")
        assert "{" in "".join(deltas)


class TestParseToolCall:
    def test_valid_json(self):
        json_str = '{"tool_call": {"name": "search", "arguments": {"keywords": ["shirt"]}}}'
        result = parse_tool_call(json_str)
        assert result is not None
        name, args = result
        assert name == "search"
        assert args == {"keywords": ["shirt"]}

    def test_empty_arguments(self):
        json_str = '{"tool_call": {"name": "ping", "arguments": {}}}'
        result = parse_tool_call(json_str)
        assert result is not None
        assert result[0] == "ping"
        assert result[1] == {}

    def test_invalid_json_returns_none(self):
        assert parse_tool_call("{broken json}") is None

    def test_missing_name_returns_none(self):
        assert parse_tool_call('{"tool_call": {"arguments": {}}}') is None

    def test_wrong_structure_returns_none(self):
        assert parse_tool_call('{"name": "foo"}') is None


@pytest.mark.asyncio
class TestDetectToolCalls:
    async def _collect(self, tokens: list[str]) -> list[tuple[str, str | None]]:
        async def gen():
            for t in tokens:
                yield t
        events = []
        async for event in detect_tool_calls(gen()):
            events.append(event)
        return events

    async def test_plain_content_emitted(self):
        events = await self._collect(["Hello", " world"])
        types = [e[0] for e in events]
        assert "content" in types
        assert "tool_call" not in types
        assert ("done", None) in events

    async def test_tool_call_detected(self):
        json_str = '{"tool_call": {"name": "f", "arguments": {}}}'
        tokens = list(json_str)
        events = await self._collect(tokens)
        tc_events = [e for e in events if e[0] == "tool_call"]
        assert len(tc_events) == 1
        assert tc_events[0][1] == json_str
        # done sentinel should NOT appear (stream closes after tool call)
        assert ("done", None) not in events

    async def test_content_then_tool_call(self):
        prefix = "Sure! "
        tc = '{"tool_call": {"name": "x", "arguments": {"a": "b"}}}'
        events = await self._collect(list(prefix + tc))
        content = "".join(p for t, p in events if t == "content")
        assert "Sure! " in content
        tc_events = [e for e in events if e[0] == "tool_call"]
        assert len(tc_events) == 1

    async def test_unclosed_braces_emitted_as_content(self):
        """Stream ends mid-accumulation — partial buffer should appear as content."""
        tokens = list('{"tool_call": {"name": "f", "arguments": {"x":')
        events = await self._collect(tokens)
        tc_events = [e for e in events if e[0] == "tool_call"]
        assert len(tc_events) == 0
        content = "".join(p for t, p in events if t == "content")
        assert '{"tool_call"' in content
        assert ("done", None) in events


# ── Buffer edge case tests ────────────────────────────────────────────────────

class TestBufferEdgeCases:
    """Edge cases for ToolCallDetector: unclosed braces, string braces, large payloads."""

    def _feed_all(self, text: str, chunk_size: int = 1) -> tuple[list[str], str | None]:
        detector = ToolCallDetector()
        all_deltas: list[str] = []
        tool_call = None
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            deltas, tc = detector.feed(chunk)
            all_deltas.extend(deltas)
            if tc:
                tool_call = tc
        if tool_call is None:
            remaining = detector.flush_lookahead()
            if detector.state == BufferState.ACCUMULATING:
                remaining += detector.flush_accumulating()
            if remaining:
                all_deltas.append(remaining)
        return all_deltas, tool_call

    def test_unclosed_braces_returns_partial_buffer(self):
        """Stream ends with unclosed braces — partial buffer returned as content."""
        text = '{"tool_call": {"name": "f", "arguments": {"x": 1'
        deltas, tc = self._feed_all(text)
        assert tc is None
        combined = "".join(deltas)
        assert '{"tool_call"' in combined

    def test_braces_inside_json_string_values(self):
        """Braces inside string values must not break depth counting."""
        json_str = '{"tool_call": {"name": "f", "arguments": {"desc": "use { and } carefully"}}}'
        deltas, tc = self._feed_all(json_str)
        assert tc == json_str

    def test_escaped_quotes_in_string_values(self):
        """Escaped quotes inside strings must not toggle string state."""
        json_str = '{"tool_call": {"name": "f", "arguments": {"msg": "say \\"hello\\" {here}"}}}'
        deltas, tc = self._feed_all(json_str)
        assert tc == json_str

    def test_large_payload_across_many_chunks(self):
        """Large tool call JSON split across many small chunks."""
        import json as jsonmod
        big_args = {f"key_{i}": f"value_{i}" for i in range(50)}
        json_str = jsonmod.dumps({"tool_call": {"name": "big_fn", "arguments": big_args}})
        deltas, tc = self._feed_all(json_str, chunk_size=3)
        assert tc == json_str

    def test_large_payload_char_by_char(self):
        import json as jsonmod
        big_args = {f"k{i}": "x" * 100 for i in range(10)}
        json_str = jsonmod.dumps({"tool_call": {"name": "f", "arguments": big_args}})
        deltas, tc = self._feed_all(json_str, chunk_size=1)
        assert tc == json_str

    def test_nested_objects_with_string_braces(self):
        """Nested objects where string values contain braces at multiple levels."""
        json_str = '{"tool_call": {"name": "f", "arguments": {"a": {"b": "text{{}}", "c": 1}}}}'
        deltas, tc = self._feed_all(json_str)
        assert tc == json_str

    def test_empty_arguments(self):
        json_str = '{"tool_call": {"name": "f", "arguments": {}}}'
        deltas, tc = self._feed_all(json_str)
        assert tc == json_str

    def test_backslash_at_end_of_string_not_escaping_quote(self):
        """A backslash that is itself escaped should not escape the following quote."""
        json_str = '{"tool_call": {"name": "f", "arguments": {"path": "C:\\\\"}}}'
        deltas, tc = self._feed_all(json_str)
        assert tc == json_str
