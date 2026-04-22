"""
buffer.py — Streaming token buffer with brace-depth–based tool call detection.

State machine:
  SCANNING     → watching for tool_call_open_token in the stream
                 content tokens pass through immediately as deltas
  ACCUMULATING → opening brace seen, buffering all tokens
                 depth counter tracks nested braces
                 braces inside JSON string literals are ignored
  DONE         → closing brace matched depth=0, buffer complete

The caller receives tokens via an async generator that yields either:
  ("content", str)    — plain text delta to forward to the client
  ("tool_call", str)  — complete JSON string ready for parsing
  ("done", None)      — stream ended with no tool call (or after tool call)
"""
from __future__ import annotations

import json
from enum import Enum, auto
from typing import AsyncIterator


class BufferState(Enum):
    SCANNING = auto()
    ACCUMULATING = auto()
    DONE = auto()


class ToolCallDetector:
    """
    Stateful brace-depth detector with string-literal awareness.
    Feed raw text chunks; call .feed() repeatedly.
    """

    OPEN_TOKEN = '{"tool_call"'

    def __init__(self, open_token: str = OPEN_TOKEN) -> None:
        self.open_token = open_token
        self._state = BufferState.SCANNING
        self._buffer = ""
        self._lookahead = ""
        self._depth = 0
        self._in_string = False
        self._escape_next = False

    # ── Public API ────────────────────────────────────────────────────────────

    def feed(self, chunk: str) -> tuple[list[str], str | None]:
        """
        Process one text chunk.

        Returns:
            content_deltas: list of plain-text strings to forward immediately
            tool_call_json: complete JSON string if a tool call was completed, else None
        """
        content_deltas: list[str] = []
        tool_call_json: str | None = None

        for char in chunk:
            if self._state == BufferState.SCANNING:
                flushed, result = self._scan(char)
                if flushed:
                    content_deltas.append(flushed)
                if result == "matched":
                    self._state = BufferState.ACCUMULATING
                    self._buffer = self.open_token
                    self._depth = self.open_token.count("{")
                    self._lookahead = ""
                    self._in_string = False
                    self._escape_next = False

            elif self._state == BufferState.ACCUMULATING:
                self._buffer += char

                if self._escape_next:
                    self._escape_next = False
                    continue

                if char == "\\" and self._in_string:
                    self._escape_next = True
                    continue

                if char == '"':
                    self._in_string = not self._in_string
                    continue

                if self._in_string:
                    continue

                if char == "{":
                    self._depth += 1
                elif char == "}":
                    self._depth -= 1
                    if self._depth == 0:
                        self._state = BufferState.DONE
                        tool_call_json = self._buffer

        return content_deltas, tool_call_json

    def flush_lookahead(self) -> str:
        """Call at stream end — any remaining lookahead is plain content."""
        remaining = self._lookahead
        self._lookahead = ""
        return remaining

    def flush_accumulating(self) -> str:
        """Call at stream end while ACCUMULATING — return partial buffer as content."""
        if self._state == BufferState.ACCUMULATING:
            remaining = self._buffer
            self._buffer = ""
            self._state = BufferState.DONE
            return remaining
        return ""

    @property
    def state(self) -> BufferState:
        return self._state

    @property
    def is_done(self) -> bool:
        return self._state == BufferState.DONE

    # ── Internal ──────────────────────────────────────────────────────────────

    def _scan(self, char: str) -> tuple[str, str]:
        """
        Incrementally match the open_token prefix.

        Returns:
            (flushed, result) where:
              flushed — content string to emit immediately (may be empty)
              result  — "matched" | "pending" | "content"
        """
        candidate = self._lookahead + char
        if self.open_token.startswith(candidate):
            self._lookahead = candidate
            if candidate == self.open_token:
                return "", "matched"
            return "", "pending"
        else:
            for start in range(1, len(candidate)):
                tail = candidate[start:]
                if self.open_token.startswith(tail):
                    flushed = candidate[:start]
                    self._lookahead = tail
                    return flushed, "pending"
            self._lookahead = ""
            return candidate, "content"


def parse_tool_call(json_str: str) -> tuple[str, dict] | None:
    """
    Parse a buffered tool call JSON string.
    Returns (function_name, arguments_dict) or None on parse error.

    Expected shape:
      {"tool_call": {"name": "...", "arguments": {...}}}
    """
    try:
        data = json.loads(json_str)
        tc = data["tool_call"]
        name = tc["name"]
        args = tc.get("arguments", {})
        return name, args
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


# ── Async generator adapter ───────────────────────────────────────────────────

async def detect_tool_calls(
    token_stream: AsyncIterator[str],
    open_token: str = ToolCallDetector.OPEN_TOKEN,
) -> AsyncIterator[tuple[str, str | None]]:
    """
    Wraps a raw token stream and yields tagged events:
      ("content", text)
      ("tool_call", json_string)
      ("done", None)
    """
    detector = ToolCallDetector(open_token=open_token)

    async for chunk in token_stream:
        if detector.is_done:
            break

        content_deltas, tool_call_json = detector.feed(chunk)

        for delta in content_deltas:
            if delta:
                yield ("content", delta)

        if tool_call_json is not None:
            yield ("tool_call", tool_call_json)
            return

    # Stream ended — flush any remaining state
    remaining = detector.flush_lookahead()

    if detector.state == BufferState.ACCUMULATING:
        partial = detector.flush_accumulating()
        if partial:
            yield ("content", partial)
    elif remaining:
        yield ("content", remaining)

    yield ("done", None)
