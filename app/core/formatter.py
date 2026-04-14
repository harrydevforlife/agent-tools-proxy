"""
formatter.py — Convert parsed tool calls and content deltas into
               OpenAI-compatible SSE data frames.

All public functions return **raw JSON strings** (or the "[DONE]" sentinel).
The caller (EventSourceResponse) handles SSE ``data:`` framing — we must NOT
add our own ``data:`` prefix here to avoid double-wrapping.

OpenAI streaming protocol for tool calls:
  1. First chunk: role="assistant", tool_calls=[{index, id, type, function.name}]
  2. Subsequent chunks: tool_calls=[{index, function.arguments: "...partial..."}]
  3. Final chunk: finish_reason="tool_calls"
  4. "[DONE]" sentinel

For content deltas the protocol is simpler:
  1. First chunk: role="assistant", content=""
  2. Subsequent chunks: content="...token..."
  3. Final chunk: finish_reason="stop"
  4. "[DONE]" sentinel
"""
from __future__ import annotations

import json
import uuid
import time


def _base_chunk(model: str = "tool-wrapper") -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }


def _to_json(chunk: dict) -> str:
    """Serialize a chunk dict to a JSON string (no SSE framing)."""
    return json.dumps(chunk)


def role_chunk(model: str = "tool-wrapper") -> str:
    """First chunk: establishes role=assistant."""
    chunk = _base_chunk(model)
    chunk["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    return _to_json(chunk)


def content_chunk(text: str, model: str = "tool-wrapper") -> str:
    """Mid-stream content delta."""
    chunk = _base_chunk(model)
    chunk["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
    return _to_json(chunk)


def content_stop_chunk(model: str = "tool-wrapper") -> str:
    """Final content chunk — signals finish_reason=stop."""
    chunk = _base_chunk(model)
    chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    return _to_json(chunk)


def tool_call_chunks(
    name: str,
    arguments: dict,
    model: str = "tool-wrapper",
) -> list[str]:
    """
    Produce the sequence of SSE chunks for a single tool call.

    OpenAI sends the function name in one chunk and the arguments
    (as a JSON string) in the next. We follow the same pattern.
    """
    tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
    args_str = json.dumps(arguments, ensure_ascii=False)

    c1 = _base_chunk(model)
    c1["choices"] = [{
        "index": 0,
        "delta": {
            "tool_calls": [{
                "index": 0,
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }]
        },
        "finish_reason": None,
    }]

    c2 = _base_chunk(model)
    c2["choices"] = [{
        "index": 0,
        "delta": {
            "tool_calls": [{
                "index": 0,
                "function": {"arguments": args_str},
            }]
        },
        "finish_reason": None,
    }]

    c3 = _base_chunk(model)
    c3["choices"] = [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]

    return [_to_json(c1), _to_json(c2), _to_json(c3)]


def done_sentinel() -> str:
    """The [DONE] sentinel that signals end-of-stream to OpenAI clients."""
    return "[DONE]"
