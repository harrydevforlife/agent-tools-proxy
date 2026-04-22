"""
formatter.py — Convert parsed tool calls and content deltas into
               OpenAI-compatible SSE data frames.

All public functions return **raw JSON strings** (or the "[DONE]" sentinel).
The caller (EventSourceResponse) handles SSE ``data:`` framing — we must NOT
add our own ``data:`` prefix here to avoid double-wrapping.

Every chunk in a single stream must share the same ``id`` and ``created``
timestamp (matching the OpenAI spec). The caller generates these once and
passes them to all formatter calls via a StreamContext.
"""
from __future__ import annotations

import json
import uuid
import time
from dataclasses import dataclass, field


@dataclass
class StreamContext:
    """Immutable identifiers shared across all chunks in a single stream."""
    chunk_id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "tool-wrapper"


def _base_chunk(ctx: StreamContext) -> dict:
    return {
        "id": ctx.chunk_id,
        "object": "chat.completion.chunk",
        "created": ctx.created,
        "model": ctx.model,
    }


def _to_json(chunk: dict) -> str:
    return json.dumps(chunk)


def role_chunk(ctx: StreamContext) -> str:
    """First chunk: establishes role=assistant."""
    chunk = _base_chunk(ctx)
    chunk["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    return _to_json(chunk)


def content_chunk(text: str, ctx: StreamContext) -> str:
    """Mid-stream content delta."""
    chunk = _base_chunk(ctx)
    chunk["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
    return _to_json(chunk)


def content_stop_chunk(ctx: StreamContext) -> str:
    """Final content chunk — signals finish_reason=stop."""
    chunk = _base_chunk(ctx)
    chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    return _to_json(chunk)


def tool_call_chunks(
    name: str,
    arguments: dict,
    ctx: StreamContext,
) -> list[str]:
    """
    Produce the sequence of SSE chunks for a single tool call.

    OpenAI sends the function name in one chunk and the arguments
    (as a JSON string) in the next. We follow the same pattern.
    """
    tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
    args_str = json.dumps(arguments, ensure_ascii=False)

    c1 = _base_chunk(ctx)
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

    c2 = _base_chunk(ctx)
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

    c3 = _base_chunk(ctx)
    c3["choices"] = [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]

    return [_to_json(c1), _to_json(c2), _to_json(c3)]


def done_sentinel() -> str:
    """The [DONE] sentinel that signals end-of-stream to OpenAI clients."""
    return "[DONE]"
