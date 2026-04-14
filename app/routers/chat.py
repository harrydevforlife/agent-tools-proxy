"""
routers/chat.py — POST /v1/chat/completions

Handles both streaming and non-streaming requests.
Backend communication is fully delegated to the BackendAdapter stored in
app.state — this router contains zero backend-specific logic.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.core.adapters import BackendAdapter
from app.core.buffer import detect_tool_calls, parse_tool_call
from app.core.formatter import (
    content_chunk,
    content_stop_chunk,
    done_sentinel,
    role_chunk,
    tool_call_chunks,
)
from app.core.prompt import build_llm_messages
from app.models.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    ToolCallFunction,
    ToolCallMessage,
    Usage,
)

log = logging.getLogger(__name__)
router = APIRouter()


# ── SSE generator ─────────────────────────────────────────────────────────────

async def _sse_generator(
    req: ChatCompletionRequest,
    client: httpx.AsyncClient,
    adapter: BackendAdapter,
) -> AsyncIterator[str]:
    """
    Main streaming generator — backend-agnostic.
    Yields SSE-formatted strings consumed by EventSourceResponse.
    """
    llm_messages = build_llm_messages(req.messages, req.tools)
    token_stream = adapter.stream_tokens(client, llm_messages)

    open_token = settings.tool_call_open_token
    emitted_role = False

    async for event_type, payload in detect_tool_calls(token_stream, open_token):
        if event_type == "content":
            if not emitted_role:
                yield role_chunk()
                emitted_role = True
            yield content_chunk(payload)  # type: ignore[arg-type]

        elif event_type == "tool_call":
            parsed = parse_tool_call(payload)  # type: ignore[arg-type]
            if parsed is None:
                log.warning("Failed to parse tool call JSON: %r", payload)
                if not emitted_role:
                    yield role_chunk()
                yield content_chunk(payload)  # type: ignore[arg-type]
                yield content_stop_chunk()
                yield done_sentinel()
                return

            name, arguments = parsed
            log.info("Tool call detected: %s(%s)", name, list(arguments.keys()))

            if not emitted_role:
                yield role_chunk()

            for chunk in tool_call_chunks(name, arguments):
                yield chunk

            yield done_sentinel()
            return  # stream ends — client sends tool result as new request

        elif event_type == "done":
            if emitted_role:
                yield content_stop_chunk()
            yield done_sentinel()


# ── Route handler ─────────────────────────────────────────────────────────────

@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> JSONResponse:
    client: httpx.AsyncClient = request.app.state.http_client
    adapter: BackendAdapter = request.app.state.adapter

    # ── Streaming ─────────────────────────────────────────────────────────────
    if req.stream:
        return EventSourceResponse(
            _sse_generator(req, client, adapter),
            media_type="text/event-stream",
        )

    # ── Non-streaming ─────────────────────────────────────────────────────────
    llm_messages = build_llm_messages(req.messages, req.tools)
    result = await adapter.complete(client, llm_messages)

    usage = Usage(**result.usage) if result.usage else None

    stripped = result.text.strip()
    if stripped.startswith(settings.tool_call_open_token):
        parsed = parse_tool_call(stripped)
        if parsed:
            name, arguments = parsed
            tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
            msg = Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCallMessage(
                        id=tool_call_id,
                        type="function",
                        function=ToolCallFunction(
                            name=name,
                            arguments=json.dumps(arguments),
                        ),
                    )
                ],
            )
            resp = ChatCompletionResponse(
                choices=[Choice(message=msg, finish_reason="tool_calls")],
                usage=usage,
            )
            return JSONResponse(resp.model_dump(exclude_none=True))

    msg = Message(role="assistant", content=result.text)
    resp = ChatCompletionResponse(
        choices=[Choice(message=msg, finish_reason="stop")],
        usage=usage,
    )
    return JSONResponse(resp.model_dump(exclude_none=True))