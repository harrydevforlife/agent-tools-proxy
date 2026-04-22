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
from app.core.adapters import BackendAdapter, SamplingParams
from app.core.buffer import detect_tool_calls, parse_tool_call
from app.core.formatter import (
    StreamContext,
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
    ToolChoice,
    ToolDef,
    Usage,
)

log = logging.getLogger(__name__)
router = APIRouter()


# ── Tool choice resolution ────────────────────────────────────────────────────

def _resolve_tools(
    req: ChatCompletionRequest,
) -> tuple[list[ToolDef] | None, str]:
    """
    Apply tool_choice logic. Returns (effective_tools, mode) where mode is
    one of "none", "auto", "required", or "forced:<name>".
    """
    if not req.tools:
        return None, "none"

    tc = req.tool_choice

    if tc == "none":
        return None, "none"

    if tc == "auto":
        return req.tools, "auto"

    if tc == "required":
        return req.tools, "required"

    if isinstance(tc, ToolChoice):
        forced_name = tc.function.get("name", "")
        matching = [t for t in req.tools if t.function.name == forced_name]
        return (matching or req.tools), f"forced:{forced_name}"

    return req.tools, "auto"


def _sampling_params(req: ChatCompletionRequest) -> SamplingParams:
    return SamplingParams(
        model=req.model if req.model != "passthrough" else None,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )


def _model_name() -> str:
    return settings.llm_model


# ── SSE generator ─────────────────────────────────────────────────────────────

async def _sse_generator(
    req: ChatCompletionRequest,
    client: httpx.AsyncClient,
    adapter: BackendAdapter,
) -> AsyncIterator[str]:
    """
    Main streaming generator — backend-agnostic.
    Yields raw JSON strings consumed by EventSourceResponse.
    """
    tools, mode = _resolve_tools(req)
    llm_messages = build_llm_messages(req.messages, tools, tool_choice_mode=mode)
    params = _sampling_params(req)
    token_stream = adapter.stream_tokens(client, llm_messages, params)

    ctx = StreamContext(model=_model_name())
    open_token = settings.tool_call_open_token
    emitted_role = False

    async for event_type, payload in detect_tool_calls(token_stream, open_token):
        if event_type == "content":
            if not emitted_role:
                yield role_chunk(ctx)
                emitted_role = True
            yield content_chunk(payload, ctx)  # type: ignore[arg-type]

        elif event_type == "tool_call":
            parsed = parse_tool_call(payload)  # type: ignore[arg-type]
            if parsed is None:
                log.warning("Failed to parse tool call JSON: %r", payload)
                if not emitted_role:
                    yield role_chunk(ctx)
                yield content_chunk(payload, ctx)  # type: ignore[arg-type]
                yield content_stop_chunk(ctx)
                yield done_sentinel()
                return

            name, arguments = parsed
            log.info("Tool call detected: %s(%s)", name, list(arguments.keys()))

            if not emitted_role:
                yield role_chunk(ctx)

            for chunk in tool_call_chunks(name, arguments, ctx):
                yield chunk

            yield done_sentinel()
            return

        elif event_type == "done":
            if emitted_role:
                yield content_stop_chunk(ctx)
            yield done_sentinel()


# ── Route handler ─────────────────────────────────────────────────────────────

@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> JSONResponse:
    client: httpx.AsyncClient = request.app.state.http_client
    adapter: BackendAdapter = request.app.state.adapter

    if req.stream:
        return EventSourceResponse(
            _sse_generator(req, client, adapter),
            media_type="text/event-stream",
        )

    # ── Non-streaming ─────────────────────────────────────────────────────────
    tools, mode = _resolve_tools(req)
    llm_messages = build_llm_messages(req.messages, tools, tool_choice_mode=mode)
    params = _sampling_params(req)
    result = await adapter.complete(client, llm_messages, params)

    model = _model_name()
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
                model=model,
                choices=[Choice(message=msg, finish_reason="tool_calls")],
                usage=usage,
            )
            return JSONResponse(resp.model_dump(exclude_none=True))

    msg = Message(role="assistant", content=result.text)
    resp = ChatCompletionResponse(
        model=model,
        choices=[Choice(message=msg, finish_reason="stop")],
        usage=usage,
    )
    return JSONResponse(resp.model_dump(exclude_none=True))
