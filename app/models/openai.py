"""
Pydantic models that mirror the OpenAI Chat Completions API spec.
We only model the fields we actually need — unknown fields are ignored.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Inbound (client → wrapper) ────────────────────────────────────────────────

class FunctionParameters(BaseModel):
    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class FunctionDef(BaseModel):
    name: str
    description: str = ""
    parameters: FunctionParameters = Field(default_factory=FunctionParameters)


class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class ToolChoice(BaseModel):
    type: Literal["function"]
    function: dict[str, str]  # {"name": "..."}


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    # tool_calls present when role == "assistant" and model called a tool
    tool_calls: list[ToolCallMessage] | None = None
    # tool_call_id + name present when role == "tool" (result message)
    tool_call_id: str | None = None
    name: str | None = None  # function name for role="tool"


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string, as per OpenAI spec


class ToolCallMessage(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


Message.model_rebuild()  # resolve forward ref


class ChatCompletionRequest(BaseModel):
    model: str = "passthrough"
    messages: list[Message]
    tools: list[ToolDef] | None = None
    tool_choice: Literal["none", "auto"] | ToolChoice = "auto"
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


# ── Outbound (wrapper → client) ───────────────────────────────────────────────

class DeltaToolCallFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(BaseModel):
    index: int = 0
    id: str | None = None
    type: Literal["function"] | None = None
    function: DeltaToolCallFunction | None = None


class Delta(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[DeltaToolCall] | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "tool-wrapper"
    choices: list[StreamChoice]


# ── Non-streaming response (for tool_choice="none" / simple pass-through) ────

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "tool-wrapper"
    choices: list[Choice]
    usage: Usage | None = None


# ── Models endpoint ──────────────────────────────────────────────────────────

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "tool-call-proxy"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]
