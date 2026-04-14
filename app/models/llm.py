"""
Internal models for talking to the LLM backend.
Default shape matches Ollama's /api/chat endpoint.
Swap out the payload builder in chat.py to support vLLM / LiteLLM / etc.
"""
from __future__ import annotations
from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMRequest(BaseModel):
    model: str
    messages: list[LLMMessage]
    stream: bool = True
    options: dict | None = None  # temperature, num_predict, etc.


class LLMStreamChunk(BaseModel):
    """One JSON line from Ollama's streaming response."""
    model: str = ""
    message: LLMMessage | None = None
    done: bool = False
    done_reason: str | None = None
