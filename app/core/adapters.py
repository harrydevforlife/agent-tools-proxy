"""
adapters.py — Pluggable backend adapters.

Each adapter knows how to:
  1. Build the HTTP request payload for its backend
  2. Parse the streaming response line-by-line into plain text tokens
  3. Parse a non-streaming response into a plain text string

Protocol
--------
BackendAdapter defines the interface. Two concrete implementations:

  OllamaAdapter   — Ollama /api/chat  (NDJSON lines)
  OpenAIAdapter   — OpenAI-compatible /v1/chat/completions  (SSE data: lines)
                    Works with: vLLM, LiteLLM, Together AI, Groq, Anyscale,
                                OpenRouter, and real OpenAI.

Factory
-------
  get_adapter(settings) → BackendAdapter

Usage in chat.py
----------------
  adapter = get_adapter(settings)
  async for token in adapter.stream_tokens(client, messages):
      ...
  text = await adapter.complete(client, messages)
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
from fastapi import HTTPException

from app.config import Settings


@dataclass
class SamplingParams:
    """Optional sampling parameters forwarded from the client request."""
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass
class CompleteResult:
    """Return value from BackendAdapter.complete()."""
    text: str
    usage: dict[str, int] = field(default_factory=dict)

log = logging.getLogger(__name__)


# ── Protocol / base class ─────────────────────────────────────────────────────

class BackendAdapter(ABC):
    """Abstract backend adapter. One instance per app lifetime."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            h["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        return h

    def _base_url(self) -> str:
        return str(self.settings.llm_base_url).rstrip("/")

    @abstractmethod
    def _stream_url(self) -> str: ...

    @abstractmethod
    def _complete_url(self) -> str: ...

    @abstractmethod
    def _stream_payload(self, messages: list[dict], params: SamplingParams) -> dict: ...

    @abstractmethod
    def _complete_payload(self, messages: list[dict], params: SamplingParams) -> dict: ...

    @abstractmethod
    def _parse_stream_line(self, line: str) -> tuple[str, bool]:
        """
        Parse one line from the streaming response.

        Returns:
            (token, done) — token is the text fragment (may be ""),
                            done=True signals the stream is finished.
        """
        ...

    @abstractmethod
    def _parse_complete_response(self, data: dict) -> CompleteResult:
        """Extract the assistant's text and usage from a non-streaming response."""
        ...

    # ── Public API ────────────────────────────────────────────────────────────

    async def stream_tokens(
        self,
        client: httpx.AsyncClient,
        messages: list[dict],
        params: SamplingParams | None = None,
    ) -> AsyncIterator[str]:
        """Yield raw text tokens from the backend's streaming response."""
        payload = self._stream_payload(messages, params or SamplingParams())

        async with client.stream(
            "POST",
            self._stream_url(),
            json=payload,
            headers=self._headers,
            timeout=120.0,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"LLM backend error {response.status_code}: "
                        f"{body.decode()[:300]}"
                    ),
                )

            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    token, done = self._parse_stream_line(line)
                except Exception as exc:
                    log.warning("Failed to parse stream line %r: %s", line, exc)
                    continue

                if token:
                    yield token
                if done:
                    break

    async def complete(
        self,
        client: httpx.AsyncClient,
        messages: list[dict],
        params: SamplingParams | None = None,
    ) -> CompleteResult:
        """Non-streaming: return the assistant text and usage stats."""
        payload = self._complete_payload(messages, params or SamplingParams())

        resp = await client.post(
            self._complete_url(),
            json=payload,
            headers=self._headers,
            timeout=120.0,
        )
        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"LLM backend error {resp.status_code}: {resp.text[:300]}",
            )

        return self._parse_complete_response(resp.json())


# ── Ollama adapter ────────────────────────────────────────────────────────────

class OllamaAdapter(BackendAdapter):
    """
    Ollama /api/chat

    Streaming format — one JSON object per line:
      {"model":"…","message":{"role":"assistant","content":"Hello"},"done":false}
      {"model":"…","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}

    Non-streaming format:
      {"model":"…","message":{"role":"assistant","content":"Hello world"},"done":true}
    """

    def _stream_url(self) -> str:
        return self._base_url() + "/api/chat"

    def _complete_url(self) -> str:
        return self._base_url() + "/api/chat"

    def _build_payload(self, messages: list[dict], params: SamplingParams, stream: bool) -> dict:
        payload: dict = {
            "model": params.model or self.settings.llm_model,
            "messages": messages,
            "stream": stream,
        }
        if params.temperature is not None:
            payload["temperature"] = params.temperature
        if params.max_tokens is not None:
            payload["num_predict"] = params.max_tokens
        return payload

    def _stream_payload(self, messages: list[dict], params: SamplingParams) -> dict:
        return self._build_payload(messages, params, stream=True)

    def _complete_payload(self, messages: list[dict], params: SamplingParams) -> dict:
        return self._build_payload(messages, params, stream=False)

    def _parse_stream_line(self, line: str) -> tuple[str, bool]:
        data = json.loads(line)
        token = data.get("message", {}).get("content", "") or ""
        done = bool(data.get("done", False))
        return token, done

    def _parse_complete_response(self, data: dict) -> CompleteResult:
        text = data.get("message", {}).get("content", "") or ""
        usage = {}
        if "eval_count" in data:
            usage["completion_tokens"] = data["eval_count"]
        if "prompt_eval_count" in data:
            usage["prompt_tokens"] = data["prompt_eval_count"]
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        return CompleteResult(text=text, usage=usage)


# ── OpenAI-compatible adapter (vLLM, LiteLLM, OpenAI, Groq, …) ───────────────

class OpenAIAdapter(BackendAdapter):
    """
    OpenAI-compatible /v1/chat/completions

    Streaming format — SSE, each data line is JSON:
      data: {"id":"…","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
      data: {"id":"…","choices":[{"delta":{},"finish_reason":"stop"}]}
      data: [DONE]

    Non-streaming format:
      {"id":"…","choices":[{"message":{"role":"assistant","content":"Hello world"}}]}

    Compatible with: OpenAI API, vLLM (--served-model-name),
                     LiteLLM proxy, Together AI, Groq, Anyscale, OpenRouter.
    """

    def _stream_url(self) -> str:
        return self._base_url() + "/v1/chat/completions"

    def _complete_url(self) -> str:
        return self._base_url() + "/v1/chat/completions"

    def _build_payload(self, messages: list[dict], params: SamplingParams, stream: bool) -> dict:
        payload: dict = {
            "model": params.model or self.settings.llm_model,
            "messages": messages,
            "stream": stream,
        }
        if params.temperature is not None:
            payload["temperature"] = params.temperature
        if params.max_tokens is not None:
            payload["max_tokens"] = params.max_tokens
        return payload

    def _stream_payload(self, messages: list[dict], params: SamplingParams) -> dict:
        return self._build_payload(messages, params, stream=True)

    def _complete_payload(self, messages: list[dict], params: SamplingParams) -> dict:
        return self._build_payload(messages, params, stream=False)

    def _parse_stream_line(self, line: str) -> tuple[str, bool]:
        # SSE lines start with "data: "
        if line.startswith("data:"):
            payload = line[5:].strip()
        else:
            # Some proxies emit raw JSON without the "data: " prefix
            payload = line

        if payload == "[DONE]":
            return "", True

        data = json.loads(payload)
        choices = data.get("choices", [])
        if not choices:
            return "", False

        choice = choices[0]
        delta = choice.get("delta", {})
        token = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")
        done = finish_reason is not None

        return token, done

    def _parse_complete_response(self, data: dict) -> CompleteResult:
        choices = data.get("choices", [])
        text = ""
        if choices:
            text = choices[0].get("message", {}).get("content", "") or ""
        usage = data.get("usage", {}) or {}
        return CompleteResult(text=text, usage=usage)


# ── Factory ───────────────────────────────────────────────────────────────────

_ADAPTERS: dict[str, type[BackendAdapter]] = {
    "ollama": OllamaAdapter,
    "openai": OpenAIAdapter,
}


def get_adapter(settings: Settings) -> BackendAdapter:
    """
    Return the configured backend adapter.
    Raises ValueError for unknown backend names.
    """
    cls = _ADAPTERS.get(settings.llm_backend)
    if cls is None:
        raise ValueError(
            f"Unknown backend {settings.llm_backend!r}. "
            f"Valid options: {list(_ADAPTERS)}"
        )
    log.info("Using backend adapter: %s → %s with base url %s", settings.llm_backend, cls.__name__, settings.llm_base_url)
    return cls(settings)