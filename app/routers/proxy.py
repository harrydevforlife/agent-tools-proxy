"""
routers/proxy.py — Catch-all reverse proxy.

Any request that doesn't match an explicitly defined route (chat completions,
models, health) is forwarded as-is to the LLM backend. This lets clients use
endpoints like /v1/embeddings, /v1/completions, /v1/audio, etc. without the
wrapper needing explicit support for each one.

Streaming responses from the backend are streamed through to the client.
"""
from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter()

_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
})


def _backend_url(path: str) -> str:
    return str(settings.llm_base_url).rstrip("/") + path


def _forward_headers(request: Request) -> dict[str, str]:
    """Build headers to send to the backend, adding auth if configured."""
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() not in _HOP_BY_HOP | {"host"}:
            headers[key] = value
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    return headers


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_passthrough(request: Request, path: str) -> Response:
    """Forward unhandled requests to the LLM backend."""
    client: httpx.AsyncClient = request.app.state.http_client
    url = _backend_url(f"/{path}")
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = _forward_headers(request)
    body = await request.body()

    log.debug("Proxying %s %s → %s", request.method, request.url.path, url)

    backend_resp = await client.request(
        method=request.method,
        url=url,
        headers=headers,
        content=body if body else None,
        timeout=120.0,
    )

    resp_headers = {
        k: v for k, v in backend_resp.headers.items()
        if k.lower() not in _HOP_BY_HOP | {"content-encoding", "content-length"}
    }

    content_type = backend_resp.headers.get("content-type", "")
    is_stream = "text/event-stream" in content_type or "chunked" in backend_resp.headers.get("transfer-encoding", "")

    if is_stream:
        return StreamingResponse(
            content=backend_resp.aiter_bytes(),
            status_code=backend_resp.status_code,
            headers=resp_headers,
            media_type=content_type,
        )

    return Response(
        content=backend_resp.content,
        status_code=backend_resp.status_code,
        headers=resp_headers,
        media_type=content_type or None,
    )
