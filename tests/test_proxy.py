"""
tests/test_proxy.py — Tests for the catch-all reverse proxy router.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _mock_response(
    status_code: int = 200,
    content: bytes = b'{"result": "ok"}',
    headers: dict | None = None,
) -> httpx.Response:
    resp_headers = {"content-type": "application/json"}
    if headers:
        resp_headers.update(headers)
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers=resp_headers,
        request=httpx.Request("GET", "http://backend/test"),
    )


class TestProxyPassthrough:
    def test_get_forwarded(self, client):
        mock_resp = _mock_response(content=b'{"data": [1,2,3]}')
        app.state.http_client.request = AsyncMock(return_value=mock_resp)

        resp = client.get("/v1/embeddings")
        assert resp.status_code == 200
        assert resp.json() == {"data": [1, 2, 3]}

        call = app.state.http_client.request.call_args
        assert call.kwargs["method"] == "GET"
        assert "/v1/embeddings" in call.kwargs["url"]

    def test_post_forwarded_with_body(self, client):
        mock_resp = _mock_response(content=b'{"embedding": [0.1]}')
        app.state.http_client.request = AsyncMock(return_value=mock_resp)

        body = {"input": "hello", "model": "text-embedding-ada-002"}
        resp = client.post("/v1/embeddings", json=body)
        assert resp.status_code == 200

        call = app.state.http_client.request.call_args
        assert call.kwargs["method"] == "POST"
        sent_body = call.kwargs["content"]
        assert b"hello" in sent_body

    def test_query_params_forwarded(self, client):
        mock_resp = _mock_response()
        app.state.http_client.request = AsyncMock(return_value=mock_resp)

        resp = client.get("/v1/files?purpose=fine-tune")
        assert resp.status_code == 200

        call = app.state.http_client.request.call_args
        assert "purpose=fine-tune" in call.kwargs["url"]

    def test_backend_error_forwarded(self, client):
        mock_resp = _mock_response(
            status_code=404,
            content=b'{"error": {"message": "not found"}}',
        )
        app.state.http_client.request = AsyncMock(return_value=mock_resp)

        resp = client.get("/v1/nonexistent")
        assert resp.status_code == 404

    def test_auth_header_injected(self, client):
        mock_resp = _mock_response()
        app.state.http_client.request = AsyncMock(return_value=mock_resp)

        from app.config import settings
        original_key = settings.llm_api_key
        try:
            settings.llm_api_key = "sk-proxy-test"
            client.get("/v1/some-endpoint")
            call = app.state.http_client.request.call_args
            assert call.kwargs["headers"]["Authorization"] == "Bearer sk-proxy-test"
        finally:
            settings.llm_api_key = original_key

    def test_explicit_routes_not_intercepted(self, client):
        """POST /v1/chat/completions should NOT hit the proxy (it has its own router)."""
        from app.core.adapters import CompleteResult
        from unittest.mock import patch

        result = CompleteResult(text="direct")
        with patch.object(app.state.adapter, "complete", new_callable=AsyncMock, return_value=result):
            resp = client.post("/v1/chat/completions", json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "direct"

    def test_deep_path_forwarded(self, client):
        mock_resp = _mock_response(content=b'{"id": "ft-123"}')
        app.state.http_client.request = AsyncMock(return_value=mock_resp)

        resp = client.get("/v1/fine_tuning/jobs/ft-123")
        assert resp.status_code == 200

        call = app.state.http_client.request.call_args
        assert "/v1/fine_tuning/jobs/ft-123" in call.kwargs["url"]
