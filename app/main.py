"""
main.py — FastAPI application factory.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.adapters import get_adapter
from app.routers.chat import router as chat_router
from app.routers.models import router as models_router
from app.routers.proxy import router as proxy_router

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create shared HTTP client and backend adapter for the app lifetime."""
    log.info(
        "Starting agent-tools-proxy  backend=%s  url=%s  model=%s",
        settings.llm_backend,
        settings.llm_base_url,
        settings.llm_model,
    )
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )
    app.state.adapter = get_adapter(settings)
    yield
    log.info("Shutting down — closing HTTP client")
    await app.state.http_client.aclose()


app = FastAPI(
    title="Tool Call Wrapper",
    description="OpenAI-compatible tool calling for non-function-calling LLMs",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(proxy_router)  # catch-all — must be last


# ── OpenAI-shaped error responses ─────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def openai_http_exception_handler(_request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
                "code": exc.status_code,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def openai_validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": "invalid_request",
            }
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backend": settings.llm_backend,
        "url": str(settings.llm_base_url),
        "model": settings.llm_model,
    }