from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="allow",
    )

    # ── LLM backend ──────────────────────────────────────────────────────────
    llm_backend: Literal["ollama", "openai"] = Field(
        default="ollama",
        description=(
            "Backend adapter to use. "
            "'ollama' for Ollama's /api/chat format. "
            "'openai' for OpenAI-compatible /v1/chat/completions format "
            "(vLLM, LiteLLM, OpenAI, Together, Groq, etc.)"
        ),
    )
    llm_base_url: AnyHttpUrl = Field(
        default="http://localhost:11434",
        description="Base URL of your LLM backend (Ollama, vLLM, LiteLLM, etc.)",
    )
    llm_model: str = Field(default="llama3.1", description="Model name passed to the backend")
    llm_api_key: str = Field(default="", description="API key if required by backend")

    # ── Prompt injection ──────────────────────────────────────────────────────
    tool_call_open_token: str = Field(
        default='{"tool_call"',
        description="Token prefix that signals the model is emitting a tool call",
    )

    # ── Streaming ─────────────────────────────────────────────────────────────
    stream_chunk_size: int = Field(default=32, description="Bytes per read from backend stream")

    # ── Server ────────────────────────────────────────────────────────────────
    log_level: str = Field(default="info")


settings = Settings()