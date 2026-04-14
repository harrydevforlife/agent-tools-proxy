import pathlib
import sys

# Make `app` importable from the project root without pip install
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.openai import FunctionDef, FunctionParameters, ToolDef

# ── App / HTTP client ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_client():
    """
    FastAPI TestClient for integration-style route tests.
    Shares a single app instance across the session.
    """
    with TestClient(app) as client:
        yield client


# ── Reusable tool definitions ─────────────────────────────────────────────────

def _make_tool(name: str, description: str, properties: dict, required: list[str]) -> ToolDef:
    return ToolDef(
        function=FunctionDef(
            name=name,
            description=description,
            parameters=FunctionParameters(
                type="object",
                properties=properties,
                required=required,
            ),
        )
    )


@pytest.fixture()
def search_tool() -> ToolDef:
    return _make_tool(
        name="search_products",
        description="Search for products using keywords",
        properties={
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Search keywords",
            }
        },
        required=["keywords"],
    )


@pytest.fixture()
def get_details_tool() -> ToolDef:
    return _make_tool(
        name="get_product_details",
        description="Get detailed information about a specific product",
        properties={
            "product_id": {
                "type": "string",
                "description": "The product ID to look up",
            }
        },
        required=["product_id"],
    )


@pytest.fixture()
def clarify_tool() -> ToolDef:
    return _make_tool(
        name="clarify_request",
        description="Ask the user to clarify their request",
        properties={
            "question": {
                "type": "string",
                "description": "The clarifying question to ask",
            }
        },
        required=["question"],
    )


@pytest.fixture()
def all_tools(search_tool, get_details_tool, clarify_tool) -> list[ToolDef]:
    return [search_tool, get_details_tool, clarify_tool]


# ── Mock LLM backend responses ────────────────────────────────────────────────

@pytest.fixture()
def mock_llm_plain_text():
    """
    Patch _stream_tokens to yield a plain-text response with no tool call.
    Usage:
        async for token in mock_llm_plain_text("Hello there"):
            ...
    """
    async def _gen(text: str):
        async def stream(*args, **kwargs):
            for char in text:
                yield char
        return stream

    return _gen


@pytest.fixture()
def tool_call_json() -> str:
    """A well-formed tool call JSON string the LLM might emit."""
    import json
    return json.dumps({
        "tool_call": {
            "name": "search_products",
            "arguments": {"keywords": ["blue", "running", "shirt"]},
        }
    })


@pytest.fixture()
def nested_tool_call_json() -> str:
    """Tool call with nested arguments — exercises brace-depth counting."""
    import json
    return json.dumps({
        "tool_call": {
            "name": "search_products",
            "arguments": {
                "keywords": ["shirt"],
                "filters": {"color": "blue", "size": "M"},
                "page": 1,
            },
        }
    })