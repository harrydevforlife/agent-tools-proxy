"""Tests for app/core/prompt.py"""
import pytest
from app.core.prompt import build_tools_block, build_llm_messages, serialize_history
from app.models.openai import Message, ToolDef, FunctionDef, FunctionParameters


def _make_tool(name: str, desc: str, props: dict, required: list[str]) -> ToolDef:
    return ToolDef(
        function=FunctionDef(
            name=name,
            description=desc,
            parameters=FunctionParameters(
                type="object",
                properties=props,
                required=required,
            ),
        )
    )


SEARCH_TOOL = _make_tool(
    "search_products",
    "Search for products using keywords",
    {"keywords": {"type": "array", "items": {"type": "string"}, "description": "Search keywords"}},
    ["keywords"],
)

GET_TOOL = _make_tool(
    "get_product_details",
    "Get detailed info about a product",
    {"product_id": {"type": "string", "description": "Product ID"}},
    ["product_id"],
)


class TestBuildToolsBlock:
    def test_contains_function_names(self):
        block = build_tools_block([SEARCH_TOOL, GET_TOOL])
        assert "search_products" in block
        assert "get_product_details" in block

    def test_contains_descriptions(self):
        block = build_tools_block([SEARCH_TOOL])
        assert "Search for products using keywords" in block

    def test_contains_tool_call_format(self):
        block = build_tools_block([SEARCH_TOOL])
        assert '{"tool_call"' in block

    def test_array_type_rendered(self):
        block = build_tools_block([SEARCH_TOOL])
        assert "list[string]" in block

    def test_required_tag(self):
        block = build_tools_block([SEARCH_TOOL])
        assert "[required]" in block

    def test_optional_tag(self):
        tool = _make_tool(
            "foo", "bar",
            {"x": {"type": "string"}, "y": {"type": "integer"}},
            ["x"],  # y is optional
        )
        block = build_tools_block([tool])
        assert "[optional]" in block

    def test_few_shot_example_present(self):
        block = build_tools_block([SEARCH_TOOL])
        assert "[TOOL RESULT:" in block


class TestSerializeHistory:
    def test_user_message(self):
        msgs = [Message(role="user", content="hello")]
        result = serialize_history(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_tool_result_becomes_user_turn(self):
        msgs = [Message(role="tool", content='{"temp": 34}', name="get_weather", tool_call_id="c1")]
        result = serialize_history(msgs)
        assert result[0]["role"] == "user"
        assert "[TOOL RESULT: get_weather]" in result[0]["content"]
        assert '{"temp": 34}' in result[0]["content"]

    def test_assistant_tool_call_serialized_as_json(self):
        from app.models.openai import ToolCallMessage, ToolCallFunction
        import json
        tc = ToolCallMessage(
            id="call_abc",
            function=ToolCallFunction(name="search_products", arguments='{"keywords": ["shirt"]}'),
        )
        msg = Message(role="assistant", tool_calls=[tc])
        result = serialize_history([msg])
        assert result[0]["role"] == "assistant"
        data = json.loads(result[0]["content"])
        assert data["tool_call"]["name"] == "search_products"


class TestBuildLLMMessages:
    def test_no_tools_passthrough(self):
        msgs = [Message(role="user", content="hi")]
        result = build_llm_messages(msgs, tools=None)
        assert result == [{"role": "user", "content": "hi"}]

    def test_tools_injected_into_system(self):
        msgs = [Message(role="user", content="find a shirt")]
        result = build_llm_messages(msgs, tools=[SEARCH_TOOL])
        system = result[0]
        assert system["role"] == "system"
        assert "search_products" in system["content"]
        assert "<tools>" in system["content"]

    def test_existing_system_message_merged(self):
        msgs = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="hi"),
        ]
        result = build_llm_messages(msgs, tools=[SEARCH_TOOL])
        system = result[0]
        assert "You are a helpful assistant." in system["content"]
        assert "search_products" in system["content"]

    def test_message_order_preserved(self):
        msgs = [
            Message(role="user", content="first"),
            Message(role="assistant", content="second"),
            Message(role="user", content="third"),
        ]
        result = build_llm_messages(msgs, tools=None)
        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "user"]
