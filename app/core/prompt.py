"""
prompt.py — Convert OpenAI tools[] + messages[] into a plain-text prompt
that a non-function-calling LLM can follow.

Strategy:
  1. Build a <tools> block injected into / prepended to the system prompt.
  2. Serialize conversation history (including past tool calls/results)
     as labelled turns so the model understands the full context.
  3. Include a concrete few-shot example so the model has seen the format.
"""
from __future__ import annotations

import json
from typing import Sequence

from app.models.openai import Message, ToolDef


# ── Few-shot example embedded in every system prompt ─────────────────────────

_FEW_SHOT = """\
Example of calling a function:
[USER]
What's the weather in Hanoi?
[ASSISTANT]
{"tool_call": {"name": "get_weather", "arguments": {"city": "Hanoi", "units": "celsius"}}}
[TOOL RESULT: get_weather]
{"temp": 34, "condition": "humid"}
[ASSISTANT]
It's currently 34 °C and humid in Hanoi."""


# ── Tool block builder ────────────────────────────────────────────────────────

def _signature(tool: ToolDef) -> str:
    """Build a Python-style signature string from the tool's parameters."""
    props = tool.function.parameters.properties
    required = set(tool.function.parameters.required)
    parts = []
    for name, schema in props.items():
        typ = schema.get("type", "any")
        # Handle array types: array of string → list[string]
        if typ == "array":
            items_type = schema.get("items", {}).get("type", "any")
            typ = f"list[{items_type}]"
        suffix = "" if name in required else " = None"
        parts.append(f"{name}: {typ}{suffix}")
    return ", ".join(parts)


def _param_lines(tool: ToolDef) -> str:
    """Bullet-point parameter descriptions."""
    props = tool.function.parameters.properties
    required = set(tool.function.parameters.required)
    lines = []
    for name, schema in props.items():
        typ = schema.get("type", "any")
        if typ == "array":
            items_type = schema.get("items", {}).get("type", "any")
            typ = f"list[{items_type}]"
        desc = schema.get("description", "")
        req_tag = "[required]" if name in required else "[optional]"
        desc_part = f" — {desc}" if desc else ""
        lines.append(f"  - {name} ({typ}) {req_tag}{desc_part}")
    return "\n".join(lines) if lines else "  (no parameters)"


def build_tools_block(
    tools: list[ToolDef],
    *,
    tool_choice_mode: str = "auto",
) -> str:
    """
    Render the <tools> system prompt block.
    This is injected at the top of the system prompt on every request.
    """
    tool_entries = []
    for tool in tools:
        fn = tool.function
        sig = _signature(tool)
        params = _param_lines(tool)
        entry = (
            f"{fn.name}({sig})\n"
            f'  """{fn.description}"""\n'
            f"  Parameters:\n{params}"
        )
        tool_entries.append(entry)

    tools_text = "\n\n".join(tool_entries)

    choice_rule = (
        "If you do not need to call a function, respond in plain text as normal.\n"
        "One function call per response — no parallel calls."
    )
    if tool_choice_mode == "required":
        choice_rule = (
            "You MUST call one of the available functions. "
            "Do not respond in plain text — always emit a tool call.\n"
            "One function call per response — no parallel calls."
        )
    elif tool_choice_mode.startswith("forced:"):
        forced_name = tool_choice_mode.split(":", 1)[1]
        choice_rule = (
            f"You MUST call the function '{forced_name}'. "
            "Do not respond in plain text — always emit a tool call.\n"
            "One function call per response — no parallel calls."
        )

    return f"""\
<tools>
You have access to the following functions. When you need to call one,
respond with ONLY a JSON object on a single line — no preamble, no commentary:

{{"tool_call": {{"name": "<function_name>", "arguments": {{...}}}}}}

After emitting the JSON, stop immediately. Do not add any text after it.
{choice_rule}

Available functions:

{tools_text}
</tools>

<tool_rules>
1. Emit the JSON ONLY when calling a function. It must be your entire response.
2. Never fabricate tool results — wait for the [TOOL RESULT] turn.
3. After receiving a [TOOL RESULT], you may call another function if needed (if the request is not clear), or respond in plain text.
4. If the request is ambiguous and no tool fits, ask for clarification in plain text.
</tool_rules>

{_FEW_SHOT}"""


# ── History serializer ────────────────────────────────────────────────────────

def serialize_history(messages: Sequence[Message]) -> list[dict[str, str]]:
    """
    Convert OpenAI messages[] (including tool call/result turns) into
    a flat list of {role, content} dicts suitable for the LLM backend.

    Tool call assistant turns  → role=assistant, content=JSON string
    Tool result turns          → role=user, content=[TOOL RESULT: name]\n{content}
    """
    out: list[dict[str, str]] = []

    for msg in messages:
        if msg.role == "system":
            # System messages are handled separately (tools block prepended)
            out.append({"role": "system", "content": msg.content or ""})

        elif msg.role == "user":
            out.append({"role": "user", "content": msg.content or ""})

        elif msg.role == "assistant":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = tc.function.arguments
                    payload = {
                        "tool_call": {
                            "name": tc.function.name,
                            "arguments": args,
                        }
                    }
                    out.append({"role": "assistant", "content": json.dumps(payload)})
            else:
                out.append({"role": "assistant", "content": msg.content or ""})

        elif msg.role == "tool":
            # Inject as a user turn with a clear header so the model
            # understands this is a tool result, not human input.
            fn_name = msg.name or "unknown"
            content = f"[TOOL RESULT: {fn_name}]\n{msg.content or ''}"
            out.append({"role": "user", "content": content})

    return out


def build_llm_messages(
    request_messages: Sequence[Message],
    tools: list[ToolDef] | None,
    *,
    tool_choice_mode: str = "auto",
) -> list[dict[str, str]]:
    """
    Full pipeline: inject tools block into system prompt, then serialize history.
    Returns a list of {role, content} ready for the LLM backend.

    tool_choice_mode controls prompt behaviour:
      "none"         — skip tool injection entirely
      "auto"         — inject tools, model decides
      "required"     — inject tools + "You MUST call a tool" rule
      "forced:<name>"— inject tools + "You MUST call <name>"
    """
    messages = list(request_messages)

    if not tools or tool_choice_mode == "none":
        return serialize_history(messages)

    tools_block = build_tools_block(tools, tool_choice_mode=tool_choice_mode)

    system_idx = next((i for i, m in enumerate(messages) if m.role == "system"), None)

    if system_idx is not None:
        original = messages[system_idx].content or ""
        print(f"original: {original}")
        merged_content = f"{tools_block}\n\n{original}".strip()
        # merged_content = f"{original}\n\n{tools_block}".strip()
        messages[system_idx] = Message(role="system", content=merged_content)
    else:
        messages.insert(0, Message(role="system", content=tools_block))

    return serialize_history(messages)
