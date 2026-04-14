# Prompt engineering spec — tool-call-wrapper

This document captures the exact prompt strategy used to teach non-function-calling
LLMs to emit tool calls reliably. It is the contract between the wrapper and the model.

---

## Why prompting works

Function calling in native models is just constrained decoding on top of a
fine-tuned format. For a base instruction-following model, we replicate that
constraint through the system prompt:

1. Describe the available functions clearly
2. Show the exact output format with a concrete example
3. Give explicit rules for when to call vs. when to respond in prose

A well-prompted 7B model hits ~85–90% format compliance on unambiguous intents.
Ambiguous intents (model unsure whether to call or not) are where compliance
drops — mitigated by a `clarify_request` tool that gives the model an escape hatch.

---

## System prompt structure

Every request to the LLM backend receives a system prompt with these sections
in this order:

```
<tools>
  [format instruction + JSON example line]
  [function list]
</tools>

<tool_rules>
  [numbered behavioral rules]
</tool_rules>

[few-shot example — complete multi-turn]

[original system message from the caller, if present]
```

Ordering rationale:
- `<tools>` first — the model needs to know what it *can* do before it reads rules
- `<tool_rules>` second — behavioral constraints after capability declaration
- few-shot last before caller content — recency bias means it's freshest in
  attention when the model generates

---

## Tool schema serialization

OpenAI sends tool schemas as JSON Schema. We render them as Python-style
function signatures for two reasons:

1. Most instruction-tuned models were trained on more Python than JSON Schema
2. Python signatures are more compact — fewer tokens, less context consumed

### Input (OpenAI JSON Schema)

```json
{
  "name": "search_products",
  "description": "Search for products using keywords",
  "parameters": {
    "type": "object",
    "properties": {
      "keywords": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Keywords to search for"
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum results to return"
      }
    },
    "required": ["keywords"]
  }
}
```

### Output (injected into system prompt)

```
search_products(keywords: list[string], max_results: integer = None)
  """Search for products using keywords."""
  Parameters:
  - keywords (list[string]) [required] — Keywords to search for
  - max_results (integer) [optional] — Maximum results to return
```

Type mapping:

| JSON Schema type | Rendered as |
|---|---|
| `string` | `string` |
| `integer` | `integer` |
| `number` | `number` |
| `boolean` | `boolean` |
| `array` of `X` | `list[X]` |
| `object` | `object` |
| unknown | `any` |

---

## Output format contract

The model must emit exactly this when calling a function — nothing else on that line:

```json
{"tool_call": {"name": "<function_name>", "arguments": {<key>: <value>, ...}}}
```

Rules encoded in the prompt:
- The JSON must be the model's **entire response** for that turn
- No preamble ("Sure! I'll search for...") — just the JSON
- No commentary after the JSON
- Stop generating after the closing `}`
- One function call per response — no arrays of calls

The `{"tool_call": ...}` envelope is intentional:
- The prefix `{"tool_call"` is a distinctive sentinel unlikely to appear in prose
- Makes brace-depth detection unambiguous
- Separates our protocol from any JSON the model might produce in normal text

---

## Few-shot example

A single concrete example is embedded in every system prompt. It shows one
complete tool call round-trip so the model has seen the format — not just read
a description of it.

```
Example of calling a function:
[USER]
What's the weather in Hanoi?
[ASSISTANT]
{"tool_call": {"name": "get_weather", "arguments": {"city": "Hanoi", "units": "celsius"}}}
[TOOL RESULT: get_weather]
{"temp": 34, "condition": "humid"}
[ASSISTANT]
It's currently 34 °C and humid in Hanoi.
```

Design choices in this example:
- Generic function (`get_weather`) — not one of the registered tools, so the
  model doesn't confuse example calls with real available tools
- Short, clear arguments
- Tool result is realistic JSON
- The final assistant turn is plain text — teaches the model to switch back to
  prose after receiving a result

---

## History serialization — multi-turn tool calls

When the client sends back a tool result and prior tool call history, we
re-serialize the conversation into a format the model can follow.

### Role mapping

| OpenAI role | Serialized as | Content |
|---|---|---|
| `system` | `system` | tools block + original system message |
| `user` | `user` | original content |
| `assistant` (text) | `assistant` | original content |
| `assistant` (tool_call) | `assistant` | raw JSON: `{"tool_call": {"name": …, "arguments": …}}` |
| `tool` (result) | `user` | `[TOOL RESULT: {name}]\n{content}` |

The `[TOOL RESULT: name]` header is critical. Without the function name, the
model cannot tell which tool result belongs to which call in multi-tool sessions.

### Example serialized history (round 2 context)

```
[SYSTEM]
<tools>…</tools>
<tool_rules>…</tool_rules>
…

[USER]
Find me a blue running shirt.

[ASSISTANT]
{"tool_call": {"name": "search_products", "arguments": {"keywords": ["blue", "running", "shirt"]}}}

[USER]
[TOOL RESULT: search_products]
{"results": [{"id": "p1", "name": "Nike Dri-FIT", "price": 39.99}]}

[ASSISTANT]   ← model generates here
```

---

## Behavioral rules (verbatim in prompt)

```
<tool_rules>
1. Emit the JSON ONLY when calling a function. It must be your entire response.
2. Never fabricate tool results — wait for the [TOOL RESULT] turn.
3. After receiving a [TOOL RESULT], continue in plain text.
4. If the request is ambiguous and no tool fits, ask for clarification in plain text.
</tool_rules>
```

Rule 2 is the most important safety rule. Without it, some models will hallucinate
a tool result and continue as if the call was made. This breaks the entire loop.

---

## Tuning for specific models

### Models that ignore format instructions

Increase few-shot examples from 1 to 2–3. Use examples that cover:
- A call with a list argument
- A call with a nested object argument  
- A plain-text response (no tool call needed)

### Models that add preamble before the JSON

Add to `<tool_rules>`:
> Do not say anything before the JSON. Do not say "Sure" or "I'll call...". Output the JSON immediately.

### Models that continue generating after the closing `}`

The buffer stops reading after depth=0, so extra tokens are discarded. However
they waste context on the next turn. Add to rules:
> After the closing `}`, stop immediately. Do not add any text.

### Models with very short context windows

Consider reducing the few-shot example or shortening parameter descriptions.
The tools block adds roughly 200–600 tokens depending on tool count and
parameter verbosity. Monitor context consumption with `LOG_LEVEL=debug`.

---

## Reliability characteristics

| Scenario | Expected compliance | Notes |
|---|---|---|
| Clear intent, matching tool | ~90% | Most common case |
| Ambiguous intent | ~70% | Model may call wrong tool or go prose |
| Intent clearly requires no tool | ~95% | Model stays in prose |
| Nested object arguments | ~80% | Drops with weaker models |
| After receiving tool result | ~85% | Occasional re-call without reason |

These estimates are for a 7B instruction-tuned model (llama3.1, mistral-7b).
Larger models (13B+) typically add 5–10% across all scenarios.
