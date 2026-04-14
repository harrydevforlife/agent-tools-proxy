from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="hehe")

# 1. Define the tool
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name"}
            },
            "required": ["location"]
        }
    }
}]

# 2. Initial request
messages = [{"role": "user", "content": "What's the weather in London?"}]
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)

# 3. Check for tool calls
tool_calls = response.choices[0].message.tool_calls
if tool_calls:
    # (Execute your function here and append result to messages)
    print(tool_calls)
