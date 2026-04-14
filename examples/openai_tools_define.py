from openai import OpenAI

from pydantic import BaseModel

import openai
from openai import OpenAI


class GetWeather(BaseModel):
    city: str
    country: str


client = OpenAI(base_url="http://localhost:8080/v1", api_key="hehe")


with client.chat.completions.stream(
    model="gemma-3-27b",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in New York?",
        },
    ],
    tools=[
        # because we're using `.parse_stream()`, the returned tool calls
        # will be automatically deserialized into this `GetWeather` type
        openai.pydantic_function_tool(GetWeather, name="get_weather"),
    ],
    parallel_tool_calls=True,
) as stream:
    for event in stream:
        # print(event)
        # print('-'*80)
        if event.type == "tool_calls.function.arguments.delta" or event.type == "tool_calls.function.arguments.done":
            print(event, end='\n\n')

print("----\n")
print(stream.get_final_completion())
