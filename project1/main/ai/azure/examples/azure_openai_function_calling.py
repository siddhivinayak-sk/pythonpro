# pip install "openai>=1.3.0" python-dotenv
#
# Azure OpenAI - Function / Tool Calling (Responses API)
# The model decides when to call a tool, we execute it locally,
# and feed the result back so the model can produce a final answer.

import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_model_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version=azure_openai_api_version,
)

# --- Local tool implementations -----------------------------------------

def get_weather(city: str, unit: str = "celsius") -> dict:
    # In production this would call a weather API.
    fake = {"paris": 18, "seattle": 12, "hyderabad": 31}
    temp_c = fake.get(city.lower(), 22)
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32, 1)
    return {"city": city, "temperature": temp, "unit": unit, "conditions": "Partly cloudy"}

TOOL_REGISTRY = {"get_weather": get_weather}

# --- Tool schema exposed to the model -----------------------------------

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["city"],
        },
    }
]

input_messages = [
    {"role": "user", "content": "What's the weather like in Paris and Hyderabad right now?"}
]

# First call: the model may respond with tool calls
response = client.responses.create(
    model=azure_model_deployment_name,
    input=input_messages,
    tools=tools,
)

# Collect any tool calls, execute them, and feed the outputs back
tool_call_items = [item for item in response.output if item.type == "function_call"]

if tool_call_items:
    # Preserve the assistant's tool-call items in the running input
    input_messages += [item.model_dump() for item in tool_call_items]

    for call in tool_call_items:
        fn = TOOL_REGISTRY[call.name]
        args = json.loads(call.arguments or "{}")
        result = fn(**args)
        print(f"[tool] {call.name}({args}) -> {result}")

        input_messages.append({
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result),
        })

    # Second call: model produces the final natural-language answer
    response = client.responses.create(
        model=azure_model_deployment_name,
        input=input_messages,
        tools=tools,
    )

print("\nFinal answer:\n" + response.output_text)
