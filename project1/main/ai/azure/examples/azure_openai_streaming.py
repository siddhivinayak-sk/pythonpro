# pip install "openai>=1.3.0" python-dotenv
#
# Azure OpenAI - Streaming Responses
# Streams incremental tokens as they are generated. Useful for chat UIs
# and any latency-sensitive scenario.

import os
import sys
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

print("Assistant: ", end="", flush=True)

with client.responses.stream(
    model=azure_model_deployment_name,
    input=[
        {"role": "system", "content": "You are a concise technical writer."},
        {"role": "user", "content": "Explain what Azure AI Foundry is in 5 short bullets."},
    ],
) as stream:
    for event in stream:
        # Print only the text-delta events; other events include tool calls,
        # audio deltas, response.completed, etc.
        if event.type == "response.output_text.delta":
            sys.stdout.write(event.delta)
            sys.stdout.flush()

    final = stream.get_final_response()

print("\n\n---")
print(f"Model: {final.model}")
if final.usage:
    print(f"Tokens (in/out/total): "
          f"{final.usage.input_tokens}/"
          f"{final.usage.output_tokens}/"
          f"{final.usage.total_tokens}")
