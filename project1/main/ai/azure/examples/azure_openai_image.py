# pip install "openai>=1.3.0" python-dotenv
#
# Azure OpenAI - Image generation
# Two supported approaches:
#   1. Responses API with the `image_generation` tool (gpt-image-1)
#   2. Classic Images API (client.images.generate)

import os
import base64
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
# Deployment name of a gpt-image-1 model in Foundry
image_deployment = os.getenv("AZURE_OPENAI_IMAGE_DEPLOYMENT")

prompt = "A modern flat illustration of a robot holding a potted plant, clean vector style, pastel colors."

# ---------------------------------------------------------------------------
# Approach 1: Responses API with the image_generation tool
# Uses the OpenAI-compatible /openai/v1/ base URL exposed by Foundry.
# ---------------------------------------------------------------------------
compat_base_url = f"{azure_openai_endpoint.rstrip('/')}/openai/v1/" if azure_openai_endpoint else None

client_compat = OpenAI(api_key=azure_openai_key, base_url=compat_base_url)

response = client_compat.responses.create(
    model=image_deployment,
    input=prompt,
    tools=[{"type": "image_generation"}],
)

image_base64 = next(
    item.result for item in response.output if item.type == "image_generation_call"
)
with open("foundry_generated.png", "wb") as f:
    f.write(base64.b64decode(image_base64))
print("Saved: foundry_generated.png")

# ---------------------------------------------------------------------------
# Approach 2: Classic Images API via the AzureOpenAI client
# ---------------------------------------------------------------------------
client_azure = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version=azure_openai_api_version,
)

img = client_azure.images.generate(
    model=image_deployment,
    prompt="A cute baby polar bear",
    n=1,
    size="1024x1024",
)

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
print("Saved: output.png")
