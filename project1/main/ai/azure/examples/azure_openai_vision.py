# pip install "openai>=1.3.0" python-dotenv

import os
from openai import AzureOpenAI

from dotenv import load_dotenv

import base64

load_dotenv()

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_model_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT") # the Azure deployment name, e.g., "gpt-4o-mini"
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

print(f"Endpoint: {azure_openai_endpoint}")
print(f"Deployment: {azure_model_deployment_name}")

client = AzureOpenAI(
    api_key=azure_openai_key,
    azure_endpoint=azure_openai_endpoint,
    api_version=azure_openai_api_version,
)

image_url = f"./static/image/zx-spectrum.jpg"
with open(image_url, "rb") as image_file:
    image = base64.b64encode(image_file.read()).decode("utf-8")

response = client.responses.create(
    model=azure_model_deployment_name,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this image? Provide 3 bullet points."},
                {"type": "input_image", "image_url": f"data:image/jpg;base64,{image}"}
            ],
        }
    ],
)

print(response.output_text)