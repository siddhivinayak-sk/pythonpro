# pip install "openai>=1.3.0" python-dotenv

import os
from openai import AzureOpenAI

from dotenv import load_dotenv

load_dotenv()

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_model_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT") # the Azure deployment name, e.g., "gpt-4o-mini"
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

print(f"Endpoint: {azure_openai_endpoint}")

client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version=azure_openai_api_version,
)

response = client.responses.create(
    model=azure_model_deployment_name,
    input=[{"role": "system", "content": "You're a helpful assistant."},
           {"role": "user", "content": "Summarize the key points from our release notes in 3 bullets."}],
    max_output_tokens=300,
    temperature=0.7
)

print(response.output_text)
