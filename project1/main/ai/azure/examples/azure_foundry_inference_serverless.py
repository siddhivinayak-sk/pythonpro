# pip install azure-ai-inference python-dotenv
#
# Azure AI Foundry - Model Catalog / Serverless Inference
# Calls a non-OpenAI model (Llama, Mistral, Phi, DeepSeek, Cohere, ...)
# deployed as a Serverless API endpoint in Foundry.
#
# The SDK is `azure.ai.inference`, and the API is OpenAI-compatible but
# unified across all catalog models -- so you can swap models by changing
# the endpoint / deployment name without rewriting call code.

import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from dotenv import load_dotenv

load_dotenv()

# Foundry gives you a per-deployment endpoint like:
#   https://<deployment>.<region>.models.ai.azure.com
# or a project-level endpoint. Both work with this SDK.
endpoint = os.getenv("AZURE_FOUNDRY_INFERENCE_ENDPOINT")
key = os.getenv("AZURE_FOUNDRY_INFERENCE_KEY")

# --- 1. Chat completion -------------------------------------------------

chat = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

response = chat.complete(
    messages=[
        SystemMessage(content="You are a concise assistant."),
        UserMessage(content="Name three open-source LLMs available in Azure AI Foundry."),
    ],
    max_tokens=300,
    temperature=0.5,
    # `model` is optional for single-model serverless endpoints;
    # required when the endpoint hosts multiple models.
    model=os.getenv("AZURE_FOUNDRY_INFERENCE_MODEL"),
)
print("Non-streaming:\n" + response.choices[0].message.content)
print(f"Tokens: prompt={response.usage.prompt_tokens} "
      f"completion={response.usage.completion_tokens}")

# --- 2. Streaming -------------------------------------------------------

print("\nStreaming:")
stream = chat.complete(
    stream=True,
    messages=[UserMessage(content="Give me a 4-line haiku about Kubernetes.")],
    model=os.getenv("AZURE_FOUNDRY_INFERENCE_MODEL"),
)
for update in stream:
    if update.choices and update.choices[0].delta.content:
        print(update.choices[0].delta.content, end="", flush=True)
print()

# --- 3. Embeddings (if the deployment is an embeddings model) -----------

embed_endpoint = os.getenv("AZURE_FOUNDRY_EMBEDDINGS_ENDPOINT")
embed_key = os.getenv("AZURE_FOUNDRY_EMBEDDINGS_KEY")
if embed_endpoint and embed_key:
    emb_client = EmbeddingsClient(endpoint=embed_endpoint, credential=AzureKeyCredential(embed_key))
    emb = emb_client.embed(input=["Azure AI Foundry model catalog is nice."])
    print(f"\nEmbedding dim: {len(emb.data[0].embedding)}")
