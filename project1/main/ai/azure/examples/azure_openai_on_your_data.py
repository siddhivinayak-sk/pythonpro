# pip install "openai>=1.3.0" python-dotenv
#
# Azure OpenAI - "On Your Data" / Bring-Your-Own-Data (RAG)
# Chat completions grounded on an Azure AI Search index. Azure OpenAI
# handles the retrieval, prompt assembly, and citations for you.
#
# Prerequisites:
#   * An Azure AI Search index that already contains your documents
#     (created via Foundry portal "Add your data" or manually).
#   * The chat deployment has permission to query the search index
#     (either via the key below or a managed identity).

import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_model_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")          # https://<name>.search.windows.net
search_index = os.getenv("AZURE_SEARCH_INDEX")                # e.g. "company-docs"
search_key = os.getenv("AZURE_SEARCH_API_KEY")                # or use system-assigned MI

client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version=azure_openai_api_version,
)

completion = client.chat.completions.create(
    model=azure_model_deployment_name,
    messages=[
        {"role": "system", "content": "Answer only from the provided documents. Cite sources."},
        {"role": "user", "content": "What is our company policy on remote work?"},
    ],
    extra_body={
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": search_endpoint,
                    "index_name": search_index,
                    "authentication": {
                        "type": "api_key",
                        "key": search_key,
                    },
                    "query_type": "vector_simple_hybrid",   # or "semantic", "vector"
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": os.getenv(
                            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
                        ),
                    },
                    "top_n_documents": 5,
                    "in_scope": True,
                    "strictness": 3,
                    "role_information": "You are an internal HR assistant.",
                },
            }
        ]
    },
)

msg = completion.choices[0].message
print("Answer:\n" + (msg.content or ""))

# Retrieved citations are exposed on the message via `context`
ctx = getattr(msg, "context", None) or completion.choices[0].message.model_extra.get("context")
if ctx and "citations" in ctx:
    print("\nCitations:")
    for i, c in enumerate(ctx["citations"], 1):
        print(f"  [{i}] {c.get('title')} — {c.get('url') or c.get('filepath')}")
