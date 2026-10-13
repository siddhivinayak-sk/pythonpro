# pip install "openai>=1.3.0" python-dotenv numpy
#
# Azure OpenAI - Text Embeddings
# Generates vector embeddings for a list of texts and shows a simple
# cosine-similarity search. Embeddings are the foundation of vector
# search / RAG on top of Azure AI Search or any vector store.

import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
# Deployment must point at an embeddings model, e.g. "text-embedding-3-small"
azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version=azure_openai_api_version,
)

corpus = [
    "Azure AI Foundry is a unified platform for building AI apps and agents.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    "Retrieval-Augmented Generation grounds LLM answers on your own data.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Azure AI Search supports vector, keyword, and hybrid retrieval.",
]

# Batch-embed the corpus in a single request
embed_response = client.embeddings.create(
    model=azure_embedding_deployment,
    input=corpus,
)
corpus_vectors = np.array([item.embedding for item in embed_response.data])
print(f"Embedded {len(corpus)} docs, dim={corpus_vectors.shape[1]}")

# Embed a query and rank by cosine similarity
query = "How do I ground an LLM on my company documents?"
query_vector = np.array(
    client.embeddings.create(model=azure_embedding_deployment, input=query).data[0].embedding
)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

scores = [(cosine(query_vector, v), doc) for v, doc in zip(corpus_vectors, corpus)]
scores.sort(key=lambda x: x[0], reverse=True)

print(f"\nQuery: {query}\n")
for score, doc in scores:
    print(f"{score:0.4f}  {doc}")
