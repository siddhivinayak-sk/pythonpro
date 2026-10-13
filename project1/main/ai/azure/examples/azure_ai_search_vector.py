# pip install azure-search-documents openai python-dotenv
#
# Azure AI Search - vector + hybrid retrieval
# Creates a small in-memory index, embeds a corpus with Azure OpenAI,
# uploads the vectors, and runs three queries:
#   1. Pure vector search
#   2. Keyword (BM25) search
#   3. Hybrid (vector + keyword) with semantic reranking
#
# This is the retrieval half of RAG; pair with azure_openai_on_your_data.py
# or your own chat pipeline for the generation half.

import os
import uuid
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
)
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

load_dotenv()

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_API_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX", "foundry-demo-idx")

aoai = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
)
embed_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
EMBED_DIM = 1536

# --- 1. (Re)create the index -------------------------------------------

index_client = SearchIndexClient(search_endpoint, AzureKeyCredential(search_key))
if index_name in [i.name for i in index_client.list_indexes()]:
    index_client.delete_index(index_name)

index = SearchIndex(
    name=index_name,
    fields=[
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBED_DIM,
            vector_search_profile_name="default-profile",
        ),
    ],
    vector_search=VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
        profiles=[VectorSearchProfile(name="default-profile", algorithm_configuration_name="default-hnsw")],
    ),
    semantic_search=SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="default-semantic",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    ),
)
index_client.create_index(index)
print(f"Index '{index_name}' created.")

# --- 2. Embed + upload documents ---------------------------------------

corpus = [
    {"title": "Foundry overview", "content": "Azure AI Foundry unifies models, agents, and evaluation for building AI apps."},
    {"title": "Content Safety", "content": "Content Safety flags harmful text and images and detects prompt injection."},
    {"title": "Vector search", "content": "Azure AI Search stores vectors alongside keywords for hybrid retrieval."},
    {"title": "Speech", "content": "Azure AI Speech provides STT, TTS, avatars, translation, and realtime voice."},
]

embeds = aoai.embeddings.create(model=embed_model, input=[d["content"] for d in corpus]).data
docs = [
    {
        "id": str(uuid.uuid4()),
        "title": d["title"],
        "content": d["content"],
        "content_vector": e.embedding,
    }
    for d, e in zip(corpus, embeds)
]

search_client = SearchClient(search_endpoint, index_name, AzureKeyCredential(search_key))
search_client.upload_documents(docs)
print(f"Uploaded {len(docs)} documents.")

# --- 3. Query --------------------------------------------------------------

query = "How do I detect jailbreak attempts?"
query_vec = aoai.embeddings.create(model=embed_model, input=query).data[0].embedding
vq = VectorizedQuery(vector=query_vec, k_nearest_neighbors=3, fields="content_vector")

print(f"\n=== Pure vector search: {query!r}")
for r in search_client.search(search_text=None, vector_queries=[vq], select=["title", "content"]):
    print(f"  score={r['@search.score']:0.4f}  {r['title']}")

print(f"\n=== Keyword (BM25) search: {query!r}")
for r in search_client.search(search_text=query, select=["title", "content"]):
    print(f"  score={r['@search.score']:0.4f}  {r['title']}")

print(f"\n=== Hybrid + semantic rerank: {query!r}")
for r in search_client.search(
    search_text=query,
    vector_queries=[vq],
    query_type="semantic",
    semantic_configuration_name="default-semantic",
    top=3,
    select=["title", "content"],
):
    print(f"  reranker={r.get('@search.reranker_score')}  {r['title']}")
