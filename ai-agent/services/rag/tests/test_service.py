"""Service-level tests for RagService (retrieval defaults), using a recording fake vector store."""

from __future__ import annotations

from typing import Any

from ai_agent_core.schemas import RetrieveRequest
from ai_agent_rag.config import (
    ChunkerProfile,
    ChunkersConfig,
    CollectionSpec,
    DirectorySource,
    EmbeddingProfile,
    EmbeddingsConfig,
    RagConfig,
    RagSettings,
    RetrievalConfig,
    VectorStoreConfig,
)
from ai_agent_rag.models import RetrievedHit
from ai_agent_rag.service import RagService


class _RecordingStore:
    """Minimal VectorStore that records the k passed to search()."""

    def __init__(self) -> None:
        self.k: int | None = None

    def ensure_collection(self, name: str, dimension: int, distance: str = "cosine") -> None:
        return None

    def upsert(self, collection: str, chunks: list, vectors: list) -> None:
        return None

    def delete_by_source(self, collection: str, source_id: str) -> None:
        return None

    def search(self, collection: str, query_vector: Any, k: int, filters=None) -> list:
        self.k = k
        return []

    def count(self, collection: str) -> int:
        return 0


def _service(default_k: int) -> tuple[RagService, _RecordingStore]:
    config = RagConfig(
        embeddings=EmbeddingsConfig(
            profiles=[EmbeddingProfile(id="e", provider="hashing", dimension=8)]
        ),
        chunkers=ChunkersConfig(profiles=[ChunkerProfile(id="ch", strategy="recursive")]),
        collections=[CollectionSpec(name="c", embedding_ref="e", dimension=8, chunker_ref="ch")],
        retrieval=RetrievalConfig(default_k=default_k),
    )
    store = _RecordingStore()
    service = RagService(config, RagSettings(index_db_path=":memory:"), vector_store=store)
    return service, store


def test_retrieve_falls_back_to_configured_default_k() -> None:
    service, store = _service(default_k=3)
    service.retrieve(RetrieveRequest(collection="c", query="hi"))  # k omitted -> None
    assert store.k == 3


def test_retrieve_uses_explicit_k_when_provided() -> None:
    service, store = _service(default_k=3)
    service.retrieve(RetrieveRequest(collection="c", query="hi", k=5))
    assert store.k == 5


# --- hybrid retrieval + reranking (end-to-end over the memory store + BM25) ------------------------
def _indexed_service(tmp_path, *, reranker=None, default_k: int = 5) -> RagService:
    (tmp_path / "hr.txt").write_text(
        "parental leave policy for employees and family leave entitlements", encoding="utf-8"
    )
    (tmp_path / "it.txt").write_text(
        "kubernetes ingress controller networking and load balancing", encoding="utf-8"
    )
    config = RagConfig(
        vector_store=VectorStoreConfig(backend="memory"),
        embeddings=EmbeddingsConfig(
            profiles=[EmbeddingProfile(id="e", provider="hashing", dimension=64)]
        ),
        chunkers=ChunkersConfig(
            profiles=[ChunkerProfile(id="ch", strategy="recursive", target_tokens=64)]
        ),
        sources=[DirectorySource(id="src", path=str(tmp_path))],
        collections=[
            CollectionSpec(
                name="c", embedding_ref="e", dimension=64, chunker_ref="ch", sources=["src"]
            )
        ],
        retrieval=RetrievalConfig(default_k=default_k),
    )
    service = RagService(config, RagSettings(index_db_path=":memory:"), reranker=reranker)
    service.run_index("c")
    return service


def test_hybrid_retrieval_populates_sparse_and_ranks_relevant_first(tmp_path) -> None:
    service = _indexed_service(tmp_path)
    # sparse index was populated during indexing
    assert service.sparse.search("c", "parental leave", k=5)

    resp = service.retrieve(RetrieveRequest(collection="c", query="parental leave", mode="hybrid"))
    assert resp.used["mode"] == "hybrid"
    assert resp.hits
    assert "leave" in resp.hits[0].text.lower()


def test_rerank_runs_when_requested(tmp_path) -> None:
    class _PinKubernetesReranker:
        """Fake reranker that pins the k8s doc first, proving rerank actually ran."""

        def rerank(self, query: str, hits: list[RetrievedHit], top_k: int) -> list[RetrievedHit]:
            pinned = [h for h in hits if "kubernetes" in h.text.lower()]
            others = [h for h in hits if "kubernetes" not in h.text.lower()]
            return (pinned + others)[:top_k]

    service = _indexed_service(tmp_path, reranker=_PinKubernetesReranker())
    resp = service.retrieve(
        RetrieveRequest(collection="c", query="parental leave", mode="hybrid", rerank=True)
    )
    assert resp.used["reranked"] is True
    assert "kubernetes" in resp.hits[0].text.lower()  # reranker reordered the results
