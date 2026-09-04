"""Tests for the dependency-free BM25 sparse retriever."""

from __future__ import annotations

from ai_agent_rag.models import Chunk
from ai_agent_rag.sparse import InMemoryBM25Retriever


def _chunk(cid: str, source: str, text: str, **meta) -> Chunk:
    return Chunk(id=cid, text=text, source_id=source, path=meta.get("path", "/x"), metadata=meta)


def _index() -> InMemoryBM25Retriever:
    r = InMemoryBM25Retriever()
    r.index(
        "c",
        [
            _chunk("a", "s1", "parental leave policy for employees", path="hr/leave.md"),
            _chunk("b", "s2", "kubernetes ingress controller networking", path="it/k8s.md"),
        ],
    )
    return r


def test_bm25_ranks_by_term_overlap() -> None:
    hits = _index().search("c", "parental leave", k=5)
    assert hits
    assert hits[0].chunk_id == "a"
    assert hits[0].score > 0


def test_bm25_no_match_returns_empty() -> None:
    assert _index().search("c", "quantum chromodynamics", k=5) == []


def test_bm25_delete_by_source() -> None:
    r = _index()
    r.delete_by_source("c", "s1")
    hits = r.search("c", "parental leave policy", k=5)
    assert all(h.source_id != "s1" for h in hits)


def test_bm25_reindex_replaces_document() -> None:
    r = _index()
    r.index(
        "c", [_chunk("a", "s1", "totally different content about vacations", path="hr/leave.md")]
    )
    # "parental" no longer present in doc a -> a should not match
    assert all(h.chunk_id != "a" for h in r.search("c", "parental", k=5))


def test_bm25_honors_metadata_filter() -> None:
    hits = _index().search("c", "policy networking", k=5, filters={"path": {"$contains": "hr/"}})
    assert [h.chunk_id for h in hits] == ["a"]


def test_bm25_unknown_collection_is_empty() -> None:
    assert InMemoryBM25Retriever().search("missing", "anything", k=3) == []


# --- backend selection (build_sparse_retriever) ----------------------------------------------------
from ai_agent_rag.config import (  # noqa: E402
    PgVectorConfig,
    RagConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from ai_agent_rag.sparse import build_sparse_retriever  # noqa: E402


def test_build_sparse_defaults_to_bm25() -> None:
    assert isinstance(build_sparse_retriever(None), InMemoryBM25Retriever)
    assert isinstance(build_sparse_retriever(RagConfig()), InMemoryBM25Retriever)


def test_build_sparse_pgfts_with_dsn() -> None:
    from ai_agent_rag.sparse_pg import PgFtsSparseRetriever

    config = RagConfig(
        retrieval=RetrievalConfig(sparse_backend="pgfts"),
        vector_store=VectorStoreConfig(
            backend="pgvector", pgvector=PgVectorConfig(dsn="postgresql://u@h/db")
        ),
    )
    assert isinstance(build_sparse_retriever(config), PgFtsSparseRetriever)


def test_build_sparse_pgfts_without_dsn_falls_back_to_bm25() -> None:
    config = RagConfig(retrieval=RetrievalConfig(sparse_backend="pgfts"))  # no pgvector dsn
    assert isinstance(build_sparse_retriever(config), InMemoryBM25Retriever)


def test_build_sparse_unknown_backend_falls_back_to_bm25() -> None:
    config = RagConfig(retrieval=RetrievalConfig(sparse_backend="wat"))
    assert isinstance(build_sparse_retriever(config), InMemoryBM25Retriever)
