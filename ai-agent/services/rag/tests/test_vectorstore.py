"""Tests for the in-memory vector store + factory."""

from __future__ import annotations

import pytest
from ai_agent_rag.config import VectorStoreConfig
from ai_agent_rag.models import Chunk
from ai_agent_rag.vectorstore import InMemoryVectorStore, build_vector_store


def _chunk(cid: str, source_id: str, text: str, **meta) -> Chunk:
    return Chunk(id=cid, text=text, source_id=source_id, path=meta.get("path", "/x"), metadata=meta)


def test_upsert_search_and_ranking() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection("kb", dimension=3)
    store.upsert(
        "kb",
        [_chunk("a", "s1", "alpha"), _chunk("b", "s2", "beta")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    hits = store.search("kb", [0.9, 0.1, 0.0], k=2)
    assert hits[0].chunk_id == "a"  # closest to the query vector
    assert hits[0].score > hits[1].score
    assert store.count("kb") == 2


def test_delete_by_source() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection("kb", dimension=2)
    store.upsert(
        "kb",
        [_chunk("a", "s1", "x"), _chunk("b", "s1", "y"), _chunk("c", "s2", "z")],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )
    store.delete_by_source("kb", "s1")
    assert store.count("kb") == 1


def test_dimension_mismatch_raises() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection("kb", dimension=3)
    with pytest.raises(ValueError):
        store.upsert("kb", [_chunk("a", "s1", "x")], [[1.0, 0.0]])


def test_metadata_filter_contains() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection("kb", dimension=2)
    store.upsert(
        "kb",
        [_chunk("a", "s1", "x", path="hr/leave.pdf"), _chunk("b", "s2", "y", path="it/assets.md")],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    hits = store.search("kb", [1.0, 0.0], k=5, filters={"path": {"$contains": "hr/"}})
    assert [h.chunk_id for h in hits] == ["a"]


def test_factory_routes_all_backends() -> None:
    # Qdrant/Milvus clients are created lazily, so the factory constructs them without their SDKs.
    from ai_agent_rag.vectorstore.milvus import MilvusVectorStore
    from ai_agent_rag.vectorstore.qdrant import QdrantVectorStore

    assert isinstance(build_vector_store(VectorStoreConfig(backend="memory")), InMemoryVectorStore)
    assert isinstance(build_vector_store(VectorStoreConfig(backend="qdrant")), QdrantVectorStore)
    assert isinstance(build_vector_store(VectorStoreConfig(backend="milvus")), MilvusVectorStore)
    with pytest.raises(ValueError):
        build_vector_store(VectorStoreConfig(backend="bogus"))


def test_qdrant_and_milvus_filter_translation() -> None:
    from ai_agent_rag.vectorstore.milvus import _to_milvus_filter
    from ai_agent_rag.vectorstore.qdrant import _to_qdrant_filter

    assert _to_qdrant_filter(None) is None
    assert _to_milvus_filter(None) == ""
    assert _to_milvus_filter({"source_id": "s1"}) == 'source_id == "s1"'


def test_memory_l2_distance_ranks_nearest_first() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection("kb", dimension=2, distance="l2")
    store.upsert(
        "kb",
        [_chunk("near", "s1", "x"), _chunk("far", "s2", "y")],
        [[0.0, 0.0], [10.0, 10.0]],
    )
    hits = store.search("kb", [0.1, 0.1], k=2)
    assert hits[0].chunk_id == "near"  # smallest L2 distance -> highest score
    assert hits[0].score >= hits[1].score


def test_pgvector_build_where_is_parameterized() -> None:
    from ai_agent_rag.vectorstore.pgvector import _build_where

    assert _build_where(None) == ("", [])
    sql, params = _build_where({"source_id": "s1"})
    assert "metadata->>%s = %s" in sql
    assert params == ["source_id", "s1"]
    sql, params = _build_where({"path": {"$contains": "hr/"}})
    assert "ILIKE" in sql
    assert params == ["path", "%hr/%"]
    # keys that aren't plain identifiers are ignored (injection guard)
    assert _build_where({"a; DROP TABLE x": "v"}) == ("", [])


def test_pgvector_filter_conditions_fragments() -> None:
    from ai_agent_rag.vectorstore.pgvector import _filter_conditions

    assert _filter_conditions(None) == ([], [])
    clauses, params = _filter_conditions({"source_id": "s1", "path": {"$contains": "hr/"}})
    assert (
        len(clauses) == 2
    )  # fragments without a leading WHERE (reused by the pgfts sparse backend)
    assert "s1" in params and "%hr/%" in params
