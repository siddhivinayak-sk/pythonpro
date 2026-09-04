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


def test_factory_and_unsupported_backends() -> None:
    assert isinstance(build_vector_store(VectorStoreConfig(backend="memory")), InMemoryVectorStore)
    with pytest.raises(NotImplementedError):
        build_vector_store(VectorStoreConfig(backend="qdrant"))
    with pytest.raises(ValueError):
        build_vector_store(VectorStoreConfig(backend="bogus"))
