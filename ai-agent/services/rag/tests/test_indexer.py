"""End-to-end tests for the indexing pipeline (temp dir → index → retrieve), all in-memory."""

from __future__ import annotations

from pathlib import Path

import pytest
from ai_agent_rag.config import (
    ChunkerProfile,
    ChunkersConfig,
    CollectionSpec,
    DirectorySource,
    EmbeddingProfile,
    EmbeddingsConfig,
    RagConfig,
    VectorStoreConfig,
)
from ai_agent_rag.embeddings import build_embedder
from ai_agent_rag.indexing import Indexer, IndexStore
from ai_agent_rag.vectorstore import InMemoryVectorStore


def _config(tmp_path: Path) -> RagConfig:
    return RagConfig(
        vector_store=VectorStoreConfig(backend="memory"),
        embeddings=EmbeddingsConfig(
            profiles=[EmbeddingProfile(id="e", provider="hashing", dimension=64)]
        ),
        chunkers=ChunkersConfig(
            profiles=[ChunkerProfile(id="c", strategy="recursive", target_tokens=40)]
        ),
        sources=[DirectorySource(id="docs", path=str(tmp_path))],
        collections=[
            CollectionSpec(
                name="kb", embedding_ref="e", dimension=64, chunker_ref="c", sources=["docs"]
            )
        ],
    )


def _indexer(tmp_path: Path):
    config = _config(tmp_path)
    store = IndexStore(":memory:")
    vector_store = InMemoryVectorStore()
    return Indexer(config, store, vector_store), vector_store


def test_index_and_retrieve_with_citation(tmp_path: Path) -> None:
    (tmp_path / "leave.md").write_text(
        "# Leave Policy\nEmployees get parental leave and vacation days.", encoding="utf-8"
    )
    (tmp_path / "it.txt").write_text(
        "Request a laptop through the IT asset portal.", encoding="utf-8"
    )
    indexer, vector_store = _indexer(tmp_path)

    result = indexer.index_collection("kb")
    assert result.changed == 2
    assert result.vectors_upserted >= 2

    embedder = build_embedder(indexer.config.embedding("e"))
    hits = vector_store.search("kb", embedder.embed_query("parental leave policy"), k=1)
    assert hits
    assert "leave.md" in hits[0].metadata["path"]


def test_incremental_skips_unchanged(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\nalpha content here", encoding="utf-8")
    indexer, _ = _indexer(tmp_path)
    assert indexer.index_collection("kb").changed == 1
    second = indexer.index_collection("kb")
    assert second.changed == 0
    assert second.scanned == 1


def test_changed_file_is_reindexed(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("# A\nalpha", encoding="utf-8")
    indexer, _ = _indexer(tmp_path)
    indexer.index_collection("kb")
    path.write_text("# A\nalpha beta gamma changed content", encoding="utf-8")
    assert indexer.index_collection("kb").changed == 1


def test_deleted_file_removes_vectors(tmp_path: Path) -> None:
    a = tmp_path / "a.md"
    a.write_text("# A\nalpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\nbeta", encoding="utf-8")
    indexer, vector_store = _indexer(tmp_path)
    indexer.index_collection("kb")
    before = vector_store.count("kb")
    a.unlink()
    result = indexer.index_collection("kb")
    assert result.deleted == 1
    assert vector_store.count("kb") < before


def test_full_mode_reindexes_all(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\nalpha", encoding="utf-8")
    indexer, _ = _indexer(tmp_path)
    indexer.index_collection("kb")
    assert indexer.index_collection("kb", mode="full").changed == 1


def test_dimension_mismatch_raises(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.collections[0].dimension = 128  # mismatch with embedding dim 64
    indexer = Indexer(config, IndexStore(":memory:"), InMemoryVectorStore())
    with pytest.raises(ValueError):
        indexer.index_collection("kb")
