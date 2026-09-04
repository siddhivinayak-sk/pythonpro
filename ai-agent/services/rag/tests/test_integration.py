"""Integration test for the pgvector adapter against a LIVE Postgres+pgvector.

Skipped unless RUN_INTEGRATION=1 and RAG_PGVECTOR_DSN points to a reachable database with the vector
extension available. Demonstrates the real-service test pattern the offline suite covers with the
in-memory store.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


def test_pgvector_upsert_and_search() -> None:
    dsn = os.environ.get("RAG_PGVECTOR_DSN")
    if not dsn:
        pytest.skip("RAG_PGVECTOR_DSN not set")

    from ai_agent_rag.models import Chunk
    from ai_agent_rag.vectorstore.pgvector import PgVectorStore

    store = PgVectorStore(dsn)
    store.ensure_collection("it_smoke", dimension=3)
    store.delete_by_source("it_smoke", "s1")
    store.upsert(
        "it_smoke",
        [
            Chunk(
                id="s1#0",
                text="hello world",
                source_id="s1",
                path="/a.txt",
                metadata={"path": "/a.txt"},
            )
        ],
        [[1.0, 0.0, 0.0]],
    )
    hits = store.search("it_smoke", [1.0, 0.0, 0.0], k=1)
    assert hits and hits[0].chunk_id == "s1#0"


def test_qdrant_upsert_and_search() -> None:
    url = os.environ.get("RAG_QDRANT_URL")
    if not url:
        pytest.skip("RAG_QDRANT_URL not set")

    from ai_agent_rag.config import QdrantConfig
    from ai_agent_rag.models import Chunk
    from ai_agent_rag.vectorstore.qdrant import QdrantVectorStore

    store = QdrantVectorStore(QdrantConfig(url=url, api_key=os.environ.get("RAG_QDRANT_API_KEY")))
    store.ensure_collection("it_smoke", dimension=3)
    store.delete_by_source("it_smoke", "s1")
    store.upsert(
        "it_smoke",
        [Chunk(id="s1#0", text="hello world", source_id="s1", path="/a.txt", metadata={})],
        [[1.0, 0.0, 0.0]],
    )
    hits = store.search("it_smoke", [1.0, 0.0, 0.0], k=1)
    assert hits and hits[0].chunk_id == "s1#0"


def test_milvus_upsert_and_search() -> None:
    uri = os.environ.get("RAG_MILVUS_URI")
    if not uri:
        pytest.skip("RAG_MILVUS_URI not set")

    from ai_agent_rag.config import MilvusConfig
    from ai_agent_rag.models import Chunk
    from ai_agent_rag.vectorstore.milvus import MilvusVectorStore

    store = MilvusVectorStore(MilvusConfig(uri=uri, token=os.environ.get("RAG_MILVUS_TOKEN")))
    store.ensure_collection("it_smoke", dimension=3)
    store.delete_by_source("it_smoke", "s1")
    store.upsert(
        "it_smoke",
        [Chunk(id="s1#0", text="hello world", source_id="s1", path="/a.txt", metadata={})],
        [[1.0, 0.0, 0.0]],
    )
    hits = store.search("it_smoke", [1.0, 0.0, 0.0], k=1)
    assert hits and hits[0].chunk_id == "s1#0"


def test_pgfts_sparse_index_and_search() -> None:
    dsn = os.environ.get("RAG_PGVECTOR_DSN")
    if not dsn:
        pytest.skip("RAG_PGVECTOR_DSN not set")

    from ai_agent_rag.models import Chunk
    from ai_agent_rag.sparse_pg import PgFtsSparseRetriever

    sparse = PgFtsSparseRetriever(dsn)
    sparse.delete_by_source("it_fts", "s1")
    sparse.index(
        "it_fts",
        [
            Chunk(
                id="s1#0",
                text="parental leave policy for employees",
                source_id="s1",
                path="/a.md",
                metadata={"path": "/a.md"},
            )
        ],
    )
    hits = sparse.search("it_fts", "parental leave", k=3)
    assert hits and hits[0].chunk_id == "s1#0"
    sparse.delete_by_source("it_fts", "s1")
    assert sparse.search("it_fts", "parental leave", k=3) == []
