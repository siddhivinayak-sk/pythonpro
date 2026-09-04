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
