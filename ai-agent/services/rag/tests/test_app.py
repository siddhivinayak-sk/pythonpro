"""End-to-end tests for the wired RAG API (index a temp dir, then retrieve with citations)."""

from __future__ import annotations

from pathlib import Path

from ai_agent_rag.app import create_app
from ai_agent_rag.config import (
    ChunkerProfile,
    ChunkersConfig,
    CollectionSpec,
    DirectorySource,
    EmbeddingProfile,
    EmbeddingsConfig,
    RagConfig,
    RagSettings,
    VectorStoreConfig,
)
from ai_agent_rag.service import RagService
from fastapi.testclient import TestClient


def _client(tmp_path: Path) -> TestClient:
    config = RagConfig(
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
    service = RagService(config, RagSettings(index_db_path=":memory:"))
    return TestClient(create_app(RagSettings(log_json=False), service=service))


def test_health_and_ready(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/healthz").json()["service"] == "rag-api"
    assert client.get("/readyz").json()["backend"] == "memory"


def test_collections_lists_kb(tmp_path: Path) -> None:
    cols = _client(tmp_path).get("/v1/collections").json()["collections"]
    assert any(c["name"] == "kb" and c["dimension"] == 64 for c in cols)


def test_index_then_retrieve_with_citation(tmp_path: Path) -> None:
    (tmp_path / "leave.md").write_text(
        "# Leave Policy\nEmployees receive parental leave and paid vacation.", encoding="utf-8"
    )
    client = _client(tmp_path)

    run = client.post("/v1/index/run", params={"collection": "kb"}).json()
    assert run["changed"] == 1
    assert run["vectors_upserted"] >= 1

    resp = client.post("/v1/retrieve", json={"collection": "kb", "query": "parental leave", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"]
    assert "leave.md" in body["hits"][0]["citation"]["path"]
    assert body["used"]["backend"] == "memory"


def test_retrieve_unknown_collection_is_404(tmp_path: Path) -> None:
    resp = _client(tmp_path).post("/v1/retrieve", json={"collection": "nope", "query": "x"})
    assert resp.status_code == 404


def test_index_status_after_run(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\nalpha", encoding="utf-8")
    client = _client(tmp_path)
    client.post("/v1/index/run", params={"collection": "kb"})
    status = client.get("/v1/index/status", params={"collection": "kb"}).json()
    assert status["collection"] == "kb"
    assert status["last_run"] is not None


def test_metrics_and_security_headers(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/metrics").status_code == 200  # Prometheus endpoint wired
    assert client.get("/healthz").headers["x-content-type-options"] == "nosniff"
