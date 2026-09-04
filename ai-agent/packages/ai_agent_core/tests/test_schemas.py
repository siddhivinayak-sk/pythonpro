"""Tests for shared DTOs."""

from __future__ import annotations

from ai_agent_core.schemas import Citation, HealthStatus, Hit, ModelInfo, RetrieveRequest


def test_retrieve_request_defaults() -> None:
    req = RetrieveRequest(collection="handbook", query="leave policy?")
    assert req.k == 6
    assert req.mode == "dense"
    assert req.rerank is False


def test_hit_with_citation_roundtrip() -> None:
    hit = Hit(
        text="policy text",
        score=0.82,
        citation=Citation(path="hr/leave.pdf", page=3, title="Leave Policy"),
        chunk_id="c1",
    )
    dumped = hit.model_dump()
    assert dumped["citation"]["page"] == 3
    assert dumped["score"] == 0.82


def test_model_info_fields() -> None:
    info = ModelInfo(
        connection_id="openai-prod",
        model_name="gpt-4o",
        display_name="GPT-4o",
        provider="openai",
        capabilities=["chat", "vision"],
    )
    assert info.capabilities == ["chat", "vision"]


def test_health_status_default_ok() -> None:
    hs = HealthStatus(service="rag-api")
    assert hs.status == "ok"
