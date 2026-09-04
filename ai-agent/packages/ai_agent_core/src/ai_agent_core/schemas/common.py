"""Shared request/response models used by more than one service.

Keeping these here lets the chat backend and the RAG API agree on the retrieval contract, and lets a
typed client live in the core library (design doc, RAG deep dive §8.4).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    status: str = "ok"
    service: str
    version: str | None = None


class ModelInfo(BaseModel):
    """Non-secret model descriptor exposed to clients (e.g. the chat model picker)."""

    connection_id: str
    model_name: str
    display_name: str
    provider: str
    capabilities: list[str] = Field(default_factory=list)
    dimensions: int | None = None
    context_window: int | None = None


class Citation(BaseModel):
    path: str | None = None
    page: int | None = None
    title: str | None = None
    url: str | None = None


class Hit(BaseModel):
    text: str
    score: float
    citation: Citation | None = None
    chunk_id: str | None = None
    source_id: str | None = None


class RetrieveRequest(BaseModel):
    collection: str
    query: str
    k: int = 6
    mode: str = "dense"  # dense | hybrid
    filters: dict | None = None
    rerank: bool = False


class RetrieveResponse(BaseModel):
    hits: list[Hit] = Field(default_factory=list)
    used: dict = Field(default_factory=dict)
