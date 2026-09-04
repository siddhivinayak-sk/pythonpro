"""Internal domain models for the RAG pipeline.

API request/response models are reused from ``ai_agent_core.schemas``; these dataclasses are the internal
currency passed between loader → chunker → embedder → vector store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedDocument:
    """A source file parsed into text plus document-level metadata."""

    source_id: str
    path: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit produced by a chunker, ready to embed and upsert."""

    id: str
    text: str
    source_id: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedHit:
    """A search result from the vector store."""

    text: str
    score: float
    chunk_id: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
