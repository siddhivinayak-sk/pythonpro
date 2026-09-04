"""Pluggable vector stores (default: pgvector)."""

from __future__ import annotations

from .base import VectorStore, build_vector_store
from .memory import InMemoryVectorStore

__all__ = ["VectorStore", "build_vector_store", "InMemoryVectorStore"]
