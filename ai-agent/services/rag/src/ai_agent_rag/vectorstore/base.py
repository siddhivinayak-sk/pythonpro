"""Vector store protocol + factory.

``memory`` is always available (tests/dev). ``pgvector`` is the production default. ``chroma`` is an
embedded option. Qdrant/Milvus are reserved for later (see docs/subprojects/rag.md §3.2).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..config import VectorStoreConfig
from ..embeddings import Vector
from ..models import Chunk, RetrievedHit


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collection(self, name: str, dimension: int, distance: str = "cosine") -> None: ...

    def upsert(self, collection: str, chunks: list[Chunk], vectors: list[Vector]) -> None: ...

    def delete_by_source(self, collection: str, source_id: str) -> None: ...

    def search(
        self, collection: str, query_vector: Vector, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedHit]: ...

    def count(self, collection: str) -> int: ...


def build_vector_store(config: VectorStoreConfig) -> VectorStore:
    """Construct the configured vector store. Backend SDKs are imported lazily."""
    backend = config.backend
    if backend == "memory":
        from .memory import InMemoryVectorStore

        return InMemoryVectorStore()
    if backend == "pgvector":
        from .pgvector import PgVectorStore

        return PgVectorStore(config.pgvector.dsn)
    if backend == "chroma":
        from .chroma import ChromaVectorStore

        return ChromaVectorStore(config.chroma.path)
    if backend in ("qdrant", "milvus"):
        raise NotImplementedError(f"vector store backend '{backend}' is reserved for a later phase")
    raise ValueError(f"unknown vector store backend '{backend}'")
