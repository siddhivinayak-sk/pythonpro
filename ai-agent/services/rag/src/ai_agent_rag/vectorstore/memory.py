"""In-memory vector store: the dependency-free backend for tests and local development.

Implements exact cosine similarity in pure Python. Fine for small corpora; production uses pgvector.
"""

from __future__ import annotations

import math
from typing import Any

from ..embeddings import Vector
from ..filters import matches
from ..models import Chunk, RetrievedHit


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _score(a: Vector, b: Vector, distance: str) -> float:
    """Similarity score (higher = closer) for the collection's configured distance metric."""
    if distance == "l2":
        return -math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))
    if distance == "ip":
        return sum(x * y for x, y in zip(a, b, strict=False))
    return _cosine(a, b)


class _Record:
    __slots__ = ("vector", "chunk")

    def __init__(self, vector: Vector, chunk: Chunk) -> None:
        self.vector = vector
        self.chunk = chunk


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._collections: dict[str, dict[str, Any]] = {}

    def ensure_collection(self, name: str, dimension: int, distance: str = "cosine") -> None:
        existing = self._collections.get(name)
        if existing is None:
            self._collections[name] = {"dimension": dimension, "distance": distance, "records": {}}
        elif existing["dimension"] != dimension:
            raise ValueError(
                f"collection '{name}' exists with dimension {existing['dimension']}, got {dimension}"
            )

    def upsert(self, collection: str, chunks: list[Chunk], vectors: list[Vector]) -> None:
        coll = self._collections[collection]
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != coll["dimension"]:
                raise ValueError(
                    f"vector dim {len(vector)} != collection '{collection}' dim {coll['dimension']}"
                )
            coll["records"][chunk.id] = _Record(vector, chunk)

    def delete_by_source(self, collection: str, source_id: str) -> None:
        coll = self._collections.get(collection)
        if not coll:
            return
        to_delete = [
            cid for cid, rec in coll["records"].items() if rec.chunk.source_id == source_id
        ]
        for cid in to_delete:
            del coll["records"][cid]

    def search(
        self,
        collection: str,
        query_vector: Vector,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedHit]:
        coll = self._collections.get(collection)
        if not coll:
            raise KeyError(f"collection '{collection}' does not exist")
        scored: list[RetrievedHit] = []
        distance = coll.get("distance", "cosine")
        for rec in coll["records"].values():
            if not matches(rec.chunk.metadata, filters):
                continue
            score = _score(query_vector, rec.vector, distance)
            scored.append(
                RetrievedHit(
                    text=rec.chunk.text,
                    score=score,
                    chunk_id=rec.chunk.id,
                    source_id=rec.chunk.source_id,
                    metadata=rec.chunk.metadata,
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def count(self, collection: str) -> int:
        coll = self._collections.get(collection)
        return len(coll["records"]) if coll else 0
