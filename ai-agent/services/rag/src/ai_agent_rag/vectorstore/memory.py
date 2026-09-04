"""In-memory vector store: the dependency-free backend for tests and local development.

Implements exact cosine similarity in pure Python. Fine for small corpora; production uses pgvector.
"""

from __future__ import annotations

import math
from typing import Any

from ..embeddings import Vector
from ..models import Chunk, RetrievedHit


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, condition in filters.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            if "$contains" in condition and (
                value is None or condition["$contains"] not in str(value)
            ):
                return False
            if "$eq" in condition and value != condition["$eq"]:
                return False
        elif value != condition:
            return False
    return True


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
        for rec in coll["records"].values():
            if not _matches(rec.chunk.metadata, filters):
                continue
            score = _cosine(query_vector, rec.vector)
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
