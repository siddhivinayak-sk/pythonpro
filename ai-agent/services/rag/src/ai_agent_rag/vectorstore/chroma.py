"""Chroma adapter (embedded option).

Requires the ``chroma`` extra (``ai-agent-rag[chroma]``). Lazy-imported; covered by the memory backend in
automated tests.
"""

from __future__ import annotations

from typing import Any

from ..embeddings import Vector
from ..models import Chunk, RetrievedHit


class ChromaVectorStore:
    def __init__(self, path: str | None = None) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self._dims: dict[str, int] = {}

    def _collection(self, name: str):
        return self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def ensure_collection(self, name: str, dimension: int, distance: str = "cosine") -> None:
        self._dims[name] = dimension
        self._collection(name)

    def upsert(self, collection: str, chunks: list[Chunk], vectors: list[Vector]) -> None:
        coll = self._collection(collection)
        coll.upsert(
            ids=[c.id for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[{"source_id": c.source_id, "path": c.path, **c.metadata} for c in chunks],
        )

    def delete_by_source(self, collection: str, source_id: str) -> None:
        self._collection(collection).delete(where={"source_id": source_id})

    def search(
        self, collection: str, query_vector: Vector, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedHit]:
        coll = self._collection(collection)
        result = coll.query(
            query_embeddings=[list(query_vector)], n_results=k, where=filters or None
        )
        hits: list[RetrievedHit] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            hits.append(
                RetrievedHit(
                    text=doc,
                    score=1.0 - float(dist),  # cosine distance -> similarity
                    chunk_id=cid,
                    source_id=(meta or {}).get("source_id", ""),
                    metadata=meta or {},
                )
            )
        return hits

    def count(self, collection: str) -> int:
        return self._collection(collection).count()
