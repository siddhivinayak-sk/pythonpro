"""Milvus adapter (via ``pymilvus`` ``MilvusClient``).

Requires the ``milvus`` extra (``ai-agent-rag[milvus]``). The client is created lazily on first use, so
constructing the store (and the factory) needs no SDK. Uses the quick-setup collection (string primary key
``id`` + ``vector`` field, dynamic field enabled) so chunk metadata is stored without a rigid schema.
Covered by the memory backend in automated tests; live Milvus is integration-only.
"""

from __future__ import annotations

from typing import Any

from ..config import MilvusConfig
from ..embeddings import Vector
from ..models import Chunk, RetrievedHit

_METRICS = {"cosine": "COSINE", "l2": "L2", "ip": "IP"}


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class MilvusVectorStore:
    def __init__(self, config: MilvusConfig) -> None:
        self._config = config
        self._client: Any = None

    def _c(self) -> Any:
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(
                uri=self._config.uri or "http://localhost:19530",
                token=self._config.token or "",
                db_name=self._config.db_name or "default",
            )
        return self._client

    def ensure_collection(self, name: str, dimension: int, distance: str = "cosine") -> None:
        client = self._c()
        if not client.has_collection(name):
            client.create_collection(
                collection_name=name,
                dimension=dimension,
                metric_type=_METRICS.get(distance, "COSINE"),
                id_type="string",
                max_length=512,
                auto_id=False,
                primary_field_name="id",
                vector_field_name="vector",
            )

    def upsert(self, collection: str, chunks: list[Chunk], vectors: list[Vector]) -> None:
        rows = [
            {
                "id": c.id,
                "vector": list(v),
                "text": c.text,
                "source_id": c.source_id,
                "path": c.path,
                **c.metadata,
            }
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._c().upsert(collection_name=collection, data=rows)

    def delete_by_source(self, collection: str, source_id: str) -> None:
        self._c().delete(collection_name=collection, filter=f'source_id == "{_escape(source_id)}"')

    def search(
        self, collection: str, query_vector: Vector, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedHit]:
        results = self._c().search(
            collection_name=collection,
            data=[list(query_vector)],
            limit=k,
            output_fields=["text", "source_id", "path"],
            filter=_to_milvus_filter(filters),
        )
        hits: list[RetrievedHit] = []
        for match in results[0] if results else []:
            entity = match.get("entity", {}) or {}
            hits.append(
                RetrievedHit(
                    text=entity.get("text", ""),
                    score=float(match.get("distance", 0.0)),  # COSINE metric -> similarity
                    chunk_id=str(match.get("id", "")),
                    source_id=entity.get("source_id", ""),
                    metadata=entity,
                )
            )
        return hits

    def count(self, collection: str) -> int:
        rows = self._c().query(collection_name=collection, filter="", output_fields=["count(*)"])
        return int(rows[0]["count(*)"]) if rows else 0


def _to_milvus_filter(filters: dict[str, Any] | None) -> str:
    """Translate simple equality filters (``{field: value}``) to a Milvus boolean expression."""
    if not filters:
        return ""
    clauses: list[str] = []
    for key, value in filters.items():
        if isinstance(value, str):
            clauses.append(f'{key} == "{_escape(value)}"')
        elif isinstance(value, (int, float, bool)):
            clauses.append(f"{key} == {value}")
    return " and ".join(clauses)
