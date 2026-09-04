"""Qdrant adapter.

Requires the ``qdrant`` extra (``ai-agent-rag[qdrant]``). The client is created lazily on first use, so
constructing the store (and the factory) needs no SDK. Server mode (``url``) and local embedded mode
(``path``) are both supported. Covered by the memory backend in automated tests; live Qdrant is
integration-only.

Qdrant point IDs must be UUIDs or ints, so the string chunk id is hashed to a stable UUID and the original
id is preserved in the payload.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..config import QdrantConfig
from ..embeddings import Vector
from ..models import Chunk, RetrievedHit

_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantVectorStore:
    def __init__(self, config: QdrantConfig) -> None:
        self._config = config
        self._client: Any = None

    def _c(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            if self._config.path:
                self._client = QdrantClient(path=self._config.path)
            elif self._config.url:
                self._client = QdrantClient(
                    url=self._config.url,
                    api_key=self._config.api_key,
                    prefer_grpc=self._config.prefer_grpc,
                )
            else:
                self._client = QdrantClient(location=":memory:")
        return self._client

    def ensure_collection(self, name: str, dimension: int, distance: str = "cosine") -> None:
        from qdrant_client.models import Distance, VectorParams

        dmap = {"cosine": Distance.COSINE, "l2": Distance.EUCLID, "ip": Distance.DOT}
        client = self._c()
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimension, distance=dmap.get(distance, Distance.COSINE)
                ),
            )

    def upsert(self, collection: str, chunks: list[Chunk], vectors: list[Vector]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=_point_id(c.id),
                vector=list(v),
                payload={
                    "chunk_id": c.id,
                    "text": c.text,
                    "source_id": c.source_id,
                    "path": c.path,
                    **c.metadata,
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._c().upsert(collection_name=collection, points=points)

    def delete_by_source(self, collection: str, source_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._c().delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            ),
        )

    def search(
        self, collection: str, query_vector: Vector, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedHit]:
        query_filter = _to_qdrant_filter(filters)
        response = self._c().query_points(
            collection_name=collection,
            query=list(query_vector),
            limit=k,
            with_payload=True,
            query_filter=query_filter,
        )
        hits: list[RetrievedHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                RetrievedHit(
                    text=payload.get("text", ""),
                    score=float(point.score),
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    source_id=payload.get("source_id", ""),
                    metadata=payload,
                )
            )
        return hits

    def count(self, collection: str) -> int:
        return int(self._c().count(collection_name=collection, exact=True).count)


def _to_qdrant_filter(filters: dict[str, Any] | None) -> Any:
    """Translate simple equality filters (``{field: value}``) to a Qdrant ``Filter``.

    Complex operator maps (e.g. ``{"$contains": ...}``) are not supported by Qdrant here and are ignored.
    """
    if not filters:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in filters.items()
        if isinstance(value, (str, int, bool))
    ]
    return Filter(must=conditions) if conditions else None
