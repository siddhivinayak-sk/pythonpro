"""pgvector adapter (production default).

Uses psycopg 3 + the pgvector extension directly for predictable SQL. One table per collection.

NOTE: this path requires a live PostgreSQL with the ``vector`` extension and the ``pgvector`` extra
installed (``ai-agent-rag[pgvector]``). It is exercised via the ``pgvector`` Docker profile; the
dependency-free ``memory`` backend is what the automated tests cover.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..embeddings import Vector
from ..models import Chunk, RetrievedHit

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_]")


def _table(collection: str) -> str:
    return "rag_" + _SAFE_NAME.sub("_", collection).lower()


def _normalize_dsn(dsn: str) -> str:
    # Accept SQLAlchemy-style DSNs (postgresql+psycopg://...) and hand psycopg a plain one.
    return dsn.replace("postgresql+psycopg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )


class PgVectorStore:
    def __init__(self, dsn: str | None) -> None:
        if not dsn:
            raise ValueError("pgvector backend requires a DSN (vector_store.pgvector.dsn)")
        self._dsn = _normalize_dsn(dsn)

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self._dsn, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def ensure_collection(self, name: str, dimension: int, distance: str = "cosine") -> None:
        table = _table(name)
        with self._connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "id TEXT PRIMARY KEY, source_id TEXT, path TEXT, content TEXT, "
                "metadata JSONB, embedding vector(%s))",
                (dimension,),
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx "
                f"ON {table} USING hnsw (embedding vector_cosine_ops)"
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_source_idx ON {table} (source_id)")

    def upsert(self, collection: str, chunks: list[Chunk], vectors: list[Vector]) -> None:
        table = _table(collection)
        rows = [
            (c.id, c.source_id, c.path, c.text, json.dumps(c.metadata), vec)
            for c, vec in zip(chunks, vectors, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {table} (id, source_id, path, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET source_id=EXCLUDED.source_id, path=EXCLUDED.path, "
                "content=EXCLUDED.content, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
                rows,
            )

    def delete_by_source(self, collection: str, source_id: str) -> None:
        table = _table(collection)
        with self._connect() as conn:
            conn.execute(f"DELETE FROM {table} WHERE source_id = %s", (source_id,))

    def search(
        self, collection: str, query_vector: Vector, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedHit]:
        table = _table(collection)
        # Cosine distance operator <=>; score = 1 - distance.
        sql = (
            f"SELECT id, source_id, content, metadata, 1 - (embedding <=> %s) AS score "
            f"FROM {table} ORDER BY embedding <=> %s LIMIT %s"
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (query_vector, query_vector, k))
            rows = cur.fetchall()
        hits: list[RetrievedHit] = []
        for row in rows:
            cid, source_id, content, metadata, score = row
            hits.append(
                RetrievedHit(
                    text=content,
                    score=float(score),
                    chunk_id=cid,
                    source_id=source_id,
                    metadata=metadata or {},
                )
            )
        return hits

    def count(self, collection: str) -> int:
        table = _table(collection)
        with self._connect() as conn:
            result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(result[0]) if result else 0
