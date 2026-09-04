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
_SAFE_KEY = re.compile(r"^[a-zA-Z0-9_]+$")

# distance -> (HNSW ops class, distance operator). ORDER BY <op> ASC yields nearest neighbours.
_DISTANCE_OPS: dict[str, tuple[str, str]] = {
    "cosine": ("vector_cosine_ops", "<=>"),
    "l2": ("vector_l2_ops", "<->"),
    "ip": ("vector_ip_ops", "<#>"),
}


def _table(collection: str) -> str:
    return "rag_" + _SAFE_NAME.sub("_", collection).lower()


def _score_expr(distance: str, op: str) -> str:
    # Higher score = more similar. Cosine distance -> 1 - d; L2/IP -> negate so nearer ranks higher
    # (pgvector's <#> already returns the negative inner product).
    if distance == "cosine":
        return f"1 - (embedding {op} %s)"
    return f"-(embedding {op} %s)"


def _filter_conditions(filters: dict[str, Any] | None) -> tuple[list[str], list[Any]]:
    """Build JSONB-metadata SQL condition fragments + bound params (injection-safe).

    Supports ``{field: value}`` and ``{field: {"$eq": value}}`` equality plus ``{field: {"$contains": s}}``
    (case-insensitive substring), matching the semantics of the in-memory store. Shared by the pgvector
    store and the pgfts sparse retriever so both filter identically.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if not filters:
        return clauses, params
    for key, condition in filters.items():
        if not _SAFE_KEY.match(str(key)):
            continue  # ignore keys that aren't plain identifiers
        if isinstance(condition, dict):
            if "$contains" in condition:
                clauses.append("metadata->>%s ILIKE %s")
                params.extend([key, f"%{condition['$contains']}%"])
            elif "$eq" in condition:
                clauses.append("metadata->>%s = %s")
                params.extend([key, str(condition["$eq"])])
        else:
            clauses.append("metadata->>%s = %s")
            params.extend([key, str(condition)])
    return clauses, params


def _build_where(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """JSONB-metadata WHERE clause (with leading ``WHERE``) + params, or ``("", [])`` when empty."""
    clauses, params = _filter_conditions(filters)
    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


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
        self._distances: dict[str, str] = {}  # collection -> distance metric (for query operator)

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self._dsn, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def ensure_collection(self, name: str, dimension: int, distance: str = "cosine") -> None:
        table = _table(name)
        ops_class, _ = _DISTANCE_OPS.get(distance, _DISTANCE_OPS["cosine"])
        self._distances[name] = distance if distance in _DISTANCE_OPS else "cosine"
        # `dimension` is a type modifier (typmod), which cannot be a bound parameter, so it is
        # interpolated as a validated integer.
        dim = int(dimension)
        with self._connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "id TEXT PRIMARY KEY, source_id TEXT, path TEXT, content TEXT, "
                f"metadata JSONB, embedding vector({dim}))"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx "
                f"ON {table} USING hnsw (embedding {ops_class})"
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
        distance = self._distances.get(collection, "cosine")
        _, op = _DISTANCE_OPS[distance]
        score_expr = _score_expr(distance, op)
        where_sql, where_params = _build_where(filters)
        sql = (
            f"SELECT id, source_id, content, metadata, {score_expr} AS score "
            f"FROM {table}{where_sql} ORDER BY embedding {op} %s LIMIT %s"
        )
        params = [query_vector, *where_params, query_vector, k]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
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
