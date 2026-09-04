"""Persistent sparse retriever backed by PostgreSQL full-text search.

Durable and multi-replica-safe (state lives in Postgres), reusing the same database as the pgvector store.
Each collection gets a table with a GENERATED ``tsvector`` column + GIN index; queries use
``websearch_to_tsquery`` + ``ts_rank_cd``. psycopg is imported lazily (needs the ``pgvector`` extra).
Metadata filtering reuses the shared JSONB condition builder so it matches the pgvector store exactly.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models import Chunk, RetrievedHit
from .vectorstore.pgvector import _SAFE_NAME, _filter_conditions, _normalize_dsn

_SAFE_LANG = re.compile(r"^[a-z_]+$")


def _fts_table(collection: str) -> str:
    return "rag_fts_" + _SAFE_NAME.sub("_", collection).lower()


class PgFtsSparseRetriever:
    def __init__(self, dsn: str | None, *, language: str = "english") -> None:
        if not dsn:
            raise ValueError("pgfts sparse retriever requires a DSN")
        self._dsn = _normalize_dsn(dsn)
        self._lang = language if _SAFE_LANG.match(language or "") else "english"
        self._ensured: set[str] = set()

    def _connect(self):
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)

    def _ensure(self, conn: Any, collection: str) -> None:
        if collection in self._ensured:
            return
        table = _fts_table(collection)
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "id TEXT PRIMARY KEY, source_id TEXT, content TEXT, metadata JSONB, "
            f"tsv tsvector GENERATED ALWAYS AS (to_tsvector('{self._lang}', coalesce(content, ''))) STORED)"
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_tsv_idx ON {table} USING GIN (tsv)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_src_idx ON {table} (source_id)")
        self._ensured.add(collection)

    def index(self, collection: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        table = _fts_table(collection)
        rows = [(c.id, c.source_id, c.text, json.dumps(c.metadata)) for c in chunks]
        with self._connect() as conn:
            self._ensure(conn, collection)
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {table} (id, source_id, content, metadata) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET source_id=EXCLUDED.source_id, "
                    "content=EXCLUDED.content, metadata=EXCLUDED.metadata",
                    rows,
                )

    def delete_by_source(self, collection: str, source_id: str) -> None:
        table = _fts_table(collection)
        with self._connect() as conn:
            self._ensure(conn, collection)
            conn.execute(f"DELETE FROM {table} WHERE source_id = %s", (source_id,))

    def search(
        self, collection: str, query: str, k: int, filters: dict | None = None
    ) -> list[RetrievedHit]:
        table = _fts_table(collection)
        tsquery = f"websearch_to_tsquery('{self._lang}', %s)"
        clauses, filter_params = _filter_conditions(filters)
        where = f"tsv @@ {tsquery}"
        if clauses:
            where += " AND " + " AND ".join(clauses)
        # ts_rank_cd (in SELECT) takes the query first, then the WHERE match, then filter params, then k.
        sql = (
            f"SELECT id, source_id, content, metadata, ts_rank_cd(tsv, {tsquery}) AS score "
            f"FROM {table} WHERE {where} ORDER BY score DESC LIMIT %s"
        )
        params = [query, query, *filter_params, k]
        with self._connect() as conn:
            # Avoid erroring on a collection that was never indexed (no table yet).
            reg = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()
            if not reg or reg[0] is None:
                return []
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        hits: list[RetrievedHit] = []
        for row in rows:
            chunk_id, source_id, content, metadata, score = row
            hits.append(
                RetrievedHit(
                    text=content,
                    score=float(score),
                    chunk_id=chunk_id,
                    source_id=source_id,
                    metadata=metadata or {},
                )
            )
        return hits
