"""Index-metadata store (SQLite): tracks indexed files (for change detection) and run history.

Uses stdlib ``sqlite3``. Holds no secrets. Path is configurable; ``:memory:`` is used in tests.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    collection    TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    path          TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    size          INTEGER,
    mtime         REAL,
    status        TEXT,
    last_indexed_at REAL,
    PRIMARY KEY (collection, source_id)
);
CREATE TABLE IF NOT EXISTS index_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    collection    TEXT NOT NULL,
    mode          TEXT NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    scanned       INTEGER,
    changed       INTEGER,
    deleted       INTEGER,
    failed        INTEGER,
    vectors_upserted INTEGER,
    status        TEXT
);
"""


@dataclass
class FileRecord:
    collection: str
    source_id: str
    path: str
    content_hash: str
    size: int
    mtime: float
    status: str


class IndexStore:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # -- files --
    def get_file(self, collection: str, source_id: str) -> FileRecord | None:
        row = self._conn.execute(
            "SELECT * FROM files WHERE collection = ? AND source_id = ?", (collection, source_id)
        ).fetchone()
        if row is None:
            return None
        return FileRecord(
            collection=row["collection"],
            source_id=row["source_id"],
            path=row["path"],
            content_hash=row["content_hash"],
            size=row["size"],
            mtime=row["mtime"],
            status=row["status"],
        )

    def upsert_file(
        self,
        collection: str,
        source_id: str,
        path: str,
        content_hash: str,
        size: int,
        mtime: float,
        status: str = "indexed",
    ) -> None:
        self._conn.execute(
            "INSERT INTO files (collection, source_id, path, content_hash, size, mtime, status, last_indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (collection, source_id) DO UPDATE SET path=excluded.path, "
            "content_hash=excluded.content_hash, size=excluded.size, mtime=excluded.mtime, "
            "status=excluded.status, last_indexed_at=excluded.last_indexed_at",
            (collection, source_id, path, content_hash, size, mtime, status, time.time()),
        )
        self._conn.commit()

    def delete_file(self, collection: str, source_id: str) -> None:
        self._conn.execute(
            "DELETE FROM files WHERE collection = ? AND source_id = ?", (collection, source_id)
        )
        self._conn.commit()

    def list_source_ids(self, collection: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT source_id FROM files WHERE collection = ?", (collection,)
        ).fetchall()
        return {row["source_id"] for row in rows}

    # -- runs --
    def record_run(
        self,
        collection: str,
        mode: str,
        *,
        started_at: float,
        scanned: int,
        changed: int,
        deleted: int,
        failed: int,
        vectors_upserted: int,
        status: str = "completed",
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO index_runs (collection, mode, started_at, finished_at, scanned, changed, "
            "deleted, failed, vectors_upserted, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                collection,
                mode,
                started_at,
                time.time(),
                scanned,
                changed,
                deleted,
                failed,
                vectors_upserted,
                status,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def last_run(self, collection: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM index_runs WHERE collection = ? ORDER BY id DESC LIMIT 1", (collection,)
        ).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self._conn.close()
