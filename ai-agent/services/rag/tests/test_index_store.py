"""Tests for the sqlite index-metadata store."""

from __future__ import annotations

import time

from ai_agent_rag.indexing import IndexStore


def test_upsert_get_and_list() -> None:
    store = IndexStore(":memory:")
    store.upsert_file("kb", "src:a.md", "/a.md", "hash1", 100, 1.0)
    rec = store.get_file("kb", "src:a.md")
    assert rec is not None
    assert rec.content_hash == "hash1"
    assert store.list_source_ids("kb") == {"src:a.md"}


def test_upsert_updates_hash() -> None:
    store = IndexStore(":memory:")
    store.upsert_file("kb", "src:a.md", "/a.md", "hash1", 100, 1.0)
    store.upsert_file("kb", "src:a.md", "/a.md", "hash2", 120, 2.0)
    assert store.get_file("kb", "src:a.md").content_hash == "hash2"
    assert len(store.list_source_ids("kb")) == 1


def test_delete_file() -> None:
    store = IndexStore(":memory:")
    store.upsert_file("kb", "src:a.md", "/a.md", "h", 1, 1.0)
    store.delete_file("kb", "src:a.md")
    assert store.get_file("kb", "src:a.md") is None


def test_record_and_read_run() -> None:
    store = IndexStore(":memory:")
    store.record_run(
        "kb",
        "incremental",
        started_at=time.time(),
        scanned=3,
        changed=2,
        deleted=1,
        failed=0,
        vectors_upserted=7,
    )
    last = store.last_run("kb")
    assert last is not None
    assert last["scanned"] == 3
    assert last["vectors_upserted"] == 7
