"""Tests for the cron scheduler wrapper (registration + job dispatch, without starting the loop)."""

from __future__ import annotations

from ai_agent_rag.config import IndexerConfig, IndexJob, RagConfig
from ai_agent_rag.scheduler import IndexScheduler


class _FakeIndexer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def index_collection(self, collection: str, mode: str):
        self.calls.append((collection, mode))


def test_registers_a_job_per_config_entry() -> None:
    config = RagConfig(indexer=IndexerConfig(jobs=[IndexJob(collection="kb", cron="*/5 * * * *")]))
    scheduler = IndexScheduler(config, _FakeIndexer())
    scheduler._register_jobs()
    assert "index-kb" in scheduler.job_ids
    scheduler.shutdown()


def test_run_job_invokes_indexer() -> None:
    fake = _FakeIndexer()
    scheduler = IndexScheduler(RagConfig(), fake)
    scheduler._run_job("kb", "full")
    assert fake.calls == [("kb", "full")]
