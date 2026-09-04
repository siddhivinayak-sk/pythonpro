"""Cron-driven indexing via APScheduler.

Registers one job per configured indexer job using standard 5-field cron expressions. Kept thin; the real
work lives in ``Indexer.index_collection``. APScheduler is imported lazily.
"""

from __future__ import annotations

from ai_agent_core import get_logger

from .config import RagConfig
from .indexing import Indexer

log = get_logger("rag-scheduler")


class IndexScheduler:
    def __init__(self, config: RagConfig, indexer: Indexer) -> None:
        from apscheduler.schedulers.background import BackgroundScheduler

        self.config = config
        self.indexer = indexer
        self._scheduler = BackgroundScheduler(timezone=config.indexer.timezone)

    def _register_jobs(self) -> None:
        from apscheduler.triggers.cron import CronTrigger

        for job in self.config.indexer.jobs:
            trigger = CronTrigger.from_crontab(job.cron, timezone=self.config.indexer.timezone)
            self._scheduler.add_job(
                self._run_job,
                trigger=trigger,
                args=[job.collection, job.mode],
                id=f"index-{job.collection}",
                replace_existing=True,
            )
        log.info("scheduler_jobs_registered", count=len(self.config.indexer.jobs))

    def _run_job(self, collection: str, mode: str) -> None:
        try:
            self.indexer.index_collection(collection, mode)
        except Exception as exc:  # noqa: BLE001 - a failing run must not kill the scheduler
            log.warning("scheduled_index_failed", collection=collection, error=str(exc))

    def start(self) -> None:
        self._register_jobs()
        if self.config.indexer.on_start_run:
            for job in self.config.indexer.jobs:
                self._run_job(job.collection, job.mode)
        self._scheduler.start()
        log.info("scheduler_started")

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    @property
    def job_ids(self) -> list[str]:
        return [job.id for job in self._scheduler.get_jobs()]
