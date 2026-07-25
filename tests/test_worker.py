"""M10 (worker half): async remember enqueues, a background worker drains it.

Two levels:
  * end-to-end on ex_mem (in-process queue, fakes): ``extract="async"`` returns a job id
    immediately with no facts, then the worker extracts and the fact becomes recallable —
    while ``extract="sync"`` stays inline even with a queue attached.
  * PgJobQueue mechanics against Postgres: enqueue → claim (FOR UPDATE SKIP LOCKED) hands
    out a job exactly once, and complete drives the status through.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest

from local_infra.backends.exmem_backend import ExMemBackend
from local_infra.backends.pg.migrate import run_migrations
from local_infra.backends.pg.repository import PgRepository
from local_infra.backends.sqlite.repository import SqliteRepository
from local_infra.config import get_settings
from local_infra.exmem.config import RetrievalConfig
from local_infra.exmem.retrieval import HybridRetriever
from local_infra.exmem.store import SQLiteMetadataStore
from local_infra.extraction.budget import Budget
from local_infra.extraction.ladder import ExtractionLadder
from local_infra.extraction.providers.base import Stage
from local_infra.ingest.queue import MemoryJobQueue, PgJobQueue

from tests._fakes import HashEmbedder, MappingProvider

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=30)

WORK_ACME = "I work at Acme."
WORK_GLOBEX = "Actually I moved to Globex."
MAPPING = {
    WORK_ACME: [dict(subject="I", predicate="works_at", object="Acme",
                     evidence_span="I work at Acme", confidence=0.9)],
    WORK_GLOBEX: [dict(subject="I", predicate="works_at", object="Globex",
                       evidence_span="moved to Globex", confidence=0.9)],
}


def _exmem_with_worker() -> ExMemBackend:
    embed = HashEmbedder(dim=64)
    store = SQLiteMetadataStore(":memory:")
    retriever = HybridRetriever(embed, store, RetrievalConfig(top_k=10))
    repo = SqliteRepository(":memory:")
    ladder = ExtractionLadder(
        {Stage.LOCAL_SMALL: MappingProvider(Stage.LOCAL_SMALL, MAPPING)},
        offline=True, budget=Budget(0.0),
    )
    return ExMemBackend(
        repo, retriever, ladder, store, offline=True, embedder=embed, queue=MemoryJobQueue()
    )


async def _wait_done(queue, job_id, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = await queue.status(job_id)
        if st in ("done", "failed", "deferred"):
            return st
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} never finished (last status {st!r})")


# --------------------------------------------------------------------------- #
async def test_async_remember_enqueues_and_worker_extracts() -> None:
    svc = _exmem_with_worker()
    await svc.start()
    try:
        r = await svc.remember(WORK_ACME, occurred_at=T0, extract="async")
        # Returned immediately: a job id, deferred, and NO facts yet.
        assert r.job_id is not None and r.deferred is True
        assert r.extracted == 0 and r.fact_ids == []
        # The episode is already durable and recallable...
        assert await svc.recall("where do I work") == []  # fact not extracted yet

        status = await _wait_done(svc._queue, r.job_id)
        assert status == "done"

        # ...and once the worker has run, the fact is there.
        hits = await svc.recall("where do I work")
        assert [h.memory.object_key for h in hits] == ["acme"]
    finally:
        await svc.close()


async def test_sync_still_inline_with_queue_attached() -> None:
    svc = _exmem_with_worker()
    await svc.start()
    try:
        r = await svc.remember(WORK_ACME, occurred_at=T0, extract="sync")
        assert r.job_id is None and r.extracted == 1 and len(r.fact_ids) == 1
        assert [h.memory.object_key for h in await svc.recall("where do I work")] == ["acme"]
    finally:
        await svc.close()


async def test_worker_processes_multiple_and_supersedes() -> None:
    svc = _exmem_with_worker()
    await svc.start()
    try:
        r1 = await svc.remember(WORK_ACME, occurred_at=T0, extract="async")
        await _wait_done(svc._queue, r1.job_id)
        r2 = await svc.remember(WORK_GLOBEX, occurred_at=T1, extract="async")
        await _wait_done(svc._queue, r2.job_id)

        # The worker ran both through the same temporal path -> globex supersedes acme.
        assert [h.memory.object_key for h in await svc.recall("where do I work")] == ["globex"]
        assert await svc._queue.pending_count() == 0
    finally:
        await svc.close()


# --------------------------------------------------------------------------- #
async def test_pg_queue_claims_once_skip_locked() -> None:
    settings = get_settings()
    try:
        repo = await PgRepository(settings.pg_dsn).connect()
        worker_conn = await asyncpg.connect(settings.pg_dsn)
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres not reachable: {exc!r}")
    await run_migrations(settings)
    await repo.conn.execute(
        "TRUNCATE extraction_attempts, extraction_jobs, memory_episodes, memory_revisions, "
        "memories, episodes, scopes RESTART IDENTITY CASCADE"
    )
    try:
        scope_id = await repo.ensure_scope("session:worker", kind="session", key="worker")
        episode = await repo.add_episode(scope_id, WORK_ACME, occurred_at=T0)

        queue = PgJobQueue(repo.conn, worker_conn)
        job_id = await queue.enqueue(episode)
        assert await queue.pending_count() == 1

        job = await queue.claim()
        assert job is not None and job.id == job_id
        assert job.content == WORK_ACME and job.occurred_at == T0
        # SKIP LOCKED + status flip: the one job is now running, so a second claim is idle.
        assert await queue.claim() is None
        assert await queue.status(job_id) == "running"

        await queue.complete(job_id, status="done")
        assert await queue.status(job_id) == "done"
        assert await queue.pending_count() == 0
    finally:
        await worker_conn.close()
        await repo.close()
