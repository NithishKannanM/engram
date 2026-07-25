"""Job queues: one in-process, one Postgres SKIP LOCKED. Same protocol, one worker."""

from __future__ import annotations

from collections import deque
from typing import Protocol
from uuid import UUID, uuid4

from local_infra.core.models import Episode
from local_infra.ingest.models import Job


class JobQueue(Protocol):
    async def enqueue(self, episode: Episode) -> UUID:
        """Register extraction work for a just-written episode. Returns the job id."""
        ...

    async def claim(self) -> Job | None:
        """Atomically take the oldest pending job (marking it running), or None if idle."""
        ...

    async def complete(self, job_id: UUID, *, status: str = "done", error: str | None = None) -> None:
        ...

    async def status(self, job_id: UUID) -> str | None:
        ...

    async def pending_count(self) -> int:
        """Jobs not yet finished (queued/deferred/running)."""
        ...

    async def close(self) -> None:
        ...


class MemoryJobQueue:
    """In-process queue for the ex_mem backend. Its SQLite connection is synchronous, so
    the worker and request handlers cannot truly interleave — no locking needed."""

    def __init__(self) -> None:
        self._queued: deque[Job] = deque()
        self._status: dict[UUID, str] = {}
        self._error: dict[UUID, str | None] = {}

    async def enqueue(self, episode: Episode) -> UUID:
        job = Job(
            id=uuid4(),
            scope_id=episode.scope_id,
            episode_id=episode.id,
            content=episode.content,
            occurred_at=episode.occurred_at,
        )
        self._queued.append(job)
        self._status[job.id] = "queued"
        return job.id

    async def claim(self) -> Job | None:
        if not self._queued:
            return None
        job = self._queued.popleft()
        self._status[job.id] = "running"
        return job

    async def complete(self, job_id: UUID, *, status: str = "done", error: str | None = None) -> None:
        self._status[job_id] = status
        self._error[job_id] = error

    async def status(self, job_id: UUID) -> str | None:
        return self._status.get(job_id)

    async def pending_count(self) -> int:
        return sum(1 for s in self._status.values() if s in ("queued", "running"))

    async def close(self) -> None:
        return None


class PgJobQueue:
    """The ``extraction_jobs`` table with ``FOR UPDATE SKIP LOCKED``.

    ``enqueue`` runs on the *request* connection (the same one that just wrote the
    episode), while ``claim``/``complete`` run on a dedicated *worker* connection — so a
    long, LLM-bound job never holds the connection the API is serving reads on.
    """

    _CLAIM = (
        "WITH j AS ("
        "  SELECT id FROM extraction_jobs WHERE status IN ('queued', 'deferred') "
        "  ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1"
        ") "
        "UPDATE extraction_jobs e "
        "SET status = 'running', locked_by = $1, locked_at = now(), "
        "    attempts = attempts + 1, updated_at = now() "
        "FROM j WHERE e.id = j.id "
        "RETURNING e.id, e.scope_id, e.episode_id"
    )

    def __init__(self, request_conn, worker_conn, *, worker_id: str = "worker-0") -> None:
        self._req = request_conn
        self._wconn = worker_conn
        self._worker_id = worker_id

    async def enqueue(self, episode: Episode) -> UUID:
        return await self._req.fetchval(
            "INSERT INTO extraction_jobs (episode_id, scope_id) VALUES ($1, $2) RETURNING id",
            episode.id, episode.scope_id,
        )

    async def claim(self) -> Job | None:
        row = await self._wconn.fetchrow(self._CLAIM, self._worker_id)
        if row is None:
            return None
        ep = await self._wconn.fetchrow(
            "SELECT content, occurred_at FROM episodes WHERE id = $1", row["episode_id"]
        )
        return Job(
            id=row["id"],
            scope_id=int(row["scope_id"]),
            episode_id=row["episode_id"],
            content=ep["content"],
            occurred_at=ep["occurred_at"],
        )

    async def complete(self, job_id: UUID, *, status: str = "done", error: str | None = None) -> None:
        await self._wconn.execute(
            "UPDATE extraction_jobs SET status = $2, last_error = $3, locked_by = NULL, "
            "updated_at = now() WHERE id = $1",
            job_id, status, error,
        )

    async def status(self, job_id: UUID) -> str | None:
        return await self._wconn.fetchval(
            "SELECT status FROM extraction_jobs WHERE id = $1", job_id
        )

    async def pending_count(self) -> int:
        return int(
            await self._wconn.fetchval(
                "SELECT count(*) FROM extraction_jobs "
                "WHERE status IN ('queued', 'deferred', 'running')"
            )
        )

    async def close(self) -> None:
        return None  # the worker connection is owned and closed by the backend
