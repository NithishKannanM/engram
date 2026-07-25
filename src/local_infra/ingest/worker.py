"""The background extraction worker: claim → process → mark done, in a loop.

One in-process asyncio task. ``process`` is the backend's extraction path (the same one
the inline case runs), so async and sync extraction can never diverge. A job that raises
is marked ``failed`` with its error, never dropped — the episode stays durable and the
failure is auditable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from local_infra.ingest.models import Job
from local_infra.ingest.queue import JobQueue

ProcessFn = Callable[[Job], Awaitable[object]]


class ExtractionWorker:
    def __init__(
        self,
        process: ProcessFn,
        queue: JobQueue,
        *,
        poll_interval: float = 0.02,
    ) -> None:
        self._process = process
        self._queue = queue
        self._poll = poll_interval
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = await self._queue.claim()
            except Exception:
                job = None
            if job is None:
                # Idle: sleep until the next poll, but wake immediately on stop.
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._poll)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await self._process(job)
                await self._queue.complete(job.id, status="done")
            except Exception as exc:  # noqa: BLE001 — a failed job is recorded, not fatal
                await self._queue.complete(job.id, status="failed", error=str(exc))
