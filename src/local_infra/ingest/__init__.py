"""Async extraction ingest: the queue + worker that make ``remember(extract="async")`` real.

``remember`` always writes the episode synchronously (durable provenance) and, when a
queue is attached, enqueues an extraction job and returns immediately with a ``job_id``.
A background :class:`ExtractionWorker` drains the queue, running the SAME extraction path
as the inline case. No information is ever lost to a slow or failed LLM call — the
episode is already on disk, and a failed job is recorded, not dropped.

Two queue implementations behind one protocol:
  * :class:`MemoryJobQueue` — in-process, for the zero-infra ex_mem backend (its SQLite
    connection is synchronous, so worker and request handlers never truly overlap).
  * :class:`PgJobQueue` — the ``extraction_jobs`` table with ``FOR UPDATE SKIP LOCKED``,
    so a job is claimed once. Its claim/complete run on a DEDICATED worker connection so
    a long, LLM-bound job never ties up the request connection.
"""

from __future__ import annotations

from local_infra.ingest.models import Job
from local_infra.ingest.queue import JobQueue, MemoryJobQueue, PgJobQueue
from local_infra.ingest.worker import ExtractionWorker

__all__ = ["Job", "JobQueue", "MemoryJobQueue", "PgJobQueue", "ExtractionWorker"]
