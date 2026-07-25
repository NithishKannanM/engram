"""Ingest value types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class Job:
    """A claimed extraction job — everything the worker needs to run the ladder and link
    provenance, with no further reads: scope, the source episode, its content, and the
    world time to stamp on extracted facts."""

    id: UUID
    scope_id: int
    episode_id: UUID
    content: str
    occurred_at: datetime
