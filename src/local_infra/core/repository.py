"""The persistence seam: the temporal *ledger*, shared by both backends.

A ``Repository`` stores episodes and the bitemporal memory ledger, finds
contradiction candidates, and applies a batch of :class:`WriteIntent`s in ONE
transaction. It does NOT do fused recall — that divergence lives in the backends
(cognee vs. ex_mem). ``PgRepository`` and ``SqliteRepository`` implement this
protocol identically, which is what lets one contract test run against both.

The single-valued integrity guard (a partial unique index in both backends)
raises :data:`IntegrityError` if two currently-valid single-valued rows would
exist for one (scope, subject, predicate) — a bug surfaces loudly instead of
silently storing two contradictory truths.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from uuid import UUID, uuid4

import asyncpg

from local_infra.core.models import (
    Candidate,
    CloseMemory,
    Episode,
    InsertMemory,
    Memory,
    NewFact,
    RecordRevision,
    TouchMemory,
    WriteIntent,
)


@dataclass
class ApplyPlan:
    """Intents grouped and ordered for safe application, shared by both backends.

    Closes run BEFORE the insert: superseding a single-valued fact would otherwise
    momentarily leave two currently-valid single-valued rows and trip the guard
    mid-transaction. The new row's id is pre-generated (UUIDs are client-side) so a
    close can set ``invalidated_by_id`` to it before the row physically exists.
    """

    new_id: UUID | None
    insert: InsertMemory | None
    closes: list[CloseMemory] = field(default_factory=list)
    touches: list[TouchMemory] = field(default_factory=list)
    revisions: list[RecordRevision] = field(default_factory=list)

    @property
    def primary_id(self) -> UUID | None:
        """The row provenance should link to: the new row, else the touched row."""
        if self.new_id is not None:
            return self.new_id
        return self.touches[0].memory_id if self.touches else None


def order_intents(intents: Sequence[WriteIntent]) -> ApplyPlan:
    insert: InsertMemory | None = None
    closes: list[CloseMemory] = []
    touches: list[TouchMemory] = []
    revisions: list[RecordRevision] = []
    for intent in intents:
        if isinstance(intent, InsertMemory):
            insert = intent
        elif isinstance(intent, CloseMemory):
            closes.append(intent)
        elif isinstance(intent, TouchMemory):
            touches.append(intent)
        elif isinstance(intent, RecordRevision):
            revisions.append(intent)
    return ApplyPlan(
        new_id=uuid4() if insert is not None else None,
        insert=insert,
        closes=closes,
        touches=touches,
        revisions=revisions,
    )

# One exception both backends raise on a guard/constraint violation, so the
# contract test can assert it without knowing which backend is underneath.
IntegrityError = (sqlite3.IntegrityError, asyncpg.exceptions.IntegrityConstraintViolationError)


class Repository(Protocol):
    async def ensure_scope(
        self, name: str, *, kind: str = "global", key: str | None = None, offline_only: bool = False
    ) -> int:
        """Return the scope id for ``name``, creating it if needed."""
        ...

    async def add_episode(
        self,
        scope_id: int,
        content: str,
        *,
        source: str = "api",
        occurred_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> Episode:
        """Durably store a raw episode (provenance). Never lost to a failed extraction."""
        ...

    async def find_candidates(
        self, scope_id: int, fact: NewFact, *, limit: int = 10, min_sim: float = 0.82
    ) -> list[Candidate]:
        """Currently-valid rows that ``fact`` might collide with: exact triple-key
        matches, plus vector neighbors above ``min_sim`` when an embedder is set."""
        ...

    async def apply(
        self,
        intents: Sequence[WriteIntent],
        *,
        scope_id: int,
        fact: NewFact | None = None,
        episode_id: UUID | None = None,
        evidence_span: str | None = None,
    ) -> UUID | None:
        """Apply write intents in one transaction. Returns the *primary* memory id
        (the newly inserted row, or the touched row) so the caller can link
        provenance. ``fact`` supplies the row contents for an InsertMemory intent."""
        ...

    async def get_memory(self, memory_id: UUID) -> Memory | None: ...

    async def facts(
        self,
        scope_id: int,
        *,
        subject_key: str | None = None,
        predicate_key: str | None = None,
        as_of: datetime | None = None,
        include_invalidated: bool = False,
        limit: int = 200,
    ) -> list[Memory]:
        """Ledger rows for a scope. Default = currently valid; ``as_of`` time-travels;
        ``include_invalidated`` returns every version."""
        ...

    async def history(self, scope_id: int, subject_key: str, predicate_key: str) -> list[Memory]:
        """Every version of one (subject, predicate), oldest first."""
        ...

    async def hard_delete(self, memory_id: UUID) -> bool:
        """The ONLY hard-delete path. Returns True if a row was removed."""
        ...

    async def all_scope_ids(self) -> list[int]:
        """Every scope id — so ``improve(scope=None)`` can sweep the whole ledger."""
        ...

    async def coextracted_pairs(
        self, scope_id: int, *, limit: int = 500
    ) -> list[tuple[UUID, UUID, UUID]]:
        """``(memory_a, memory_b, shared_episode)`` for facts pulled from the SAME
        episode — the provenance edges of the memory graph.

        Ordered pairs with ``a < b`` so an edge appears once, not twice.
        """
        ...

    async def scope_summaries(self) -> list[tuple[str, str, int]]:
        """``(name, kind, currently-valid memory count)`` per scope, name-ordered.

        Counts exclude closed rows: a client listing scopes wants how much is true
        now, not how many revisions the ledger has accumulated.
        """
        ...

    async def consolidate(
        self,
        canonical_id: UUID,
        duplicate_ids: Sequence[UUID],
        *,
        reason: str,
        decided_by: str,
    ) -> None:
        """Fold ``duplicate_ids`` into ``canonical_id`` in ONE transaction: add their
        summed ``seen_count`` and max ``confidence`` onto the canonical, then close each
        duplicate (``valid_until=now``, ``invalidated_by_id=canonical``, ``reason``) and
        record a revision. The consolidation write primitive — no fact is lost, the
        canonical survives, and the closed rows stay as a replayable audit trail."""
        ...

    async def close(self) -> None: ...
