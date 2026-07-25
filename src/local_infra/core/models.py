"""Backend-agnostic domain types + the write-intent vocabulary.

These are plain, mostly-frozen dataclasses with no DB or LLM dependency so the
pure temporal resolver (``core/temporal.py``) can be unit-tested in milliseconds.
A ``WriteIntent`` describes *what* to change; the repository decides *how* and
applies a batch in one transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

Cardinality = Literal["single", "multi"]


class Relation(str, Enum):
    """A contradiction judge's verdict about a new fact vs. an existing one."""

    CONTRADICTS = "contradicts"
    DUPLICATE = "duplicate"
    INDEPENDENT = "independent"


@dataclass(frozen=True)
class Verdict:
    relation: Relation
    confidence: float = 1.0


@dataclass(frozen=True)
class NewFact:
    """A candidate fact to be written, already extracted and key-normalized."""

    subject: str
    predicate: str
    object: str
    subject_key: str
    predicate_key: str
    object_key: str
    statement: str  # natural-language rendering; the thing embedded / full-text indexed
    observed_at: datetime  # world time the fact became true / was asserted
    cardinality: Cardinality = "multi"
    confidence: float = 1.0
    memory_type: str = "fact"


@dataclass
class Memory:
    """A persisted memory row, read back from either backend's ledger."""

    id: UUID
    scope_id: int
    subject: str
    predicate: str
    object: str
    subject_key: str
    predicate_key: str
    object_key: str
    statement: str
    valid_from: datetime
    valid_until: datetime | None
    recorded_at: datetime
    last_seen_at: datetime
    seen_count: int
    confidence: float
    memory_type: str = "fact"
    attrs: dict = field(default_factory=dict)
    extraction_stage: str | None = None
    embed_model: str | None = None
    supersedes_id: UUID | None = None
    invalidated_by_id: UUID | None = None
    invalidation_reason: str | None = None

    @property
    def is_current(self) -> bool:
        return self.valid_until is None


@dataclass
class Episode:
    """A durable, raw provenance record — written before any extraction runs."""

    id: UUID
    scope_id: int
    content: str
    source: str
    occurred_at: datetime
    recorded_at: datetime
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """A currently-valid memory row that might collide with a NewFact."""

    id: UUID
    subject_key: str
    predicate_key: str
    object_key: str
    statement: str
    valid_from: datetime
    similarity: float = 0.0  # cosine(new.statement, candidate.statement), 0..1


# --------------------------------------------------------------------------- #
# Write intents — the output alphabet of the pure resolver.                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InsertMemory:
    """Insert a new memory row. ``valid_until`` set means insert already-closed
    (a retroactive/historical fact that must not override the present)."""

    fact: NewFact
    valid_from: datetime
    valid_until: datetime | None = None
    supersedes_id: UUID | None = None


@dataclass(frozen=True)
class CloseMemory:
    """Invalidate a currently-valid row. The repository applies this with a
    ``WHERE valid_until IS NULL`` guard so concurrent closes serialize, never
    silently yielding two current contradictory truths."""

    memory_id: UUID
    valid_until: datetime
    invalidation_reason: str


@dataclass(frozen=True)
class TouchMemory:
    """Bump last_seen_at / seen_count / confidence on a re-observed fact."""

    memory_id: UUID


@dataclass(frozen=True)
class RecordRevision:
    """Append a replayable audit row. ``memory_id`` is None when it refers to a
    not-yet-inserted new memory (the repository backfills the id)."""

    revision_type: Literal["insert", "close", "touch"]
    reason: str
    decided_by: str
    observed_at: datetime
    memory_id: UUID | None = None


WriteIntent = InsertMemory | CloseMemory | TouchMemory | RecordRevision


@dataclass(frozen=True)
class Resolution:
    """Output of the resolver.

    ``needs_judge`` non-empty means the decision is blocked on a contradiction
    judgment for those candidate ids: the caller runs the judge (via the
    extraction ladder) and re-invokes ``resolve`` with the verdicts filled in.
    When ``needs_judge`` is empty, ``intents`` is the complete transaction.
    """

    intents: tuple[WriteIntent, ...] = ()
    needs_judge: tuple[UUID, ...] = ()
    inserts_new: bool = False  # convenience: does this resolution create a new row?
