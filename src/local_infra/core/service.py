"""The ``MemoryService`` ABC — the swap seam between backends and everything above.

Both backends (``ExMemBackend`` now, ``CogneeBackend`` next) implement this identical
interface, which is what lets one contract suite run against both unchanged. The API
and desktop shell depend only on this ABC — never on cognee or ex_mem directly, so the
implementation stays swappable and the temporal ledger stays authoritative.

Every method is async because the concrete backends do I/O (SQLite/Postgres, Ollama,
cognee). The result types here are plain dataclasses so callers — tests, the CLI, the
REST layer — never import a backend to read a return value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import UUID

from local_infra.core.models import Episode, Memory

# The default scope. Scope strings are ``global`` | ``project:<key>`` | ``session:<id>``;
# the same string also names the cognee dataset in the cognee backend.
GLOBAL = "global"

ExtractMode = Literal["sync", "async", "never"]


def parse_scope(scope: str) -> tuple[str, str, str]:
    """Split a scope string into ``(name, kind, key)`` for :meth:`Repository.ensure_scope`.

    ``global`` -> the singleton global scope; ``project:web`` / ``session:abc`` carry a
    kind and key. Anything else is treated as a named scope of kind ``global``.
    """
    if not scope or scope == GLOBAL:
        return (GLOBAL, "global", GLOBAL)
    kind, sep, key = scope.partition(":")
    if sep and kind in ("project", "session") and key:
        return (scope, kind, key)
    return (scope, "global", scope)


# --------------------------------------------------------------------------- #
# Result types                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class RecallHit:
    """One retrieved item, fused-ranked. ``kind`` says whether it is an extracted
    memory (the default) or a raw episode (only when ``include_episodes``)."""

    score: float
    kind: Literal["memory", "episode"] = "memory"
    memory: Memory | None = None
    episode: Episode | None = None

    @property
    def text(self) -> str:
        if self.memory is not None:
            return self.memory.statement
        return self.episode.content if self.episode is not None else ""


@dataclass
class RememberResult:
    """Outcome of a ``remember`` call. The episode is ALWAYS written (durable
    provenance); ``fact_ids`` are the memories extraction committed this call."""

    episode_id: UUID
    scope_id: int
    fact_ids: list[UUID] = field(default_factory=list)
    extracted: int = 0
    final_stage: str | None = None
    deferred: bool = False
    job_id: UUID | None = None  # set when extraction was enqueued (extract="async")


@dataclass
class ForgetResult:
    memory_id: UUID
    ok: bool
    hard: bool
    reason: str


@dataclass
class ImproveReport:
    ops: tuple[str, ...] = ()
    dry_run: bool = False
    reindexed: int = 0
    deduped: int = 0
    merged: int = 0
    pruned: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class ScopeSummary:
    """One scope and how many facts are currently true in it."""

    name: str
    kind: str
    memories: int


@dataclass
class HealthReport:
    backend: str
    ok: bool
    memories: int
    episodes: int
    indexed_chunks: int
    offline: bool
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The interface                                                                #
# --------------------------------------------------------------------------- #
class MemoryService(ABC):
    """The one interface both backends implement. See module docstring."""

    backend_name: str = "base"

    @abstractmethod
    async def remember(
        self,
        content: str,
        *,
        scope: str = GLOBAL,
        metadata: dict | None = None,
        source: str = "api",
        occurred_at: datetime | None = None,
        extract: ExtractMode = "async",
    ) -> RememberResult:
        """Write an episode synchronously (never lost to a failed LLM call), then
        extract facts per ``extract``. No information is discarded on extraction failure."""
        ...

    @abstractmethod
    async def recall(
        self,
        query: str,
        *,
        scope: str = GLOBAL,
        limit: int = 10,
        memory_types: list[str] | None = None,
        as_of: datetime | None = None,
        include_invalidated: bool = False,
        include_episodes: bool = False,
    ) -> list[RecallHit]:
        """Fused recall over a scope. Default returns currently-valid memories only;
        ``as_of`` time-travels, ``include_invalidated`` returns superseded rows too."""
        ...

    @abstractmethod
    async def forget(
        self, memory_id: UUID, *, reason: str = "user_forget", hard: bool = False
    ) -> ForgetResult:
        """Default: tombstone via ``valid_until = now()`` — history stays intact.
        ``hard=True`` is the ONLY path that deletes a row."""
        ...

    @abstractmethod
    async def improve(
        self,
        *,
        scope: str | None = None,
        ops: tuple[str, ...] = ("dedup", "merge", "prune", "reindex"),
        dry_run: bool = False,
    ) -> ImproveReport:
        """Consolidation pass. No valid fact is ever lost to it."""
        ...

    @abstractmethod
    async def get(self, memory_id: UUID) -> Memory | None:
        ...

    @abstractmethod
    async def history(
        self, subject_key: str, predicate_key: str, *, scope: str = GLOBAL
    ) -> list[Memory]:
        """Every version of one (subject, predicate) in a scope, oldest first."""
        ...

    @abstractmethod
    async def history_for(self, memory_id: UUID) -> list[Memory]:
        """The temporal chain a given memory belongs to (resolved from its own
        scope/subject/predicate) — what ``GET /memories/{id}/history`` needs."""
        ...

    @abstractmethod
    async def health(self) -> HealthReport:
        ...

    async def scopes(self) -> list[ScopeSummary]:
        """Every scope with its live fact count. Backends that own a Repository get
        this for free; the default is empty so a fake service need not implement it."""
        return []

    async def start(self) -> None:
        """Start background work (e.g. the async extraction worker). No-op by default;
        the API lifespan calls it after building the backend."""
        return None

    async def close(self) -> None:
        """Release resources. Overridden by backends that hold connections."""
        return None
