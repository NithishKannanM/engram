"""LedgerBackend — the write path both backends share.

The plan is explicit that ``remember``/``forget``/``history`` are nearly identical
across backends and only ``recall``/``improve`` diverge (cognee's graph channel vs.
ex_mem's FAISS). This base captures that shared write path against the ``Repository``
protocol and the pure :func:`temporal.resolve`, and exposes two indexing hooks so a
subclass can mirror facts/episodes into whatever recall index it owns:

  * :meth:`_on_episode` — a durable episode was written
  * :meth:`_on_fact`    — a *new* memory row was inserted (not a duplicate touch)

Postgres remains the record of truth for temporal facts regardless of backend; a
subclass never re-decides supersession — it only chooses how to *recall*.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from local_infra.core.models import (
    CloseMemory,
    Memory,
    NewFact,
    RecordRevision,
    Relation,
    Verdict,
)
from local_infra.core.normalize import normalize_key
from local_infra.core.predicates import cardinality_of
from local_infra.core.repository import Repository
from local_infra.core.service import (
    GLOBAL,
    ExtractMode,
    ForgetResult,
    MemoryService,
    RememberResult,
    parse_scope,
)
from local_infra.core.temporal import resolve
from local_infra.extraction.ladder import ExtractionLadder, LadderResult, ScoredFact
from local_infra.extraction.providers.base import Stage
from local_infra.extraction.registry import close_providers
from local_infra.ingest.queue import JobQueue
from local_infra.ingest.worker import ExtractionWorker


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LedgerBackend(MemoryService):
    """Shared write path. Subclasses implement ``recall``/``improve``/``health`` and
    may override the ``_on_episode``/``_on_fact`` indexing hooks."""

    def __init__(
        self,
        repo: Repository,
        ladder: ExtractionLadder,
        *,
        offline: bool = False,
        queue: "JobQueue | None" = None,
    ) -> None:
        self._repo = repo
        self._ladder = ladder
        self._offline = offline
        self._queue = queue
        self._worker: "ExtractionWorker | None" = None

    # ---- remember --------------------------------------------------------- #
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
        name, kind, key = parse_scope(scope)
        scope_id = await self._repo.ensure_scope(
            name, kind=kind, key=key, offline_only=self._offline
        )
        episode = await self._repo.add_episode(
            scope_id, content, source=source, occurred_at=occurred_at, metadata=metadata
        )
        await self._on_episode(scope_id, episode)

        result = RememberResult(episode_id=episode.id, scope_id=scope_id)
        if extract == "never":
            return result

        # With a queue attached, "async" enqueues and returns immediately — a worker
        # drains it via the SAME extraction path. Without one (or for "sync"), extract
        # inline so fact_ids come back on the response.
        if extract == "async" and self._queue is not None:
            result.job_id = await self._queue.enqueue(episode)
            result.deferred = True
            return result

        ladder_result, fact_ids = await self._extract_and_apply(
            scope_id, content, episode_id=episode.id, occurred_at=episode.occurred_at
        )
        result.final_stage = ladder_result.final_stage.value
        result.deferred = ladder_result.final_stage is Stage.DEFER or bool(ladder_result.deferred)
        result.fact_ids = fact_ids
        result.extracted = len(fact_ids)
        return result

    async def _extract_and_apply(
        self,
        scope_id: int,
        content: str,
        *,
        episode_id: UUID,
        occurred_at: datetime,
        repo: Repository | None = None,
    ) -> tuple[LadderResult, list[UUID]]:
        """Run the ladder and apply each fact against ``repo`` (defaults to the request
        repo; the worker passes its own so a long job never shares the request handle)."""
        repo = repo or self._repo
        ladder_result = await self._ladder.run(content, source_text=content)
        fact_ids: list[UUID] = []
        for sf in ladder_result.facts:
            fact = self._to_new_fact(sf, observed_at=occurred_at)
            mid = await self._write_fact(
                repo, scope_id, fact, episode_id=episode_id, evidence_span=sf.fact.evidence_span
            )
            if mid is not None:  # only newly-inserted rows are returned (not touches)
                fact_ids.append(mid)
        return ladder_result, fact_ids

    @staticmethod
    def _to_new_fact(sf: ScoredFact, *, observed_at: datetime) -> NewFact:
        f = sf.fact
        predicate_key = normalize_key(f.predicate)
        statement = f"{f.subject} {f.predicate.replace('_', ' ')} {f.object}".strip()
        return NewFact(
            subject=f.subject,
            predicate=f.predicate,
            object=f.object,
            subject_key=normalize_key(f.subject),
            predicate_key=predicate_key,
            object_key=normalize_key(f.object),
            statement=statement,
            observed_at=observed_at,
            cardinality=cardinality_of(predicate_key),
            confidence=sf.score.composite,
        )

    async def _write_fact(
        self, repo: Repository, scope_id: int, fact: NewFact, *, episode_id: UUID, evidence_span: str
    ) -> UUID | None:
        """Apply one fact against ``repo``. Returns the new memory id on an insert, else
        None (a duplicate touch resolves onto an existing row and is not reported new)."""
        cands = await repo.find_candidates(scope_id, fact)
        res = resolve(fact, cands)
        if res.needs_judge:
            # No local contradiction judge yet (a later milestone). Bias to INDEPENDENT
            # so an unresolved multi-valued collision coexists rather than silently
            # deleting a true fact — the resolver's documented safe default.
            verdicts = {cid: Verdict(Relation.INDEPENDENT, 1.0) for cid in res.needs_judge}
            res = resolve(fact, cands, verdicts)
        mid = await repo.apply(
            res.intents,
            scope_id=scope_id,
            fact=fact,
            episode_id=episode_id,
            evidence_span=evidence_span,
        )
        if res.inserts_new and mid is not None:
            await self._on_fact(scope_id, mid, fact)
            return mid
        return None

    # ---- forget ----------------------------------------------------------- #
    async def forget(
        self, memory_id: UUID, *, reason: str = "user_forget", hard: bool = False
    ) -> ForgetResult:
        mem = await self._repo.get_memory(memory_id)
        if mem is None:
            return ForgetResult(memory_id, ok=False, hard=hard, reason=reason)

        if hard:
            ok = await self._repo.hard_delete(memory_id)
            return ForgetResult(memory_id, ok=ok, hard=True, reason=reason)

        # Soft: tombstone by closing the row. History and provenance stay intact; a
        # current recall excludes it via the valid_until filter.
        now = _now()
        intents = (
            CloseMemory(memory_id=memory_id, valid_until=now, invalidation_reason=reason),
            RecordRevision(
                revision_type="close",
                reason=reason,
                decided_by=f"user:{reason}",
                observed_at=now,
                memory_id=memory_id,
            ),
        )
        await self._repo.apply(intents, scope_id=mem.scope_id)
        return ForgetResult(memory_id, ok=True, hard=False, reason=reason)

    # ---- reads ------------------------------------------------------------ #
    async def get(self, memory_id: UUID) -> Memory | None:
        return await self._repo.get_memory(memory_id)

    async def history(
        self, subject_key: str, predicate_key: str, *, scope: str = GLOBAL
    ) -> list[Memory]:
        name, kind, key = parse_scope(scope)
        scope_id = await self._repo.ensure_scope(name, kind=kind, key=key)
        return await self._repo.history(
            scope_id, normalize_key(subject_key), normalize_key(predicate_key)
        )

    async def history_for(self, memory_id: UUID) -> list[Memory]:
        mem = await self._repo.get_memory(memory_id)
        if mem is None:
            return []
        # Walk by the row's OWN scope/subject/predicate — no scope-string round-trip,
        # so it works for any scope the memory actually lives in.
        return await self._repo.history(mem.scope_id, mem.subject_key, mem.predicate_key)

    # ---- worker lifecycle ------------------------------------------------- #
    async def start(self) -> None:
        """Start the background extraction worker, if a queue is attached. Called by the
        API lifespan; a no-op for backends built without a queue (tests, sync-only use)."""
        if self._queue is not None and self._worker is None:
            self._worker = ExtractionWorker(self._process_job, self._queue)
            await self._worker.start()

    async def _process_job(self, job) -> list[UUID]:
        _, fact_ids = await self._extract_and_apply(
            job.scope_id, job.content,
            episode_id=job.episode_id, occurred_at=job.occurred_at,
            repo=self._worker_repo(),
        )
        return fact_ids

    def _worker_repo(self) -> Repository:
        """Repo the worker writes facts through. Defaults to the request repo (safe for
        the synchronous SQLite backend); the pg backend overrides with its own connection."""
        return self._repo

    async def close(self) -> None:
        if self._worker is not None:
            await self._worker.stop()
            self._worker = None
        await close_providers(dict(self._ladder.providers))
        await self._repo.close()

    async def _scope_ids(self, scope: str | None) -> list[int]:
        """Scope ids for a consolidation pass: one named scope, or the whole ledger."""
        if scope is None:
            return await self._repo.all_scope_ids()
        name, kind, key = parse_scope(scope)
        return [await self._repo.ensure_scope(name, kind=kind, key=key)]

    # ---- filtering shared by recall implementations ----------------------- #
    @staticmethod
    def _visible(mem: Memory, *, as_of: datetime | None, include_invalidated: bool) -> bool:
        if as_of is not None:
            return mem.valid_from <= as_of and (mem.valid_until is None or mem.valid_until > as_of)
        if include_invalidated:
            return True
        return mem.valid_until is None

    # ---- indexing hooks (override to mirror into a recall index) ----------- #
    async def _on_episode(self, scope_id: int, episode) -> None:
        return None

    async def _on_fact(self, scope_id: int, memory_id: UUID, fact: NewFact) -> None:
        return None
