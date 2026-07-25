"""ExMemBackend — the zero-external-infra MemoryService implementation.

Inherits the shared write path from :class:`LedgerBackend` (episode + extraction +
temporal resolution against a ``Repository``) and adds the two things that are ex_mem's
own: a hybrid FAISS+BM25 recall index and the indexing hooks that keep it in sync.

Recall design: extracted fact statements are indexed into the retriever keyed by their
memory id; the ledger is then the temporal filter. Superseded facts stay in the index
(so ``as_of`` time-travel can still find them) and are dropped from a current recall by
the ``valid_until`` check — never by mutating the index. Episodes are indexed too but
excluded from a default recall unless ``include_episodes`` is set.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from local_infra.backends.base import LedgerBackend
from local_infra.backends.sqlite.repository import SqliteRepository
from local_infra.consolidate import consolidate
from local_infra.config import Settings, get_settings
from local_infra.core.models import Episode, Memory, NewFact
from local_infra.core.repository import Repository
from local_infra.core.service import (
    GLOBAL,
    HealthReport,
    ImproveReport,
    RecallHit,
    parse_scope,
)
from local_infra.embed.base import Embedder
from local_infra.embed.ollama import OllamaEmbedder
from local_infra.exmem.config import RetrievalConfig
from local_infra.exmem.retrieval import HybridRetriever
from local_infra.exmem.store import Chunk, SQLiteMetadataStore
from local_infra.extraction.ladder import ExtractionLadder
from local_infra.extraction.registry import build_ladder
from local_infra.ingest.queue import MemoryJobQueue


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class ExMemBackend(LedgerBackend):
    backend_name = "exmem"

    def __init__(
        self,
        repo: Repository,
        retriever: HybridRetriever,
        ladder: ExtractionLadder,
        store: SQLiteMetadataStore,
        *,
        offline: bool = False,
        embedder: Embedder | None = None,
        retrieval_cfg: RetrievalConfig | None = None,
        queue: MemoryJobQueue | None = None,
    ) -> None:
        super().__init__(repo, ladder, offline=offline, queue=queue)
        self._retriever = retriever
        self._store = store
        self._embed = embedder
        self._cfg = retrieval_cfg or RetrievalConfig()

    # ---- factory ---------------------------------------------------------- #
    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        offline: bool | None = None,
        ledger_path: str | None = None,
        index_path: str | None = None,
    ) -> "ExMemBackend":
        """Wire a production backend from settings (Ollama embedder + real ladder,
        file-backed SQLite so the ledger and index survive a restart)."""
        settings = settings or get_settings()
        offline = settings.offline if offline is None else offline
        state = Path(settings.state_dir)
        state.mkdir(parents=True, exist_ok=True)
        embed: Embedder = OllamaEmbedder()
        cfg = RetrievalConfig()
        store = SQLiteMetadataStore(index_path or str(state / "exmem_index.db"))
        retriever = HybridRetriever(embed, store, cfg)
        # The ledger uses the same embedder so fuzzy (differently-phrased) duplicate
        # candidates are caught, not just exact triple-key matches.
        repo = SqliteRepository(ledger_path or str(state / "exmem_ledger.db"), embedder=embed)
        ladder = build_ladder(settings, offline=offline)
        # In-process queue: the SQLite connection is synchronous, so the worker reuses the
        # request repo (no contention) — see LedgerBackend._worker_repo.
        queue = MemoryJobQueue()
        return cls(
            repo, retriever, ladder, store,
            offline=offline, embedder=embed, retrieval_cfg=cfg, queue=queue,
        )

    # ---- recall ----------------------------------------------------------- #
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
        name, kind, key = parse_scope(scope)
        scope_id = await self._repo.ensure_scope(name, kind=kind, key=key)

        # Over-fetch, then filter by scope + kind + temporal validity, then truncate:
        # the single shared index spans all scopes and keeps superseded rows.
        raw = self._retriever.retrieve(query, top_k=max(limit * 4, 20))
        hits: list[RecallHit] = []
        for sc in raw:
            md = sc.chunk.metadata
            if md.get("scope_id") != scope_id:
                continue
            if md.get("kind") == "episode":
                if not include_episodes:
                    continue
                hits.append(
                    RecallHit(
                        score=sc.score,
                        kind="episode",
                        episode=self._episode_from(md, sc.chunk, scope_id),
                    )
                )
            else:
                mem = await self._repo.get_memory(UUID(sc.chunk.id))
                if mem is None:
                    continue  # hard-deleted row; index entry is stale but harmless
                if memory_types is not None and mem.memory_type not in memory_types:
                    continue
                if not self._visible(mem, as_of=as_of, include_invalidated=include_invalidated):
                    continue
                hits.append(RecallHit(score=sc.score, kind="memory", memory=mem))
            if len(hits) >= limit:
                break
        return hits[:limit]

    # ---- improve ---------------------------------------------------------- #
    async def improve(
        self,
        *,
        scope: str | None = None,
        ops: tuple[str, ...] = ("dedup", "merge", "prune", "reindex"),
        dry_run: bool = False,
    ) -> ImproveReport:
        report = ImproveReport(ops=tuple(ops), dry_run=dry_run)
        cons = await consolidate(
            self._repo, scope_ids=await self._scope_ids(scope),
            ops=ops, dry_run=dry_run, embedder=self._embed,
        )
        report.deduped, report.merged, report.pruned = cons.deduped, cons.merged, cons.pruned
        report.notes.extend(cons.notes)
        if "reindex" in ops:
            if self._embed is None:
                report.notes.append("reindex: no embedder configured")
            elif not dry_run:
                # Rebuild the retriever from the store so it reflects the consolidated
                # ledger (closed dup chunks are already filtered out of recall by the
                # valid_until check; this drops them from the index too on next use).
                self._retriever = HybridRetriever(self._embed, self._store, self._cfg)
                report.reindexed = self._store.count()
        return report

    # ---- health ----------------------------------------------------------- #
    async def health(self) -> HealthReport:
        chunks = self._store.all_chunks()
        memories = sum(1 for c in chunks if c.metadata.get("kind") == "memory")
        episodes = sum(1 for c in chunks if c.metadata.get("kind") == "episode")
        return HealthReport(
            backend=self.backend_name,
            ok=True,
            memories=memories,
            episodes=episodes,
            indexed_chunks=len(chunks),
            offline=self._offline,
            detail={"max_stage": self._ladder.max_stage().value},
        )

    async def close(self) -> None:
        await super().close()
        self._store.close()
        close = getattr(self._embed, "close", None)
        if close is not None:
            close()

    # ---- indexing hooks --------------------------------------------------- #
    async def _on_fact(self, scope_id: int, memory_id: UUID, fact: NewFact) -> None:
        self._index(
            Chunk(
                id=str(memory_id),
                text=fact.statement,
                turn_start=0,
                turn_end=0,
                role="memory",
                metadata={"kind": "memory", "scope_id": scope_id, "memory_id": str(memory_id)},
            )
        )

    async def _on_episode(self, scope_id: int, episode) -> None:
        self._index(
            Chunk(
                id=f"ep:{episode.id}",
                text=episode.content,
                turn_start=0,
                turn_end=0,
                role="episode",
                metadata={
                    "kind": "episode",
                    "scope_id": scope_id,
                    "episode_id": str(episode.id),
                    "source": episode.source,
                    "occurred_at": episode.occurred_at.isoformat(),
                },
            )
        )

    def _index(self, chunk: Chunk) -> None:
        # Add to the retriever BEFORE persisting to the store. The retriever builds its
        # FAISS/BM25 index lazily from the store's contents on first use; if this chunk
        # were already in the store at that moment, the lazy build would embed it once
        # and this add a second time. Adding first, persisting second, keeps it single.
        self._retriever.add_chunks([chunk])
        self._store.upsert(chunk)

    @staticmethod
    def _episode_from(md: dict, chunk: Chunk, scope_id: int) -> Episode:
        occurred = _parse_dt(md.get("occurred_at")) or _now()
        return Episode(
            id=UUID(md["episode_id"]),
            scope_id=scope_id,
            content=chunk.text,
            source=md.get("source", "api"),
            occurred_at=occurred,
            recorded_at=occurred,
        )
