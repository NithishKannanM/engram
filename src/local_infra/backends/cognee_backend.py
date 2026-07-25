"""CogneeBackend — Postgres temporal ledger + cognee graph as a third recall channel.

Shares the entire write path with the ex_mem backend via :class:`LedgerBackend`; the
only divergence is here, in ``recall`` and ``improve`` (as the plan requires):

  * ``recall``  fuses THREE rankers over the *same* pg ledger rows — pgvector dense,
    tsvector sparse, and the cognee graph channel — with Reciprocal Rank Fusion. The
    cognee channel only re-ranks ledger rows it corroborates; it can never inject a row
    that is not in the ledger, so Postgres stays the record of truth.
  * ``improve`` runs cognee ``cognify`` to (re)build the graph from ingested episodes.

Episodes are fed to the cognee channel on write (cheap ``add``); the expensive graph
build is deferred to ``improve``. Datasets are keyed by scope id.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from local_infra.backends.base import LedgerBackend
from local_infra.backends.cognee_channel import CogneeChannel, NullCogneeChannel
from local_infra.backends.pg.repository import PgRepository
from local_infra.consolidate import consolidate
from local_infra.config import Settings, get_settings
from local_infra.core.fusion import rrf_fuse
from local_infra.core.models import Memory
from local_infra.core.service import (
    GLOBAL,
    HealthReport,
    ImproveReport,
    RecallHit,
    parse_scope,
)
from local_infra.exmem.config import RetrievalConfig
from local_infra.extraction.ladder import ExtractionLadder
from local_infra.extraction.registry import build_ladder
from local_infra.ingest.queue import JobQueue, PgJobQueue

COGNEE_WEIGHT = 0.5  # graph channel weight in the RRF blend (dense/sparse from cfg)


class CogneeBackend(LedgerBackend):
    backend_name = "cognee"

    def __init__(
        self,
        repo: PgRepository,
        ladder: ExtractionLadder,
        *,
        channel: CogneeChannel | None = None,
        offline: bool = False,
        retrieval_cfg: RetrievalConfig | None = None,
        queue: JobQueue | None = None,
        worker_repo: PgRepository | None = None,
    ) -> None:
        super().__init__(repo, ladder, offline=offline, queue=queue)
        self._pg = repo  # typed alias for pg-specific recall helpers
        self._cognee = channel or NullCogneeChannel()
        self._cfg = retrieval_cfg or RetrievalConfig()
        # Dedicated connection the worker writes facts through, so a long LLM-bound job
        # never contends with the request connection. None -> no async worker configured.
        self._wrepo = worker_repo

    # ---- factory ---------------------------------------------------------- #
    @classmethod
    async def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        offline: bool | None = None,
        channel: CogneeChannel | None = None,
    ) -> "CogneeBackend":
        settings = settings or get_settings()
        offline = settings.offline if offline is None else offline
        from local_infra.embed.ollama import OllamaEmbedder

        embedder = OllamaEmbedder()
        repo = await PgRepository(settings.pg_dsn, embedder=embedder).connect()
        # A second connection dedicated to the worker: it claims jobs and writes facts
        # there, isolated from the request connection.
        worker_repo = await PgRepository(settings.pg_dsn, embedder=embedder).connect()
        ladder = build_ladder(settings, offline=offline)
        queue = PgJobQueue(repo.conn, worker_repo.conn)
        if channel is None:
            from local_infra.backends.cognee_channel import CogneeGraphChannel

            channel = CogneeGraphChannel(
                pg_host=settings.pg_host,
                pg_port=settings.pg_port,
                pg_user=settings.pg_user,
                pg_password=settings.pg_password,
                pg_database=settings.pg_database,
                ollama_base_url=settings.ollama_base_url,
                llm_model=settings.extract_model_large,
                embed_model=settings.embed_model,
                embed_dim=settings.embed_dim,
                state_dir=str(settings.state_dir),
            )
        return cls(
            repo, ladder, channel=channel, offline=offline, queue=queue, worker_repo=worker_repo
        )

    def _worker_repo(self) -> PgRepository:
        # The worker writes through its own connection (never the request connection).
        return self._wrepo or self._pg

    # ---- write-side hook: feed raw episodes to the graph channel ---------- #
    async def _on_episode(self, scope_id: int, episode) -> None:
        await self._cognee.add(self._dataset(scope_id), episode.content)

    @staticmethod
    def _dataset(scope_id: int) -> str:
        return f"scope_{scope_id}"

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
        scope_id = await self._pg.ensure_scope(name, kind=kind, key=key)
        pool_n = max(limit * 4, 20)

        dense = await self._pg.dense_ranked(
            scope_id, query, limit=pool_n, as_of=as_of,
            include_invalidated=include_invalidated, memory_types=memory_types,
        )
        sparse = await self._pg.sparse_ranked(
            scope_id, query, limit=pool_n, as_of=as_of,
            include_invalidated=include_invalidated, memory_types=memory_types,
        )

        # Fetch the candidate pool once; both the cognee re-rank and the final hits reuse it.
        pool_ids = list(dict.fromkeys([*dense, *sparse]))
        mems: dict[UUID, Memory] = {}
        for mid in pool_ids:
            m = await self._pg.get_memory(mid)
            if m is not None and self._visible(
                m, as_of=as_of, include_invalidated=include_invalidated
            ):
                mems[mid] = m

        cognee_ids = await self._cognee_rerank(scope_id, query, mems, top_k=pool_n)

        fused = rrf_fuse(
            [
                (self._cfg.dense_weight, [i for i in dense if i in mems]),
                (self._cfg.bm25_weight, [i for i in sparse if i in mems]),
                (COGNEE_WEIGHT, cognee_ids),
            ]
        )
        hits: list[RecallHit] = []
        for rank, mid in enumerate(fused):
            hits.append(RecallHit(score=1.0 / (rank + 1), kind="memory", memory=mems[mid]))
            if len(hits) >= limit:
                break

        if include_episodes and len(hits) < limit:
            for ep in await self._pg.episode_search(scope_id, query, limit=limit):
                hits.append(RecallHit(score=0.0, kind="episode", episode=ep))
                if len(hits) >= limit:
                    break
        return hits[:limit]

    async def _cognee_rerank(
        self, scope_id: int, query: str, mems: dict[UUID, Memory], *, top_k: int
    ) -> list[UUID]:
        """Turn the graph channel's text snippets into a ranking over pool rows: a row is
        corroborated if its object or full statement appears in the returned snippets.
        Never introduces a non-ledger id."""
        snippets = await self._cognee.search(self._dataset(scope_id), query, top_k=top_k)
        if not snippets:
            return []
        blob = "\n".join(s.lower() for s in snippets)
        return [
            mid
            for mid, m in mems.items()
            if m.object_key in blob or m.statement.lower() in blob
        ]

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
            self._pg, scope_ids=await self._scope_ids(scope),
            ops=ops, dry_run=dry_run, embedder=self._pg.embedder,
        )
        report.deduped, report.merged, report.pruned = cons.deduped, cons.merged, cons.pruned
        report.notes.extend(cons.notes)
        if "reindex" in ops and not dry_run:
            name, kind, key = parse_scope(scope or GLOBAL)
            scope_id = await self._pg.ensure_scope(name, kind=kind, key=key)
            # Enrichment is best-effort and never authoritative: a graph build that
            # fails (e.g. a local model tripping cognee's structured-output summarizer)
            # must not fail the consolidation pass — recall still has the pg channels.
            try:
                report.reindexed = await self._cognee.cognify(self._dataset(scope_id))
            except Exception as exc:  # noqa: BLE001 — degrade, don't propagate
                report.notes.append(f"cognify failed, degraded to pg-only recall: {exc!r}")
        return report

    # ---- health ----------------------------------------------------------- #
    async def health(self) -> HealthReport:
        memories, episodes = await self._pg.counts()
        return HealthReport(
            backend=self.backend_name,
            ok=True,
            memories=memories,
            episodes=episodes,
            indexed_chunks=memories,
            offline=self._offline,
            detail={"channel": self._cognee.name, "max_stage": self._ladder.max_stage().value},
        )

    async def close(self) -> None:
        await super().close()  # stops the worker first, then closes the request repo
        if self._wrepo is not None:
            await self._wrepo.close()
        await self._cognee.close()
