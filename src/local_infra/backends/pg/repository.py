"""Postgres implementation of the temporal ledger (asyncpg).

Same protocol and same ``order_intents`` ordering as the SQLite repository, so the
contract suite asserts identical behavior against both. Uses pgvector for
vector-neighbor candidates and a jsonb type codec so ``attrs`` round-trips as a dict.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID, uuid4

import asyncpg

from local_infra.core.models import Candidate, Episode, Memory, NewFact, WriteIntent
from local_infra.core.repository import order_intents
from local_infra.embed.base import Embedder

_MEM_COLS = (
    "id, scope_id, subject, predicate, object, subject_key, predicate_key, object_key, "
    "statement, memory_type, confidence, extraction_stage, attrs, embed_model, "
    "valid_from, valid_until, recorded_at, last_seen_at, seen_count, "
    "supersedes_id, invalidated_by_id, invalidation_reason"
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PgRepository:
    def __init__(self, dsn: str, *, embedder: Embedder | None = None) -> None:
        self._dsn = dsn
        self._embedder = embedder
        self._conn: asyncpg.Connection | None = None

    async def connect(self) -> "PgRepository":
        self._conn = await asyncpg.connect(self._dsn)
        # jsonb <-> dict so attrs/metadata are plain dicts on both sides.
        await self._conn.set_type_codec(
            "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )
        return self

    @property
    def conn(self) -> asyncpg.Connection:
        if self._conn is None:
            raise RuntimeError("PgRepository.connect() was not awaited")
        return self._conn

    @property
    def embedder(self) -> Embedder | None:
        """The ledger's embedder — reused for the similarity-merge consolidation pass."""
        return self._embedder

    # -- scopes / episodes --

    async def ensure_scope(
        self, name: str, *, kind: str = "global", key: str | None = None, offline_only: bool = False
    ) -> int:
        row = await self.conn.fetchrow("SELECT id FROM scopes WHERE name = $1", name)
        if row:
            return int(row["id"])
        return int(
            await self.conn.fetchval(
                "INSERT INTO scopes (name, kind, key, offline_only) VALUES ($1,$2,$3,$4) "
                "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
                name, kind, key if key is not None else name, offline_only,
            )
        )

    async def add_episode(
        self,
        scope_id: int,
        content: str,
        *,
        source: str = "api",
        occurred_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> Episode:
        occurred = occurred_at or _now()
        row = await self.conn.fetchrow(
            "INSERT INTO episodes (scope_id, content, source, metadata, occurred_at) "
            "VALUES ($1,$2,$3,$4,$5) RETURNING id, recorded_at",
            scope_id, content, source, metadata or {}, occurred,
        )
        return Episode(
            id=row["id"], scope_id=scope_id, content=content, source=source,
            occurred_at=occurred, recorded_at=row["recorded_at"], metadata=metadata or {},
        )

    # -- candidates --

    async def find_candidates(
        self, scope_id: int, fact: NewFact, *, limit: int = 10, min_sim: float = 0.82
    ) -> list[Candidate]:
        rows = await self.conn.fetch(
            "SELECT id, subject_key, predicate_key, object_key, statement, valid_from "
            "FROM memories WHERE scope_id = $1 AND valid_until IS NULL "
            "AND subject_key = $2 AND predicate_key = $3",
            scope_id, fact.subject_key, fact.predicate_key,
        )
        found: dict[UUID, Candidate] = {
            r["id"]: Candidate(
                id=r["id"], subject_key=r["subject_key"], predicate_key=r["predicate_key"],
                object_key=r["object_key"], statement=r["statement"],
                valid_from=r["valid_from"], similarity=0.0,
            )
            for r in rows
        }

        if self._embedder is not None:
            q = self._vec_literal(fact.statement)
            nb = await self.conn.fetch(
                "SELECT id, subject_key, predicate_key, object_key, statement, valid_from, "
                "1 - (embedding <=> $2::vector) AS similarity "
                "FROM memories WHERE scope_id = $1 AND valid_until IS NULL AND embedding IS NOT NULL "
                "ORDER BY embedding <=> $2::vector LIMIT $3",
                scope_id, q, limit,
            )
            for r in nb:
                if float(r["similarity"]) < min_sim:
                    break
                found.setdefault(
                    r["id"],
                    Candidate(
                        id=r["id"], subject_key=r["subject_key"], predicate_key=r["predicate_key"],
                        object_key=r["object_key"], statement=r["statement"],
                        valid_from=r["valid_from"], similarity=float(r["similarity"]),
                    ),
                )
        return list(found.values())

    def _vec_literal(self, text: str) -> str:
        vec = self._embedder.encode_one(text)
        return "[" + ",".join(f"{x:.7f}" for x in vec.tolist()) + "]"

    # -- apply --

    async def apply(
        self,
        intents: Sequence[WriteIntent],
        *,
        scope_id: int,
        fact: NewFact | None = None,
        episode_id: UUID | None = None,
        evidence_span: str | None = None,
    ) -> UUID | None:
        plan = order_intents(intents)
        now = _now()
        async with self.conn.transaction():
            # Close first (guard-safe), but leave invalidated_by_id NULL — the row it
            # points at is inserted next, and the FK is checked immediately.
            closed: list = []
            for c in plan.closes:
                status = await self.conn.execute(
                    "UPDATE memories SET valid_until = $1, invalidation_reason = $2 "
                    "WHERE id = $3 AND valid_until IS NULL",
                    c.valid_until, c.invalidation_reason, c.memory_id,
                )
                if not status.endswith("0"):
                    closed.append(c.memory_id)
            if plan.insert is not None:
                await self._insert_memory(plan.new_id, scope_id, plan.insert)
                # Now that the new row exists, back-fill the invalidation pointer.
                for mid in closed:
                    await self.conn.execute(
                        "UPDATE memories SET invalidated_by_id = $1 WHERE id = $2",
                        plan.new_id, mid,
                    )
            for t in plan.touches:
                await self.conn.execute(
                    "UPDATE memories SET last_seen_at = $1, seen_count = seen_count + 1, "
                    "confidence = GREATEST(confidence, $2) WHERE id = $3",
                    now, fact.confidence if fact else 1.0, t.memory_id,
                )
            for rev in plan.revisions:
                await self.conn.execute(
                    "INSERT INTO memory_revisions (memory_id, revision_type, reason, decided_by, "
                    "observed_at, payload) VALUES ($1,$2,$3,$4,$5,$6)",
                    rev.memory_id or plan.new_id, rev.revision_type, rev.reason,
                    rev.decided_by, rev.observed_at, {},
                )
            if plan.primary_id is not None and episode_id is not None:
                await self.conn.execute(
                    "INSERT INTO memory_episodes (memory_id, episode_id, evidence_span) "
                    "VALUES ($1,$2,$3) ON CONFLICT DO NOTHING",
                    plan.primary_id, episode_id, evidence_span,
                )
        return plan.primary_id

    async def _insert_memory(self, new_id: UUID, scope_id: int, intent) -> None:
        f: NewFact = intent.fact
        attrs = {"cardinality": f.cardinality}
        embedding = None
        embed_model = None
        if self._embedder is not None:
            embedding = self._vec_literal(f.statement)
            embed_model = getattr(self._embedder, "model", "embedder")
        await self.conn.execute(
            "INSERT INTO memories (id, scope_id, subject, predicate, object, subject_key, "
            "predicate_key, object_key, statement, memory_type, confidence, attrs, embed_model, "
            "embedding, valid_from, valid_until, supersedes_id) VALUES "
            "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14::vector,$15,$16,$17)",
            new_id, scope_id, f.subject, f.predicate, f.object, f.subject_key,
            f.predicate_key, f.object_key, f.statement, f.memory_type, f.confidence, attrs,
            embed_model, embedding, intent.valid_from, intent.valid_until, intent.supersedes_id,
        )

    # -- reads --

    @staticmethod
    def _row_to_memory(r: asyncpg.Record) -> Memory:
        return Memory(
            id=r["id"], scope_id=int(r["scope_id"]), subject=r["subject"], predicate=r["predicate"],
            object=r["object"], subject_key=r["subject_key"], predicate_key=r["predicate_key"],
            object_key=r["object_key"], statement=r["statement"], memory_type=r["memory_type"],
            confidence=float(r["confidence"]), extraction_stage=r["extraction_stage"],
            attrs=r["attrs"], embed_model=r["embed_model"],
            valid_from=r["valid_from"], valid_until=r["valid_until"], recorded_at=r["recorded_at"],
            last_seen_at=r["last_seen_at"], seen_count=int(r["seen_count"]),
            supersedes_id=r["supersedes_id"], invalidated_by_id=r["invalidated_by_id"],
            invalidation_reason=r["invalidation_reason"],
        )

    async def get_memory(self, memory_id: UUID) -> Memory | None:
        row = await self.conn.fetchrow(
            f"SELECT {_MEM_COLS} FROM memories WHERE id = $1", memory_id
        )
        return self._row_to_memory(row) if row else None

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
        where = ["scope_id = $1"]
        params: list = [scope_id]

        def add(value) -> str:
            """Append a bind value and return its positional placeholder."""
            params.append(value)
            return f"${len(params)}"

        if as_of is not None:
            p = add(as_of)
            where.append(f"valid_from <= {p} AND (valid_until IS NULL OR valid_until > {p})")
        elif not include_invalidated:
            where.append("valid_until IS NULL")
        if subject_key is not None:
            where.append(f"subject_key = {add(subject_key)}")
        if predicate_key is not None:
            where.append(f"predicate_key = {add(predicate_key)}")
        add(limit)
        rows = await self.conn.fetch(
            f"SELECT {_MEM_COLS} FROM memories WHERE {' AND '.join(where)} "
            f"ORDER BY valid_from DESC LIMIT ${len(params)}",
            *params,
        )
        return [self._row_to_memory(r) for r in rows]

    async def history(self, scope_id: int, subject_key: str, predicate_key: str) -> list[Memory]:
        rows = await self.conn.fetch(
            f"SELECT {_MEM_COLS} FROM memories WHERE scope_id = $1 AND subject_key = $2 "
            "AND predicate_key = $3 ORDER BY valid_from ASC, recorded_at ASC",
            scope_id, subject_key, predicate_key,
        )
        return [self._row_to_memory(r) for r in rows]

    # -- recall channels (pg-specific; not part of the Repository protocol) --

    def _validity_where(self, where: list[str], add, *, as_of, include_invalidated) -> None:
        """Append the temporal-visibility predicate to ``where`` (mutates in place)."""
        if as_of is not None:
            p = add(as_of)
            where.append(f"valid_from <= {p} AND (valid_until IS NULL OR valid_until > {p})")
        elif not include_invalidated:
            where.append("valid_until IS NULL")

    async def dense_ranked(
        self,
        scope_id: int,
        query: str,
        *,
        limit: int = 10,
        as_of: datetime | None = None,
        include_invalidated: bool = False,
        memory_types: list[str] | None = None,
    ) -> list[UUID]:
        """Currently-visible memory ids ordered by pgvector cosine to ``query``.

        Empty without an embedder. The partial HNSW index covers only current rows; an
        ``as_of``/``include_invalidated`` recall falls back to an exact scan over the
        historical rows, which is fine for a local single-user store."""
        if self._embedder is None:
            return []
        params: list = [scope_id]

        def add(value) -> str:
            params.append(value)
            return f"${len(params)}"

        where = ["scope_id = $1", "embedding IS NOT NULL"]
        self._validity_where(where, add, as_of=as_of, include_invalidated=include_invalidated)
        if memory_types:
            where.append(f"memory_type = ANY({add(list(memory_types))})")
        qvec = add(self._vec_literal(query))
        lim = add(limit)
        rows = await self.conn.fetch(
            f"SELECT id FROM memories WHERE {' AND '.join(where)} "
            f"ORDER BY embedding <=> {qvec}::vector LIMIT {lim}",
            *params,
        )
        return [r["id"] for r in rows]

    async def sparse_ranked(
        self,
        scope_id: int,
        query: str,
        *,
        limit: int = 10,
        as_of: datetime | None = None,
        include_invalidated: bool = False,
        memory_types: list[str] | None = None,
    ) -> list[UUID]:
        """Currently-visible memory ids ordered by tsvector ``ts_rank`` against ``query``."""
        params: list = [scope_id]

        def add(value) -> str:
            params.append(value)
            return f"${len(params)}"

        where = ["scope_id = $1"]
        self._validity_where(where, add, as_of=as_of, include_invalidated=include_invalidated)
        if memory_types:
            where.append(f"memory_type = ANY({add(list(memory_types))})")
        q = add(query)
        where.append(f"tsv @@ plainto_tsquery('english', {q})")
        lim = add(limit)
        rows = await self.conn.fetch(
            f"SELECT id FROM memories WHERE {' AND '.join(where)} "
            f"ORDER BY ts_rank(tsv, plainto_tsquery('english', {q})) DESC LIMIT {lim}",
            *params,
        )
        return [r["id"] for r in rows]

    async def episode_search(
        self, scope_id: int, query: str, *, limit: int = 10
    ) -> list[Episode]:
        """Full-text over raw episode content (on-the-fly tsvector; episodes are the
        durable provenance layer and carry no generated index)."""
        rows = await self.conn.fetch(
            "SELECT id, content, source, occurred_at, recorded_at, metadata FROM episodes "
            "WHERE scope_id = $1 "
            "AND to_tsvector('english', content) @@ plainto_tsquery('english', $2) "
            "ORDER BY ts_rank(to_tsvector('english', content), plainto_tsquery('english', $2)) DESC "
            "LIMIT $3",
            scope_id, query, limit,
        )
        return [
            Episode(
                id=r["id"], scope_id=scope_id, content=r["content"], source=r["source"],
                occurred_at=r["occurred_at"], recorded_at=r["recorded_at"], metadata=r["metadata"],
            )
            for r in rows
        ]

    async def counts(self) -> tuple[int, int]:
        """(memories, episodes) row counts, for a health report."""
        m = await self.conn.fetchval("SELECT count(*) FROM memories")
        e = await self.conn.fetchval("SELECT count(*) FROM episodes")
        return int(m), int(e)

    async def hard_delete(self, memory_id: UUID) -> bool:
        async with self.conn.transaction():
            await self.conn.execute("DELETE FROM memory_episodes WHERE memory_id = $1", memory_id)
            await self.conn.execute("DELETE FROM memory_revisions WHERE memory_id = $1", memory_id)
            status = await self.conn.execute("DELETE FROM memories WHERE id = $1", memory_id)
        return status.endswith("1")

    # -- consolidation --

    async def all_scope_ids(self) -> list[int]:
        rows = await self.conn.fetch("SELECT id FROM scopes ORDER BY id")
        return [int(r["id"]) for r in rows]

    async def consolidate(
        self, canonical_id: UUID, duplicate_ids, *, reason: str, decided_by: str
    ) -> None:
        dids = list(duplicate_ids)
        if not dids:
            return
        now = _now()
        async with self.conn.transaction():
            agg = await self.conn.fetchrow(
                "SELECT COALESCE(SUM(seen_count), 0) AS s, COALESCE(MAX(confidence), 0) AS c "
                "FROM memories WHERE id = ANY($1::uuid[]) AND valid_until IS NULL",
                dids,
            )
            await self.conn.execute(
                "UPDATE memories SET seen_count = seen_count + $1, "
                "confidence = GREATEST(confidence, $2), last_seen_at = $3 WHERE id = $4",
                int(agg["s"]), float(agg["c"]), now, canonical_id,
            )
            await self.conn.execute(
                "UPDATE memories SET valid_until = $1, invalidated_by_id = $2, "
                "invalidation_reason = $3 WHERE id = ANY($4::uuid[]) AND valid_until IS NULL",
                now, canonical_id, reason, dids,
            )
            for did in dids:
                await self.conn.execute(
                    "INSERT INTO memory_revisions (memory_id, revision_type, reason, decided_by, "
                    "observed_at, payload) VALUES ($1,'close',$2,$3,$4,$5)",
                    did, reason, decided_by, now, {},
                )

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
