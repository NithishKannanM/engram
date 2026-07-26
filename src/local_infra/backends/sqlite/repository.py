"""SQLite implementation of the temporal ledger (zero external infra).

Runs sync sqlite under async methods — operations are in-process microseconds, so
there is no loop-blocking worth threading around for a local single-user service.
Vector-neighbor candidates are computed on the fly from an injected embedder;
with no embedder (the contract test) only exact triple-key candidates are returned.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np

from local_infra.backends.sqlite.schema import SQLITE_DDL
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


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _uuid(s: str | None) -> UUID | None:
    return UUID(s) if s else None


class SqliteRepository:
    def __init__(self, db_path: str = ":memory:", *, embedder: Embedder | None = None) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SQLITE_DDL)
        self._conn.commit()
        self._embedder = embedder

    # -- scopes / episodes --

    async def ensure_scope(
        self, name: str, *, kind: str = "global", key: str | None = None, offline_only: bool = False
    ) -> int:
        cur = self._conn.execute("SELECT id FROM scopes WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            return int(row[0])
        cur = self._conn.execute(
            "INSERT INTO scopes (name, kind, key, offline_only, created_at) VALUES (?,?,?,?,?)",
            (name, kind, key if key is not None else name, int(offline_only), _iso(_now())),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    async def add_episode(
        self,
        scope_id: int,
        content: str,
        *,
        source: str = "api",
        occurred_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> Episode:
        eid = uuid4()
        now = _now()
        occurred = occurred_at or now
        self._conn.execute(
            "INSERT INTO episodes (id, scope_id, content, source, metadata, occurred_at, recorded_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (str(eid), scope_id, content, source, json.dumps(metadata or {}), _iso(occurred), _iso(now)),
        )
        self._conn.commit()
        return Episode(
            id=eid, scope_id=scope_id, content=content, source=source,
            occurred_at=occurred, recorded_at=now, metadata=metadata or {},
        )

    # -- candidates --

    async def find_candidates(
        self, scope_id: int, fact: NewFact, *, limit: int = 10, min_sim: float = 0.82
    ) -> list[Candidate]:
        rows = self._conn.execute(
            "SELECT id, subject_key, predicate_key, object_key, statement, valid_from "
            "FROM memories WHERE scope_id = ? AND valid_until IS NULL "
            "AND subject_key = ? AND predicate_key = ?",
            (scope_id, fact.subject_key, fact.predicate_key),
        ).fetchall()
        found: dict[str, Candidate] = {
            r[0]: Candidate(
                id=UUID(r[0]), subject_key=r[1], predicate_key=r[2], object_key=r[3],
                statement=r[4], valid_from=_dt(r[5]), similarity=0.0,
            )
            for r in rows
        }

        if self._embedder is not None:
            for cand in self._vector_neighbors(scope_id, fact, limit=limit, min_sim=min_sim):
                found.setdefault(str(cand.id), cand)
        return list(found.values())

    def _vector_neighbors(
        self, scope_id: int, fact: NewFact, *, limit: int, min_sim: float
    ) -> list[Candidate]:
        rows = self._conn.execute(
            "SELECT id, subject_key, predicate_key, object_key, statement, valid_from "
            "FROM memories WHERE scope_id = ? AND valid_until IS NULL",
            (scope_id,),
        ).fetchall()
        if not rows:
            return []
        q = self._embedder.encode_one(fact.statement)
        mat = self._embedder.encode([r[4] for r in rows])  # normalized -> dot == cosine
        sims = mat @ q
        order = np.argsort(sims)[::-1][:limit]
        out: list[Candidate] = []
        for i in order:
            if float(sims[i]) < min_sim:
                break
            r = rows[i]
            out.append(
                Candidate(
                    id=UUID(r[0]), subject_key=r[1], predicate_key=r[2], object_key=r[3],
                    statement=r[4], valid_from=_dt(r[5]), similarity=float(sims[i]),
                )
            )
        return out

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
        try:
            cur = self._conn.cursor()
            # 1. Close first (guard-safe), invalidated_by_id left NULL for now.
            closed: list = []
            for c in plan.closes:
                cur.execute(
                    "UPDATE memories SET valid_until = ?, invalidation_reason = ? "
                    "WHERE id = ? AND valid_until IS NULL",
                    (_iso(c.valid_until), c.invalidation_reason, str(c.memory_id)),
                )
                if cur.rowcount > 0:
                    closed.append(c.memory_id)
            # 2. Insert the new row, then back-fill the invalidation pointer.
            if plan.insert is not None:
                self._insert_memory(cur, plan.new_id, scope_id, plan.insert)
                for mid in closed:
                    cur.execute(
                        "UPDATE memories SET invalidated_by_id = ? WHERE id = ?",
                        (str(plan.new_id), str(mid)),
                    )
            # 3. Touch re-observed rows.
            for t in plan.touches:
                cur.execute(
                    "UPDATE memories SET last_seen_at = ?, seen_count = seen_count + 1, "
                    "confidence = max(confidence, ?) WHERE id = ?",
                    (_iso(now), fact.confidence if fact else 1.0, str(t.memory_id)),
                )
            # 4. Audit revisions (None memory_id -> the new row).
            for rev in plan.revisions:
                mid = rev.memory_id or plan.new_id
                cur.execute(
                    "INSERT INTO memory_revisions (memory_id, revision_type, reason, decided_by, "
                    "observed_at, payload, created_at) VALUES (?,?,?,?,?,?,?)",
                    (str(mid) if mid else None, rev.revision_type, rev.reason, rev.decided_by,
                     _iso(rev.observed_at), "{}", _iso(now)),
                )
            # 5. Provenance link.
            if plan.primary_id is not None and episode_id is not None:
                cur.execute(
                    "INSERT OR IGNORE INTO memory_episodes (memory_id, episode_id, evidence_span, created_at) "
                    "VALUES (?,?,?,?)",
                    (str(plan.primary_id), str(episode_id), evidence_span, _iso(now)),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return plan.primary_id

    def _insert_memory(self, cur, new_id: UUID, scope_id: int, intent) -> None:
        f: NewFact = intent.fact
        attrs = {"cardinality": f.cardinality}
        embed_model = None
        if self._embedder is not None:
            # embedding lives in FAISS for recall; the ledger just records provenance.
            embed_model = getattr(self._embedder, "model", "embedder")
        cur.execute(
            f"INSERT INTO memories ({_MEM_COLS}) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(new_id), scope_id, f.subject, f.predicate, f.object,
                f.subject_key, f.predicate_key, f.object_key, f.statement,
                f.memory_type, f.confidence, None, json.dumps(attrs), embed_model,
                _iso(intent.valid_from), _iso(intent.valid_until), _iso(_now()), _iso(_now()), 1,
                str(intent.supersedes_id) if intent.supersedes_id else None, None, None,
            ),
        )

    # -- reads --

    def _row_to_memory(self, r: tuple) -> Memory:
        return Memory(
            id=UUID(r[0]), scope_id=int(r[1]), subject=r[2], predicate=r[3], object=r[4],
            subject_key=r[5], predicate_key=r[6], object_key=r[7], statement=r[8],
            memory_type=r[9], confidence=float(r[10]), extraction_stage=r[11],
            attrs=json.loads(r[12]), embed_model=r[13],
            valid_from=_dt(r[14]), valid_until=_dt(r[15]), recorded_at=_dt(r[16]),
            last_seen_at=_dt(r[17]), seen_count=int(r[18]),
            supersedes_id=_uuid(r[19]), invalidated_by_id=_uuid(r[20]), invalidation_reason=r[21],
        )

    async def get_memory(self, memory_id: UUID) -> Memory | None:
        row = self._conn.execute(
            f"SELECT {_MEM_COLS} FROM memories WHERE id = ?", (str(memory_id),)
        ).fetchone()
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
        where = ["scope_id = ?"]
        params: list = [scope_id]
        if as_of is not None:
            where.append("valid_from <= ? AND (valid_until IS NULL OR valid_until > ?)")
            params += [_iso(as_of), _iso(as_of)]
        elif not include_invalidated:
            where.append("valid_until IS NULL")
        if subject_key is not None:
            where.append("subject_key = ?")
            params.append(subject_key)
        if predicate_key is not None:
            where.append("predicate_key = ?")
            params.append(predicate_key)
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT {_MEM_COLS} FROM memories WHERE {' AND '.join(where)} "
            "ORDER BY valid_from DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def history(self, scope_id: int, subject_key: str, predicate_key: str) -> list[Memory]:
        rows = self._conn.execute(
            f"SELECT {_MEM_COLS} FROM memories WHERE scope_id = ? AND subject_key = ? "
            "AND predicate_key = ? ORDER BY valid_from ASC, recorded_at ASC",
            (scope_id, subject_key, predicate_key),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def hard_delete(self, memory_id: UUID) -> bool:
        self._conn.execute("DELETE FROM memory_episodes WHERE memory_id = ?", (str(memory_id),))
        self._conn.execute("DELETE FROM memory_revisions WHERE memory_id = ?", (str(memory_id),))
        cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (str(memory_id),))
        self._conn.commit()
        return cur.rowcount > 0

    # -- consolidation --

    async def all_scope_ids(self) -> list[int]:
        rows = self._conn.execute("SELECT id FROM scopes ORDER BY id").fetchall()
        return [int(r[0]) for r in rows]

    async def scope_summaries(self) -> list[tuple[str, str, int]]:
        rows = self._conn.execute(
            """
            SELECT s.name, s.kind, COUNT(m.id)
              FROM scopes s
              LEFT JOIN memories m
                ON m.scope_id = s.id AND m.valid_until IS NULL
             GROUP BY s.id, s.name, s.kind
             ORDER BY s.name
            """
        ).fetchall()
        return [(str(r[0]), str(r[1]), int(r[2])) for r in rows]

    async def consolidate(
        self, canonical_id: UUID, duplicate_ids, *, reason: str, decided_by: str
    ) -> None:
        dids = [str(d) for d in duplicate_ids]
        if not dids:
            return
        now = _now()
        placeholders = ",".join("?" * len(dids))
        try:
            cur = self._conn.cursor()
            row = cur.execute(
                f"SELECT COALESCE(SUM(seen_count), 0), COALESCE(MAX(confidence), 0) "
                f"FROM memories WHERE id IN ({placeholders}) AND valid_until IS NULL",
                dids,
            ).fetchone()
            seen_sum, conf_max = int(row[0]), float(row[1])
            cur.execute(
                "UPDATE memories SET seen_count = seen_count + ?, "
                "confidence = max(confidence, ?), last_seen_at = ? WHERE id = ?",
                (seen_sum, conf_max, _iso(now), str(canonical_id)),
            )
            for did in dids:
                cur.execute(
                    "UPDATE memories SET valid_until = ?, invalidated_by_id = ?, "
                    "invalidation_reason = ? WHERE id = ? AND valid_until IS NULL",
                    (_iso(now), str(canonical_id), reason, did),
                )
                cur.execute(
                    "INSERT INTO memory_revisions (memory_id, revision_type, reason, decided_by, "
                    "observed_at, payload, created_at) VALUES (?,?,?,?,?,?,?)",
                    (did, "close", reason, decided_by, _iso(now), "{}", _iso(now)),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    async def close(self) -> None:
        self._conn.close()
