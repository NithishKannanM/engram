"""M10 verification: improve() consolidation loses no valid fact.

Backend-agnostic (both backends share the ledger + consolidation), so it is parametrized
exactly like the service contract suite; ``-k exmem`` needs no infra, ``-k cognee`` needs
Postgres (skips otherwise).

The pathological state — 50 currently-valid rows for ONE fact — cannot be produced
through the public API (the temporal resolver collapses an exact re-observation into a
touch), so it is planted directly through the repository, simulating the accumulation
from re-extraction / concurrent writes that improve() exists to clean up. Alongside the
50 dups sit 3 genuinely distinct facts. The assertions are the plan's gate: the dups
collapse, and every distinct fact is still current afterward.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import asyncpg
import numpy as np
import pytest

from local_infra.backends.cognee_backend import CogneeBackend
from local_infra.backends.cognee_channel import NullCogneeChannel
from local_infra.backends.exmem_backend import ExMemBackend
from local_infra.backends.pg.migrate import run_migrations
from local_infra.backends.pg.repository import PgRepository
from local_infra.backends.sqlite.repository import SqliteRepository
from local_infra.config import get_settings
from local_infra.core.models import InsertMemory, NewFact
from local_infra.core.normalize import normalize_key
from local_infra.exmem.config import RetrievalConfig
from local_infra.exmem.retrieval import HybridRetriever
from local_infra.exmem.store import SQLiteMetadataStore
from local_infra.extraction.budget import Budget
from local_infra.extraction.ladder import ExtractionLadder
from local_infra.extraction.providers.base import Stage

from tests._fakes import HashEmbedder, MappingProvider

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
SCOPE = "project:improve"

N_DUPS = 50
DISTINCT = ["rust", "go", "python"]  # 3 genuinely distinct facts that must survive


def _fact(obj: str, *, predicate: str = "knows_language", statement: str | None = None) -> NewFact:
    # A MULTI predicate so many current rows for the same triple can coexist in the
    # ledger (the single-valued guard would otherwise forbid the planted duplicates).
    return NewFact(
        subject="nithish", predicate=predicate, object=obj,
        subject_key="nithish", predicate_key=normalize_key(predicate), object_key=normalize_key(obj),
        statement=statement or f"nithish knows {obj}", observed_at=T0, cardinality="multi",
    )


async def _plant(repo, scope_id: int) -> None:
    """Insert 50 exact-duplicate rows for one fact, plus 3 distinct facts, bypassing
    the resolver (which would touch instead of insert)."""
    dup = _fact("clojure")
    for _ in range(N_DUPS):
        await repo.apply(
            (InsertMemory(fact=dup, valid_from=dup.observed_at, valid_until=None),),
            scope_id=scope_id, fact=dup,
        )
    for obj in DISTINCT:
        f = _fact(obj)
        await repo.apply(
            (InsertMemory(fact=f, valid_from=f.observed_at, valid_until=None),),
            scope_id=scope_id, fact=f,
        )


@pytest.fixture(params=["exmem", "cognee"])
async def backend(request):
    if request.param == "exmem":
        embed = HashEmbedder(dim=64)
        store = SQLiteMetadataStore(":memory:")
        retriever = HybridRetriever(embed, store, RetrievalConfig(top_k=10))
        repo = SqliteRepository(":memory:")
        ladder = ExtractionLadder(
            {Stage.LOCAL_SMALL: MappingProvider(Stage.LOCAL_SMALL, {})},
            offline=True, budget=Budget(0.0),
        )
        svc = ExMemBackend(repo, retriever, ladder, store, offline=True, embedder=embed)
        yield svc
        await svc.close()
        return

    settings = get_settings()
    try:
        repo = await PgRepository(settings.pg_dsn, embedder=HashEmbedder(dim=768)).connect()
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres not reachable: {exc!r}")
    await run_migrations(settings)
    await repo.conn.execute(
        "TRUNCATE memory_episodes, memory_revisions, memories, episodes, scopes "
        "RESTART IDENTITY CASCADE"
    )
    ladder = ExtractionLadder(
        {Stage.LOCAL_SMALL: MappingProvider(Stage.LOCAL_SMALL, {})},
        offline=True, budget=Budget(0.0),
    )
    svc = CogneeBackend(repo, ladder, channel=NullCogneeChannel(), offline=True)
    yield svc
    await svc.close()


async def _current(backend) -> set[str]:
    name, kind, key = ("project:improve", "project", "improve")
    scope_id = await backend._repo.ensure_scope(name, kind=kind, key=key)
    rows = await backend._repo.facts(scope_id, limit=1_000_000)
    return {m.object_key for m in rows}


# --------------------------------------------------------------------------- #
async def test_dedup_collapses_and_loses_nothing(backend) -> None:
    scope_id = await backend._repo.ensure_scope("project:improve", kind="project", key="improve")
    await _plant(backend._repo, scope_id)

    # Before: 50 dup rows + 3 distinct = 53 current, but only 4 distinct facts.
    before = await backend._repo.facts(scope_id, limit=1_000_000)
    assert len(before) == N_DUPS + len(DISTINCT)
    assert await _current(backend) == {"clojure", *DISTINCT}

    report = await backend.improve(scope=SCOPE, ops=("dedup",))

    # Collapse count: the 49 redundant copies of the one fact were closed.
    assert report.deduped == N_DUPS - 1
    # No valid fact lost: every distinct fact — including the deduped one — is still current.
    after = await backend._repo.facts(scope_id, limit=1_000_000)
    assert len(after) == 1 + len(DISTINCT)
    assert await _current(backend) == {"clojure", *DISTINCT}


async def test_dedup_folds_seen_count(backend) -> None:
    scope_id = await backend._repo.ensure_scope("project:improve", kind="project", key="improve")
    await _plant(backend._repo, scope_id)

    await backend.improve(scope=SCOPE, ops=("dedup",))

    survivors = [
        m for m in await backend._repo.facts(scope_id, limit=1_000_000) if m.object_key == "clojure"
    ]
    assert len(survivors) == 1
    # The canonical absorbed every duplicate's seen_count (each planted row was 1).
    assert survivors[0].seen_count == N_DUPS


async def test_dry_run_changes_nothing(backend) -> None:
    scope_id = await backend._repo.ensure_scope("project:improve", kind="project", key="improve")
    await _plant(backend._repo, scope_id)

    report = await backend.improve(scope=SCOPE, ops=("dedup",), dry_run=True)
    assert report.deduped == N_DUPS - 1  # it still reports what it *would* collapse
    # ...but nothing was actually closed.
    after = await backend._repo.facts(scope_id, limit=1_000_000)
    assert len(after) == N_DUPS + len(DISTINCT)


async def test_prune_removes_only_consolidated_rows(backend) -> None:
    scope_id = await backend._repo.ensure_scope("project:improve", kind="project", key="improve")
    await _plant(backend._repo, scope_id)

    await backend.improve(scope=SCOPE, ops=("dedup", "prune"))

    # The 49 closed dup rows are hard-deleted; the canonical + distinct facts remain,
    # and they are the only rows left (no orphaned tombstones).
    every_version = await backend._repo.facts(scope_id, include_invalidated=True, limit=1_000_000)
    assert len(every_version) == 1 + len(DISTINCT)
    assert await _current(backend) == {"clojure", *DISTINCT}


# --------------------------------------------------------------------------- #
# merge (similarity) — exercised on ex_mem only, since it needs a clustering embedder.
# --------------------------------------------------------------------------- #
class _ClusterEmbedder:
    """Deterministic embedder that puts every statement mentioning 'globex' on the SAME
    unit vector (cosine 1.0) and everything else on its own — so near-dup company names
    with *different* object_keys cluster and merge, while distinct facts do not."""

    model = "cluster-test"

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def encode(self, texts):
        texts = list(texts)
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = 1 if "globex" in t.lower() else int.from_bytes(
                hashlib.sha256(t.encode()).digest()[:8], "little"
            )
            v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
            out[i] = v / (float(np.linalg.norm(v)) or 1.0)
        return out

    def encode_one(self, text):
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        return self._dim


async def test_merge_collapses_near_duplicates_by_similarity() -> None:
    embed = _ClusterEmbedder()
    store = SQLiteMetadataStore(":memory:")
    retriever = HybridRetriever(embed, store, RetrievalConfig(top_k=10))
    ladder = ExtractionLadder(
        {Stage.LOCAL_SMALL: MappingProvider(Stage.LOCAL_SMALL, {})}, offline=True, budget=Budget(0.0)
    )
    svc = ExMemBackend(SqliteRepository(":memory:"), retriever, ladder, store, offline=True, embedder=embed)
    try:
        scope_id = await svc._repo.ensure_scope("project:improve", kind="project", key="improve")
        # Two near-dup company names (different object_key) + two distinct past employers.
        for obj in ("Globex", "Globex Corp"):
            f = _fact(obj, predicate="worked_at", statement=f"nithish worked at {obj}")
            await svc._repo.apply(
                (InsertMemory(fact=f, valid_from=f.observed_at, valid_until=None),),
                scope_id=scope_id, fact=f,
            )
        for obj in ("Acme", "Initech"):
            f = _fact(obj, predicate="worked_at", statement=f"nithish worked at {obj}")
            await svc._repo.apply(
                (InsertMemory(fact=f, valid_from=f.observed_at, valid_until=None),),
                scope_id=scope_id, fact=f,
            )

        report = await svc.improve(scope=SCOPE, ops=("merge",))
        assert report.merged == 1  # the two Globex variants folded into one

        current = {m.object_key for m in await svc._repo.facts(scope_id, limit=1_000_000)}
        # Exactly one Globex variant remains, and both distinct employers survive.
        assert len(current & {"globex", "globex corp"}) == 1
        assert {"acme", "initech"} <= current
    finally:
        await svc.close()
