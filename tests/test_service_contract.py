"""M7 verification: the MemoryService contract, run against the ex_mem backend.

The whole point of the ``MemoryService`` seam is that this suite is backend-agnostic
— every test speaks only the ABC. It is parametrized over a backend name so M8 can add
``"cognee"`` to the fixture and re-run *these same tests unchanged*. Select the ex_mem
run with ``pytest -k exmem``.

Extraction and embeddings are faked (``MappingProvider`` + ``HashEmbedder``) so the
suite is fully offline and deterministic — no Ollama, no sockets. The temporal ledger,
the pure resolver, and the hybrid retriever are all real.

Flow mirrors the plan's acceptance path: remember -> recall -> contradict -> recall
(only current) -> time-travel -> forget (tombstone, history intact) -> history chain.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import asyncpg
import pytest

from local_infra.backends.cognee_backend import CogneeBackend
from local_infra.backends.cognee_channel import NullCogneeChannel
from local_infra.backends.exmem_backend import ExMemBackend
from local_infra.backends.pg.migrate import run_migrations
from local_infra.backends.pg.repository import PgRepository
from local_infra.backends.sqlite.repository import SqliteRepository
from local_infra.config import get_settings
from local_infra.exmem.config import RetrievalConfig
from local_infra.exmem.retrieval import HybridRetriever
from local_infra.exmem.store import SQLiteMetadataStore
from local_infra.extraction.budget import Budget
from local_infra.extraction.ladder import ExtractionLadder
from local_infra.extraction.providers.base import Stage

from tests._fakes import HashEmbedder, MappingProvider

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=30)

# Controlled inputs. Each maps to the fact(s) a real extractor would produce; every
# evidence_span is a verbatim substring of its content so groundedness scores 1.0.
WORK_ACME = "I work at Acme."
WORK_GLOBEX = "Actually I moved to Globex."
LANG_PY = "I know Python."
LANG_RUST = "I also know Rust."

MAPPING = {
    WORK_ACME: [
        dict(subject="I", predicate="works_at", object="Acme",
             evidence_span="I work at Acme", confidence=0.9),
    ],
    WORK_GLOBEX: [
        dict(subject="I", predicate="works_at", object="Globex",
             evidence_span="moved to Globex", confidence=0.9),
    ],
    LANG_PY: [
        dict(subject="I", predicate="knows_language", object="Python",
             evidence_span="I know Python", confidence=0.9),
    ],
    LANG_RUST: [
        dict(subject="I", predicate="knows_language", object="Rust",
             evidence_span="also know Rust", confidence=0.9),
    ],
}


def _fake_ladder():
    provider = MappingProvider(Stage.LOCAL_SMALL, MAPPING)
    return ExtractionLadder({Stage.LOCAL_SMALL: provider}, offline=True, budget=Budget(0.0))


@pytest.fixture(params=["exmem", "cognee"])
async def service(request):
    """A fresh MemoryService per test, wired offline with fakes. Parametrized over both
    backends: the same test bodies must pass against each (the seam's whole point)."""
    if request.param == "exmem":
        embed = HashEmbedder(dim=64)
        store = SQLiteMetadataStore(":memory:")
        retriever = HybridRetriever(embed, store, RetrievalConfig(top_k=10))
        # No embedder on the ledger: exact triple-key candidates only, so dedup/supersede
        # is decided by keys (not by the test's arbitrary hash geometry).
        repo = SqliteRepository(":memory:")
        svc = ExMemBackend(repo, retriever, _fake_ladder(), store, offline=True, embedder=embed)
        yield svc
        await svc.close()
        return

    # cognee: a REAL Postgres temporal ledger + a null graph channel (deterministic, no
    # LLM, no graph build) — recall correctness comes from the pg ledger, exactly as the
    # ex_mem suite fakes extraction/embeddings. 768-dim hash embedder so pgvector's
    # vector(768) column (and the dense recall channel) are exercised.
    settings = get_settings()
    try:
        repo = await PgRepository(settings.pg_dsn, embedder=HashEmbedder(dim=768)).connect()
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres not reachable: {exc!r}")
    await run_migrations(settings)
    # Isolate each test: wipe the ledger so hardcoded GLOBAL-scope writes never collide
    # across tests on the shared database (predicates seed is left intact).
    await repo.conn.execute(
        "TRUNCATE memory_episodes, memory_revisions, memories, episodes, scopes "
        "RESTART IDENTITY CASCADE"
    )
    svc = CogneeBackend(repo, _fake_ladder(), channel=NullCogneeChannel(), offline=True)
    yield svc
    await svc.close()


# --------------------------------------------------------------------------- #
async def test_remember_writes_episode_and_extracts_fact(service) -> None:
    r = await service.remember(WORK_ACME, occurred_at=T0, extract="sync")
    assert r.episode_id is not None
    assert r.extracted == 1 and len(r.fact_ids) == 1
    m = await service.get(r.fact_ids[0])
    assert m.object_key == "acme" and m.predicate_key == "works_at" and m.is_current


async def test_recall_returns_current_fact(service) -> None:
    await service.remember(WORK_ACME, occurred_at=T0, extract="sync")
    hits = await service.recall("where do I work")
    assert [h.memory.object_key for h in hits] == ["acme"]
    assert all(h.kind == "memory" for h in hits)


async def test_contradiction_supersedes_current_view(service) -> None:
    await service.remember(WORK_ACME, occurred_at=T0, extract="sync")
    await service.remember(WORK_GLOBEX, occurred_at=T1, extract="sync")

    hits = await service.recall("where do I work")
    objs = [h.memory.object_key for h in hits]
    assert objs == ["globex"]  # only the current fact — acme was superseded


async def test_time_travel_recall(service) -> None:
    await service.remember(WORK_ACME, occurred_at=T0, extract="sync")
    await service.remember(WORK_GLOBEX, occurred_at=T1, extract="sync")

    past = await service.recall("where do I work", as_of=T0 + timedelta(days=1))
    assert [h.memory.object_key for h in past] == ["acme"]
    now = await service.recall("where do I work")
    assert [h.memory.object_key for h in now] == ["globex"]


async def test_history_chains_versions(service) -> None:
    await service.remember(WORK_ACME, occurred_at=T0, extract="sync")
    await service.remember(WORK_GLOBEX, occurred_at=T1, extract="sync")

    hist = await service.history("I", "works_at")
    assert [m.object_key for m in hist] == ["acme", "globex"]  # oldest first
    assert hist[0].invalidated_by_id == hist[1].id
    assert hist[1].supersedes_id == hist[0].id


async def test_soft_forget_tombstones_but_keeps_history(service) -> None:
    await service.remember(WORK_ACME, occurred_at=T0, extract="sync")
    r = await service.remember(WORK_GLOBEX, occurred_at=T1, extract="sync")
    globex_id = r.fact_ids[0]

    f = await service.forget(globex_id)
    assert f.ok and not f.hard

    # No current work fact remains (acme superseded, globex tombstoned)...
    assert await service.recall("where do I work") == []
    # ...but the row is tombstoned, not deleted, and the chain is intact.
    assert (await service.get(globex_id)).valid_until is not None
    hist = await service.history("I", "works_at")
    assert [m.object_key for m in hist] == ["acme", "globex"]


async def test_hard_delete_removes_row(service) -> None:
    r = await service.remember(WORK_ACME, occurred_at=T0, extract="sync")
    mid = r.fact_ids[0]

    f = await service.forget(mid, hard=True)
    assert f.ok and f.hard
    assert await service.get(mid) is None
    assert await service.recall("where do I work") == []


async def test_multi_valued_predicates_coexist(service) -> None:
    await service.remember(LANG_PY, occurred_at=T0, extract="sync")
    await service.remember(LANG_RUST, occurred_at=T1, extract="sync")

    hits = await service.recall("what languages do I know", limit=10)
    assert sorted(h.memory.object_key for h in hits) == ["python", "rust"]


async def test_never_mode_writes_episode_only(service) -> None:
    r = await service.remember(WORK_ACME, occurred_at=T0, extract="never")
    assert r.extracted == 0 and r.fact_ids == []

    assert await service.recall("where do I work") == []  # no fact indexed
    eps = await service.recall("where do I work", include_episodes=True)
    assert any(h.kind == "episode" and "Acme" in h.text for h in eps)


async def test_health_counts_indexed_items(service) -> None:
    await service.remember(WORK_ACME, occurred_at=T0, extract="sync")
    h = await service.health()
    assert h.backend == service.backend_name and h.ok and h.offline
    assert h.episodes == 1 and h.memories == 1


async def test_scopes_are_isolated(service) -> None:
    await service.remember(WORK_ACME, scope="project:a", occurred_at=T0, extract="sync")
    await service.remember(WORK_GLOBEX, scope="project:b", occurred_at=T0, extract="sync")

    a = await service.recall("where do I work", scope="project:a")
    b = await service.recall("where do I work", scope="project:b")
    assert [h.memory.object_key for h in a] == ["acme"]
    assert [h.memory.object_key for h in b] == ["globex"]


# --------------------------------------------------------------------------- #
# cognee-specific: enrichment is best-effort and must never break the core. A real
# local model can trip cognee's structured-output summarizer during cognify (observed
# with llama3.1:8b); improve() must degrade to pg-only recall, not raise.
# --------------------------------------------------------------------------- #
class _RaisingChannel(NullCogneeChannel):
    name = "raising"

    async def cognify(self, dataset: str) -> int:
        raise RuntimeError("cognee summarizer produced invalid structured output")


async def test_cognee_improve_degrades_when_cognify_fails() -> None:
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
    svc = CogneeBackend(repo, _fake_ladder(), channel=_RaisingChannel(), offline=True)
    try:
        await svc.remember(WORK_ACME, occurred_at=T0, extract="sync")
        report = await svc.improve(ops=("reindex",))  # must not raise
        assert report.reindexed == 0
        assert any("cognify failed" in n for n in report.notes)
        # The authoritative pg channels still recall the fact.
        hits = await svc.recall("where do I work")
        assert [h.memory.object_key for h in hits] == ["acme"]
    finally:
        await svc.close()
