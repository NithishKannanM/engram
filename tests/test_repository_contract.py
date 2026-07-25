"""M4 verification: one contract suite, run against BOTH repositories identically.

The parametrized ``repo`` fixture yields a SqliteRepository (always) and a
PgRepository (skipped if Postgres is unreachable). Every assertion is
backend-agnostic — that identical pass/fail is the whole point of the seam.

Flow mirrors production: add_episode -> find_candidates -> resolve -> apply.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import asyncpg
import pytest

from local_infra.backends.pg.migrate import run_migrations
from local_infra.backends.pg.repository import PgRepository
from local_infra.backends.sqlite.repository import SqliteRepository
from local_infra.core.models import InsertMemory, NewFact
from local_infra.core.normalize import normalize_key
from local_infra.core.repository import IntegrityError
from local_infra.core.temporal import resolve
from local_infra.config import get_settings

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)


def make_fact(obj, *, predicate="works_at", cardinality="single", observed=T0, subject="nithish"):
    return NewFact(
        subject=subject, predicate=predicate, object=obj,
        subject_key=normalize_key(subject), predicate_key=normalize_key(predicate),
        object_key=normalize_key(obj), statement=f"{subject} {predicate} {obj}",
        observed_at=observed, cardinality=cardinality,
    )


async def remember(repo, scope_id, fact):
    """add_episode -> find_candidates -> resolve -> apply. Returns (resolution, primary_id)."""
    ep = await repo.add_episode(scope_id, fact.statement)
    cands = await repo.find_candidates(scope_id, fact)
    res = resolve(fact, cands)
    assert not res.needs_judge, "contract tests use rule-decidable facts only"
    mid = await repo.apply(res.intents, scope_id=scope_id, fact=fact, episode_id=ep.id)
    return res, mid


# --------------------------------------------------------------------------- #
@pytest.fixture(params=["sqlite", "pg"])
async def repo(request):
    if request.param == "sqlite":
        r = SqliteRepository(":memory:")
        yield r
        await r.close()
        return
    settings = get_settings()
    try:
        r = await PgRepository(settings.pg_dsn).connect()
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres not reachable: {exc!r}")
    await run_migrations(settings)
    yield r
    await r.close()


@pytest.fixture
async def scope(repo):
    # Unique per test so pg rows never collide across tests or reruns.
    return await repo.ensure_scope(f"test-{uuid4()}", kind="project", key="t")


# --------------------------------------------------------------------------- #
async def test_insert_and_get(repo, scope) -> None:
    res, mid = await remember(repo, scope, make_fact("globex"))
    assert res.inserts_new and mid is not None
    m = await repo.get_memory(mid)
    assert m.object_key == "globex"
    assert m.valid_until is None and m.seen_count == 1
    assert m.attrs.get("cardinality") == "single"


async def test_supersede_closes_old_and_chains(repo, scope) -> None:
    _, acme_id = await remember(repo, scope, make_fact("acme", observed=T0))
    _, globex_id = await remember(repo, scope, make_fact("globex", observed=T0 + timedelta(days=30)))

    current = await repo.facts(scope)
    assert len(current) == 1 and current[0].object_key == "globex"

    acme = await repo.get_memory(acme_id)
    globex = await repo.get_memory(globex_id)
    assert acme.valid_until is not None
    assert acme.invalidated_by_id == globex_id
    assert globex.supersedes_id == acme_id


async def test_single_valued_guard_blocks_two_current(repo, scope) -> None:
    await remember(repo, scope, make_fact("acme"))
    # Bypass resolve: insert a second current single-valued row without closing acme.
    rogue = make_fact("globex")
    intents = (InsertMemory(fact=rogue, valid_from=rogue.observed_at, valid_until=None),)
    with pytest.raises(IntegrityError):
        await repo.apply(intents, scope_id=scope, fact=rogue)
    # The guard protected the ledger: acme is still the one current fact.
    current = await repo.facts(scope)
    assert len(current) == 1 and current[0].object_key == "acme"


async def test_history_and_as_of(repo, scope) -> None:
    t1 = T0 + timedelta(days=30)
    await remember(repo, scope, make_fact("acme", observed=T0))
    await remember(repo, scope, make_fact("globex", observed=t1))

    hist = await repo.history(scope, normalize_key("nithish"), normalize_key("works_at"))
    assert [m.object_key for m in hist] == ["acme", "globex"]  # oldest first

    before = await repo.facts(scope, as_of=T0 + timedelta(days=1))
    after = await repo.facts(scope, as_of=t1 + timedelta(days=1))
    assert [m.object_key for m in before] == ["acme"]
    assert [m.object_key for m in after] == ["globex"]


async def test_retroactive_insert_spares_present(repo, scope) -> None:
    t1 = T0 + timedelta(days=30)
    await remember(repo, scope, make_fact("globex", observed=t1))  # current belief
    # Learned now but true earlier: must not disturb the present.
    await remember(repo, scope, make_fact("acme", observed=T0))

    current = await repo.facts(scope)
    assert [m.object_key for m in current] == ["globex"]
    mid = await repo.facts(scope, as_of=T0 + timedelta(days=1))
    assert [m.object_key for m in mid] == ["acme"]


async def test_duplicate_touch_bumps_seen_count(repo, scope) -> None:
    _, mid = await remember(repo, scope, make_fact("globex"))
    res2, mid2 = await remember(repo, scope, make_fact("globex"))
    assert not res2.inserts_new and mid2 == mid

    current = await repo.facts(scope)
    assert len(current) == 1
    assert (await repo.get_memory(mid)).seen_count == 2


async def test_hard_delete(repo, scope) -> None:
    _, mid = await remember(repo, scope, make_fact("globex"))
    assert await repo.hard_delete(mid) is True
    assert await repo.get_memory(mid) is None
