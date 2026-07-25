"""M1 verification: the migrations produce exactly the schema the design depends on.

Asserts the pgvector extension, every core table, each named load-bearing index,
and that the predicate registry is seeded. Requires a live Postgres (``make up``)
with migrations applied (``make migrate``). Skips cleanly if the DB is unreachable
so the rest of the suite still runs offline.
"""

from __future__ import annotations

import asyncpg
import pytest

from local_infra.backends.pg.migrate import run_migrations
from local_infra.config import get_settings

EXPECTED_TABLES = {
    "scopes",
    "episodes",
    "memories",
    "memory_episodes",
    "predicates",
    "memory_revisions",
    "extraction_jobs",
    "extraction_attempts",
    "schema_migrations",
}

EXPECTED_INDEXES = {
    "idx_mem_single_current",
    "idx_mem_vec_current",
    "idx_mem_triple_current",
    "idx_mem_tsv_current",
    "idx_mem_asof",
}


@pytest.fixture
async def conn() -> asyncpg.Connection:
    # Function-scoped: one connection per test, bound to that test's event loop
    # (pytest-asyncio gives each test its own loop by default).
    settings = get_settings()
    try:
        connection = await asyncpg.connect(settings.pg_dsn, timeout=5)
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres not reachable at {settings.pg_dsn}: {exc!r}")
    # Idempotent: brings a fresh DB up to date, no-op if already migrated.
    await run_migrations(settings)
    yield connection
    await connection.close()


async def test_vector_extension_present(conn: asyncpg.Connection) -> None:
    row = await conn.fetchrow("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    assert row is not None, "pgvector extension not installed"


async def test_all_core_tables_exist(conn: asyncpg.Connection) -> None:
    rows = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    )
    present = {r["tablename"] for r in rows}
    missing = EXPECTED_TABLES - present
    assert not missing, f"missing tables: {sorted(missing)}"


async def test_named_indexes_exist(conn: asyncpg.Connection) -> None:
    rows = await conn.fetch(
        "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
    )
    present = {r["indexname"] for r in rows}
    missing = EXPECTED_INDEXES - present
    assert not missing, f"missing indexes: {sorted(missing)}"


async def test_single_current_index_is_unique(conn: asyncpg.Connection) -> None:
    # The integrity guard must be UNIQUE or it enforces nothing.
    row = await conn.fetchrow(
        """
        SELECT ix.indisunique
        FROM pg_class c
        JOIN pg_index ix ON ix.indexrelid = c.oid
        WHERE c.relname = 'idx_mem_single_current'
        """
    )
    assert row is not None and row["indisunique"], "single-current guard is not UNIQUE"


async def test_embedding_column_is_768_dim(conn: asyncpg.Connection) -> None:
    # A dimension mismatch here would silently corrupt every similarity score.
    dim = await conn.fetchval(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'memories' AND a.attname = 'embedding'
        """
    )
    assert dim == 768, f"embedding vector dimension is {dim}, expected 768"


async def test_predicates_seeded(conn: asyncpg.Connection) -> None:
    total = await conn.fetchval("SELECT count(*) FROM predicates")
    assert total >= 25, f"predicate registry looks unseeded ({total} rows)"
    single = await conn.fetchval(
        "SELECT cardinality FROM predicates WHERE key = 'works_at'"
    )
    assert single == "single", "works_at must be single-valued"
