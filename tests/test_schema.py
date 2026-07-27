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
    # 005: documents
    "documents",
    "document_chunks",
    "document_jobs",
}

EXPECTED_INDEXES = {
    "idx_mem_single_current",
    "idx_mem_vec_current",
    "idx_mem_triple_current",
    "idx_mem_tsv_current",
    "idx_mem_asof",
    # 005: chunk retrieval — deliberately NOT partial (a chunk has no valid_until).
    "idx_doc_chunks_vec",
    "idx_doc_chunks_tsv",
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


# --- 005: documents -------------------------------------------------------- #


async def _scope(conn: asyncpg.Connection, name: str) -> int:
    """Create (or reuse) a scope. Deliberately not `SELECT ... WHERE name = 'global'`:
    the contract suite TRUNCATEs `scopes`, so the migration's seeded global row is not
    something another test file may assume is still there."""
    return int(
        await conn.fetchval(
            "INSERT INTO scopes (name, kind, key) VALUES ($1, 'project', $1) "
            "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
            name,
        )
    )


async def test_chunk_embedding_matches_the_memory_embedding_dimension(
    conn: asyncpg.Connection,
) -> None:
    """Chunks and memories are fused in one ranking, so they must be embedded by the
    same model — a dimension mismatch would make every cross-channel score meaningless."""
    dim = await conn.fetchval(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'document_chunks' AND a.attname = 'embedding'
        """
    )
    assert dim == 768, f"chunk vector dimension is {dim}, expected 768"


async def test_same_file_is_idempotent_per_scope_but_distinct_across_scopes(
    conn: asyncpg.Connection,
) -> None:
    """Two rows for one upload would double that document's weight in every recall."""
    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    one = await _scope(conn, "project:schema_doc_a")
    two = await _scope(conn, "project:schema_doc_b")
    insert = (
        "INSERT INTO documents (scope_id, filename, kind, sha256, size_bytes) "
        "VALUES ($1, 'a.txt', 'txt', $2, 10) RETURNING id"
    )
    try:
        first = await conn.fetchval(insert, one, sha)
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.fetchval(insert, one, sha)
        # Same bytes, different scope: a separate document. Scopes are the isolation
        # boundary, so sharing a row across them would leak one project into another.
        second = await conn.fetchval(insert, two, sha)
        assert first != second
    finally:
        await conn.execute("DELETE FROM documents WHERE sha256 = $1", sha)
        await conn.execute("DELETE FROM scopes WHERE id = ANY($1::bigint[])", [one, two])


async def test_deleting_a_document_cascades_to_its_chunks(conn: asyncpg.Connection) -> None:
    """Chunks are re-derivable from the blob, so CASCADE is right here — the opposite
    of memory_episodes, where RESTRICT protects irreplaceable provenance."""
    scope_id = await _scope(conn, "project:schema_doc_cascade")
    try:
        doc_id = await conn.fetchval(
            "INSERT INTO documents (scope_id, filename, kind, sha256, size_bytes) "
            "VALUES ($1, 'cascade.txt', 'txt', 'sha-cascade-test', 3) RETURNING id",
            scope_id,
        )
        await conn.execute(
            "INSERT INTO document_chunks (document_id, scope_id, chunk_index, content, "
            "char_start, char_end) VALUES ($1, $2, 0, 'body', 0, 4)",
            doc_id, scope_id,
        )
        await conn.execute(
            "INSERT INTO document_jobs (document_id, scope_id) VALUES ($1, $2)",
            doc_id, scope_id,
        )

        await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)

        assert await conn.fetchval(
            "SELECT count(*) FROM document_chunks WHERE document_id = $1", doc_id
        ) == 0
        assert await conn.fetchval(
            "SELECT count(*) FROM document_jobs WHERE document_id = $1", doc_id
        ) == 0
    finally:
        await conn.execute("DELETE FROM scopes WHERE id = $1", scope_id)
