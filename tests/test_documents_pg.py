"""``PgDocumentStore`` against real Postgres + pgvector.

The `FakeDocumentStore` proves the plumbing; this proves the SQL — the idempotent
upsert, the pgvector and tsvector rankings, the CASCADE, and the SKIP LOCKED claim.
Skips cleanly when Postgres is unreachable, like the other pg-backed suites.

Each test runs in its own uniquely-named scope rather than truncating shared tables:
the ledger suites share this database, and a TRUNCATE here would delete their rows
mid-run.
"""

from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

from local_infra.backends.pg.documents import PgDocumentStore
from local_infra.backends.pg.migrate import run_migrations
from local_infra.config import get_settings
from local_infra.documents.models import PendingChunk

from tests._fakes import HashEmbedder

# The chunk column is vector(768) — a fake at any other width would be rejected by
# Postgres, which is itself the check that both sides agree on the embedding model.
EMBED_DIM = 768


@pytest.fixture
async def store():
    settings = get_settings()
    try:
        await run_migrations(settings)
        s = await PgDocumentStore(
            settings.pg_dsn, embedder=HashEmbedder(dim=EMBED_DIM)
        ).connect()
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres not reachable: {exc!r}")
    yield s
    await s.close()


@pytest.fixture
def scope() -> str:
    return f"project:doc_test_{uuid4().hex[:10]}"


def _chunks(store: PgDocumentStore, texts: list[str]) -> list[PendingChunk]:
    vectors = store.embedder.encode(texts)
    offset = 0
    out = []
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        out.append(
            PendingChunk(
                chunk_index=i,
                content=text,
                char_start=offset,
                char_end=offset + len(text),
                page_from=i + 1,
                tokens=len(text) // 4,
                embedding=vec.tolist(),
            )
        )
        offset += len(text) + 2
    return out


async def _cleanup(store: PgDocumentStore, scope: str) -> None:
    await store.conn.execute(
        "DELETE FROM documents WHERE scope_id IN (SELECT id FROM scopes WHERE name = $1)",
        scope,
    )
    await store.conn.execute("DELETE FROM scopes WHERE name = $1", scope)


async def test_create_is_idempotent_per_scope(store: PgDocumentStore, scope: str):
    try:
        first, created = await store.create(
            scope=scope, filename="a.md", kind="md", sha256="sha-a", size_bytes=10
        )
        assert created is True

        again, created_again = await store.create(
            scope=scope, filename="a.md", kind="md", sha256="sha-a", size_bytes=10
        )
        assert created_again is False
        assert again.id == first.id
        assert again.scope == scope
    finally:
        await _cleanup(store, scope)


async def test_chunks_round_trip_with_offsets_and_pages(store: PgDocumentStore, scope: str):
    try:
        document, _ = await store.create(
            scope=scope, filename="h.md", kind="md", sha256="sha-b", size_bytes=99
        )
        written = await store.replace_chunks(
            document.id,
            document.scope_id,
            _chunks(store, ["deployment window is Tuesday", "rollbacks use make rollback"]),
        )
        assert written == 2

        chunks = await store.chunks_for(document.id)
        assert [c.chunk_index for c in chunks] == [0, 1]
        assert chunks[0].char_start == 0 and chunks[0].char_end > 0
        assert chunks[0].page_from == 1 and chunks[1].page_from == 2
    finally:
        await _cleanup(store, scope)


async def test_replace_chunks_is_a_swap_not_an_append(store: PgDocumentStore, scope: str):
    """Re-ingesting must be repeatable. Appending would double every chunk's weight in
    the ranking on each retry."""
    try:
        document, _ = await store.create(
            scope=scope, filename="h.md", kind="md", sha256="sha-c", size_bytes=99
        )
        await store.replace_chunks(document.id, document.scope_id, _chunks(store, ["one", "two"]))
        await store.replace_chunks(document.id, document.scope_id, _chunks(store, ["three"]))

        chunks = await store.chunks_for(document.id)
        assert [c.content for c in chunks] == ["three"]
    finally:
        await _cleanup(store, scope)


async def test_dense_and_sparse_rankings_find_the_right_chunk(
    store: PgDocumentStore, scope: str
):
    try:
        document, _ = await store.create(
            scope=scope, filename="h.md", kind="md", sha256="sha-d", size_bytes=99
        )
        texts = [
            "The deployment window is every Tuesday at 14:00 UTC.",
            "Rollbacks are performed with make rollback.",
            "On-call rotation is weekly and starts on Monday.",
        ]
        await store.replace_chunks(document.id, document.scope_id, _chunks(store, texts))

        # HashEmbedder gives identical text cosine 1.0, so an exact query is the only
        # meaningful dense probe — it proves the pgvector operator and the scope join,
        # which is what this test is for.
        dense = await store.dense_ranked(scope, texts[1], limit=3)
        assert dense, "pgvector returned nothing"
        top = {c.id: c for c in await store.chunks_for(document.id)}
        assert top[dense[0]].content == texts[1]

        sparse = await store.sparse_ranked(scope, "rollback", limit=3)
        assert sparse and top[sparse[0]].content == texts[1]
    finally:
        await _cleanup(store, scope)


async def test_rankings_do_not_leak_across_scopes(store: PgDocumentStore, scope: str):
    """Scopes are the isolation boundary. A chunk surfacing in another project's search
    is the one failure mode of this feature that is genuinely a privacy problem."""
    other = f"{scope}_other"
    try:
        mine, _ = await store.create(
            scope=scope, filename="mine.md", kind="md", sha256="sha-e", size_bytes=10
        )
        theirs, _ = await store.create(
            scope=other, filename="theirs.md", kind="md", sha256="sha-f", size_bytes=10
        )
        await store.replace_chunks(
            mine.id, mine.scope_id, _chunks(store, ["alpha secret rollback plan"])
        )
        await store.replace_chunks(
            theirs.id, theirs.scope_id, _chunks(store, ["beta secret rollback plan"])
        )

        ids = await store.sparse_ranked(scope, "rollback", limit=10)
        hits = await store.hydrate(ids)
        assert hits and all(h.document_id == mine.id for h in hits)
    finally:
        await _cleanup(store, scope)
        await _cleanup(store, other)


async def test_hydrate_preserves_the_callers_ranking(store: PgDocumentStore, scope: str):
    """The caller's order IS the fused ranking. A SQL ORDER BY cannot reproduce it, so
    re-sorting here would silently discard the fusion."""
    try:
        document, _ = await store.create(
            scope=scope, filename="h.md", kind="md", sha256="sha-g", size_bytes=99
        )
        await store.replace_chunks(
            document.id, document.scope_id, _chunks(store, ["first", "second", "third"])
        )
        chunks = await store.chunks_for(document.id)
        reversed_ids = [c.id for c in reversed(chunks)]

        hits = await store.hydrate(reversed_ids)
        assert [h.chunk.id for h in hits] == reversed_ids
        assert hits[0].filename == "h.md"
    finally:
        await _cleanup(store, scope)


async def test_hydrate_skips_ids_deleted_since_ranking(store: PgDocumentStore, scope: str):
    try:
        document, _ = await store.create(
            scope=scope, filename="h.md", kind="md", sha256="sha-h", size_bytes=99
        )
        await store.replace_chunks(document.id, document.scope_id, _chunks(store, ["only"]))
        real = (await store.chunks_for(document.id))[0].id

        hits = await store.hydrate([uuid4(), real])
        assert [h.chunk.id for h in hits] == [real]
    finally:
        await _cleanup(store, scope)


async def test_delete_cascades_and_reports_blob_reuse(store: PgDocumentStore, scope: str):
    """`sha_in_use` is what stops a delete in one scope from unlinking the blob another
    scope still points at."""
    other = f"{scope}_other"
    try:
        mine, _ = await store.create(
            scope=scope, filename="s.md", kind="md", sha256="sha-shared", size_bytes=10
        )
        await store.create(
            scope=other, filename="s.md", kind="md", sha256="sha-shared", size_bytes=10
        )
        await store.replace_chunks(mine.id, mine.scope_id, _chunks(store, ["body"]))

        deleted = await store.delete(mine.id)
        assert deleted is not None and deleted.sha256 == "sha-shared"
        assert await store.chunks_for(mine.id) == []
        assert await store.sha_in_use("sha-shared") is True
    finally:
        await _cleanup(store, scope)
        await _cleanup(store, other)


async def test_job_is_claimed_once(store: PgDocumentStore, scope: str):
    """FOR UPDATE SKIP LOCKED: a second claim must not hand out the same job."""
    try:
        document, _ = await store.create(
            scope=scope, filename="j.md", kind="md", sha256="sha-j", size_bytes=10
        )
        job_id = await store.enqueue(document.id, document.scope_id)
        assert await store.job_status(job_id) == "queued"

        first = await store.claim()
        assert first is not None and first.id == job_id
        assert first.document_id == document.id
        assert await store.claim() is None

        await store.complete(job_id, status="done")
        assert await store.job_status(job_id) == "done"
    finally:
        await _cleanup(store, scope)


async def test_mark_does_not_blank_fields_it_was_not_given(
    store: PgDocumentStore, scope: str
):
    """A later status-only update must not wipe the page count an earlier step set —
    that would show '0 pages' on a perfectly good document."""
    try:
        document, _ = await store.create(
            scope=scope, filename="m.md", kind="md", sha256="sha-m", size_bytes=10
        )
        await store.mark(document.id, "ready", pages=12, chunk_count=7)
        await store.mark(document.id, "processing")

        refreshed = await store.get(document.id)
        assert refreshed.status == "processing"
        assert refreshed.pages == 12 and refreshed.chunk_count == 7
    finally:
        await _cleanup(store, scope)
