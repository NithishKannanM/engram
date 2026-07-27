"""Postgres implementation of :class:`DocumentStore` (asyncpg + pgvector).

Mirrors ``PgRepository``'s shape — ``_vec_literal``, ``dense_ranked``, ``sparse_ranked``,
the jsonb codec — but on its **own connection**. ``PgRepository`` holds a single asyncpg
connection, and asyncpg raises ``another operation is in progress`` if two coroutines use
one concurrently; a document ingest holds its connection for seconds while embedding, so
sharing would turn every concurrent recall into a 500. (A pool refactor across the whole
backend is the right long-term answer and is flagged as a follow-up, not done here.)

For the same reason the request path and the ingest worker each construct their own
store — exactly the split ``PgRepository``/``worker_repo`` already makes for the ledger.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from uuid import UUID

import asyncpg

from local_infra.core.service import parse_scope
from local_infra.documents.models import (
    ChunkHit,
    Document,
    DocumentChunk,
    DocumentJob,
    PendingChunk,
)
from local_infra.embed.base import Embedder

_DOC_FIELDS = (
    "id, scope_id, filename, kind, sha256, size_bytes, pages, chunk_count, status, "
    "error, metadata, created_at, updated_at"
)
#: The same fields plus the scope *name*, which lives on `scopes` and is what callers
#: actually pass around ("project:web", not a numeric id).
_DOC_COLS = ", ".join(f"d.{f}" for f in _DOC_FIELDS.split(", ")) + ", s.name AS scope"

_CHUNK_COLS = (
    "id, document_id, scope_id, chunk_index, content, char_start, char_end, "
    "page_from, tokens"
)


def _document(row: asyncpg.Record, scope: str | None = None) -> Document:
    """``scope`` is supplied when the row came from an ``INSERT ... RETURNING`` (which
    cannot join `scopes`); joined SELECTs carry it on the row."""
    return Document(
        id=row["id"],
        scope_id=int(row["scope_id"]),
        scope=scope if scope is not None else row["scope"],
        filename=row["filename"],
        kind=row["kind"],
        sha256=row["sha256"],
        size_bytes=int(row["size_bytes"]),
        status=row["status"],
        pages=row["pages"],
        chunk_count=int(row["chunk_count"]),
        error=row["error"],
        metadata=row["metadata"] or {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _chunk(row: asyncpg.Record) -> DocumentChunk:
    return DocumentChunk(
        id=row["id"],
        document_id=row["document_id"],
        scope_id=int(row["scope_id"]),
        chunk_index=int(row["chunk_index"]),
        content=row["content"],
        char_start=int(row["char_start"]),
        char_end=int(row["char_end"]),
        page_from=row["page_from"],
        tokens=int(row["tokens"]),
    )


class PgDocumentStore:
    def __init__(
        self,
        dsn: str,
        *,
        embedder: Embedder | None = None,
        worker_id: str = "doc-worker-0",
    ) -> None:
        self._dsn = dsn
        self._embedder = embedder
        self._worker_id = worker_id
        self._conn: asyncpg.Connection | None = None

    async def connect(self) -> "PgDocumentStore":
        self._conn = await asyncpg.connect(self._dsn)
        await self._conn.set_type_codec(
            "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )
        return self

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> asyncpg.Connection:
        if self._conn is None:
            raise RuntimeError("PgDocumentStore.connect() was not awaited")
        return self._conn

    @property
    def embedder(self) -> Embedder | None:
        return self._embedder

    def _vec_literal(self, values: Sequence[float]) -> str:
        return "[" + ",".join(f"{x:.7f}" for x in values) + "]"

    async def _scope_id(self, scope: str) -> int:
        """Resolve (creating if needed) — a document may be the first thing ever written
        to a project scope, so this cannot assume the ledger got there first."""
        name, kind, key = parse_scope(scope)
        row = await self.conn.fetchrow("SELECT id FROM scopes WHERE name = $1", name)
        if row:
            return int(row["id"])
        return int(
            await self.conn.fetchval(
                "INSERT INTO scopes (name, kind, key) VALUES ($1,$2,$3) "
                "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
                name, kind, key,
            )
        )

    # -- documents ---------------------------------------------------------- #

    async def create(
        self,
        *,
        scope: str,
        filename: str,
        kind: str,
        sha256: str,
        size_bytes: int,
        metadata: dict | None = None,
    ) -> tuple[Document, bool]:
        scope_id = await self._scope_id(scope)
        # Two statements rather than one data-modifying CTE: a CTE's INSERT is NOT
        # visible to the same statement's SELECT (they share one snapshot), so the
        # `WITH ins AS (INSERT ...) SELECT FROM documents` form returns NOTHING for a
        # brand-new row — silently, and only for the case that matters most.
        row = await self.conn.fetchrow(
            f"INSERT INTO documents (scope_id, filename, kind, sha256, size_bytes, metadata) "
            f"VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (scope_id, sha256) DO NOTHING "
            f"RETURNING {_DOC_FIELDS}",
            scope_id, filename, kind, sha256, size_bytes, metadata or {},
        )
        if row is not None:
            return _document(row, scope), True

        existing = await self.conn.fetchrow(
            f"SELECT {_DOC_COLS} FROM documents d JOIN scopes s ON s.id = d.scope_id "
            "WHERE d.scope_id = $1 AND d.sha256 = $2",
            scope_id, sha256,
        )
        return _document(existing), False

    async def get(self, document_id: UUID) -> Document | None:
        row = await self.conn.fetchrow(
            f"SELECT {_DOC_COLS} FROM documents d JOIN scopes s ON s.id = d.scope_id "
            "WHERE d.id = $1",
            document_id,
        )
        return _document(row) if row else None

    async def list_documents(self, scope: str, *, limit: int = 100) -> list[Document]:
        name, _, _ = parse_scope(scope)
        rows = await self.conn.fetch(
            f"SELECT {_DOC_COLS} FROM documents d JOIN scopes s ON s.id = d.scope_id "
            "WHERE s.name = $1 ORDER BY d.created_at DESC LIMIT $2",
            name, limit,
        )
        return [_document(r) for r in rows]

    async def mark(
        self,
        document_id: UUID,
        status: str,
        *,
        error: str | None = None,
        pages: int | None = None,
        chunk_count: int | None = None,
    ) -> None:
        # COALESCE so a status-only update never blanks pages/chunk_count that an
        # earlier step already established.
        await self.conn.execute(
            "UPDATE documents SET status = $2, error = $3, "
            "pages = COALESCE($4, pages), chunk_count = COALESCE($5, chunk_count), "
            "updated_at = now() WHERE id = $1",
            document_id, status, error, pages, chunk_count,
        )

    async def delete(self, document_id: UUID) -> Document | None:
        document = await self.get(document_id)
        if document is None:
            return None
        # Chunks and jobs cascade (they are derived and re-derivable).
        await self.conn.execute("DELETE FROM documents WHERE id = $1", document_id)
        return document

    async def sha_in_use(self, sha256: str) -> bool:
        return bool(
            await self.conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM documents WHERE sha256 = $1)", sha256
            )
        )

    # -- chunks ------------------------------------------------------------- #

    async def replace_chunks(
        self, document_id: UUID, scope_id: int, chunks: Sequence[PendingChunk]
    ) -> int:
        model = getattr(self._embedder, "model", None) if self._embedder else None
        async with self.conn.transaction():
            await self.conn.execute(
                "DELETE FROM document_chunks WHERE document_id = $1", document_id
            )
            for chunk in chunks:
                await self.conn.execute(
                    "INSERT INTO document_chunks (document_id, scope_id, chunk_index, "
                    "content, char_start, char_end, page_from, tokens, embedding, embed_model) "
                    "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::vector,$10)",
                    document_id,
                    scope_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.page_from,
                    chunk.tokens,
                    self._vec_literal(chunk.embedding) if chunk.embedding else None,
                    model if chunk.embedding else None,
                )
        return len(chunks)

    async def chunks_for(self, document_id: UUID) -> list[DocumentChunk]:
        rows = await self.conn.fetch(
            f"SELECT {_CHUNK_COLS} FROM document_chunks WHERE document_id = $1 "
            "ORDER BY chunk_index",
            document_id,
        )
        return [_chunk(r) for r in rows]

    async def hydrate(self, chunk_ids: Sequence[UUID]) -> list[ChunkHit]:
        ids = list(chunk_ids)
        if not ids:
            return []
        rows = await self.conn.fetch(
            f"SELECT {', '.join('c.' + c for c in _CHUNK_COLS.split(', '))}, "
            "d.filename AS filename "
            "FROM document_chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE c.id = ANY($1::uuid[])",
            ids,
        )
        by_id = {r["id"]: r for r in rows}
        hits: list[ChunkHit] = []
        # Caller order is the fused ranking; a SQL ORDER BY cannot reproduce it, and
        # re-sorting here by anything else would silently discard the ranking.
        for chunk_id in ids:
            row = by_id.get(chunk_id)
            if row is None:
                continue  # deleted between ranking and hydration
            chunk = _chunk(row)
            hits.append(
                ChunkHit(
                    score=0.0,
                    chunk=chunk,
                    document_id=chunk.document_id,
                    filename=row["filename"],
                    page_from=chunk.page_from,
                )
            )
        return hits

    async def dense_ranked(self, scope: str, query: str, *, limit: int = 10) -> list[UUID]:
        """Chunk ids by pgvector cosine. Empty without an embedder — the sparse channel
        still answers, so retrieval degrades rather than failing."""
        if self._embedder is None:
            return []
        name, _, _ = parse_scope(scope)
        qvec = self._vec_literal(self._embedder.encode_one(query).tolist())
        rows = await self.conn.fetch(
            "SELECT c.id FROM document_chunks c JOIN scopes s ON s.id = c.scope_id "
            "WHERE s.name = $1 AND c.embedding IS NOT NULL "
            "ORDER BY c.embedding <=> $2::vector LIMIT $3",
            name, qvec, limit,
        )
        return [r["id"] for r in rows]

    async def sparse_ranked(self, scope: str, query: str, *, limit: int = 10) -> list[UUID]:
        name, _, _ = parse_scope(scope)
        rows = await self.conn.fetch(
            "SELECT c.id FROM document_chunks c JOIN scopes s ON s.id = c.scope_id "
            "WHERE s.name = $1 AND c.tsv @@ plainto_tsquery('english', $2) "
            "ORDER BY ts_rank(c.tsv, plainto_tsquery('english', $2)) DESC LIMIT $3",
            name, query, limit,
        )
        return [r["id"] for r in rows]

    # -- jobs --------------------------------------------------------------- #

    _CLAIM = (
        "WITH j AS ("
        "  SELECT id FROM document_jobs WHERE status IN ('queued', 'deferred') "
        "  ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1"
        ") "
        "UPDATE document_jobs e "
        "SET status = 'running', locked_by = $1, locked_at = now(), "
        "    attempts = attempts + 1, updated_at = now() "
        "FROM j WHERE e.id = j.id "
        "RETURNING e.id, e.document_id, e.scope_id"
    )

    async def enqueue(self, document_id: UUID, scope_id: int) -> UUID:
        return await self.conn.fetchval(
            "INSERT INTO document_jobs (document_id, scope_id) VALUES ($1,$2) RETURNING id",
            document_id, scope_id,
        )

    async def claim(self) -> DocumentJob | None:
        row = await self.conn.fetchrow(self._CLAIM, self._worker_id)
        if row is None:
            return None
        return DocumentJob(
            id=row["id"],
            document_id=row["document_id"],
            scope_id=int(row["scope_id"]),
        )

    async def complete(
        self, job_id: UUID, *, status: str = "done", error: str | None = None
    ) -> None:
        # The job row cascades away with its document, so a delete during processing
        # makes this a harmless no-op rather than an error.
        await self.conn.execute(
            "UPDATE document_jobs SET status = $2, last_error = $3, locked_by = NULL, "
            "updated_at = now() WHERE id = $1",
            job_id, status, error,
        )

    async def job_status(self, job_id: UUID) -> str | None:
        return await self.conn.fetchval(
            "SELECT status FROM document_jobs WHERE id = $1", job_id
        )
