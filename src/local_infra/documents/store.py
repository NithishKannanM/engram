"""The ``DocumentStore`` protocol — a *separate* seam from ``Repository``.

Deliberately not extra methods on ``core/repository.py::Repository``: that protocol is
implemented by both SQLite and Postgres and is exercised against both in
``test_repository_contract.py``. Adding document methods there would force either a
second SQLite implementation (with no vector column to put embeddings in) or a stub that
silently answers "no documents" — which reads to a user exactly like broken retrieval.

Documents are a Postgres-only capability. One implementation, ``PgDocumentStore``; the
exmem backend has none and the routes answer 501 with a reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from local_infra.documents.models import (
    ChunkHit,
    Document,
    DocumentChunk,
    DocumentJob,
    PendingChunk,
)


class DocumentStore(Protocol):
    async def close(self) -> None: ...

    # -- documents --

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
        """Insert (or return the existing) document for ``(scope, sha256)``.

        The bool is ``True`` when a row was created. Re-uploading the same bytes into
        the same scope is idempotent by design — two rows would double that document's
        contribution to every ranking.
        """
        ...

    async def get(self, document_id: UUID) -> Document | None: ...

    async def list_documents(self, scope: str, *, limit: int = 100) -> list[Document]: ...

    async def mark(
        self,
        document_id: UUID,
        status: str,
        *,
        error: str | None = None,
        pages: int | None = None,
        chunk_count: int | None = None,
    ) -> None: ...

    async def delete(self, document_id: UUID) -> Document | None:
        """Delete the row (chunks and jobs cascade) and return what was deleted, so the
        caller knows which sha256 to consider unlinking."""
        ...

    async def sha_in_use(self, sha256: str) -> bool:
        """Is any document — in any scope — still backed by this blob?"""
        ...

    # -- chunks --

    async def replace_chunks(
        self, document_id: UUID, scope_id: int, chunks: Sequence[PendingChunk]
    ) -> int:
        """Atomically swap this document's chunk set. Re-ingestion is then repeatable
        rather than additive."""
        ...

    async def chunks_for(self, document_id: UUID) -> list[DocumentChunk]: ...

    async def hydrate(self, chunk_ids: Sequence[UUID]) -> list[ChunkHit]:
        """Load chunks by id, preserving the caller's order — the fused ranking's."""
        ...

    async def dense_ranked(self, scope: str, query: str, *, limit: int = 10) -> list[UUID]: ...

    async def sparse_ranked(self, scope: str, query: str, *, limit: int = 10) -> list[UUID]: ...

    # -- jobs --

    async def enqueue(self, document_id: UUID, scope_id: int) -> UUID: ...

    async def claim(self) -> DocumentJob | None: ...

    async def complete(
        self, job_id: UUID, *, status: str = "done", error: str | None = None
    ) -> None: ...

    async def job_status(self, job_id: UUID) -> str | None: ...
