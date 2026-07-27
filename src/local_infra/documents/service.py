"""Request-side document orchestration: upload, list, delete — and the ingest worker.

Everything slow lives behind the queue. ``upload`` does only bounded work (validate,
hash, write a blob, insert a row, enqueue) and returns 202; :class:`DocumentPipeline`
does the rest on the worker.

**Two store connections, on purpose.** ``PgDocumentStore`` wraps a single asyncpg
connection, and asyncpg raises ``another operation is in progress`` when two coroutines
share one. An ingest holds its connection for seconds while embedding, so the worker
gets its own — the same split ``LedgerBackend`` already makes between its request repo
and ``worker_repo``.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from local_infra.documents.blobs import BlobStore
from local_infra.documents.models import Document, DocumentChunk
from local_infra.documents.pipeline import DocumentPipeline
from local_infra.documents.store import DocumentStore
from local_infra.documents.validate import validate_upload
from local_infra.ingest.worker import ExtractionWorker


class DocumentsUnavailable(RuntimeError):
    """Raised (and translated to 501) when the active backend has no document store.

    A clear 501 rather than a 500 or an empty 200: "this backend cannot do documents"
    and "your document contains nothing" are different facts, and an empty 200 reads to
    a user exactly like broken retrieval.
    """


class DocumentService:
    def __init__(
        self,
        store: DocumentStore,
        blobs: BlobStore,
        *,
        worker_store: DocumentStore | None = None,
        pipeline: DocumentPipeline | None = None,
    ) -> None:
        self._store = store
        self._blobs = blobs
        self._worker_store = worker_store
        self._pipeline = pipeline
        self._worker: ExtractionWorker | None = None
        if pipeline is not None and worker_store is not None:
            self._worker = ExtractionWorker(pipeline, worker_store)

    @property
    def store(self) -> DocumentStore:
        return self._store

    async def start(self) -> None:
        if self._worker is not None:
            await self._worker.start()

    async def close(self) -> None:
        if self._worker is not None:
            await self._worker.stop()
        if self._worker_store is not None:
            await self._worker_store.close()
        await self._store.close()

    # -- write path --------------------------------------------------------- #

    async def upload(
        self,
        data: bytes,
        *,
        filename: str | None,
        scope: str,
        metadata: dict | None = None,
    ) -> tuple[Document, UUID | None]:
        """Validate, store the blob, register the document, enqueue ingestion.

        Returns ``(document, job_id)``; ``job_id`` is ``None`` when the same bytes were
        already ingested into this scope, since re-splitting and re-embedding an
        identical file would burn GPU time to produce byte-identical chunks.
        """
        kind, name = validate_upload(data, filename)
        sha = await asyncio.to_thread(self._blobs.put, data)

        document, created = await self._store.create(
            scope=scope,
            filename=name,
            kind=kind,
            sha256=sha,
            size_bytes=len(data),
            metadata=metadata,
        )
        # Re-enqueue a document that never finished: a failed ingest (Ollama was down)
        # or one interrupted mid-flight by a restart would otherwise be stuck forever
        # with no way to retry short of deleting and re-uploading.
        if created or document.status in ("failed", "pending"):
            job_id = await self._store.enqueue(document.id, document.scope_id)
            if not created:
                await self._store.mark(document.id, "pending", error=None)
                document.status = "pending"
                document.error = None
            return document, job_id
        return document, None

    async def delete(self, document_id: UUID) -> Document | None:
        """Delete the document; unlink its blob only if no other scope still uses it.

        The reference check is the database's, not the caller's: the same file uploaded
        into two projects is one blob, and unlinking on the first delete would leave the
        second document pointing at nothing — a failure that only surfaces later, on
        re-ingest, far from the delete that caused it.
        """
        document = await self._store.delete(document_id)
        if document is None:
            return None
        if not await self._store.sha_in_use(document.sha256):
            await asyncio.to_thread(self._blobs.unlink, document.sha256)
        return document

    # -- read path ---------------------------------------------------------- #

    async def get(self, document_id: UUID) -> Document | None:
        return await self._store.get(document_id)

    async def list_documents(self, scope: str, *, limit: int = 100) -> list[Document]:
        return await self._store.list_documents(scope, limit=limit)

    async def chunks(self, document_id: UUID) -> list[DocumentChunk]:
        return await self._store.chunks_for(document_id)
