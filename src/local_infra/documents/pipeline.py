"""Document ingestion: blob → text → chunks → embeddings → store.

Runs **in the background worker, never in the request**. Ingesting a 40-page PDF means
one PDF parse plus dozens of embedding round-trips; doing that inside ``POST /documents``
would hold the HTTP connection for a minute and time out behind any proxy. The route
does bounded work and returns 202; this runs after.

Three properties worth stating outright, because each one hides a failure that is
invisible until it bites:

* **Every blocking call is in ``asyncio.to_thread``.** pypdf, python-docx and
  ``OllamaEmbedder`` are all synchronous; a direct call blocks the event loop, which
  stalls every other request in the process — including the health check.
* **Embeddings are batched.** ``OllamaEmbedder`` is one httpx client with a 60s timeout;
  a single call with 800 chunks exceeds it and fails the *whole* document.
* **The GPU lock is taken per batch, not per document.** Holding it for an entire
  document would block chat for the full ingest; releasing between batches lets a chat
  turn interleave at the cost of a little embedding throughput.
"""

from __future__ import annotations

import asyncio
import logging

import asyncpg

from local_infra.documents.blobs import BlobStore
from local_infra.documents.models import DocumentJob, PendingChunk
from local_infra.documents.reader import extract_text
from local_infra.documents.splitter import page_of, split_text
from local_infra.documents.store import DocumentStore
from local_infra.documents.validate import ValidationError
from local_infra.embed.base import Embedder
from local_infra.llm.gpu import OLLAMA_LOCK

log = logging.getLogger(__name__)

#: One embedding request per batch. Sized against `OllamaEmbedder`'s 60s timeout with
#: margin for a cold model load, not against any throughput measurement.
EMBED_BATCH = 32


class DocumentPipeline:
    def __init__(
        self,
        store: DocumentStore,
        blobs: BlobStore,
        *,
        embedder: Embedder | None = None,
        max_tokens: int = 350,
        min_tokens: int = 80,
        overlap_tokens: int = 48,
    ) -> None:
        self._store = store
        self._blobs = blobs
        self._embedder = embedder
        self._max_tokens = max_tokens
        self._min_tokens = min_tokens
        self._overlap_tokens = overlap_tokens

    async def __call__(self, job: DocumentJob) -> None:
        """Worker entry point. Raising marks the job failed; the document is marked
        failed here too so the *user-visible* status carries the reason, not just the
        job row nobody surfaces."""
        try:
            await self.process(job)
        except asyncpg.ForeignKeyViolationError:
            # The document was deleted while we were embedding. Its chunks and job row
            # cascaded away with it, so there is nothing left to fail — and marking a
            # deleted document 'failed' would resurrect nothing and log a false error.
            log.info("document %s deleted during ingestion; dropping job", job.document_id)
        except Exception as exc:  # noqa: BLE001 — recorded on the document, then re-raised
            await self._fail(job, exc)
            raise

    async def _fail(self, job: DocumentJob, exc: Exception) -> None:
        try:
            await self._store.mark(job.document_id, "failed", error=str(exc)[:2000])
        except Exception:  # noqa: BLE001 — the original error is the one that matters
            log.exception("could not record failure for document %s", job.document_id)

    async def process(self, job: DocumentJob) -> int:
        document = await self._store.get(job.document_id)
        if document is None:
            return 0  # deleted before we claimed the job
        await self._store.mark(job.document_id, "processing", error=None)

        data = await asyncio.to_thread(self._blobs.read, document.sha256)
        extracted = await asyncio.to_thread(extract_text, data, document.kind)

        chunks = await asyncio.to_thread(
            split_text,
            extracted.text,
            max_tokens=self._max_tokens,
            min_tokens=self._min_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        if not chunks:
            # A scanned PDF with no text layer, or an empty document. 'ready' with zero
            # chunks and an explicit note beats 'failed': nothing went wrong, there is
            # simply nothing to retrieve, and OCR is out of scope.
            await self._store.replace_chunks(job.document_id, document.scope_id, [])
            await self._store.mark(
                job.document_id,
                "ready",
                error="no extractable text (a scanned PDF needs OCR, which is out of scope)",
                pages=extracted.pages,
                chunk_count=0,
            )
            return 0

        vectors = await self._embed([c.text for c in chunks])
        pending = [
            PendingChunk(
                chunk_index=chunk.index,
                content=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                # Attribute the chunk to the page its OWN content starts on, not the
                # overlap's — the lead-in is duplicated from the previous page and would
                # pull the citation back one page on every boundary chunk.
                page_from=page_of(extracted.page_starts, chunk.core_start),
                tokens=chunk.tokens,
                embedding=vectors[i] if vectors else None,
            )
            for i, chunk in enumerate(chunks)
        ]
        written = await self._store.replace_chunks(
            job.document_id, document.scope_id, pending
        )
        await self._store.mark(
            job.document_id,
            "ready",
            error=None,
            pages=extracted.pages,
            chunk_count=written,
        )
        return written

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Batched, off the event loop, one GPU permit at a time. Returns ``[]`` with no
        embedder — the sparse (tsvector) channel still works, so retrieval degrades to
        keyword-only instead of failing the ingest."""
        if self._embedder is None:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            async with OLLAMA_LOCK:
                matrix = await asyncio.to_thread(self._embedder.encode, batch)
            out.extend(row.tolist() for row in matrix)
        return out


__all__ = ["DocumentPipeline", "EMBED_BATCH", "ValidationError"]
