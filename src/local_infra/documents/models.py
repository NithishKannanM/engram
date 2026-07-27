"""Value types for the document subsystem.

Plain dataclasses, like `core/service.py`'s result types: the API returns these directly
(FastAPI's jsonable_encoder serialises them), so there is one source of truth for the
shape instead of a dataclass and a mirrored pydantic response model that drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from local_infra.documents.validate import DocumentKind

#: pending → processing → ready | failed. Only the worker moves a document past
#: `pending`; the request path never does slow work.
DocumentStatus = str


@dataclass
class Document:
    id: UUID
    scope_id: int
    scope: str
    filename: str
    kind: DocumentKind
    sha256: str
    size_bytes: int
    status: DocumentStatus
    pages: int | None = None
    chunk_count: int = 0
    error: str | None = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class DocumentChunk:
    """One embedded slice of a document, carrying enough provenance for a citation to
    point at the exact source span (and page, where the format has pages)."""

    id: UUID
    document_id: UUID
    scope_id: int
    chunk_index: int
    content: str
    char_start: int
    char_end: int
    page_from: int | None = None
    tokens: int = 0


@dataclass(frozen=True)
class PendingChunk:
    """A chunk on its way into the store: text plus its embedding, not yet persisted."""

    chunk_index: int
    content: str
    char_start: int
    char_end: int
    page_from: int | None
    tokens: int
    embedding: list[float] | None = None


@dataclass
class ChunkHit:
    """A retrieved chunk with the document it came from, ready to cite."""

    score: float
    chunk: DocumentChunk
    document_id: UUID
    filename: str
    page_from: int | None = None

    @property
    def text(self) -> str:
        return self.chunk.content


@dataclass(frozen=True)
class DocumentJob:
    """A claimed ingestion job. Structurally compatible with `ingest.models.Job` where
    the worker cares (`.id`), but it references a document, not an episode — which is
    exactly why it cannot live in `extraction_jobs` (whose `episode_id` is NOT NULL)."""

    id: UUID
    document_id: UUID
    scope_id: int
