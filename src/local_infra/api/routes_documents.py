"""Document upload/list/delete and the fused ``/rag`` endpoint.

``POST /documents`` returns **202** with a job id: the response means "accepted and
queued", not "indexed". Anything else would either block the request for a minute on a
40-page PDF or lie about the document being searchable.

The whole router is unavailable on the exmem backend, which has no document store, and
says so with **501** — never a 500, and never an empty 200, which a user cannot
distinguish from "retrieval is broken".
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from local_infra.api.schemas import RagRequest
from local_infra.core.service import GLOBAL
from local_infra.documents.models import Document
from local_infra.documents.rag import RagResult, RagService
from local_infra.documents.service import DocumentService
from local_infra.documents.validate import MAX_BYTES, ValidationError, enforce_size

router = APIRouter(tags=["documents"])

_UNAVAILABLE = (
    "documents are not available on this backend — set MEMORY_BACKEND=cognee "
    "(Postgres) and run `make up && make migrate`"
)


def get_documents(request: Request) -> DocumentService:
    documents = getattr(request.app.state, "documents", None)
    if documents is None:
        raise HTTPException(status_code=501, detail=_UNAVAILABLE)
    return documents


def get_rag(request: Request) -> RagService:
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=501, detail=_UNAVAILABLE)
    return rag


async def _read_capped(upload: UploadFile) -> bytes:
    """Read the body in chunks, aborting as soon as the cap is passed.

    Checking ``len(await upload.read())`` afterwards is not a limit — it is a
    memory-exhaustion primitive, since the whole body is already resident by then.
    """
    parts: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        enforce_size(total)
        parts.append(chunk)
    return b"".join(parts)


@router.post("/documents", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    scope: str = Form(GLOBAL),
    documents: DocumentService = Depends(get_documents),
) -> dict:
    try:
        data = await _read_capped(file)
        document, job_id = await documents.upload(
            data, filename=file.filename, scope=scope
        )
    except ValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    finally:
        await file.close()

    return {
        "document_id": str(document.id),
        "status": document.status,
        "job_id": str(job_id) if job_id else None,
        "filename": document.filename,
        "kind": document.kind,
        "size_bytes": document.size_bytes,
        "scope": document.scope,
        # False when the same bytes were already ingested here: the client asked to
        # upload and got an existing document back, which is worth saying out loud.
        "queued": job_id is not None,
        "max_bytes": MAX_BYTES,
    }


@router.get("/documents")
async def list_documents(
    scope: str = GLOBAL,
    limit: int = 100,
    documents: DocumentService = Depends(get_documents),
) -> list[Document]:
    return await documents.list_documents(scope, limit=limit)


@router.get("/documents/{document_id}")
async def get_document(
    document_id: UUID, documents: DocumentService = Depends(get_documents)
) -> Document:
    document = await documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return document


@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: UUID, documents: DocumentService = Depends(get_documents)
) -> list:
    """The chunks as stored — what a citation drawer resolves a ``[doc:n]`` against."""
    if await documents.get(document_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    return await documents.chunks(document_id)


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: UUID, documents: DocumentService = Depends(get_documents)
) -> dict:
    document = await documents.delete(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return {"document_id": str(document_id), "deleted": True}


@router.post("/rag")
async def rag_query(
    body: RagRequest, rag: RagService = Depends(get_rag)
) -> RagResult:
    return await rag.query(
        body.query,
        scope=body.scope,
        limit=body.limit,
        synthesize=body.synthesize,
        alias=body.alias,
    )
