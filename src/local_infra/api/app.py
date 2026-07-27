"""FastAPI application factory + the per-subsystem dependencies.

``create_app(service=..., chat=...)`` injects ready subsystems (tests pass fake-wired
ones); anything left as ``None`` is built by the lifespan at startup and closed at
shutdown. Routes never construct a subsystem — they resolve it from ``app.state`` via
a dependency, so the same handlers serve either backend unchanged.

**Note the shape of the lifespan.** It used to early-``return`` when a service was
injected, which meant anything added *after* that point silently did not exist for
injected apps — and seven tests take exactly that path. So each subsystem is now
resolved independently, and ownership (who closes what) is tracked explicitly rather
than implied by which branch ran.

A module-level ``app = create_app()`` is provided so ``uvicorn local_infra.api.app:app``
just works for the smoke script and production.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from typing import TYPE_CHECKING

from local_infra.config import get_settings
from local_infra.core.registry import build_backend
from local_infra.core.service import MemoryService
from local_infra.llm.chat import ChatService

if TYPE_CHECKING:  # imported lazily at runtime; see _build_documents
    from local_infra.documents.rag import RagService
    from local_infra.documents.service import DocumentService


def get_service(request: Request) -> MemoryService:
    return request.app.state.service


def get_chat(request: Request) -> ChatService:
    """Chat is optional — ``MEMORY_CHAT_ENABLED=0`` turns it off entirely.

    A disabled subsystem answers 503 with a reason rather than 500-ing on a missing
    attribute, so the desktop can show "chat unavailable" instead of "server error".
    """
    chat = getattr(request.app.state, "chat", None)
    if chat is None:
        raise HTTPException(
            status_code=503,
            detail="chat is disabled (set MEMORY_CHAT_ENABLED=1 to enable it)",
        )
    return chat


async def _build_documents(service: MemoryService) -> "DocumentService | None":
    """Build the document subsystem, or ``None`` when this backend cannot host it.

    Documents need Postgres (pgvector chunks). The exmem backend has no store, so the
    routes answer 501 with a reason rather than pretending an empty result set.
    """
    settings = get_settings()
    if not settings.documents_enabled or settings.backend != "cognee":
        return None

    # Deferred: importing the document stack pulls in pypdf/python-docx, which a
    # chat-only or exmem deployment should never pay for.
    from local_infra.backends.pg.documents import PgDocumentStore
    from local_infra.documents.blobs import BlobStore
    from local_infra.documents.pipeline import DocumentPipeline
    from local_infra.documents.service import DocumentService

    embedder = getattr(service, "embedder", None)
    blobs = BlobStore(settings.state_dir / "documents")
    # Two connections. An ingest holds its own for seconds while embedding, and a single
    # asyncpg connection used concurrently raises "another operation is in progress" —
    # the same split `LedgerBackend` already makes with its `worker_repo`.
    store = await PgDocumentStore(settings.pg_dsn, embedder=embedder).connect()
    worker_store = await PgDocumentStore(settings.pg_dsn, embedder=embedder).connect()
    pipeline = DocumentPipeline(
        worker_store,
        blobs,
        embedder=embedder,
        max_tokens=settings.doc_chunk_tokens,
        min_tokens=settings.doc_chunk_min_tokens,
        overlap_tokens=settings.doc_chunk_overlap_tokens,
    )
    return DocumentService(store, blobs, worker_store=worker_store, pipeline=pipeline)


def _build_rag(
    service: MemoryService, documents: "DocumentService", chat: ChatService | None
) -> "RagService":
    from local_infra.documents.rag import RagService

    return RagService(
        service,
        documents.store,
        chat=chat,
        limit=get_settings().rag_limit,
    )


def _build_chat(service: MemoryService) -> ChatService | None:
    settings = get_settings()
    if not settings.chat_enabled:
        return None
    # Deferred so that importing this module never pulls in the llm registry's
    # dependency graph for a deployment that has chat switched off.
    from local_infra.llm.registry import build_chat_router

    return ChatService(
        service,
        build_chat_router(settings),
        recall_limit=settings.chat_recall_limit,
    )


def create_app(
    service: MemoryService | None = None,
    *,
    chat: ChatService | None = None,
    documents: "DocumentService | None" = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Only subsystems this app BUILT get closed; injected ones belong to the caller.
        owned: list[object] = []

        if service is not None:
            app.state.service = service
        else:
            built = await build_backend()
            await built.start()  # launch the async extraction worker, if any
            app.state.service = built
            owned.append(built)

        # Built in both branches — an injected service still gets a working /chat.
        app.state.chat = chat if chat is not None else _build_chat(app.state.service)

        if documents is not None:
            app.state.documents = documents
        else:
            app.state.documents = await _build_documents(app.state.service)
            if app.state.documents is not None:
                owned.append(app.state.documents)
        if app.state.documents is not None:
            await app.state.documents.start()  # the ingest worker

        # RAG needs both halves; without documents there is nothing to fuse WITH, and
        # /rag would just be a slower /recall pretending to be more.
        app.state.rag = (
            _build_rag(app.state.service, app.state.documents, app.state.chat)
            if app.state.documents is not None
            else None
        )

        try:
            yield
        finally:
            for subsystem in owned:
                await subsystem.close()

    app = FastAPI(title="local_infra memory service", version="0.1.0", lifespan=lifespan)

    # The desktop shell calls this service from a webview origin, so preflight and
    # actual requests both need CORS headers. Origins are an explicit allow-list
    # (never "*") because this service is unauthenticated and binds localhost.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Imported here (not at module top) to avoid a circular import: the routers import
    # get_service / get_chat from this module.
    from local_infra.api.routes_admin import router as admin_router
    from local_infra.api.routes_chat import router as chat_router
    from local_infra.api.routes_documents import router as documents_router
    from local_infra.api.routes_memory import router as memory_router

    app.include_router(memory_router)
    app.include_router(admin_router)
    app.include_router(chat_router)
    # Always mounted, even on a backend that cannot serve it: a registered route that
    # answers 501 with a reason is discoverable; a missing route is a 404 that reads as
    # "wrong URL" and sends the client looking in the wrong place.
    app.include_router(documents_router)
    return app


app = create_app()
