"""FastAPI application factory + the service dependency.

``create_app(service=...)`` injects a ready backend (tests pass a fake-wired one); with
no argument the lifespan builds the configured backend at startup and closes it at
shutdown. Routes never construct a backend — they resolve it from ``app.state`` via the
``get_service`` dependency, so the same handlers serve either backend unchanged.

A module-level ``app = create_app()`` is provided so ``uvicorn local_infra.api.app:app``
just works for the smoke script and production.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from local_infra.config import get_settings
from local_infra.core.registry import build_backend
from local_infra.core.service import MemoryService


def get_service(request: Request) -> MemoryService:
    return request.app.state.service


def create_app(service: MemoryService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            # Injected (tests): the caller owns its lifecycle, so we don't close it.
            app.state.service = service
            yield
            return
        built = await build_backend()
        await built.start()  # launch the async extraction worker, if the backend has one
        app.state.service = built
        try:
            yield
        finally:
            await built.close()

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
    # get_service from this module.
    from local_infra.api.routes_admin import router as admin_router
    from local_infra.api.routes_memory import router as memory_router

    app.include_router(memory_router)
    app.include_router(admin_router)
    return app


app = create_app()
