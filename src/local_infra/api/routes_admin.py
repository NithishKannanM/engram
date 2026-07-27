"""Admin routes: consolidation and health."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query

from local_infra.api.app import get_service
from local_infra.api.schemas import ImproveRequest
from local_infra.core.service import (
    GLOBAL,
    HealthReport,
    ImproveReport,
    MemoryGraph,
    MemoryService,
    ScopeSummary,
)

router = APIRouter(tags=["admin"])


@router.post("/improve")
async def improve(
    body: ImproveRequest, svc: MemoryService = Depends(get_service)
) -> ImproveReport:
    return await svc.improve(scope=body.scope, ops=tuple(body.ops), dry_run=body.dry_run)


@router.get("/health")
async def health(svc: MemoryService = Depends(get_service)) -> HealthReport:
    return await svc.health()


@router.get("/scopes")
async def scopes(svc: MemoryService = Depends(get_service)) -> list[ScopeSummary]:
    """Every scope with its live fact count — lets a client list projects."""
    return await svc.scopes()


@router.get("/graph")
async def graph(
    scope: str = GLOBAL,
    as_of: datetime | None = None,
    limit: int = Query(200, ge=1, le=2000),
    svc: MemoryService = Depends(get_service),
) -> MemoryGraph:
    """Nodes and edges for the memory graph.

    Includes superseded rows on purpose — they are the history the time slider
    animates. ``current`` is recomputed against ``as_of``, so scrubbing backwards
    revives a fact that is superseded today.
    """
    return await svc.graph(scope, as_of=as_of, limit=limit)
