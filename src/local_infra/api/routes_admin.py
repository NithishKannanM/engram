"""Admin routes: consolidation and health."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from local_infra.api.app import get_service
from local_infra.api.schemas import ImproveRequest
from local_infra.core.service import (
    HealthReport,
    ImproveReport,
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
