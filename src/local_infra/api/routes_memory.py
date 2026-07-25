"""Memory routes: the acceptance-path surface (remember / recall / get / history / forget)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from local_infra.api.app import get_service
from local_infra.api.schemas import RememberRequest
from local_infra.core.service import GLOBAL, MemoryService, RecallHit, RememberResult
from local_infra.core.models import Memory

router = APIRouter(tags=["memory"])


@router.post("/remember")
async def remember(
    body: RememberRequest, svc: MemoryService = Depends(get_service)
) -> RememberResult:
    return await svc.remember(
        body.content,
        scope=body.scope,
        metadata=body.metadata,
        source=body.source,
        occurred_at=body.occurred_at,
        extract=body.extract,
    )


@router.get("/recall")
async def recall(
    q: str = Query(..., description="the recall query"),
    scope: str = GLOBAL,
    limit: int = Query(10, ge=1, le=100),
    as_of: datetime | None = None,
    include_invalidated: bool = False,
    include_episodes: bool = False,
    memory_types: list[str] | None = Query(None),
    svc: MemoryService = Depends(get_service),
) -> list[RecallHit]:
    return await svc.recall(
        q,
        scope=scope,
        limit=limit,
        memory_types=memory_types,
        as_of=as_of,
        include_invalidated=include_invalidated,
        include_episodes=include_episodes,
    )


@router.get("/memories/{memory_id}")
async def get_memory(memory_id: UUID, svc: MemoryService = Depends(get_service)) -> Memory:
    mem = await svc.get(memory_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return mem


@router.get("/memories/{memory_id}/history")
async def memory_history(
    memory_id: UUID, svc: MemoryService = Depends(get_service)
) -> list[Memory]:
    mem = await svc.get(memory_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return await svc.history_for(memory_id)


@router.delete("/memories/{memory_id}")
async def forget(
    memory_id: UUID,
    hard: bool = False,
    reason: str = "user_forget",
    svc: MemoryService = Depends(get_service),
):
    result = await svc.forget(memory_id, reason=reason, hard=hard)
    if not result.ok:
        raise HTTPException(status_code=404, detail="memory not found")
    return result
