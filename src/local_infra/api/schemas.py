"""Request bodies for the REST layer.

Only *inputs* need declared schemas (validation + coercion). Responses return the
service's own dataclasses (``RememberResult``, ``Memory``, ``RecallHit``, …) directly;
FastAPI's ``jsonable_encoder`` serializes them (UUID→str, datetime→ISO, Enum→value), so
there is one source of truth for the shapes and no hand-maintained mirror.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from local_infra.core.service import GLOBAL


class RememberRequest(BaseModel):
    content: str = Field(min_length=1)
    scope: str = GLOBAL
    metadata: dict | None = None
    source: str = "api"
    occurred_at: datetime | None = None
    extract: Literal["sync", "async", "never"] = "async"


class ImproveRequest(BaseModel):
    scope: str | None = None
    ops: list[str] = Field(default_factory=lambda: ["dedup", "merge", "prune", "reindex"])
    dry_run: bool = False
