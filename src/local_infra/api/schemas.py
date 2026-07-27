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


class ChatTurn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    scope: str = GLOBAL
    # An ALIAS ("default", "fast"), never a model id — a request must not be able to
    # name a cloud model and thereby route around the offline guarantee.
    alias: str = "default"
    history: list[ChatTurn] = Field(default_factory=list)


class RagRequest(BaseModel):
    query: str = Field(min_length=1)
    scope: str = GLOBAL
    limit: int = Field(default=8, ge=1, le=50)
    #: Retrieval-only by default. Synthesis costs a generation, and the citations alone
    #: are already useful — the desktop renders them without waiting for prose.
    synthesize: bool = False
    alias: str = "default"


class ImproveRequest(BaseModel):
    scope: str | None = None
    ops: list[str] = Field(default_factory=lambda: ["dedup", "merge", "prune", "reindex"])
    dry_run: bool = False
