"""Backend factory — the one place a backend name becomes a live MemoryService.

Imports are deferred so selecting ``exmem`` never imports cognee (heavy, and it reads
env at import time), and vice-versa. The API and CLI call this; nothing else needs to
know which concrete backend is running behind the seam.
"""

from __future__ import annotations

from local_infra.config import Settings, get_settings
from local_infra.core.service import MemoryService


async def build_backend(settings: Settings | None = None) -> MemoryService:
    """Construct the backend named by ``settings.backend`` (``exmem`` | ``cognee``)."""
    settings = settings or get_settings()
    name = settings.backend.lower()
    if name == "exmem":
        from local_infra.backends.exmem_backend import ExMemBackend

        return ExMemBackend.from_settings(settings)
    if name == "cognee":
        from local_infra.backends.cognee_backend import CogneeBackend

        return await CogneeBackend.from_settings(settings)
    raise ValueError(f"unknown backend {settings.backend!r} (expected 'exmem' or 'cognee')")
