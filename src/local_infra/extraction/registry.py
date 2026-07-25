"""Ladder assembly.

Builds an :class:`ExtractionLadder` wired to real providers from settings. Cloud
providers are constructed ONLY when not offline — so in offline mode there is no
cloud object to call by accident (the ladder's ``_max_index`` cap is the second
layer of the same guarantee).
"""

from __future__ import annotations

from local_infra.config import Settings, get_settings
from local_infra.extraction.budget import Budget
from local_infra.extraction.ladder import ExtractionLadder
from local_infra.extraction.providers.base import ExtractionProvider, Stage
from local_infra.extraction.providers.ollama import OllamaExtractionProvider


def build_local_providers(settings: Settings) -> dict[Stage, ExtractionProvider]:
    return {
        Stage.LOCAL_SMALL: OllamaExtractionProvider(
            Stage.LOCAL_SMALL, settings.extract_model_small, base_url=settings.ollama_base_url
        ),
        Stage.LOCAL_REPAIR: OllamaExtractionProvider(
            Stage.LOCAL_REPAIR, settings.extract_model_small,
            base_url=settings.ollama_base_url, repair=True,
        ),
        Stage.LOCAL_LARGE: OllamaExtractionProvider(
            Stage.LOCAL_LARGE, settings.extract_model_large, base_url=settings.ollama_base_url
        ),
    }


def build_ladder(settings: Settings | None = None, *, offline: bool | None = None) -> ExtractionLadder:
    settings = settings or get_settings()
    offline = settings.offline if offline is None else offline
    providers = build_local_providers(settings)
    # Cloud rung intentionally omitted here — added by a later milestone, and never
    # when offline. Budget still caps spend once it exists.
    return ExtractionLadder(providers, offline=offline, budget=Budget(settings.cloud_budget_usd))


async def close_providers(providers: dict[Stage, ExtractionProvider]) -> None:
    for p in providers.values():
        close = getattr(p, "close", None)
        if close is not None:
            await close()
