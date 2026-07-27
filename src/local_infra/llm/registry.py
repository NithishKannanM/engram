"""Assembly point for the chat router — and layer 1 of the offline guarantee.

This mirrors ``extraction/registry.py`` exactly, including *why* it is shaped this
way: when offline, a cloud deployment is **never constructed**. Not "constructed and
skipped", not "constructed behind an if" — never built, so there is no object to call
by accident. A boolean check somewhere downstream can be bypassed by a refactor; a
missing object cannot.

Importing this module must NOT import litellm. ``tests/test_chat_router.py`` asserts
that, because litellm fetches its cost map over the network at import time.
"""

from __future__ import annotations

from local_infra.config import Settings, get_settings
from local_infra.extraction.budget import Budget
from local_infra.llm.client import CompletionClient, LiteLLMClient
from local_infra.llm.router import ChatRouter
from local_infra.llm.types import Deployment

#: Stable names the API exposes. Clients send one of these, never a model id.
ALIAS_DEFAULT = "default"
ALIAS_FAST = "fast"


def build_deployments(settings: Settings | None = None) -> list[Deployment]:
    """Ordered deployments. Position within an alias is the fallback order."""
    s = settings or get_settings()
    base = s.ollama_base_url

    deployments = [
        # `default` prefers the 8B, then degrades to the 3B rather than failing:
        # a smaller answer beats no answer when VRAM is contended.
        Deployment(ALIAS_DEFAULT, s.chat_model_default, api_base=base, num_ctx=s.chat_num_ctx),
        Deployment(ALIAS_DEFAULT, s.chat_model_fast, api_base=base, num_ctx=s.chat_num_ctx),
        Deployment(ALIAS_FAST, s.chat_model_fast, api_base=base, num_ctx=s.chat_num_ctx),
    ]

    if not s.offline and s.chat_model_cloud:
        key = s.anthropic_api_key or s.openai_api_key
        cloud = Deployment(ALIAS_DEFAULT, s.chat_model_cloud, api_key=key)
        # Last resort only: local first, always.
        deployments.append(cloud)
        deployments.append(Deployment(ALIAS_FAST, s.chat_model_cloud, api_key=key))

    return deployments


def build_chat_router(
    settings: Settings | None = None,
    *,
    client: CompletionClient | None = None,
) -> ChatRouter:
    s = settings or get_settings()
    return ChatRouter(
        build_deployments(s),
        offline=s.offline,
        budget=Budget(s.chat_budget_usd),
        client=client or LiteLLMClient(),
        temperature=s.chat_temperature,
        max_tokens=s.chat_max_tokens,
        timeout_s=s.chat_timeout_s,
    )
