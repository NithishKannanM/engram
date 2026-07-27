"""Chat routing: alias resolution, ordered fallback, budget gating, offline structure.

Nothing here touches the network or a model. The most important test in the file is
``test_importing_registry_does_not_import_litellm`` — see its docstring.
"""

from __future__ import annotations

import sys

import pytest

from local_infra.config import Settings
from local_infra.extraction.budget import Budget
from local_infra.llm.registry import ALIAS_DEFAULT, build_chat_router, build_deployments
from local_infra.llm.router import ChatRouter
from local_infra.llm.types import (
    ChatMessage,
    ChatUnavailable,
    ChatUsage,
    Deployment,
)
from tests._fakes import FakeCompletionClient

LOCAL = "http://127.0.0.1:11434"
MSGS = [ChatMessage("user", "where do I work?")]


def local_dep(model: str = "ollama_chat/llama3.1:8b", alias: str = ALIAS_DEFAULT) -> Deployment:
    return Deployment(alias, model, api_base=LOCAL)


def cloud_dep(model: str = "anthropic/claude-sonnet-5", alias: str = ALIAS_DEFAULT) -> Deployment:
    return Deployment(alias, model, api_key="sk-test")


def router(deployments, *, offline=True, budget=0.0, client=None) -> ChatRouter:
    return ChatRouter(
        deployments,
        offline=offline,
        budget=Budget(budget),
        client=client or FakeCompletionClient(),
    )


# --------------------------------------------------------------------------- #
# Deployment locality
# --------------------------------------------------------------------------- #
def test_locality_needs_both_namespace_and_host() -> None:
    assert local_dep().is_local
    assert Deployment("a", "ollama_chat/llama3.2").is_local  # no api_base = local
    # Local namespace, remote host -> NOT local.
    assert not Deployment("a", "ollama/llama3.2", api_base="http://10.0.0.5:11434").is_local
    # Cloud model does not become local by pointing at localhost.
    assert not Deployment("a", "anthropic/claude-sonnet-5", api_base=LOCAL).is_local


# --------------------------------------------------------------------------- #
# Offline is structural
# --------------------------------------------------------------------------- #
def test_offline_router_refuses_a_cloud_deployment() -> None:
    with pytest.raises(ValueError, match="non-local"):
        router([local_dep(), cloud_dep()], offline=True)


def test_offline_registry_never_builds_a_cloud_deployment() -> None:
    settings = Settings(
        offline=True,
        chat_model_cloud="anthropic/claude-sonnet-5",
        anthropic_api_key="sk-test",
    )
    deployments = build_deployments(settings)
    assert deployments, "expected local deployments"
    assert all(d.is_local for d in deployments)
    # And the whole router builds without raising, because layer 1 held.
    assert build_chat_router(settings, client=FakeCompletionClient()).aliases()


def test_online_registry_appends_cloud_as_last_resort() -> None:
    settings = Settings(
        offline=False,
        chat_model_cloud="anthropic/claude-sonnet-5",
        anthropic_api_key="sk-test",
    )
    chain = [d for d in build_deployments(settings) if d.alias == ALIAS_DEFAULT]
    assert chain[0].is_local, "local must be preferred"
    assert not chain[-1].is_local, "cloud must be the fallback, not the default"


def test_importing_registry_does_not_import_litellm() -> None:
    """litellm fetches its model-cost map over the NETWORK at import time unless
    LITELLM_LOCAL_MODEL_COST_MAP is set first, and costs ~2.2s / 213MB besides.

    tests/test_offline_guard.py cannot catch that: it patches socket.connect inside a
    test, long after imports run at collection. So the guarantee is pinned here, by
    asserting the module is simply never imported as a side effect of the chat layer.
    """
    for module in ("local_infra.llm.registry", "local_infra.llm.router", "local_infra.llm.chat"):
        assert module in sys.modules or __import__(module)
    assert "litellm" not in sys.modules


def test_client_sets_the_cost_map_guard_before_any_import() -> None:
    import os

    import local_infra.llm.client as client  # noqa: F401 — import is the point

    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"


# --------------------------------------------------------------------------- #
# Alias resolution + fallback
# --------------------------------------------------------------------------- #
async def test_unknown_alias_raises_keyerror_not_a_generic_failure() -> None:
    with pytest.raises(KeyError):
        await router([local_dep()]).complete("nope", MSGS)


async def test_first_deployment_answers() -> None:
    client = FakeCompletionClient([("in Chennai", ChatUsage("ollama_chat/llama3.1:8b"))])
    result = await router([local_dep()], client=client).complete(ALIAS_DEFAULT, MSGS)

    assert result.text == "in Chennai"
    assert result.fallback_used is False
    assert client.calls == ["ollama_chat/llama3.1:8b"]


async def test_falls_back_in_declaration_order() -> None:
    client = FakeCompletionClient([
        ConnectionError("ollama is down"),
        ("smaller answer", ChatUsage("ollama_chat/llama3.2:latest")),
    ])
    chain = [local_dep(), local_dep("ollama_chat/llama3.2:latest")]
    result = await router(chain, client=client).complete(ALIAS_DEFAULT, MSGS)

    assert result.text == "smaller answer"
    assert result.model == "ollama_chat/llama3.2:latest"
    assert result.fallback_used is True
    # The exact chain that was tried, in order.
    assert client.calls == ["ollama_chat/llama3.1:8b", "ollama_chat/llama3.2:latest"]


async def test_all_deployments_failing_raises_chat_unavailable_with_reasons() -> None:
    client = FakeCompletionClient([ConnectionError("down"), TimeoutError("slow")])
    chain = [local_dep(), local_dep("ollama_chat/llama3.2:latest")]

    with pytest.raises(ChatUnavailable) as excinfo:
        await router(chain, client=client).complete(ALIAS_DEFAULT, MSGS)

    # The error names what was tried and why, so a 503 can explain itself.
    assert "ollama_chat/llama3.1:8b" in str(excinfo.value)
    assert "down" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Costing + budget
# --------------------------------------------------------------------------- #
async def test_local_completion_is_free() -> None:
    client = FakeCompletionClient([("hi", ChatUsage("ollama_chat/llama3.1:8b", 1000, 500))])
    result = await router([local_dep()], client=client).complete(ALIAS_DEFAULT, MSGS)

    assert result.usage.cost_usd == 0.0
    assert result.usage.cost_unknown is False


async def test_cloud_completion_is_priced_from_our_table() -> None:
    client = FakeCompletionClient([
        ("hi", ChatUsage("anthropic/claude-sonnet-5", 1_000_000, 1_000_000))
    ])
    r = router([cloud_dep()], offline=False, budget=100.0, client=client)
    result = await r.complete(ALIAS_DEFAULT, MSGS)

    # $3/1M in + $15/1M out — our numbers, not litellm's (which prices ollama/* at 0).
    assert result.usage.cost_usd == 18.0


async def test_exhausted_budget_skips_cloud_rather_than_overspending() -> None:
    client = FakeCompletionClient([("local", ChatUsage("ollama_chat/llama3.1:8b"))])
    # Cloud first, but a zero ceiling means it is never attempted.
    chain = [cloud_dep(), local_dep()]
    r = router(chain, offline=False, budget=0.0, client=client)
    result = await r.complete(ALIAS_DEFAULT, MSGS)

    assert result.model == "ollama_chat/llama3.1:8b"
    assert client.calls == ["ollama_chat/llama3.1:8b"], "cloud must not be called"


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
async def test_stream_yields_deltas_then_a_final_result() -> None:
    client = FakeCompletionClient([
        (["Hello", " ", "world"], ChatUsage("ollama_chat/llama3.1:8b", 10, 3))
    ])
    from local_infra.llm.types import ChatResult, StreamDelta

    items = [i async for i in router([local_dep()], client=client).stream(ALIAS_DEFAULT, MSGS)]

    assert [i.text for i in items if isinstance(i, StreamDelta)] == ["Hello", " ", "world"]
    final = items[-1]
    assert isinstance(final, ChatResult)
    assert final.usage.output_tokens == 3


async def test_stream_falls_back_only_before_the_first_token() -> None:
    """Once a token is on screen we cannot silently switch models — the partial
    answer is already visible. Pre-token failures fall back; mid-stream ones do not."""
    client = FakeCompletionClient([
        ConnectionError("died before any token"),
        (["recovered"], ChatUsage("ollama_chat/llama3.2:latest")),
    ])
    chain = [local_dep(), local_dep("ollama_chat/llama3.2:latest")]
    from local_infra.llm.types import StreamDelta

    items = [i async for i in router(chain, client=client).stream(ALIAS_DEFAULT, MSGS)]
    assert [i.text for i in items if isinstance(i, StreamDelta)] == ["recovered"]


async def test_stream_releases_the_gpu_lock_on_failure() -> None:
    """A client disconnect mid-stream must not wedge the GPU for the process life."""
    from local_infra.llm.gpu import OLLAMA_LOCK

    client = FakeCompletionClient([ConnectionError("boom"), ConnectionError("boom2")])
    chain = [local_dep(), local_dep("ollama_chat/llama3.2:latest")]

    with pytest.raises(ChatUnavailable):
        async for _ in router(chain, client=client).stream(ALIAS_DEFAULT, MSGS):
            pass

    assert not OLLAMA_LOCK.locked()
