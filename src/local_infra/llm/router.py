"""Alias -> deployment routing with ordered fallback, budget gating and costing.

**Why this is ~120 lines of our own code instead of ``litellm.Router``.** Router
carries per-deployment cooldown state: with one deployment per alias (which is this
setup), a single transient Ollama blip cools that deployment down and subsequent
requests fail in a way that reads as "the model vanished". It also brings background
health checks, global mutable state, and a ``model_list`` schema that shifts between
litellm versions. The policy actually needed here is: *try deployments in order, on
failure try the next, stop at the first success.* Owning that keeps the offline
guarantee structural and makes the whole thing trivially fake-injectable.

The offline guarantee has four layers; this module is layers 2 and 4:

  0. ``client.py`` sets ``LITELLM_LOCAL_MODEL_COST_MAP`` and imports litellm lazily.
  1. ``registry.py`` never *constructs* a cloud deployment when offline.
  2. ``ChatRouter.__init__`` raises if it is handed one anyway.
  3. ``Deployment.is_local`` checks the model namespace AND the endpoint host.
  4. Callers pass an **alias**, never a model id, so a request cannot smuggle one in.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from local_infra.bench.pricing import cost_usd_or_none
from local_infra.extraction.budget import Budget
from local_infra.llm.client import CompletionClient
from local_infra.llm.gpu import OLLAMA_LOCK
from local_infra.llm.types import (
    ChatMessage,
    ChatResult,
    ChatUnavailable,
    ChatUsage,
    Deployment,
    StreamDelta,
)

# What one cloud call might cost, for the pre-flight budget check. Deliberately
# pessimistic: better to refuse a call we could have afforded than to overrun.
CLOUD_ESTIMATE_USD = 0.02


class ChatRouter:
    def __init__(
        self,
        deployments: Sequence[Deployment],
        *,
        offline: bool,
        budget: Budget,
        client: CompletionClient,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout_s: float = 120.0,
    ) -> None:
        cloud = [d for d in deployments if not d.is_local]
        if offline and cloud:
            # Layer 2. registry.py should never build these while offline; if one
            # reaches here, fail loudly at construction rather than at call time.
            raise ValueError(
                "offline router was given non-local deployments: "
                + ", ".join(d.model for d in cloud)
            )
        self._offline = offline
        self._budget = budget
        self._client = client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s

        # Preserve declaration order within each alias — that IS the fallback order.
        self._chains: dict[str, list[Deployment]] = {}
        for d in deployments:
            self._chains.setdefault(d.alias, []).append(d)

    # ---- introspection ---------------------------------------------------- #
    def aliases(self) -> list[str]:
        return list(self._chains)

    def describe(self) -> list[dict[str, object]]:
        """What ``GET /chat/models`` reports. Never includes a key or a key's presence
        per-deployment beyond a boolean — the desktop shows status, not secrets."""
        return [
            {
                "alias": alias,
                "model": chain[0].model,
                "is_local": chain[0].is_local,
                "fallbacks": [d.model for d in chain[1:]],
                "available": True,
            }
            for alias, chain in self._chains.items()
        ]

    def chain(self, alias: str) -> list[Deployment]:
        if alias not in self._chains:
            raise KeyError(alias)
        return self._chains[alias]

    # ---- costing ---------------------------------------------------------- #
    def _price(self, usage: ChatUsage) -> ChatUsage:
        cost = cost_usd_or_none(usage.model, usage.input_tokens, usage.output_tokens)
        if cost is None:
            usage.cost_unknown = True
            usage.cost_usd = 0.0
        else:
            usage.cost_usd = cost
            if cost:
                self._budget.record(cost)
        return usage

    def _affordable(self, deployment: Deployment) -> bool:
        return deployment.is_local or self._budget.can_spend(CLOUD_ESTIMATE_USD)

    # ---- completion ------------------------------------------------------- #
    async def complete(
        self, alias: str, messages: Sequence[ChatMessage]
    ) -> ChatResult:
        chain = self.chain(alias)
        attempts: list[str] = []
        errors: list[tuple[str, str]] = []

        for deployment in chain:
            if not self._affordable(deployment):
                errors.append((deployment.model, "cloud budget exhausted"))
                continue
            attempts.append(deployment.model)
            try:
                text, usage = await self._call(deployment, messages)
            except Exception as exc:  # noqa: BLE001 — any transport failure falls back
                errors.append((deployment.model, f"{type(exc).__name__}: {exc}"))
                continue
            return ChatResult(
                text=text,
                alias=alias,
                model=deployment.model,
                usage=self._price(usage),
                fallback_used=deployment is not chain[0],
                attempts=attempts,
            )

        raise ChatUnavailable(alias, errors)

    async def _call(
        self, deployment: Deployment, messages: Sequence[ChatMessage]
    ) -> tuple[str, ChatUsage]:
        """Local generation holds the one GPU semaphore; cloud calls do not — they
        use no VRAM, and blocking them behind a local generation would serialize the
        very fallback that exists to route around a busy or dead local model."""
        if not deployment.is_local:
            return await self._client.complete(
                deployment,
                messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout_s=self._timeout_s,
            )
        async with OLLAMA_LOCK:
            return await self._client.complete(
                deployment,
                messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout_s=self._timeout_s,
            )

    # ---- streaming -------------------------------------------------------- #
    async def stream(
        self, alias: str, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[StreamDelta | ChatResult]:
        """Yield deltas, then one final ``ChatResult`` carrying usage.

        **Fallback is weaker here than in ``complete``, deliberately.** Once a token
        has been emitted to the client we cannot silently switch models — the partial
        answer is already on screen. So fallback applies only to failures *before* the
        first delta; a mid-stream failure propagates and ends the stream.
        """
        chain = self.chain(alias)
        errors: list[tuple[str, str]] = []

        for deployment in chain:
            if not self._affordable(deployment):
                errors.append((deployment.model, "cloud budget exhausted"))
                continue

            started = False
            usage = ChatUsage(model=deployment.model)
            acquired = deployment.is_local
            if acquired:
                # Held for the whole generation; released in the finally below so a
                # client disconnect cannot wedge the GPU for the process lifetime.
                await OLLAMA_LOCK.acquire()
            try:
                async for item in self._client.stream(
                    deployment,
                    messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    timeout_s=self._timeout_s,
                ):
                    if isinstance(item, ChatUsage):
                        usage = item
                        continue
                    started = True
                    yield StreamDelta(item)
            except Exception as exc:  # noqa: BLE001
                errors.append((deployment.model, f"{type(exc).__name__}: {exc}"))
                if started:
                    raise
                continue
            finally:
                if acquired:
                    OLLAMA_LOCK.release()

            yield ChatResult(
                text="",
                alias=alias,
                model=deployment.model,
                usage=self._price(usage),
                fallback_used=deployment is not chain[0],
                attempts=[deployment.model],
            )
            return

        raise ChatUnavailable(alias, errors)
