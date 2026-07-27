"""Transport to the model provider, and the only module that touches litellm.

Three things about ``import litellm`` forced this shape, and all three are silent
failures rather than errors, so they are worth stating plainly:

1. **It fetches over the network at import time.** litellm downloads its
   model-cost map from raw.githubusercontent.com unless
   ``LITELLM_LOCAL_MODEL_COST_MAP`` is set first, using the bundled copy otherwise.
   That is a real violation of this service's offline guarantee — and
   ``tests/test_offline_guard.py`` would NOT catch it, because its
   ``socket.connect`` patch is applied inside a test, long after imports run at
   collection time.
2. **It costs ~2.2s and ~213MB RSS.** Paying that at API startup, or on every
   ``pytest`` collection, is pure waste on a box with 6GB of usable RAM.
3. **It reads ``ANTHROPIC_API_KEY``/``OPENAI_API_KEY`` from the ambient
   environment.** A developer's shell export would silently become a cloud
   capability, so keys are passed explicitly per call instead.

Hence: the env guard is set at module import (cheap, no litellm yet), and litellm
itself is imported lazily inside ``_litellm()``. Nothing else in the codebase may
import it. ``tests/test_chat_router.py`` asserts ``"litellm" not in sys.modules``
after importing the registry, which pins all of this.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

from local_infra.llm.types import ChatMessage, ChatUsage, Deployment

# Must be set BEFORE litellm is first imported, which is why it lives at module
# scope here rather than inside the lazy import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_litellm_module: Any = None


def _litellm() -> Any:
    """Import litellm on first use and quiet its global side effects."""
    global _litellm_module
    if _litellm_module is None:
        import litellm  # noqa: PLC0415 — deliberately lazy; see the module docstring

        litellm.telemetry = False
        litellm.suppress_debug_info = True
        # Providers disagree about which sampling params they accept; dropping the
        # unsupported ones beats a 400 from one backend mid-fallback.
        litellm.drop_params = True
        _litellm_module = litellm
    return _litellm_module


class CompletionClient(Protocol):
    """Pure transport, mirroring ``ExtractionProvider``: it performs one call and
    reports what happened. It makes no routing, fallback or budget decisions."""

    async def complete(
        self,
        deployment: Deployment,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> tuple[str, ChatUsage]:
        ...

    def stream(
        self,
        deployment: Deployment,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> AsyncIterator[str | ChatUsage]:
        """Yields text chunks, then exactly one final ``ChatUsage``."""
        ...


def _kwargs(
    deployment: Deployment,
    messages: Sequence[ChatMessage],
    *,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": deployment.model,
        "messages": [m.as_dict() for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout_s,
    }
    if deployment.api_base:
        kwargs["api_base"] = deployment.api_base
    if deployment.api_key:
        # Explicit, never ambient — see hazard 3 in the module docstring.
        kwargs["api_key"] = deployment.api_key
    if deployment.num_ctx is not None:
        kwargs["num_ctx"] = deployment.num_ctx
    return kwargs


def _usage_from(response: Any, model: str) -> ChatUsage:
    usage = getattr(response, "usage", None)
    return ChatUsage(
        model=model,
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


class LiteLLMClient:
    """Real transport. Costing is NOT done here — the router prices with our own
    table, because litellm's map reports ``ollama/*`` at zero and is exactly the
    thing the benchmark-honesty claim must not depend on."""

    async def complete(
        self,
        deployment: Deployment,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> tuple[str, ChatUsage]:
        litellm = _litellm()
        response = await litellm.acompletion(
            **_kwargs(
                deployment,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
        )
        text = response.choices[0].message.content or ""
        return text, _usage_from(response, deployment.model)

    async def stream(
        self,
        deployment: Deployment,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> AsyncIterator[str | ChatUsage]:
        litellm = _litellm()
        kwargs = _kwargs(
            deployment,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        # stream_options asks OpenAI-compatible backends to send a final usage chunk;
        # Ollama ignores it, so token counts fall back to 0 and cost stays $0 anyway.
        response = await litellm.acompletion(
            **kwargs, stream=True, stream_options={"include_usage": True}
        )
        usage = ChatUsage(model=deployment.model)
        async for chunk in response:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _usage_from(chunk, deployment.model)
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta else None
            if text:
                yield text
        yield usage
