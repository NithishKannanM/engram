"""Real per-model pricing for benchmark cost accounting.

The ex_mem rig this is ported from carried an *invented* pricing table; the plan's
"benchmark honesty" note says to replace it before reporting any cost number. So:

  * **Local Ollama inference is free.** The whole local-first design exists so that
    extraction and embeddings cost nothing — the ladder's local rungs (``llama3.2``,
    ``llama3.1:8b``, ``nomic-embed-text``) are $0.
  * **Cloud rungs use real Anthropic list prices.** ``CLOUD_PRICING`` maps real model
    IDs to (input, output) USD per 1M tokens. Cloud escalation is disabled offline
    (``MEMORY_OFFLINE=1``) and gated by ``cloud_budget_usd=0`` by default, so a normal
    local benchmark run accrues no cloud spend — but if a run is configured to
    escalate, the cost is computed against these real numbers, not a placeholder.

**Why this module is stricter than it looks.** It used to treat *any* id absent from
``CLOUD_PRICING`` as local and free. That is safe only while every model id is a bare
Ollama tag. The moment a router speaks provider-qualified ids — ``anthropic/claude-sonnet-5``,
``openrouter/anthropic/claude-sonnet-5``, ``vertex_ai/claude-sonnet-5`` — the old rule
priced every one of them at **$0.00**, silently falsifying the "an all-local run is
$0.00" claim. Unknown ids now *raise* rather than defaulting to free: a wrong number is
worse than no number when the whole point is cost honesty. Callers that must not fail
use :func:`cost_usd_or_none` and report the gap explicitly.

Prices are list rates per 1,000,000 tokens (USD), current as of the 2026 knowledge
cutoff. Update from platform.claude.com/docs/en/pricing before quoting a cloud number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# model id -> (input $/1M tokens, output $/1M tokens)
CLOUD_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Routers namespace local inference behind these. Anything under them runs on this
# machine and is free no matter what the model is called.
LOCAL_PROVIDER_PREFIXES: tuple[str, ...] = (
    "ollama/",
    "ollama_chat/",
    "huggingface/local/",
)

# Provider namespaces that must be peeled before a CLOUD_PRICING lookup. Order does
# not matter; stripping repeats until it reaches a bare id (``openrouter/anthropic/x``).
CLOUD_PROVIDER_PREFIXES: tuple[str, ...] = (
    "anthropic/",
    "openai/",
    "azure/",
    "vertex_ai/",
    "bedrock/",
    "openrouter/",
    "gemini/",
    "groq/",
    "mistral/",
    "cohere/",
    "deepseek/",
    "together_ai/",
    "fireworks_ai/",
    "xai/",
)

# Bare ids that *look* like hosted models. A bare id is normally assumed to be an
# Ollama tag and free — but "claude-sonet-5" (typo) must not quietly bill $0.00, so
# anything in a known hosted family that is missing from CLOUD_PRICING is an error.
CLOUD_FAMILY_PREFIXES: tuple[str, ...] = (
    "claude-",
    "gpt-",
    "gemini-",
    "o1-",
    "o3-",
    "o4-",
)


class UnknownModelPricing(LookupError):
    """Raised when a model's cost cannot be established.

    Deliberately not a silent 0.0 — see the module docstring.
    """


@dataclass(frozen=True)
class Usage:
    """One model invocation's token accounting, tagged with the model that ran it."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0


def _strip_provider(model: str) -> str:
    """Peel known cloud provider namespaces until a bare model id remains."""
    current = model
    changed = True
    while changed:
        changed = False
        for prefix in CLOUD_PROVIDER_PREFIXES:
            if current.startswith(prefix):
                current = current[len(prefix) :]
                changed = True
    return current


def is_local(model: str) -> bool:
    """True when the model runs on this machine and is therefore free.

    A provider-qualified id is local only under an explicitly local namespace; a bare
    id is assumed to be an Ollama tag unless it is a known hosted model or looks like
    one (see ``CLOUD_FAMILY_PREFIXES``).
    """
    if model.startswith(LOCAL_PROVIDER_PREFIXES):
        return True
    if "/" in model:
        return False
    if model in CLOUD_PRICING or model.startswith(CLOUD_FAMILY_PREFIXES):
        return False
    return True


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one invocation in USD.

    Local models are free. Raises :class:`UnknownModelPricing` for a model whose price
    is not known, rather than reporting it as free.
    """
    if is_local(model):
        return 0.0
    prices = CLOUD_PRICING.get(_strip_provider(model))
    if prices is None:
        raise UnknownModelPricing(
            f"no price for {model!r}; add it to CLOUD_PRICING or route it under "
            f"one of {LOCAL_PROVIDER_PREFIXES} if it runs locally"
        )
    p_in, p_out = prices
    return input_tokens / 1_000_000 * p_in + output_tokens / 1_000_000 * p_out


def cost_usd_or_none(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """:func:`cost_usd`, but ``None`` instead of raising on an unknown model.

    For reporting paths (the bench runner) that must complete and surface the gap as
    ``cost_unknown`` rather than abort a whole run over one unrecognised id.
    """
    try:
        return cost_usd(model, input_tokens, output_tokens)
    except UnknownModelPricing:
        return None


def total_cost_usd(usages: Iterable[Usage]) -> float:
    """Sum the cost of a stream of invocations (local ones contribute $0)."""
    return sum(cost_usd(u.model, u.input_tokens, u.output_tokens) for u in usages)
