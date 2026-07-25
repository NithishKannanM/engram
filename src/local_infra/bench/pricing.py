"""Real per-model pricing for benchmark cost accounting.

The ex_mem rig this is ported from carried an *invented* pricing table; the plan's
"benchmark honesty" note says to replace it before reporting any cost number. So:

  * **Local Ollama inference is free.** The whole local-first design exists so that
    extraction and embeddings cost nothing — the ladder's local rungs (``llama3.2``,
    ``llama3.1:8b``, ``nomic-embed-text``) are $0. Any model id NOT in the cloud table
    below is treated as a local, zero-cost model.
  * **Cloud rungs use real Anthropic list prices.** ``CLOUD_PRICING`` maps real model
    IDs to (input, output) USD per 1M tokens. Cloud escalation is disabled offline
    (``MEMORY_OFFLINE=1``) and gated by ``cloud_budget_usd=0`` by default, so a normal
    local benchmark run accrues no cloud spend — but if a run is configured to
    escalate, the cost is computed against these real numbers, not a placeholder.

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


@dataclass(frozen=True)
class Usage:
    """One model invocation's token accounting, tagged with the model that ran it."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one invocation. Local (non-cloud) models are free -> 0.0."""
    prices = CLOUD_PRICING.get(model)
    if prices is None:
        return 0.0  # local Ollama model — no charge
    p_in, p_out = prices
    return input_tokens / 1_000_000 * p_in + output_tokens / 1_000_000 * p_out


def total_cost_usd(usages: Iterable[Usage]) -> float:
    """Sum the cost of a stream of invocations (local ones contribute $0)."""
    return sum(cost_usd(u.model, u.input_tokens, u.output_tokens) for u in usages)


def is_local(model: str) -> bool:
    """True when the model is a free, local Ollama model (not in the cloud table)."""
    return model not in CLOUD_PRICING
