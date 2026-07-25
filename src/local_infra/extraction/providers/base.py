"""The extraction-provider seam + the stage vocabulary.

A provider is pure transport: given text (and an optional focus = the spans that
failed a lower rung), it returns the model's raw JSON string, or a transport
failure. All parsing, validation, grounding, and scoring live in the ladder — a
provider never decides whether an extraction is good enough. That split is what
lets the whole ladder be tested with fakes and no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence


class Stage(str, Enum):
    GATE = "gate"  # pure-Python pre-filter, cost 0
    LOCAL_SMALL = "local_small"  # llama3.2 via Ollama format
    LOCAL_REPAIR = "local_repair"  # same small model + the error text
    LOCAL_LARGE = "local_large"  # llama3.1:8b, last fully-offline rung
    CLOUD = "cloud"  # Anthropic/OpenAI strict structured output
    DEFER = "defer"  # give up for now; the episode is already durable


# The climbable rungs, in order. GATE is pre-loop; DEFER is terminal.
LADDER: tuple[Stage, ...] = (
    Stage.LOCAL_SMALL,
    Stage.LOCAL_REPAIR,
    Stage.LOCAL_LARGE,
    Stage.CLOUD,
)


@dataclass
class ProviderResult:
    """What a provider returns. ``raw is None`` (or ``ok is False``) is a transport
    failure the ladder treats exactly like unparseable output: escalate."""

    raw: str | None
    ok: bool = True
    error: str | None = None
    latency_ms: int = 0
    cost_usd: float = 0.0


class ExtractionProvider(Protocol):
    stage: Stage

    async def extract(
        self, text: str, *, focus: Sequence[str] | None = None, seed: int = 0
    ) -> ProviderResult:
        """Return the model's raw JSON for ``text``. ``focus`` narrows a re-extraction
        to just the spans that failed below (per-fact escalation)."""
        ...
