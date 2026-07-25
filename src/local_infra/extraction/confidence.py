"""Composite confidence scoring.

A JSON grammar guarantees syntax, not truth — a small model will happily emit a
structurally perfect, semantically invented fact. So no single signal is trusted;
groundedness (is the evidence actually in the source?) carries the most weight and
can veto on its own. Every term is persisted (M6) so these weights get tuned
against benchmark ground truth instead of guessed forever.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

# Weights (sum to 1.0 with all four terms present).
W_GROUND = 0.45
W_CONSISTENCY = 0.25
W_STRUCTURAL = 0.15
W_SELF = 0.15

# A fact whose evidence isn't really in the source is treated as hallucinated and
# must escalate regardless of how confident everything else looks.
GROUND_FLOOR = 0.6
# Self-consistency (2-seed agreement) is only worth its cost in this ambiguous band.
CONSISTENCY_BAND = (0.55, 0.80)


def groundedness(evidence_span: str, source_text: str) -> float:
    """1.0 on exact substring match, else rapidfuzz partial_ratio / 100."""
    span = evidence_span.strip()
    if not span:
        return 0.0
    if span in source_text:
        return 1.0
    return fuzz.partial_ratio(span, source_text) / 100.0


@dataclass
class Score:
    groundedness: float
    structural: float
    self_report: float
    self_consistency: float | None
    composite: float

    @property
    def grounded(self) -> bool:
        return self.groundedness >= GROUND_FLOOR


def composite(
    ground: float, structural: float, self_report: float, self_consistency: float | None = None
) -> float:
    """Weighted blend, normalized over whichever terms are present. Self-consistency
    is absent unless it was measured (only inside CONSISTENCY_BAND)."""
    num = W_GROUND * ground + W_STRUCTURAL * structural + W_SELF * self_report
    den = W_GROUND + W_STRUCTURAL + W_SELF
    if self_consistency is not None:
        num += W_CONSISTENCY * self_consistency
        den += W_CONSISTENCY
    return num / den


def score_fact(
    evidence_span: str,
    source_text: str,
    *,
    structural: float,
    self_report: float,
    self_consistency: float | None = None,
) -> Score:
    g = groundedness(evidence_span, source_text)
    return Score(
        groundedness=g,
        structural=structural,
        self_report=self_report,
        self_consistency=self_consistency,
        composite=composite(g, structural, self_report, self_consistency),
    )


def in_consistency_band(value: float) -> bool:
    return CONSISTENCY_BAND[0] <= value <= CONSISTENCY_BAND[1]
