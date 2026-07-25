"""Stage 0 — the deterministic pre-filter.

Decides *whether* to spend any model call at all, and flags documents complex
enough to skip straight to the large local model (a 4B round trip on a 2000-char
transcript is wasted latency). Pure Python, no I/O, fully deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# First-person / stative cues that suggest an assertable fact is present.
_CUES = (
    r"\bi\b", r"\bi'm\b", r"\bi've\b", r"\bmy\b", r"\bme\b", r"\bwe\b", r"\bour\b",
    r"\bis\b", r"\bare\b", r"\bwas\b", r"\bwere\b", r"\bam\b", r"\bhas\b", r"\bhave\b",
    r"\bwork", r"\blive", r"\buse", r"\bprefer", r"\bname", r"\blike", r"\bmoved\b",
)
_CUE_RE = re.compile("|".join(_CUES), re.IGNORECASE)
_SENT_SPLIT = re.compile(r"[.!?\n]+")

COMPLEX_CHARS = 1500
COMPLEX_FACTS = 6


@dataclass
class GateDecision:
    should_extract: bool
    predicted_facts: int
    is_complex: bool  # fast-path straight to LOCAL_LARGE
    reason: str


def gate(text: str) -> GateDecision:
    t = text.strip()
    if len(t) < 3:
        return GateDecision(False, 0, False, "too short")

    predicted = sum(1 for s in _SENT_SPLIT.split(t) if s.strip() and _CUE_RE.search(s))
    if predicted == 0:
        return GateDecision(False, 0, False, "no assertion cues")

    is_complex = len(t) > COMPLEX_CHARS or predicted > COMPLEX_FACTS
    return GateDecision(True, predicted, is_complex, "assertions present")
