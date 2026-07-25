"""Recall judges for IRR (Information Retention Rate).

The ex_mem rig judged recall with an Anthropic LLM — non-deterministic and costly.
The plan's "benchmark honesty" note asks for a **local judge for offline runs** and
for judgments to be **cached by (fact, retrieved_text) hash**. Both are here.

:class:`LocalJudge` is the default: deterministic, offline, free. It decides "did the
retrieved context contain this fact's answer?" by exact normalized-substring match
first, then a rapidfuzz partial-ratio fallback for paraphrases — the same groundedness
idea the extraction ladder already uses (see ``extraction/confidence.py``), so the
whole rig stays local. A cloud LLM judge can be slotted in behind the same protocol
later; it is intentionally not wired up, because a cloud judge would reintroduce the
cost and nondeterminism this milestone is meant to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rapidfuzz import fuzz

from local_infra.bench.generator import Fact


@dataclass(frozen=True)
class Judgment:
    recalled: bool
    score: float  # 0..1 groundedness of the answer within the retrieved context
    reason: str


class Judge(Protocol):
    """Scores whether ``answer_text`` (the retrieved context) recalls ``fact.answer``."""

    def judge(self, fact: Fact, answer_text: str) -> Judgment: ...


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


class LocalJudge:
    """Deterministic, offline judge. Exact normalized-substring match scores 1.0;
    otherwise rapidfuzz partial_ratio/100 of the answer against the context. Recalled
    when the score meets ``threshold`` (default 0.75, matching the ladder's accept bar)."""

    def __init__(self, threshold: float = 0.75) -> None:
        self._threshold = threshold

    def judge(self, fact: Fact, answer_text: str) -> Judgment:
        answer = _norm(fact.answer)
        ctx = _norm(answer_text)
        if not answer:
            return Judgment(False, 0.0, "empty ground-truth answer")
        if not ctx:
            return Judgment(False, 0.0, "no retrieved context")
        if answer in ctx:
            return Judgment(True, 1.0, "exact substring match")
        score = fuzz.partial_ratio(answer, ctx) / 100.0
        return Judgment(
            score >= self._threshold,
            score,
            f"fuzzy partial_ratio={score:.2f} (threshold {self._threshold:.2f})",
        )


class CachingJudge:
    """Wraps a judge and memoizes by (fact.id, retrieved_text) — the plan's caching
    requirement. Deterministic judges gain nothing but repeat-call speed; a future
    nondeterministic/cloud judge gets stable, reproducible results within a run."""

    def __init__(self, inner: Judge) -> None:
        self._inner = inner
        self._cache: dict[tuple[str, str], Judgment] = {}

    def judge(self, fact: Fact, answer_text: str) -> Judgment:
        key = (fact.id, answer_text)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._inner.judge(fact, answer_text)
            self._cache[key] = hit
        return hit
