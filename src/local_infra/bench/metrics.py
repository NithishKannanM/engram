"""Benchmark metrics: recall@k, IRR, latency percentiles, cost.

Definitions (chosen so the two accuracy numbers measure different things):

  * **recall@k** — retrieval quality. For each probe the runner fetches the top-k
    recalled memories; a probe counts as a hit when the fact's ground-truth answer
    appears (normalized substring) in ANY of those k statements. ``recall@k`` is the
    fraction of facts hit. It asks: *did retrieval surface the right fact in the top k?*
  * **IRR (Information Retention Rate)** — end-to-end retention, via the judge. The
    retrieved top-k statements are joined into an "answer context" and scored by the
    judge (local + fuzzy by default). ``irr`` is the fraction the judge marks recalled.
    It is more forgiving than recall@k (paraphrase-tolerant), matching the prior rig's
    IRR semantics but computed locally.

Latency is measured on the wall clock, split into ingest (``remember``) and query
(``recall``); p50/p95 are reported for both. Cost sums real per-model pricing and is
$0.00 for an all-local run (see :mod:`bench.pricing`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from local_infra.bench.generator import Fact


@dataclass
class RecallResult:
    fact: Fact
    hit_at_k: bool  # answer present in the top-k retrieved statements
    recalled: bool  # judge verdict (IRR)
    judge_score: float
    top_text: str  # the retrieved context that was judged


@dataclass
class BenchmarkResult:
    backend: str
    facts: int
    k: int
    recall_at_k: float
    irr: float
    p50_ingest_ms: float
    p95_ingest_ms: float
    p50_recall_ms: float
    p95_recall_ms: float
    total_extracted: int
    cost_usd: float = 0.0
    notes: str = ""
    per_fact: list[RecallResult] = field(default_factory=list)

    def summary_row(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "facts": str(self.facts),
            "recall@k": f"{self.recall_at_k:.1%}",
            "IRR": f"{self.irr:.1%}",
            "p50_ingest_ms": f"{self.p50_ingest_ms:.0f}",
            "p95_ingest_ms": f"{self.p95_ingest_ms:.0f}",
            "p50_recall_ms": f"{self.p50_recall_ms:.0f}",
            "p95_recall_ms": f"{self.p95_recall_ms:.0f}",
            "extracted": str(self.total_extracted),
            "cost_usd": f"${self.cost_usd:.4f}",
        }


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,1]); 0.0 for an empty sequence.

    No numpy dependency — the rig must stay light. Matches numpy's default
    ('linear') method on the samples the runner produces."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def answer_present(answer: str, statements: Sequence[str]) -> bool:
    """Normalized-substring hit test for recall@k: is ``answer`` inside any statement?"""
    norm_answer = " ".join(answer.lower().split())
    if not norm_answer:
        return False
    return any(norm_answer in " ".join(s.lower().split()) for s in statements)
