"""Reciprocal Rank Fusion — the one place multiple rankers are combined.

RRF:  score(d) = Σ_i  w_i / (k + rank_i(d)),  rank 1-based, k=60 by convention
(Cormack et al., 2009). Rank-based, so it fuses rankers with incomparable score
scales — pgvector cosine, tsvector ts_rank, a graph channel — without normalization.
Pure and generic over any hashable id, so it unit-tests without a database.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Hashable, TypeVar

T = TypeVar("T", bound=Hashable)

RRF_K = 60


def rrf_fuse(rankings: Sequence[tuple[float, Sequence[T]]], *, k: int = RRF_K) -> list[T]:
    """Fuse weighted rankings (best-first id lists) into one, highest score first.

    ``rankings`` is a list of ``(weight, ids)``. An id absent from a ranking simply
    contributes nothing from it. Ties break by first appearance, so the ordering is
    deterministic for a fixed input.
    """
    scores: dict[T, float] = {}
    order: dict[T, int] = {}
    seq = 0
    for weight, ids in rankings:
        for rank, item in enumerate(ids):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)
            if item not in order:
                order[item] = seq
                seq += 1
    return sorted(scores, key=lambda it: (-scores[it], order[it]))
