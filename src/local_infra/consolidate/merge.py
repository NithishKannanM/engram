"""Similarity merge: collapse near-identical currently-valid rows within a predicate.

Where dedup needs an exact triple match, merge catches the same fact phrased or objected
differently ("Globex" vs "Globex Corp") by clustering statement embeddings above a high
cosine threshold. Embedder-gated: with no embedder it is a reported no-op, never a guess
— a bad merge would silently drop a true, distinct fact, so the threshold is strict and
merge only ever runs within one (subject, predicate).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from local_infra.core.models import Memory
from local_infra.consolidate.dedup import canonical_of

MERGE_SIM = 0.93  # strict: only near-identical statements collapse


def _cluster(members: list[Memory], vecs: np.ndarray, threshold: float) -> list[list[Memory]]:
    """Union-find clustering by pairwise cosine over L2-normalized ``vecs``."""
    n = len(members)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sims = vecs @ vecs.T
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                parent[find(i)] = find(j)

    clusters: dict[int, list[Memory]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(members[i])
    return list(clusters.values())


async def merge_scope(
    repo, scope_id: int, embedder, *, dry_run: bool = False
) -> tuple[int, str | None]:
    """Collapse near-duplicate statements in one scope. Returns (rows closed, note)."""
    if embedder is None:
        return 0, "merge: skipped (no embedder)"

    rows = await repo.facts(scope_id, limit=1_000_000)
    by_sp: dict[tuple[str, str], list[Memory]] = defaultdict(list)
    for m in rows:
        by_sp[(m.subject_key, m.predicate_key)].append(m)

    merged = 0
    for members in by_sp.values():
        if len(members) < 2:
            continue
        vecs = embedder.encode([m.statement for m in members])  # normalized -> dot == cosine
        for cluster in _cluster(members, vecs, MERGE_SIM):
            if len(cluster) < 2:
                continue
            canonical = canonical_of(cluster)
            dup_ids = [m.id for m in cluster if m.id != canonical.id]
            merged += len(dup_ids)
            if not dry_run:
                await repo.consolidate(
                    canonical.id, dup_ids, reason="merge:similar", decided_by="improve:merge"
                )
    return merged, None
