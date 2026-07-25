"""Exact-triple dedup: collapse redundant currently-valid rows for the same fact.

Rows that share (scope, subject_key, predicate_key, object_key) and are all currently
valid are the SAME fact stored more than once — they accumulate from re-extraction or
concurrent writes that bypass the resolver's touch. Keep one canonical, fold the rest.
Deterministic and embedder-free, which is why it is the improve() gate.
"""

from __future__ import annotations

from collections import defaultdict

from local_infra.core.models import Memory


def canonical_of(members: list[Memory]) -> Memory:
    """Pick the survivor: most-seen, then most-confident, then oldest (stable tie-break)."""
    return max(members, key=lambda m: (m.seen_count, m.confidence, -m.recorded_at.timestamp()))


async def dedup_scope(repo, scope_id: int, *, dry_run: bool = False) -> int:
    """Collapse exact-triple duplicates in one scope. Returns rows closed as dups."""
    rows = await repo.facts(scope_id, limit=1_000_000)  # currently valid only
    groups: dict[tuple[str, str, str], list[Memory]] = defaultdict(list)
    for m in rows:
        groups[(m.subject_key, m.predicate_key, m.object_key)].append(m)

    collapsed = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = canonical_of(members)
        dup_ids = [m.id for m in members if m.id != canonical.id]
        collapsed += len(dup_ids)
        if not dry_run:
            await repo.consolidate(
                canonical.id, dup_ids, reason="dedup:exact", decided_by="improve:dedup"
            )
    return collapsed
