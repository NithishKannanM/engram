"""Ledger consolidation — the work behind ``improve()``.

Backend-agnostic: every op runs against the ``Repository`` protocol, so the same
consolidation serves both backends (only the surrounding ``reindex`` differs). The
load-bearing invariant, enforced by construction, is **no valid fact is ever lost**:

  * ``dedup``  collapses currently-valid rows sharing the EXACT (subject,predicate,object)
    into one canonical — a redundant row, never a distinct fact.
  * ``merge``  collapses currently-valid rows in the same (subject,predicate) whose
    statements are near-identical by cosine (embedder-gated) — the semantic near-dup.
  * ``prune``  hard-deletes ONLY rows that dedup/merge already closed (reason
    ``dedup:``/``merge:``) — never a genuine temporal supersession.

Both dedup and merge fold via :meth:`Repository.consolidate`, which keeps the canonical,
sums ``seen_count``, and leaves the closed rows as a replayable audit until pruned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from local_infra.consolidate.dedup import dedup_scope
from local_infra.consolidate.merge import merge_scope
from local_infra.consolidate.prune import prune_scope


@dataclass
class ConsolidationResult:
    deduped: int = 0
    merged: int = 0
    pruned: int = 0
    notes: list[str] = field(default_factory=list)


async def consolidate(
    repo,
    *,
    scope_ids,
    ops,
    dry_run: bool = False,
    embedder=None,
) -> ConsolidationResult:
    """Run the requested ops over each scope, in the order dedup → merge → prune (so
    prune only ever removes rows the earlier passes folded)."""
    result = ConsolidationResult()
    for scope_id in scope_ids:
        if "dedup" in ops:
            result.deduped += await dedup_scope(repo, scope_id, dry_run=dry_run)
        if "merge" in ops:
            merged, note = await merge_scope(repo, scope_id, embedder, dry_run=dry_run)
            result.merged += merged
            if note and note not in result.notes:
                result.notes.append(note)
        if "prune" in ops:
            result.pruned += await prune_scope(repo, scope_id, dry_run=dry_run)
    return result
