"""Prune: hard-delete the redundant rows dedup/merge already folded away.

The ONLY rows prune removes are ones closed by consolidation (``invalidation_reason``
starting ``dedup:`` or ``merge:``) — pure duplicates whose content survives in their
canonical. It never touches a currently-valid row, and never a genuine supersession
(``superseded:*``) or user tombstone, so the temporal history a user can time-travel to
stays intact. This is what keeps "no valid fact lost" true even as the ledger compacts.
"""

from __future__ import annotations

from local_infra.core.models import Memory

_CONSOLIDATION_REASONS = ("dedup:", "merge:")


async def prune_scope(repo, scope_id: int, *, dry_run: bool = False) -> int:
    """Hard-delete consolidation-closed duplicate rows in one scope. Returns rows removed."""
    rows = await repo.facts(scope_id, include_invalidated=True, limit=1_000_000)
    victims: list[Memory] = [
        m
        for m in rows
        if m.valid_until is not None
        and (m.invalidation_reason or "").startswith(_CONSOLIDATION_REASONS)
    ]
    if not dry_run:
        for m in victims:
            await repo.hard_delete(m.id)
    return len(victims)
