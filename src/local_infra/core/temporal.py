"""The pure temporal resolver.

``resolve(new, candidates, verdicts)`` takes a new fact and the currently-valid
rows it might collide with, and returns a :class:`Resolution` — a batch of write
intents the repository applies in ONE transaction, or a request for contradiction
judgments it cannot make itself. No I/O, no LLM, no DB: every branch unit-tests in
milliseconds, which is the whole point of locking this before wiring M4/M5.

Decision cascade, cheapest first (LLM only reached at the very end):

  1. DUPLICATE  — same object_key, or statement cosine >= 0.95. Touch, no new row.
  2. RETROACTIVE — new is older than current belief (observed_at < earliest
                   valid_from). Insert already-closed; never invalidate the present.
  3. SUPERSEDE-BY-RULE — single-valued predicate, object differs. Close old, insert
                   new. Deterministic; no judge.
  4. JUDGE      — multi/unknown predicate, object differs. Ask for a verdict per
                   colliding candidate; independent -> coexist, contradicts ->
                   supersede, duplicate -> touch. Bias to independent: a verdict
                   below INVALIDATE_MIN confidence is downgraded to independent so a
                   shaky judge never silently deletes a true fact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from local_infra.core.models import (
    Candidate,
    CloseMemory,
    InsertMemory,
    NewFact,
    Relation,
    Resolution,
    RecordRevision,
    TouchMemory,
    Verdict,
    WriteIntent,
)

# statement-cosine at/above which two facts are the *same* assertion, no judge.
DUPLICATE_SIM = 0.95
# a contradiction/duplicate verdict below this confidence is treated as independent.
INVALIDATE_MIN = 0.7


def _effective_relation(verdict: Verdict) -> Relation:
    """Downgrade low-confidence contradict/duplicate to independent (bias to keep)."""
    if verdict.relation is Relation.INDEPENDENT:
        return Relation.INDEPENDENT
    if verdict.confidence < INVALIDATE_MIN:
        return Relation.INDEPENDENT
    return verdict.relation


def resolve(
    new: NewFact,
    candidates: Sequence[Candidate],
    verdicts: Mapping[UUID, Verdict] | None = None,
) -> Resolution:
    verdicts = verdicts or {}

    # --- 1. DUPLICATE (across ALL candidates, incl. differently-phrased neighbors) ---
    for c in candidates:
        if c.object_key == new.object_key or c.similarity >= DUPLICATE_SIM:
            return Resolution(
                intents=(
                    TouchMemory(c.id),
                    RecordRevision(
                        revision_type="touch",
                        reason="duplicate",
                        decided_by="rule:duplicate",
                        observed_at=new.observed_at,
                        memory_id=c.id,
                    ),
                ),
            )

    # Only same subject+predicate rows can supersede/coexist along a temporal chain.
    triple = [
        c
        for c in candidates
        if c.subject_key == new.subject_key and c.predicate_key == new.predicate_key
    ]

    # --- 2. RETROACTIVE: the new fact predates current belief -> historical insert ---
    if triple:
        earliest = min(triple, key=lambda c: c.valid_from)
        if new.observed_at < earliest.valid_from:
            return _historical_insert(new, closed_at=earliest.valid_from)

    # --- 3. SUPERSEDE BY RULE: single-valued predicate, object differs ---
    if new.cardinality == "single":
        return _supersede(new, triple, reason="single_valued", decided_by="rule:single_valued")

    # --- 4. JUDGE: multi/unknown predicate, object differs ---
    unresolved = [c.id for c in triple if c.id not in verdicts]
    if unresolved:
        return Resolution(needs_judge=tuple(unresolved))

    dup = [c for c in triple if _effective_relation(verdicts[c.id]) is Relation.DUPLICATE]
    if dup:
        target = dup[0]
        return Resolution(
            intents=(
                TouchMemory(target.id),
                RecordRevision(
                    revision_type="touch",
                    reason="duplicate",
                    decided_by="judge:duplicate",
                    observed_at=new.observed_at,
                    memory_id=target.id,
                ),
            ),
        )

    contradicting = [
        c for c in triple if _effective_relation(verdicts[c.id]) is Relation.CONTRADICTS
    ]
    if contradicting:
        return _supersede(new, contradicting, reason="contradicts", decided_by="judge:contradicts")

    # independent (or nothing to collide with) -> coexist as a plain insert.
    return _plain_insert(new, reason="coexist", decided_by="judge:independent")


# --------------------------------------------------------------------------- #
# Intent builders                                                              #
# --------------------------------------------------------------------------- #
def _plain_insert(new: NewFact, *, reason: str, decided_by: str) -> Resolution:
    return Resolution(
        intents=(
            InsertMemory(fact=new, valid_from=new.observed_at, valid_until=None),
            RecordRevision(
                revision_type="insert",
                reason=reason,
                decided_by=decided_by,
                observed_at=new.observed_at,
            ),
        ),
        inserts_new=True,
    )


def _historical_insert(new: NewFact, *, closed_at) -> Resolution:
    """Insert an already-closed row spanning [observed_at, closed_at). The present
    is left untouched — an out-of-order fact must not invalidate current belief."""
    return Resolution(
        intents=(
            InsertMemory(fact=new, valid_from=new.observed_at, valid_until=closed_at),
            RecordRevision(
                revision_type="insert",
                reason="retroactive",
                decided_by="rule:out_of_order",
                observed_at=new.observed_at,
            ),
        ),
        inserts_new=True,
    )


def _supersede(
    new: NewFact, targets: Sequence[Candidate], *, reason: str, decided_by: str
) -> Resolution:
    """Close every ``target`` at ``new.observed_at`` and insert ``new`` as current,
    chaining ``supersedes_id`` to the first closed row."""
    supersedes_id = targets[0].id if targets else None
    intents: list[WriteIntent] = [
        InsertMemory(
            fact=new, valid_from=new.observed_at, valid_until=None, supersedes_id=supersedes_id
        ),
        RecordRevision(
            revision_type="insert",
            reason=reason,
            decided_by=decided_by,
            observed_at=new.observed_at,
        ),
    ]
    for c in targets:
        intents.append(
            CloseMemory(
                memory_id=c.id,
                valid_until=new.observed_at,
                invalidation_reason=f"superseded:{reason}",
            )
        )
        intents.append(
            RecordRevision(
                revision_type="close",
                reason=f"superseded:{reason}",
                decided_by=decided_by,
                observed_at=new.observed_at,
                memory_id=c.id,
            )
        )
    return Resolution(intents=tuple(intents), inserts_new=True)
