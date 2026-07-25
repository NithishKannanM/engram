"""M3 verification: the pure temporal resolver, no infrastructure.

Covers duplicate, single-valued supersede, multi coexist (via judge), retroactive
insert, low-confidence downgrade, and the concurrent-guard intent shape. Every test
runs in microseconds — that is the payoff for keeping ``resolve`` pure.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from local_infra.core.models import (
    Candidate,
    CloseMemory,
    InsertMemory,
    NewFact,
    Relation,
    TouchMemory,
    Verdict,
)
from local_infra.core.normalize import normalize_key
from local_infra.core.temporal import resolve

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)


def make_fact(obj: str, *, predicate="works_at", cardinality="single", observed=T0, subject="nithish") -> NewFact:
    return NewFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        subject_key=normalize_key(subject),
        predicate_key=normalize_key(predicate),
        object_key=normalize_key(obj),
        statement=f"{subject} {predicate} {obj}",
        observed_at=observed,
        cardinality=cardinality,
    )


def make_candidate(obj: str, *, predicate="works_at", valid_from=T0, similarity=0.0, subject="nithish") -> Candidate:
    return Candidate(
        id=uuid4(),
        subject_key=normalize_key(subject),
        predicate_key=normalize_key(predicate),
        object_key=normalize_key(obj),
        statement=f"{subject} {predicate} {obj}",
        valid_from=valid_from,
        similarity=similarity,
    )


def _of(intents, kind):
    return [i for i in intents if isinstance(i, kind)]


# --------------------------------------------------------------------------- #
def test_duplicate_same_object_key_touches_only() -> None:
    cand = make_candidate("globex")
    res = resolve(make_fact("Globex"), [cand])  # normalizes to same object_key
    assert not res.inserts_new
    touches = _of(res.intents, TouchMemory)
    assert len(touches) == 1 and touches[0].memory_id == cand.id
    assert not _of(res.intents, InsertMemory)
    assert not _of(res.intents, CloseMemory)


def test_duplicate_by_high_similarity_touches() -> None:
    cand = make_candidate("globex corp", similarity=0.97)  # different object_key
    res = resolve(make_fact("globex corporation"), [cand])
    assert _of(res.intents, TouchMemory)
    assert not _of(res.intents, InsertMemory)


# --------------------------------------------------------------------------- #
def test_single_valued_supersede_shape() -> None:
    later = T0 + timedelta(days=30)
    old = make_candidate("acme", valid_from=T0)
    res = resolve(make_fact("globex", observed=later), [old])

    inserts = _of(res.intents, InsertMemory)
    closes = _of(res.intents, CloseMemory)
    assert res.inserts_new
    assert len(inserts) == 1 and inserts[0].supersedes_id == old.id
    assert inserts[0].valid_until is None  # new row is current
    # Concurrent-guard shape: the old row is closed exactly at the new observation.
    assert len(closes) == 1
    assert closes[0].memory_id == old.id
    assert closes[0].valid_until == later
    assert "single_valued" in closes[0].invalidation_reason


def test_single_valued_with_no_candidates_is_plain_insert() -> None:
    res = resolve(make_fact("globex"), [])
    assert res.inserts_new
    inserts = _of(res.intents, InsertMemory)
    assert len(inserts) == 1 and inserts[0].supersedes_id is None
    assert not _of(res.intents, CloseMemory)


# --------------------------------------------------------------------------- #
def test_multi_predicate_requests_judge_then_coexists() -> None:
    old = make_candidate("python", predicate="knows_language")
    new = make_fact("rust", predicate="knows_language", cardinality="multi")

    blocked = resolve(new, [old])
    assert blocked.needs_judge == (old.id,)
    assert not blocked.intents

    coexist = resolve(new, [old], {old.id: Verdict(Relation.INDEPENDENT)})
    assert coexist.inserts_new
    assert _of(coexist.intents, InsertMemory)
    assert not _of(coexist.intents, CloseMemory)  # both languages stay valid


def test_multi_judge_contradiction_supersedes() -> None:
    old = make_candidate("vim", predicate="prefers_editor")
    new = make_fact("neovim", predicate="prefers_editor", cardinality="multi")
    res = resolve(new, [old], {old.id: Verdict(Relation.CONTRADICTS, 0.9)})
    assert _of(res.intents, CloseMemory)[0].memory_id == old.id
    assert _of(res.intents, InsertMemory)[0].supersedes_id == old.id


def test_low_confidence_contradiction_downgraded_to_independent() -> None:
    old = make_candidate("vim", predicate="prefers_editor")
    new = make_fact("neovim", predicate="prefers_editor", cardinality="multi")
    res = resolve(new, [old], {old.id: Verdict(Relation.CONTRADICTS, 0.5)})
    # Below INVALIDATE_MIN -> keep the old fact, coexist.
    assert not _of(res.intents, CloseMemory)
    assert _of(res.intents, InsertMemory)


# --------------------------------------------------------------------------- #
def test_retroactive_fact_inserts_closed_and_spares_present() -> None:
    old = make_candidate("globex", valid_from=T0)  # current belief since June
    # A fact learned now but true earlier (January): must not invalidate the present.
    new = make_fact("acme", observed=datetime(2026, 1, 1, tzinfo=timezone.utc))
    res = resolve(new, [old])

    inserts = _of(res.intents, InsertMemory)
    assert len(inserts) == 1
    assert inserts[0].valid_from == new.observed_at
    assert inserts[0].valid_until == old.valid_from  # closed where current belief begins
    assert not _of(res.intents, CloseMemory)  # the present row is untouched
