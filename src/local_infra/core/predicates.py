"""Predicate cardinality registry — the Python-side single source of truth.

The pure ``temporal.resolve()`` decides supersede-by-rule from ``NewFact.cardinality``,
so extraction must stamp each fact's cardinality *before* it reaches the resolver.
This list mirrors the seed in ``migrations/003_temporal.sql`` exactly; keep the two in
sync (the DB copy documents the same policy for pg-side queries). Unknown predicates
default to MULTI so an unseen predicate coexists rather than silently deleting a prior
fact — the resolver is meant to bias toward keeping.
"""

from __future__ import annotations

from local_infra.core.models import Cardinality
from local_infra.core.normalize import normalize_key

# Single-valued predicates: a new object supersedes the old by rule (no LLM judge).
# Mirrors migrations/003_temporal.sql.
_SINGLE_VALUED = (
    "lives_in",
    "works_at",
    "current_role",
    "current_project",
    "prefers_editor",
    "prefers_shell",
    "prefers_language",
    "uses_os",
    "primary_email",
    "phone_number",
    "timezone",
    "marital_status",
    "employer",
    "job_title",
    "city",
    "country",
    "age",
    "birthday",
    "full_name",
    "nickname",
    "default_branch",
    "current_goal",
    "owns_device",
    "gpu",
    "salary",
)

SINGLE_VALUED: frozenset[str] = frozenset(normalize_key(k) for k in _SINGLE_VALUED)


def cardinality_of(predicate_key: str) -> Cardinality:
    """SINGLE for a known single-valued predicate, else MULTI (the safe default)."""
    return "single" if normalize_key(predicate_key) in SINGLE_VALUED else "multi"
