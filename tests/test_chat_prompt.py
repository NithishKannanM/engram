"""The prompt contract: citation numbering, superseded marking, context bounding.

Numbering is the load-bearing part. The desktop resolves ``[n]`` markers positionally
against the returned citation list, so if the prompt numbers memories differently from
the order the answer carries, facts get mis-attributed to the wrong memory.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from local_infra.core.models import Memory
from local_infra.core.service import RecallHit
from local_infra.llm.prompt import NO_MEMORIES, SYSTEM, build_messages, render_memories

NOW = datetime(2026, 7, 26, tzinfo=timezone.utc)


def memory(statement: str, *, valid_until: datetime | None = None) -> Memory:
    return Memory(
        id=uuid4(),
        scope_id=1,
        subject="s",
        predicate="p",
        object="o",
        subject_key="s",
        predicate_key="p",
        object_key="o",
        statement=statement,
        valid_from=NOW,
        valid_until=valid_until,
        recorded_at=NOW,
        last_seen_at=NOW,
        seen_count=1,
        confidence=0.9,
    )


def hit(statement: str, *, score: float = 1.0, valid_until: datetime | None = None) -> RecallHit:
    return RecallHit(score=score, kind="memory", memory=memory(statement, valid_until=valid_until))


def test_memories_are_numbered_from_one_in_rank_order() -> None:
    rendered = render_memories([hit("first"), hit("second"), hit("third")])
    assert rendered.splitlines() == ["[1] first", "[2] second", "[3] third"]


def test_superseded_memories_are_marked() -> None:
    rendered = render_memories([hit("port is 8000", valid_until=NOW)])
    assert "(superseded)" in rendered
    # The instruction that makes the model handle it correctly must be present.
    assert "superseded" in SYSTEM


def test_no_memories_uses_the_no_fabrication_system_prompt() -> None:
    messages = build_messages("where do I work?", [])
    assert messages[0].content == NO_MEMORIES
    assert messages[-1].content == "where do I work?"
    # No MEMORIES block at all, so there is nothing to cite.
    assert "MEMORIES:" not in messages[-1].content


def test_grounded_turn_carries_both_memories_and_question() -> None:
    messages = build_messages("where do I work?", [hit("Nithish works at Acme")])
    user = messages[-1].content
    assert "MEMORIES:" in user
    assert "[1] Nithish works at Acme" in user
    assert "QUESTION: where do I work?" in user
    assert messages[0].content == SYSTEM


def test_history_is_preserved_between_system_and_the_grounded_turn() -> None:
    from local_infra.llm.types import ChatMessage

    history = [ChatMessage("user", "hi"), ChatMessage("assistant", "hello")]
    messages = build_messages("q", [hit("m")], history=history)

    assert messages[0].role == "system"
    assert [m.content for m in messages[1:3]] == ["hi", "hello"]
    assert messages[-1].content.endswith("QUESTION: q")


def test_context_is_bounded_so_ollama_cannot_truncate_away_the_question() -> None:
    """Ollama truncates from the LEFT. An unbounded memory block would evict the
    system prompt and the question, yielding an ungrounded answer that looks exactly
    like a retrieval failure."""
    hits = [hit("x" * 200) for _ in range(200)]
    messages = build_messages("the question", hits, max_context_chars=1000)
    user = messages[-1].content

    assert len(user) < 2000
    assert "QUESTION: the question" in user, "the question must survive bounding"
