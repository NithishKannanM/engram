"""Turning recalled memories into a grounded, citable prompt.

Pure functions only — no I/O, no model — so the prompt contract is unit-testable
without a network or a GPU.

The citation numbering here is load-bearing: the desktop already parses ``[n]``
markers out of assistant text (``src/lib/citations.ts``) and resolves them against
an ordered citation list. So the model is told to cite with ``[n]`` and the *same*
ordering is returned alongside the answer. Numbering that drifts between the prompt
and the response list would mis-attribute facts, which is worse than no citation.
"""

from __future__ import annotations

from collections.abc import Sequence

from local_infra.core.service import RecallHit
from local_infra.llm.types import ChatMessage

SYSTEM = (
    "You are Engram, a memory assistant. You answer using the numbered MEMORIES "
    "provided below, which were retrieved from the user's own long-term store.\n\n"
    "Rules:\n"
    "- Cite every claim you take from a memory with its bracketed number, like [1].\n"
    "- If the memories do not answer the question, say so plainly. Never invent a "
    "fact and never attach a citation to something the memories do not support.\n"
    "- A memory marked (superseded) was true earlier and is not current. Mention the "
    "change rather than stating the old value as fact.\n"
    "- Be concise and direct. Do not restate the question."
)

NO_MEMORIES = (
    "You are Engram, a memory assistant. Nothing in the user's store matched this "
    "question. Say so plainly, answer from general knowledge only if that is clearly "
    "useful, and do not fabricate citations."
)


def render_memories(hits: Sequence[RecallHit]) -> str:
    """Number hits from 1, in rank order, matching the returned citation list."""
    lines: list[str] = []
    for i, hit in enumerate(hits, start=1):
        memory = hit.memory
        if memory is None:
            # An episode hit (only present with include_episodes) — still citable.
            text = hit.text.strip().replace("\n", " ")
            lines.append(f"[{i}] {text[:400]}")
            continue
        status = " (superseded)" if memory.valid_until is not None else ""
        lines.append(f"[{i}] {memory.statement}{status}")
    return "\n".join(lines)


def build_messages(
    question: str,
    hits: Sequence[RecallHit],
    *,
    history: Sequence[ChatMessage] = (),
    max_context_chars: int = 6000,
) -> list[ChatMessage]:
    """Assemble system + history + the grounded user turn.

    ``max_context_chars`` is a guard against a large recall blowing past
    ``chat_num_ctx`` (4096 tokens ≈ 16k chars); Ollama truncates from the LEFT, so an
    unbounded memory block would silently evict the system prompt and the question —
    producing an ungrounded answer that looks like a retrieval failure.
    """
    if not hits:
        return [ChatMessage("system", NO_MEMORIES), *history, ChatMessage("user", question)]

    rendered = render_memories(hits)
    if len(rendered) > max_context_chars:
        kept: list[str] = []
        budget = max_context_chars
        for line in rendered.split("\n"):
            if len(line) + 1 > budget:
                break
            kept.append(line)
            budget -= len(line) + 1
        rendered = "\n".join(kept)

    user = f"MEMORIES:\n{rendered}\n\nQUESTION: {question}"
    return [ChatMessage("system", SYSTEM), *history, ChatMessage("user", user)]


# --------------------------------------------------------------------------- #
# RAG: the same [n] contract, over mixed sources                               #
# --------------------------------------------------------------------------- #
RAG_SYSTEM = (
    "You are Engram. You answer using the numbered SOURCES below, which come from the "
    "user's own memory store and their uploaded documents.\n\n"
    "Rules:\n"
    "- Cite every claim with its bracketed number, like [1]. A document source names "
    "its file, so the citation is checkable.\n"
    "- If the sources do not answer the question, say so plainly. Never invent a fact "
    "and never attach a citation to something the sources do not support.\n"
    "- Where a memory and a document disagree, say so instead of silently picking one.\n"
    "- Be concise and direct. Do not restate the question."
)

NO_SOURCES = (
    "You are Engram. Nothing in the user's memories or documents matched this question. "
    "Say so plainly and do not fabricate citations."
)


def render_sources(sources: Sequence[tuple[str, str]]) -> str:
    """Number ``(label, text)`` pairs from 1, in rank order.

    The label ("memory", "report.pdf p.3") is part of the prompt because a document
    quote and a stored fact carry different authority, and the model cannot tell them
    apart from the text alone.
    """
    return "\n".join(
        f"[{i}] ({label}) {text.strip()[:1200]}"
        for i, (label, text) in enumerate(sources, start=1)
    )


def build_rag_messages(
    question: str,
    sources: Sequence[tuple[str, str]],
    *,
    history: Sequence[ChatMessage] = (),
    max_context_chars: int = 6000,
) -> list[ChatMessage]:
    """Same shape and same truncation guard as :func:`build_messages`, over sources that
    may be memories or document chunks. Chunks are far longer than facts, so the
    character budget does real work here rather than being a formality."""
    if not sources:
        return [ChatMessage("system", NO_SOURCES), *history, ChatMessage("user", question)]

    rendered = render_sources(sources)
    if len(rendered) > max_context_chars:
        kept: list[str] = []
        budget = max_context_chars
        for line in rendered.split("\n"):
            if len(line) + 1 > budget:
                break
            kept.append(line)
            budget -= len(line) + 1
        rendered = "\n".join(kept)

    user = f"SOURCES:\n{rendered}\n\nQUESTION: {question}"
    return [ChatMessage("system", RAG_SYSTEM), *history, ChatMessage("user", user)]
