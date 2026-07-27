"""Retrieval over memories *and* documents, fused into one ranked, citable list.

Two decisions here are load-bearing:

1. **The memory channel is ``MemoryService.recall()``, not re-implemented SQL.** Recall
   owns the temporal-visibility rule (a superseded fact must not come back as current)
   and differs per backend. Re-deriving it here would fork that invariant into a second
   place and quietly resurrect old facts on one backend only.
2. **Fusion is the existing ``rrf_fuse`` over *tagged* keys** — ``("memory", uuid)`` vs
   ``("chunk", uuid)``. Both id spaces are UUIDs, so an untagged fusion would happily
   collide them, and rank-based RRF is what lets a pgvector distance and a ``ts_rank``
   be combined at all without inventing a normalisation between incomparable scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from local_infra.core.fusion import rrf_fuse
from local_infra.core.service import GLOBAL, MemoryService, RecallHit
from local_infra.documents.store import DocumentStore
from local_infra.llm.prompt import build_rag_messages

#: Channel weights. Memories are curated, deduplicated facts, so they outrank raw prose;
#: dense beats sparse on paraphrase, which is the common case for a question against a
#: document. These are ranking weights, not probabilities — they need not sum to 1.
W_MEMORY = 1.0
W_CHUNK_DENSE = 0.9
W_CHUNK_SPARSE = 0.6


@dataclass
class RagCitation:
    """One cited source. ``kind`` decides which half of the payload is populated —
    mirroring ``RecallHit``, so the desktop's citation drawer can stay one component."""

    kind: str  # "memory" | "document"
    text: str
    score: float
    # memory
    memory_id: str | None = None
    statement: str | None = None
    valid_until: str | None = None
    # document
    document_id: str | None = None
    filename: str | None = None
    chunk_index: int | None = None
    page_from: int | None = None
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class RagResult:
    scope: str
    query: str
    citations: list[RagCitation] = field(default_factory=list)
    answer: str | None = None
    synthesized: bool = False
    model: str | None = None
    usage: dict[str, object] = field(default_factory=dict)
    fallback_used: bool = False
    #: Channels that could not run on this backend or in this configuration, e.g.
    #: ["dense"] with no embedder. Reported rather than silently returning less.
    unavailable: list[str] = field(default_factory=list)


def _label(citation: RagCitation) -> str:
    if citation.kind == "memory":
        return "memory (superseded)" if citation.valid_until else "memory"
    page = f" p.{citation.page_from}" if citation.page_from else ""
    return f"{citation.filename}{page}"


class RagService:
    def __init__(
        self,
        memory: MemoryService,
        store: DocumentStore,
        *,
        chat=None,
        limit: int = 8,
    ) -> None:
        self._memory = memory
        self._store = store
        self._chat = chat
        self._limit = limit

    async def query(
        self,
        question: str,
        *,
        scope: str = GLOBAL,
        limit: int | None = None,
        synthesize: bool = False,
        alias: str = "default",
    ) -> RagResult:
        top = limit or self._limit
        # Over-fetch per channel: fusion is only meaningful if each ranker contributes
        # more candidates than the final cut.
        depth = max(top * 2, 10)

        hits: list[RecallHit] = list(
            await self._memory.recall(question, scope=scope, limit=depth)
        )
        dense = await self._store.dense_ranked(scope, question, limit=depth)
        sparse = await self._store.sparse_ranked(scope, question, limit=depth)

        unavailable: list[str] = []
        if not dense and getattr(self._store, "embedder", None) is None:
            unavailable.append("dense")

        by_memory: dict[UUID, RecallHit] = {
            hit.memory.id: hit for hit in hits if hit.memory is not None
        }
        fused = rrf_fuse(
            [
                (W_MEMORY, [("memory", mid) for mid in by_memory]),
                (W_CHUNK_DENSE, [("chunk", cid) for cid in dense]),
                (W_CHUNK_SPARSE, [("chunk", cid) for cid in sparse]),
            ]
        )[:top]

        chunk_ids = [key[1] for key in fused if key[0] == "chunk"]
        chunks = {hit.chunk.id: hit for hit in await self._store.hydrate(chunk_ids)}

        citations: list[RagCitation] = []
        for rank, (channel, item_id) in enumerate(fused):
            # Score is the fused *rank* expressed as a descending value, not a channel
            # score: the underlying scales are not comparable, which is why RRF is used.
            score = 1.0 / (rank + 1)
            if channel == "memory":
                hit = by_memory.get(item_id)
                if hit is None or hit.memory is None:
                    continue
                citations.append(
                    RagCitation(
                        kind="memory",
                        text=hit.memory.statement,
                        score=score,
                        memory_id=str(hit.memory.id),
                        statement=hit.memory.statement,
                        valid_until=(
                            hit.memory.valid_until.isoformat()
                            if hit.memory.valid_until
                            else None
                        ),
                    )
                )
                continue
            chunk_hit = chunks.get(item_id)
            if chunk_hit is None:
                continue  # deleted between ranking and hydration
            chunk = chunk_hit.chunk
            citations.append(
                RagCitation(
                    kind="document",
                    text=chunk.content,
                    score=score,
                    document_id=str(chunk.document_id),
                    filename=chunk_hit.filename,
                    chunk_index=chunk.chunk_index,
                    page_from=chunk.page_from,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                )
            )

        result = RagResult(
            scope=scope,
            query=question,
            citations=citations,
            unavailable=unavailable,
        )
        if synthesize and self._chat is not None:
            await self._synthesize(result, question, alias=alias)
        elif synthesize:
            result.unavailable.append("synthesis")
        return result

    async def _synthesize(self, result: RagResult, question: str, *, alias: str) -> None:
        """Generate prose over the already-fused sources.

        Goes through the router directly rather than ``ChatService.ask``: that method
        runs its own memory-only recall, which would discard the document half of this
        fusion and answer from a different evidence set than the one being cited.
        """
        messages = build_rag_messages(
            question, [(_label(c), c.text) for c in result.citations]
        )
        completion = await self._chat.router.complete(alias, messages)
        result.answer = completion.text
        result.synthesized = True
        result.model = completion.model
        result.fallback_used = completion.fallback_used
        result.usage = {
            "model": completion.usage.model,
            "input_tokens": completion.usage.input_tokens,
            "output_tokens": completion.usage.output_tokens,
            "cost_usd": completion.usage.cost_usd,
            "cost_unknown": completion.usage.cost_unknown,
        }
