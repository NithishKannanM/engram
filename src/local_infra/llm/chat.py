"""Chat over the memory service: recall, ground, generate, cite.

Deliberately NOT a method on ``MemoryService``. That ABC is the backend swap seam
with eight abstract methods; adding a ninth would break ``ExMemBackend``,
``CogneeBackend`` and every fake in the contract suite for a capability that has
nothing to do with being a ledger. Chat composes a ``MemoryService`` instead of
extending it, and is wired onto ``app.state.chat`` behind its own dependency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from local_infra.core.service import GLOBAL, MemoryService, RecallHit
from local_infra.llm.prompt import build_messages
from local_infra.llm.router import ChatRouter
from local_infra.llm.types import ChatMessage, ChatResult, StreamDelta


@dataclass
class ChatAnswer:
    """A generated answer plus the memories it was grounded in.

    ``citations`` is ordered to match the ``[n]`` markers in ``text`` — index 0 is
    ``[1]``. The desktop resolves markers positionally, so this ordering is contract.
    """

    text: str
    alias: str
    model: str
    scope: str
    citations: list[RecallHit] = field(default_factory=list)
    usage: dict[str, object] = field(default_factory=dict)
    fallback_used: bool = False


def _usage_dict(result: ChatResult) -> dict[str, object]:
    u = result.usage
    return {
        "model": u.model,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cost_usd": u.cost_usd,
        "cost_unknown": u.cost_unknown,
    }


class ChatService:
    def __init__(
        self,
        memory: MemoryService,
        router: ChatRouter,
        *,
        recall_limit: int = 6,
    ) -> None:
        self._memory = memory
        self._router = router
        self._recall_limit = recall_limit

    @property
    def router(self) -> ChatRouter:
        """The routing policy, exposed so RAG synthesis can reuse it.

        RAG must NOT call :meth:`ask` — that runs its own memory-only recall, which
        would answer from a different evidence set than the fused memory+document one
        it is meant to cite.
        """
        return self._router

    def aliases(self) -> list[dict[str, object]]:
        return self._router.describe()

    async def _ground(self, question: str, scope: str) -> tuple[list[RecallHit], list[ChatMessage]]:
        hits = await self._memory.recall(question, scope=scope, limit=self._recall_limit)
        return list(hits), build_messages(question, hits)

    async def ask(
        self,
        question: str,
        *,
        scope: str = GLOBAL,
        alias: str = "default",
        history: Sequence[ChatMessage] = (),
    ) -> ChatAnswer:
        hits = list(
            await self._memory.recall(question, scope=scope, limit=self._recall_limit)
        )
        messages = build_messages(question, hits, history=history)
        result = await self._router.complete(alias, messages)
        return ChatAnswer(
            text=result.text,
            alias=result.alias,
            model=result.model,
            scope=scope,
            citations=hits,
            usage=_usage_dict(result),
            fallback_used=result.fallback_used,
        )

    async def ask_stream(
        self,
        question: str,
        *,
        scope: str = GLOBAL,
        alias: str = "default",
        history: Sequence[ChatMessage] = (),
    ) -> AsyncIterator[StreamDelta | ChatAnswer]:
        """Yield deltas, then one final :class:`ChatAnswer` with usage and citations.

        Citations are known *before* the first token (recall runs first), but they are
        emitted at the end alongside usage so the client has a single completion frame
        to act on.
        """
        hits = list(
            await self._memory.recall(question, scope=scope, limit=self._recall_limit)
        )
        messages = build_messages(question, hits, history=history)

        async for item in self._router.stream(alias, messages):
            if isinstance(item, StreamDelta):
                yield item
                continue
            yield ChatAnswer(
                text="",
                alias=item.alias,
                model=item.model,
                scope=scope,
                citations=hits,
                usage=_usage_dict(item),
                fallback_used=item.fallback_used,
            )
