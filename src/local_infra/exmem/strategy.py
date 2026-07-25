"""Vector (FAISS + BM25 + RRF) memory strategy (vendored from ex_mem).

BUG FIX ON PORT — persistence across restart. The original hardcoded ``:memory:``
in both ``__init__`` and ``reset``, making ``VectorConfig.db_path`` dead config: no
state survived process exit. Here the store is opened at ``cfg.db_path`` (overridable
per-instance), so a new strategy pointed at the same file recovers its chunks and the
retriever rebuilds from them on first query.

Also decoupled from sentence-transformers: the embedder and token counter are
injected (an :class:`Embedder` and a ``str -> int`` callable), so the strategy is
testable with fakes and swappable per the design's embedder seam.
"""

from __future__ import annotations

import time
from typing import Callable

from local_infra.embed.base import Embedder
from local_infra.exmem.chunker import SemanticChunker
from local_infra.exmem.config import RetrievalConfig, VectorConfig
from local_infra.exmem.retrieval import HybridRetriever
from local_infra.exmem.store import SQLiteMetadataStore
from local_infra.exmem.types import ContextWindow, MemoryStrategy, StrategyStats, Turn


def default_token_counter() -> Callable[[str], int]:
    """tiktoken cl100k_base if available, else a whitespace word count."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    except Exception:
        return lambda text: len(text.split())


class VectorStrategy(MemoryStrategy):
    name = "vector"

    def __init__(
        self,
        cfg: VectorConfig,
        embed: Embedder,
        retrieval_cfg: RetrievalConfig,
        *,
        token_counter: Callable[[str], int] | None = None,
        db_path: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._retrieval_cfg = retrieval_cfg
        self._embed = embed
        self._count = token_counter or default_token_counter()
        # FIX: honor cfg.db_path (was always ":memory:").
        self._db_path = db_path if db_path is not None else cfg.db_path

        self._store = SQLiteMetadataStore(self._db_path)
        self._chunker = SemanticChunker(embed, cfg)
        self._retriever = HybridRetriever(embed, self._store, retrieval_cfg)

        self._active: list[Turn] = []
        self._stats = StrategyStats(strategy_name=self.name)
        self._turn_idx = 0

    # -- write --

    def add_turn(self, role: str, content: str) -> None:
        t0 = time.perf_counter()
        tokens = self._count(content)
        turn = Turn(role=role, content=content, turn_index=self._turn_idx, token_count=tokens)
        self._turn_idx += 1
        self._active.append(turn)
        self._stats.total_turns += 1
        self._stats.total_tokens_active += tokens

        if sum(t.token_count for t in self._active) > self._cfg.max_active_tokens:
            self._evict()

        self._stats.turn_latency_ms.append((time.perf_counter() - t0) * 1000)

    # -- read --

    def get_context(self, query: str = "") -> ContextWindow:
        t0 = time.perf_counter()
        retrieved: list[str] = []

        if query and self._store.count() > 0:
            results = self._retriever.retrieve(query)
            retrieved = [r.chunk.text for r in results]
            self._stats.total_retrieval_calls += 1
            self._stats.total_tokens_retrieved += sum(self._count(c) for c in retrieved)
            self._stats.retrieval_latency_ms.append((time.perf_counter() - t0) * 1000)

        active_tokens = sum(t.token_count for t in self._active)
        retrieved_tokens = sum(self._count(c) for c in retrieved)
        return ContextWindow(
            turns=list(self._active),
            retrieved_chunks=retrieved,
            total_tokens=active_tokens + retrieved_tokens,
        )

    # -- evict --

    def _evict(self) -> None:
        """Move oldest turns out of the active buffer into the vector store."""
        keep_recent = max(4, len(self._active) // 3)
        to_evict = self._active[:-keep_recent]
        if not to_evict:
            return

        chunks = self._chunker.chunk(to_evict)
        for chunk in chunks:
            self._store.upsert(chunk)
        # Incremental index update (embeds only the new chunks).
        self._retriever.add_chunks(chunks)

        self._stats.embedding_calls += 1
        self._active = self._active[-keep_recent:]

    # -- util --

    def get_stats(self) -> StrategyStats:
        self._stats.total_evictions = self._store.count()
        return self._stats

    def reset(self) -> None:
        self._active.clear()
        self._stats = StrategyStats(strategy_name=self.name)
        self._turn_idx = 0
        # FIX: re-open at the configured path, not a forced ":memory:".
        self._store = SQLiteMetadataStore(self._db_path)
        self._chunker = SemanticChunker(self._embed, self._cfg)
        self._retriever = HybridRetriever(self._embed, self._store, self._retrieval_cfg)
