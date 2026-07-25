"""Hybrid retriever: FAISS dense + BM25 sparse fused with RRF (vendored from ex_mem).

Reciprocal Rank Fusion:  score(d) = Σ w_i / (k + rank_i(d)),  k=60 by convention.

BUG FIX ON PORT — incremental indexing. The original re-embedded *every* chunk on
any ``mark_dirty()``, so streaming N chunks in one at a time cost O(N²) embedding
calls. Here the FAISS index is grown incrementally: adding a chunk embeds only that
chunk (one encode call). BM25 has no incremental API, so its lightweight tokenized
corpus is rebuilt each add — that is CPU-only string work, never an embedding call,
which is the cost the fix targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from local_infra.embed.base import Embedder
from local_infra.exmem.config import RetrievalConfig
from local_infra.exmem.store import Chunk, SQLiteMetadataStore


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_rank: int = -1
    sparse_rank: int = -1


class HybridRetriever:
    def __init__(
        self,
        embed: Embedder,
        store: SQLiteMetadataStore,
        cfg: RetrievalConfig,
    ) -> None:
        self._embed = embed
        self._store = store
        self._cfg = cfg
        self._faiss_index = None
        self._bm25 = None
        self._indexed_chunks: list[Chunk] = []  # parallel to FAISS rows
        self._built = False

    # -- build --

    def _ensure_built(self) -> None:
        """Lazily build the index from the store's current contents (one batch encode)."""
        if self._built:
            return
        self._built = True  # set first so _add_vectors doesn't recurse
        self._faiss_index = None
        self._bm25 = None
        self._indexed_chunks = []
        chunks = self._store.all_chunks()
        if chunks:
            self._add_vectors(chunks)

    def _add_vectors(self, chunks: list[Chunk]) -> None:
        import faiss
        from rank_bm25 import BM25Okapi

        vecs = self._embed.encode([c.text for c in chunks])  # <-- the only encode call
        if self._faiss_index is None:
            self._faiss_index = faiss.IndexFlatIP(vecs.shape[1])  # cosine on normalized vecs
        self._faiss_index.add(vecs)
        self._indexed_chunks.extend(chunks)
        # BM25 rebuild = tokenization only, no embedding.
        self._bm25 = BM25Okapi([c.text.lower().split() for c in self._indexed_chunks])

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Incrementally index new chunks, embedding ONLY those chunks."""
        self._ensure_built()
        chunks = list(chunks)
        if not chunks:
            return
        self._add_vectors(chunks)

    # -- query --

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        self._ensure_built()
        if not self._indexed_chunks:
            return []

        k = top_k or self._cfg.top_k
        n = len(self._indexed_chunks)
        effective_k = min(k, n)

        # Dense.
        q_vec = self._embed.encode_one(query).reshape(1, -1)
        _, idxs = self._faiss_index.search(q_vec, effective_k)
        dense_ranking: dict[int, int] = {}
        for rank, idx in enumerate(idxs[0]):
            if idx >= 0:
                dense_ranking[int(idx)] = rank + 1

        # Sparse (BM25).
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_order = np.argsort(bm25_scores)[::-1][:effective_k]
        sparse_ranking: dict[int, int] = {
            int(idx): rank + 1 for rank, idx in enumerate(bm25_order)
        }

        # RRF fusion.
        rrf_k = self._cfg.rrf_k
        fused: list[tuple[float, int]] = []
        for idx in set(dense_ranking) | set(sparse_ranking):
            dr = dense_ranking.get(idx, n + 1)
            sr = sparse_ranking.get(idx, n + 1)
            rrf = self._cfg.dense_weight / (rrf_k + dr) + self._cfg.bm25_weight / (rrf_k + sr)
            fused.append((rrf, idx))

        fused.sort(key=lambda x: x[0], reverse=True)
        return [
            ScoredChunk(
                chunk=self._indexed_chunks[idx],
                score=score,
                dense_rank=dense_ranking.get(idx, -1),
                sparse_rank=sparse_ranking.get(idx, -1),
            )
            for score, idx in fused[:k]
        ]
