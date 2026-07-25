"""Topic-boundary chunker (vendored from ex_mem).

Embeds each turn, fires a chunk boundary when consecutive-turn cosine distance
exceeds a threshold, keeping thematically related turns together.

BUG FIX ON PORT — first-turn token accounting. The original seeded
``current_token_count = 0`` and only added ``turns[i]`` inside a loop starting at
``i = 1``, so turn 0's tokens were never counted toward ``force_split`` (and, after
each split, the boundary turn's tokens were likewise dropped). A single turn larger
than ``chunk_max_tokens`` therefore never forced a split. Fixed by seeding the
running total with the segment's first turn and resetting to the boundary turn's
tokens after each split.
"""

from __future__ import annotations

import time
import uuid
from typing import Sequence

import numpy as np

from local_infra.embed.base import Embedder
from local_infra.exmem.config import VectorConfig
from local_infra.exmem.store import Chunk
from local_infra.exmem.types import Turn


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity for normalized vectors (fast path)."""
    return float(1.0 - np.dot(a, b))


class SemanticChunker:
    """Converts a stream of Turns into Chunks using embedding-based topic boundaries."""

    def __init__(self, embed: Embedder, cfg: VectorConfig) -> None:
        self._embed = embed
        self._cfg = cfg

    def chunk(self, turns: Sequence[Turn]) -> list[Chunk]:
        if not turns:
            return []

        texts = [t.content for t in turns]
        vecs = self._embed.encode(texts)

        boundaries: list[int] = [0]
        # FIX: seed with turn 0's tokens so a large first turn can force a split.
        seg_tokens = turns[0].token_count

        for i in range(1, len(turns)):
            dist = _cosine_distance(vecs[i - 1], vecs[i])
            seg_tokens += turns[i].token_count

            force_split = seg_tokens >= self._cfg.chunk_max_tokens
            topic_shift = dist >= self._cfg.topic_shift_threshold

            if topic_shift or force_split:
                boundaries.append(i)
                # FIX: the new segment starts at turn i, so its running total
                # begins at turn i's tokens (not 0, which dropped the boundary turn).
                seg_tokens = turns[i].token_count

        boundaries.append(len(turns))  # sentinel

        chunks: list[Chunk] = []
        for start, end in zip(boundaries, boundaries[1:]):
            span = turns[start:end]
            combined = "\n".join(f"[{t.role.upper()}] {t.content}" for t in span)
            total_tokens = sum(t.token_count for t in span)

            if total_tokens < self._cfg.chunk_min_tokens and chunks:
                # Too small — merge into previous chunk.
                prev = chunks[-1]
                chunks[-1] = Chunk(
                    id=prev.id,
                    text=prev.text + "\n" + combined,
                    turn_start=prev.turn_start,
                    turn_end=span[-1].turn_index,
                    role="mixed",
                    timestamp=prev.timestamp,
                    importance=prev.importance,
                )
            else:
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        text=combined,
                        turn_start=span[0].turn_index,
                        turn_end=span[-1].turn_index,
                        role=span[0].role if len({t.role for t in span}) == 1 else "mixed",
                        timestamp=time.time(),
                        importance=1.0,
                    )
                )

        return chunks
