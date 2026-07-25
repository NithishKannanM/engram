"""M2 verification: the three known ex_mem bugs are fixed on port.

One test per bug:
  1. persistence across restart  (VectorStrategy honored db_path)
  2. first-turn token counting    (SemanticChunker force_split)
  3. one encode call per add       (HybridRetriever incremental indexing)

All use a deterministic FakeEmbedder so no model, GPU, or network is involved.
"""

from __future__ import annotations

import hashlib

import numpy as np

from local_infra.exmem.chunker import SemanticChunker
from local_infra.exmem.config import RetrievalConfig, VectorConfig
from local_infra.exmem.retrieval import HybridRetriever
from local_infra.exmem.store import Chunk, SQLiteMetadataStore
from local_infra.exmem.strategy import VectorStrategy
from local_infra.exmem.types import Turn

DIM = 8


class FakeEmbedder:
    """Deterministic unit vectors from a text hash; records embedding calls."""

    def __init__(self, *, constant: bool = False) -> None:
        self._constant = constant  # True -> every text maps to the same vector
        self.encode_calls = 0
        self.texts_embedded = 0

    def _vec(self, text: str) -> np.ndarray:
        key = "" if self._constant else text
        seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
        v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        self.encode_calls += 1
        self.texts_embedded += len(texts)
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        return DIM

    def reset_counts(self) -> None:
        self.encode_calls = 0
        self.texts_embedded = 0


# --------------------------------------------------------------------------- #
# Bug 1: persistence across restart                                            #
# --------------------------------------------------------------------------- #
def test_vector_strategy_persists_across_restart(tmp_path) -> None:
    db = str(tmp_path / "store.db")
    cfg = VectorConfig(max_active_tokens=10, chunk_min_tokens=0, chunk_max_tokens=1000, db_path=db)
    rcfg = RetrievalConfig()

    # First "process": stream enough turns to force evictions into the file store.
    strat1 = VectorStrategy(cfg, FakeEmbedder(), rcfg, token_counter=lambda _t: 10)
    for i in range(8):
        strat1.add_turn("user", f"fact {i} apple banana cherry")
    persisted = strat1._store.count()
    assert persisted > 0, "eviction produced no persisted chunks — nothing to restart from"

    # Second "process": a brand-new strategy on the SAME path must recover state,
    # and its retriever must rebuild the index from the persisted store.
    strat2 = VectorStrategy(cfg, FakeEmbedder(), rcfg, token_counter=lambda _t: 10)
    assert strat2._store.count() == persisted, "restart lost persisted chunks"
    ctx = strat2.get_context(query="apple banana")
    assert ctx.retrieved_chunks, "restarted strategy retrieved nothing from persisted store"


# --------------------------------------------------------------------------- #
# Bug 2: first-turn token counting                                             #
# --------------------------------------------------------------------------- #
def test_chunker_counts_first_turn_tokens() -> None:
    # constant embedder -> zero cosine distance -> no topic shifts, so ONLY the
    # token-based force_split can create a boundary.
    cfg = VectorConfig(chunk_min_tokens=0, chunk_max_tokens=100, topic_shift_threshold=0.35)
    chunker = SemanticChunker(FakeEmbedder(constant=True), cfg)

    turns = [
        Turn(role="user", content="x", turn_index=0, token_count=150),  # alone exceeds max
        Turn(role="user", content="x", turn_index=1, token_count=1),
    ]
    chunks = chunker.chunk(turns)

    # With the bug, turn 0's 150 tokens were ignored and only turn 1's 1 token
    # counted -> no force_split -> a single chunk. The fix forces the split.
    assert len(chunks) == 2, f"first-turn tokens not counted toward force_split ({len(chunks)} chunks)"


# --------------------------------------------------------------------------- #
# Bug 3: incremental indexing (one encode call per add)                        #
# --------------------------------------------------------------------------- #
def test_retriever_adds_incrementally_without_reembedding() -> None:
    embed = FakeEmbedder()
    store = SQLiteMetadataStore(":memory:")
    retriever = HybridRetriever(embed, store, RetrievalConfig())

    base = [
        Chunk(id=str(i), text=f"chunk number {i} content", turn_start=i, turn_end=i, role="user")
        for i in range(500)
    ]
    retriever.add_chunks(base)  # one batch encode of 500

    embed.reset_counts()
    retriever.add_chunks(
        [Chunk(id="new", text="the freshly added chunk", turn_start=500, turn_end=500, role="user")]
    )

    # The old code re-embedded all 501 chunks on the next build; the fix embeds
    # only the newcomer.
    assert embed.encode_calls == 1, f"adding 1 chunk made {embed.encode_calls} encode calls"
    assert embed.texts_embedded == 1, f"re-embedded {embed.texts_embedded} texts, expected 1"
    assert len(retriever._indexed_chunks) == 501
