"""Engine configuration (vendored from ex_mem, converted to plain dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    top_k: int = 5
    rrf_k: int = 60  # RRF constant — higher = flatter ranking (Cormack et al., 2009)
    bm25_weight: float = 0.4
    dense_weight: float = 0.6


@dataclass
class VectorConfig:
    max_active_tokens: int = 4_000
    chunk_min_tokens: int = 50
    chunk_max_tokens: int = 300
    topic_shift_threshold: float = 0.35  # cosine distance delta -> new chunk
    # Honored on port (was dead config): file path persists the SQLite store so
    # chunks survive a restart. ":memory:" keeps the old ephemeral behavior.
    db_path: str = "exmem_store.db"
