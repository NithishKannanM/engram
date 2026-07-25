"""Optional sentence-transformers embedder (ported from ex_mem).

Kept behind the same :class:`Embedder` protocol as an A/B control and an
offline-without-Ollama fallback. Not imported by default — sentence-transformers
(and torch) are only pulled in if this module is actually used.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class SentenceTransformerEmbedder:
    """bge-m3 (or any sentence-transformers model), lazily loaded on first use."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        *,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._model = None  # lazy: importing torch is expensive.

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name, device=self._device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        self._load()
        return int(self._model.get_sentence_embedding_dimension())
