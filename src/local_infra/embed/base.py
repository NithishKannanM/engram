"""The embedder seam.

Every component that turns text into vectors (chunker, retriever, backends)
depends on this Protocol, never on a concrete engine. The default is
:class:`~local_infra.embed.ollama.OllamaEmbedder` (nomic-embed-text, 768-dim,
no torch); sentence-transformers is an optional swap-in behind the same shape.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Encode text to L2-normalized float32 vectors (inner product == cosine)."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(N, dim)`` float32 array for ``texts`` (empty -> ``(0, dim)``)."""
        ...

    def encode_one(self, text: str) -> np.ndarray:
        """Return a single ``(dim,)`` vector."""
        ...

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        ...
