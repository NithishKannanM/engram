"""Default embedder: Ollama ``nomic-embed-text`` over the local HTTP API.

Chosen over sentence-transformers/bge to avoid loading torch (~2.5GB) on a box
with 8.7GB free RAM, and to reuse the already-running Ollama process. Vectors are
L2-normalized so a FAISS ``IndexFlatIP`` computes cosine similarity directly.

Endpoint note (cost real debugging in the M0.5 spike): the modern ``/api/embed``
takes ``{"model", "input": [...]}`` and returns ``{"embeddings": [[...]]}``. The
older ``/api/embeddings`` wants ``prompt`` and silently returns nothing for this
payload — surfacing much later as a dimension-mismatch on insert.
"""

from __future__ import annotations

from typing import Sequence

import httpx
import numpy as np

from local_infra.config import get_settings


class OllamaEmbedder:
    """Synchronous embedder; the chunker/retriever call it on the worker thread."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        dim: int | None = None,
        *,
        normalize: bool = True,
        timeout: float = 60.0,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.embed_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._dim = dim or settings.embed_dim
        self._normalize = normalize
        self._client = httpx.Client(timeout=timeout)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        resp = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        vecs = np.asarray(resp.json()["embeddings"], dtype=np.float32)
        if vecs.shape[1] != self._dim:
            raise ValueError(
                f"{self.model} returned dim {vecs.shape[1]}, expected {self._dim}; "
                f"a model change requires a full reindex."
            )
        if self._normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vecs = vecs / norms
        return vecs

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        return self._dim

    def close(self) -> None:
        self._client.close()
