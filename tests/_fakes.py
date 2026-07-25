"""Scripted fake extraction providers for the ladder tests (no model, no network)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from local_infra.extraction.providers.base import ProviderResult, Stage
from local_infra.extraction.schemas import ExtractedFact, ExtractionResult

# A source with two clearly-locatable facts, used for grounding.
SRC = "Nithish works at Globex as a backend engineer. He lives in Chennai."

GOOD = dict(subject="nithish", predicate="works_at", object="globex",
            evidence_span="Nithish works at Globex", confidence=0.9)
GOOD2 = dict(subject="nithish", predicate="lives_in", object="chennai",
             evidence_span="lives in Chennai", confidence=0.9)
# Unrelated span -> groundedness ~0.45, composite <0.5 -> jumps two rungs.
HALLUC = dict(subject="nithish", predicate="studies", object="quantum",
              evidence_span="quantum tunneling in graphene lattices", confidence=0.0)
# Partial-match span -> groundedness 0.5 (<floor) but composite in [0.5,0.75] -> jump one.
MED = dict(subject="nithish", predicate="hobby", object="skydiving",
           evidence_span="Nithish enjoys skydiving on weekends", confidence=0.9)


class FakeProvider:
    def __init__(self, stage: Stage, scripts: Sequence) -> None:
        self.stage = stage
        self._scripts = list(scripts)
        self.calls: list[list[str] | None] = []  # focus per call

    async def extract(self, text, *, focus=None, seed=0) -> ProviderResult:
        self.calls.append(list(focus) if focus else None)
        script = self._scripts.pop(0)
        return script(focus) if callable(script) else script


def facts_result(*facts: dict, cost: float = 0.0) -> ProviderResult:
    payload = ExtractionResult(facts=[ExtractedFact(**f) for f in facts]).model_dump_json()
    return ProviderResult(raw=payload, cost_usd=cost)


def empty_result() -> ProviderResult:
    return ProviderResult(raw=ExtractionResult(facts=[]).model_dump_json())


def transport_fail() -> ProviderResult:
    return ProviderResult(raw=None, ok=False, error="connection refused")


def bad_json() -> ProviderResult:
    return ProviderResult(raw="{ this is not valid json")


class MappingProvider:
    """Content-addressed fake extractor: returns the facts registered for an input
    string, or none. Lets a service-contract test drive extraction deterministically
    with real inputs (unlike positional scripts), still with no model and no network."""

    def __init__(self, stage: Stage, mapping: dict[str, list[dict]]) -> None:
        self.stage = stage
        self._mapping = mapping
        self.calls: list[str] = []

    async def extract(self, text, *, focus=None, seed=0) -> ProviderResult:
        self.calls.append(text)
        facts = self._mapping.get(text.strip(), [])
        return facts_result(*facts)


class HashEmbedder:
    """Deterministic, offline embedder for tests: a stable hash seeds a normalized
    random vector per text. Identical text -> identical vector (cosine 1.0); unrelated
    text -> cosine ~0. Implements the :class:`Embedder` protocol, no torch, no Ollama."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.model = "hash-test"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            v = rng.standard_normal(self._dim).astype(np.float32)
            norm = float(np.linalg.norm(v)) or 1.0
            out[i] = v / norm
        return out

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        return self._dim
