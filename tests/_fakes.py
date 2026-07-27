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


# --------------------------------------------------------------------------- #
# Chat transport fakes.
#
# Mirrors FakeProvider's scripted style. No test may import litellm — it fetches a
# model-cost map over the network at import time, so the whole chat layer is tested
# against these instead.
# --------------------------------------------------------------------------- #
class FakeCompletionClient:
    """Scripted completion transport.

    ``scripts`` entries are either a ``(text, ChatUsage)`` tuple to return or an
    ``Exception`` instance to raise. ``self.calls`` records the model of every
    attempt in order, so a fallback test can assert the exact chain that was tried.
    """

    def __init__(self, scripts: Sequence[object] | None = None) -> None:
        self._scripts = list(scripts or [])
        self.calls: list[str] = []
        self.messages: list[list[dict[str, str]]] = []

    def _next(self, model: str):
        self.calls.append(model)
        if not self._scripts:
            raise AssertionError(f"FakeCompletionClient ran out of scripts at {model!r}")
        return self._scripts.pop(0)

    async def complete(self, deployment, messages, *, temperature, max_tokens, timeout_s):
        self.messages.append([m.as_dict() for m in messages])
        item = self._next(deployment.model)
        if isinstance(item, Exception):
            raise item
        text, usage = item
        return text, usage

    async def stream(self, deployment, messages, *, temperature, max_tokens, timeout_s):
        self.messages.append([m.as_dict() for m in messages])
        item = self._next(deployment.model)
        if isinstance(item, Exception):
            raise item
        chunks, usage = item
        for chunk in chunks:
            yield chunk
        yield usage


# --------------------------------------------------------------------------- #
# Document subsystem fake.
#
# An in-memory DocumentStore so the routes, the pipeline and the RAG fusion can be
# tested with no Postgres and no Ollama. It is also a second implementation of the
# `DocumentStore` protocol, which is the cheapest available check that the protocol
# describes something implementable rather than PgDocumentStore's exact SQL.
# --------------------------------------------------------------------------- #
class FakeDocumentStore:
    """In-memory documents, chunks and jobs.

    Ranking is deliberately crude — substring for sparse, cosine over the injected
    embedder for dense. The point of these tests is the FUSION and the plumbing, not
    Postgres' ranking quality, which `test_documents_pg.py` covers against real SQL.
    """

    def __init__(self, embedder=None) -> None:
        self.embedder = embedder
        self.documents: dict[object, object] = {}
        self.chunks: dict[object, list] = {}
        self.jobs: dict[object, tuple] = {}
        self._queue: list[object] = []
        self._scopes: dict[str, int] = {}
        self.closed = False

    # -- helpers --
    def _scope_id(self, scope: str) -> int:
        return self._scopes.setdefault(scope, len(self._scopes) + 1)

    async def close(self) -> None:
        self.closed = True

    # -- documents --
    async def create(self, *, scope, filename, kind, sha256, size_bytes, metadata=None):
        from local_infra.documents.models import Document

        for doc in self.documents.values():
            if doc.scope == scope and doc.sha256 == sha256:
                return doc, False
        import uuid

        doc = Document(
            id=uuid.uuid4(),
            scope_id=self._scope_id(scope),
            scope=scope,
            filename=filename,
            kind=kind,
            sha256=sha256,
            size_bytes=size_bytes,
            status="pending",
            metadata=metadata or {},
        )
        self.documents[doc.id] = doc
        self.chunks[doc.id] = []
        return doc, True

    async def get(self, document_id):
        return self.documents.get(document_id)

    async def list_documents(self, scope, *, limit: int = 100):
        return [d for d in self.documents.values() if d.scope == scope][:limit]

    async def mark(self, document_id, status, *, error=None, pages=None, chunk_count=None):
        doc = self.documents.get(document_id)
        if doc is None:
            return
        doc.status = status
        doc.error = error
        if pages is not None:
            doc.pages = pages
        if chunk_count is not None:
            doc.chunk_count = chunk_count

    async def delete(self, document_id):
        doc = self.documents.pop(document_id, None)
        if doc is not None:
            self.chunks.pop(document_id, None)
        return doc

    async def sha_in_use(self, sha256) -> bool:
        return any(d.sha256 == sha256 for d in self.documents.values())

    # -- chunks --
    async def replace_chunks(self, document_id, scope_id, chunks):
        from local_infra.documents.models import DocumentChunk
        import uuid

        if document_id not in self.documents:
            import asyncpg

            # Mirrors the real store: the FK is what makes a delete-during-embed race
            # surface as an error the pipeline can recognise instead of a silent write.
            raise asyncpg.ForeignKeyViolationError("document was deleted")
        stored = [
            DocumentChunk(
                id=uuid.uuid4(),
                document_id=document_id,
                scope_id=scope_id,
                chunk_index=c.chunk_index,
                content=c.content,
                char_start=c.char_start,
                char_end=c.char_end,
                page_from=c.page_from,
                tokens=c.tokens,
            )
            for c in chunks
        ]
        self._vectors = getattr(self, "_vectors", {})
        for stored_chunk, pending in zip(stored, chunks):
            self._vectors[stored_chunk.id] = pending.embedding
        self.chunks[document_id] = stored
        return len(stored)

    async def chunks_for(self, document_id):
        return list(self.chunks.get(document_id, []))

    def _all_chunks(self, scope: str | None = None):
        for doc_id, chunks in self.chunks.items():
            doc = self.documents.get(doc_id)
            if doc is None or (scope is not None and doc.scope != scope):
                continue
            for chunk in chunks:
                yield doc, chunk

    async def hydrate(self, chunk_ids):
        from local_infra.documents.models import ChunkHit

        index = {chunk.id: (doc, chunk) for doc, chunk in self._all_chunks()}
        hits = []
        for chunk_id in chunk_ids:
            found = index.get(chunk_id)
            if found is None:
                continue
            doc, chunk = found
            hits.append(
                ChunkHit(
                    score=0.0,
                    chunk=chunk,
                    document_id=doc.id,
                    filename=doc.filename,
                    page_from=chunk.page_from,
                )
            )
        return hits

    async def dense_ranked(self, scope, query, *, limit: int = 10):
        if self.embedder is None:
            return []
        vectors = getattr(self, "_vectors", {})
        q = self.embedder.encode_one(query)
        scored = []
        for _, chunk in self._all_chunks(scope):
            vec = vectors.get(chunk.id)
            if vec is None:
                continue
            scored.append((float(np.dot(q, np.array(vec, dtype=np.float32))), chunk.id))
        scored.sort(key=lambda pair: -pair[0])
        return [cid for _, cid in scored[:limit]]

    async def sparse_ranked(self, scope, query, *, limit: int = 10):
        terms = [t for t in query.lower().split() if len(t) > 2]
        scored = []
        for _, chunk in self._all_chunks(scope):
            body = chunk.content.lower()
            overlap = sum(1 for t in terms if t in body)
            if overlap:
                scored.append((overlap, chunk.id))
        scored.sort(key=lambda pair: -pair[0])
        return [cid for _, cid in scored[:limit]]

    # -- jobs --
    async def enqueue(self, document_id, scope_id):
        import uuid

        job_id = uuid.uuid4()
        self.jobs[job_id] = ("queued", None)
        self._queue.append((job_id, document_id, scope_id))
        return job_id

    async def claim(self):
        from local_infra.documents.models import DocumentJob

        if not self._queue:
            return None
        job_id, document_id, scope_id = self._queue.pop(0)
        self.jobs[job_id] = ("running", None)
        return DocumentJob(id=job_id, document_id=document_id, scope_id=scope_id)

    async def complete(self, job_id, *, status="done", error=None):
        self.jobs[job_id] = (status, error)

    async def job_status(self, job_id):
        entry = self.jobs.get(job_id)
        return entry[0] if entry else None
