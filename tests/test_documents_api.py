"""T6 acceptance over HTTP: upload → ingest → /rag → cite → delete.

Offline and deterministic: a `FakeDocumentStore`, a `HashEmbedder`, and the same fake
extraction ladder the other API tests use. No Postgres, no Ollama, no network.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi.testclient import TestClient

from local_infra.api.app import create_app
from local_infra.backends.exmem_backend import ExMemBackend
from local_infra.backends.sqlite.repository import SqliteRepository
from local_infra.documents.blobs import BlobStore
from local_infra.documents.models import DocumentJob
from local_infra.documents.pipeline import DocumentPipeline
from local_infra.documents.rag import RagService
from local_infra.documents.service import DocumentService
from local_infra.exmem.config import RetrievalConfig
from local_infra.exmem.retrieval import HybridRetriever
from local_infra.exmem.store import SQLiteMetadataStore
from local_infra.extraction.budget import Budget
from local_infra.extraction.ladder import ExtractionLadder
from local_infra.extraction.providers.base import Stage

from tests._fakes import (
    FakeCompletionClient,
    FakeDocumentStore,
    HashEmbedder,
    MappingProvider,
)

HANDBOOK = (
    "# Platform Handbook\n\n"
    "The deployment window is every Tuesday at 14:00 UTC. Releases outside the window "
    "require an exception approved by the platform lead.\n\n"
    "Rollbacks are performed with `make rollback` and always target the previous "
    "released tag. A rollback never requires an exception.\n\n"
    "On-call rotation is weekly and starts on Monday morning.\n"
)


@pytest.fixture
def wiring(tmp_path):
    """A fully wired app plus handles on the pieces, so a test can drive the worker
    deterministically instead of sleeping and hoping."""
    embed = HashEmbedder(dim=64)

    store = SQLiteMetadataStore(":memory:")
    retriever = HybridRetriever(embed, store, RetrievalConfig(top_k=10))
    ladder = ExtractionLadder(
        {Stage.LOCAL_SMALL: MappingProvider(Stage.LOCAL_SMALL, {})},
        offline=True,
        budget=Budget(0.0),
    )
    memory = ExMemBackend(
        SqliteRepository(":memory:"), retriever, ladder, store, offline=True, embedder=embed
    )

    doc_store = FakeDocumentStore(embedder=embed)
    blobs = BlobStore(tmp_path / "blobs")
    pipeline = DocumentPipeline(doc_store, blobs, embedder=embed)
    documents = DocumentService(doc_store, blobs)
    rag = RagService(memory, doc_store, limit=8)

    app = create_app(memory, chat=None, documents=documents)
    with TestClient(app) as client:
        app.state.rag = rag  # the lifespan only builds RAG when it built documents
        yield client, doc_store, pipeline


def _drain(store: FakeDocumentStore, pipeline: DocumentPipeline) -> None:
    """Run every queued ingest job to completion, synchronously."""

    async def run() -> None:
        while (job := await store.claim()) is not None:
            await pipeline(job)
            await store.complete(job.id)

    asyncio.run(run())


def _upload(client: TestClient, body: bytes, name: str, scope: str = "global"):
    return client.post(
        "/documents",
        files={"file": (name, io.BytesIO(body), "application/octet-stream")},
        data={"scope": scope},
    )


# --- upload ---------------------------------------------------------------- #


def test_upload_accepts_and_queues_without_doing_the_work(wiring):
    """202 means accepted, not indexed. A synchronous ingest would hold the request for
    the length of a PDF parse plus dozens of embedding round-trips."""
    client, store, _ = wiring
    response = _upload(client, HANDBOOK.encode(), "handbook.md")

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["queued"] is True and body["job_id"]
    assert body["kind"] == "md" and body["filename"] == "handbook.md"
    # Nothing has been chunked yet — that is the worker's job.
    doc_id = list(store.documents)[0]
    assert store.chunks[doc_id] == []


def test_multipart_parsing_works(wiring):
    """`python-multipart` is a REQUEST-time dependency: without it FastAPI raises only
    when a multipart body arrives, not at import. This test is what makes its absence
    break the suite instead of the user's first upload."""
    client, _, _ = wiring
    assert _upload(client, b"a small note", "note.txt").status_code == 202


def test_reupload_of_identical_bytes_is_idempotent(wiring):
    client, store, pipeline = wiring
    first = _upload(client, HANDBOOK.encode(), "handbook.md").json()
    _drain(store, pipeline)

    second = _upload(client, HANDBOOK.encode(), "handbook.md").json()
    assert second["document_id"] == first["document_id"]
    # Already ingested: no second job, because re-embedding identical bytes would burn
    # GPU time to produce byte-identical chunks.
    assert second["queued"] is False and second["job_id"] is None
    assert len(store.documents) == 1


def test_same_bytes_in_another_scope_is_a_separate_document(wiring):
    client, store, _ = wiring
    a = _upload(client, HANDBOOK.encode(), "handbook.md", scope="project:alpha").json()
    b = _upload(client, HANDBOOK.encode(), "handbook.md", scope="project:beta").json()
    assert a["document_id"] != b["document_id"]


def test_failed_ingest_can_be_retried_by_re_uploading(wiring):
    """A failed ingest (Ollama was down) would otherwise be stuck forever with no way
    to retry short of deleting the document first."""
    client, store, _ = wiring
    body = _upload(client, HANDBOOK.encode(), "handbook.md").json()
    doc_id = list(store.documents)[0]
    asyncio.run(store.mark(doc_id, "failed", error="ollama unreachable"))

    retried = _upload(client, HANDBOOK.encode(), "handbook.md").json()
    assert retried["document_id"] == body["document_id"]
    assert retried["queued"] is True
    assert store.documents[doc_id].status == "pending"
    assert store.documents[doc_id].error is None


# --- rejection ------------------------------------------------------------- #


def test_uploading_an_executable_is_415_not_500(wiring):
    client, _, _ = wiring
    response = _upload(client, b"\x7fELF\x02\x01\x01" + b"\x00" * 64, "payload.txt")
    assert response.status_code == 415
    assert "ELF" in response.json()["detail"]


def test_uploading_a_zip_bomb_is_413(wiring):
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", b"<w:document/>")
        zf.writestr("bomb.bin", b"\x00" * 40_000_000)

    client, _, _ = wiring
    response = _upload(client, buf.getvalue(), "bomb.docx")
    assert response.status_code == 413


def test_empty_upload_is_400(wiring):
    client, _, _ = wiring
    assert _upload(client, b"", "empty.txt").status_code == 400


# --- ingest + retrieval ---------------------------------------------------- #


def test_ingest_chunks_embeds_and_marks_ready(wiring):
    client, store, pipeline = wiring
    doc_id = _upload(client, HANDBOOK.encode(), "handbook.md").json()["document_id"]
    _drain(store, pipeline)

    document = client.get(f"/documents/{doc_id}").json()
    assert document["status"] == "ready"
    assert document["chunk_count"] >= 1
    assert document["error"] is None

    chunks = client.get(f"/documents/{doc_id}/chunks").json()
    assert chunks and chunks[0]["content"]
    # Offsets index into the extracted text, which is what makes a citation checkable.
    assert chunks[0]["char_start"] == 0


def test_rag_returns_document_citations_with_provenance(wiring):
    client, store, pipeline = wiring
    _upload(client, HANDBOOK.encode(), "handbook.md")
    _drain(store, pipeline)

    result = client.post("/rag", json={"query": "rollback", "scope": "global"}).json()
    document_hits = [c for c in result["citations"] if c["kind"] == "document"]
    assert document_hits, result
    hit = document_hits[0]
    assert "rollback" in hit["text"].lower()
    assert hit["filename"] == "handbook.md"
    assert hit["char_start"] is not None and hit["chunk_index"] is not None
    assert result["synthesized"] is False and result["answer"] is None


def test_rag_fuses_memories_and_documents(wiring):
    """The point of /rag: one ranked list over both stores. A memory-only answer here
    would mean the document channel silently contributed nothing."""
    client, store, pipeline = wiring
    client.post(
        "/remember",
        json={
            "content": "The deployment window is owned by Priya.",
            "scope": "global",
            "extract": "never",
        },
    )
    _upload(client, HANDBOOK.encode(), "handbook.md")
    _drain(store, pipeline)

    result = client.post(
        "/rag",
        json={"query": "deployment window", "scope": "global", "limit": 8},
    ).json()
    kinds = {c["kind"] for c in result["citations"]}
    assert "document" in kinds, result


def test_rag_reports_an_unavailable_dense_channel_instead_of_returning_less(wiring, tmp_path):
    """Without an embedder, dense retrieval is impossible. Saying so beats quietly
    returning keyword-only results that look like a worse ranker."""
    client, _, _ = wiring
    memory = client.app.state.service
    store = FakeDocumentStore(embedder=None)
    client.app.state.rag = RagService(memory, store)

    result = client.post("/rag", json={"query": "anything"}).json()
    assert result["unavailable"] == ["dense"]


def test_rag_without_a_chat_service_says_synthesis_is_unavailable(wiring):
    """Asked to synthesize with no chat configured, the honest answer is 'that channel
    is not available' — not an empty answer field the client reads as a bad model."""
    client, store, pipeline = wiring
    _upload(client, HANDBOOK.encode(), "handbook.md")
    _drain(store, pipeline)

    result = client.post(
        "/rag", json={"query": "rollback", "synthesize": True}
    ).json()
    assert result["synthesized"] is False
    assert "synthesis" in result["unavailable"]
    assert result["citations"], "retrieval must still work without synthesis"


def test_rag_synthesis_grounds_on_the_fused_sources_not_a_second_recall(wiring):
    """The trap this pins: routing synthesis through `ChatService.ask` would make it run
    its own memory-only recall, so the model would answer from a DIFFERENT evidence set
    than the one returned as citations — a citation that does not support the sentence
    it is attached to."""
    from local_infra.extraction.budget import Budget as _Budget
    from local_infra.llm.chat import ChatService
    from local_infra.llm.router import ChatRouter
    from local_infra.llm.types import ChatUsage, Deployment

    client, store, pipeline = wiring
    _upload(client, HANDBOOK.encode(), "handbook.md")
    _drain(store, pipeline)

    transport = FakeCompletionClient(
        [("Rollbacks use `make rollback` [1].", ChatUsage(model="ollama_chat/llama3.2"))]
    )
    router = ChatRouter(
        [Deployment("default", "ollama_chat/llama3.2", api_base="http://127.0.0.1:11434")],
        offline=True,
        budget=_Budget(0.0),
        client=transport,
    )
    memory = client.app.state.service
    client.app.state.rag = RagService(
        memory, store, chat=ChatService(memory, router), limit=8
    )

    result = client.post(
        "/rag", json={"query": "how do rollbacks work", "synthesize": True}
    ).json()

    assert result["synthesized"] is True
    assert result["answer"].startswith("Rollbacks use")
    assert result["usage"]["cost_usd"] == 0.0

    # The prompt the model actually saw must contain the DOCUMENT text and name the
    # file, or the [1] in its answer points at nothing.
    prompt = transport.messages[0][-1]["content"]
    assert "rollback" in prompt.lower()
    assert "handbook.md" in prompt
    assert "SOURCES:" in prompt


# --- delete ---------------------------------------------------------------- #


def test_delete_removes_chunks_and_unlinks_the_blob(wiring):
    client, store, pipeline = wiring
    doc_id = _upload(client, HANDBOOK.encode(), "handbook.md").json()["document_id"]
    _drain(store, pipeline)
    sha = list(store.documents.values())[0].sha256
    blobs = BlobStore(client.app.state.documents._blobs.root)
    assert blobs.exists(sha)

    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert client.get(f"/documents/{doc_id}").status_code == 404
    assert not blobs.exists(sha)

    result = client.post("/rag", json={"query": "rollback"}).json()
    assert not [c for c in result["citations"] if c["kind"] == "document"]


def test_delete_keeps_the_blob_when_another_scope_still_uses_it(wiring):
    """The same file in two projects is ONE blob. Unlinking on the first delete leaves
    the second document pointing at nothing — a failure that only surfaces later, on
    re-ingest, far from the delete that caused it."""
    client, store, pipeline = wiring
    a = _upload(client, HANDBOOK.encode(), "h.md", scope="project:alpha").json()
    _upload(client, HANDBOOK.encode(), "h.md", scope="project:beta")
    _drain(store, pipeline)
    sha = list(store.documents.values())[0].sha256

    client.delete(f"/documents/{a['document_id']}")
    assert BlobStore(client.app.state.documents._blobs.root).exists(sha)


def test_deleting_a_document_mid_ingest_is_not_an_error(wiring):
    """The delete-during-embed race. The chunks and job cascade away with the document,
    so there is nothing left to fail — and marking a deleted document 'failed' would
    log an error for correct behaviour."""
    client, store, pipeline = wiring
    _upload(client, HANDBOOK.encode(), "handbook.md")

    async def run() -> None:
        job = await store.claim()
        assert job is not None
        await store.delete(job.document_id)
        await pipeline(job)  # must not raise

    asyncio.run(run())


def test_delete_of_an_unknown_document_is_404(wiring):
    client, _, _ = wiring
    response = client.delete("/documents/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


# --- unsupported backend --------------------------------------------------- #


def test_document_routes_answer_501_when_the_backend_has_no_store():
    """501 with a reason, never a 500 and never an empty 200 — a user cannot tell an
    empty 200 apart from broken retrieval."""
    memory = ExMemBackend(
        SqliteRepository(":memory:"),
        HybridRetriever(
            HashEmbedder(dim=64), SQLiteMetadataStore(":memory:"), RetrievalConfig(top_k=5)
        ),
        ExtractionLadder({}, offline=True, budget=Budget(0.0)),
        SQLiteMetadataStore(":memory:"),
        offline=True,
    )
    with TestClient(create_app(memory, chat=None, documents=None)) as client:
        client.app.state.documents = None
        client.app.state.rag = None

        assert client.get("/documents").status_code == 501
        assert client.post("/rag", json={"query": "x"}).status_code == 501
        upload = _upload(client, b"hello", "a.txt")
        assert upload.status_code == 501
        assert "MEMORY_BACKEND=cognee" in upload.json()["detail"]


# --- pipeline edge cases --------------------------------------------------- #


def test_a_document_with_no_extractable_text_is_ready_not_failed(wiring, tmp_path):
    """A scanned PDF has no text layer. Nothing went wrong — there is simply nothing to
    retrieve, and OCR is out of scope. 'failed' would send the user debugging."""
    client, store, pipeline = wiring
    blobs = client.app.state.documents._blobs
    sha = blobs.put(b"   \n\n   \n")
    document, _ = asyncio.run(
        store.create(
            scope="global", filename="scan.txt", kind="txt", sha256=sha, size_bytes=9
        )
    )

    async def run() -> None:
        await pipeline.process(
            DocumentJob(id=document.id, document_id=document.id, scope_id=document.scope_id)
        )

    asyncio.run(run())

    assert store.documents[document.id].status == "ready"
    assert store.documents[document.id].chunk_count == 0
    assert "OCR" in store.documents[document.id].error
