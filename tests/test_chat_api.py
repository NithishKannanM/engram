"""The HTTP chat surface, over the same offline fakes as the rest of the API suite.

Also guards the ``create_app`` restructure: the lifespan used to early-``return`` when
a service was injected, so anything wired after that point silently did not exist for
injected apps. ``test_injected_service_still_gets_chat`` is that regression test.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from local_infra.api.app import create_app
from local_infra.backends.exmem_backend import ExMemBackend
from local_infra.backends.sqlite.repository import SqliteRepository
from local_infra.exmem.config import RetrievalConfig
from local_infra.exmem.retrieval import HybridRetriever
from local_infra.exmem.store import SQLiteMetadataStore
from local_infra.extraction.budget import Budget
from local_infra.extraction.ladder import ExtractionLadder
from local_infra.extraction.providers.base import Stage
from local_infra.llm.chat import ChatService
from local_infra.llm.router import ChatRouter
from local_infra.llm.types import ChatUsage, Deployment

from tests._fakes import FakeCompletionClient, HashEmbedder, MappingProvider

WORK = "I work at Acme as a backend engineer."
MAPPING = {
    WORK: [
        dict(subject="I", predicate="works_at", object="Acme",
             evidence_span="I work at Acme", confidence=0.9),
    ],
}
MODEL = "ollama_chat/llama3.1:8b"


def _backend() -> ExMemBackend:
    embed = HashEmbedder(dim=64)
    store = SQLiteMetadataStore(":memory:")
    retriever = HybridRetriever(embed, store, RetrievalConfig(top_k=10))
    repo = SqliteRepository(":memory:")
    provider = MappingProvider(Stage.LOCAL_SMALL, MAPPING)
    ladder = ExtractionLadder({Stage.LOCAL_SMALL: provider}, offline=True, budget=Budget(0.0))
    return ExMemBackend(repo, retriever, ladder, store, offline=True, embedder=embed)


def _chat(backend: ExMemBackend, scripts) -> ChatService:
    router = ChatRouter(
        [Deployment("default", MODEL, api_base="http://127.0.0.1:11434")],
        offline=True,
        budget=Budget(0.0),
        client=FakeCompletionClient(scripts),
    )
    return ChatService(backend, router, recall_limit=6)


@pytest.fixture
def client_and_backend():
    backend = _backend()
    chat = _chat(backend, [("You work at Acme [1].", ChatUsage(MODEL, 120, 8))])
    with TestClient(create_app(service=backend, chat=chat)) as c:
        yield c, backend


def test_injected_service_still_gets_chat(client_and_backend) -> None:
    """The lifespan's injected-service branch must wire every subsystem, not just
    the service — this is the exact shape that used to be skipped by an early return."""
    client, _ = client_and_backend
    assert client.get("/chat/models").status_code == 200


def test_chat_models_lists_aliases_without_leaking_keys(client_and_backend) -> None:
    client, _ = client_and_backend
    body = client.get("/chat/models").json()

    assert body[0]["alias"] == "default"
    assert body[0]["is_local"] is True
    serialized = json.dumps(body)
    assert "api_key" not in serialized and "sk-" not in serialized


def test_chat_answers_and_returns_the_memories_it_was_grounded_in(client_and_backend) -> None:
    client, _ = client_and_backend
    assert client.post("/remember", json={"content": WORK, "extract": "sync"}).status_code == 200

    res = client.post("/chat", json={"question": "where do I work?"})
    assert res.status_code == 200
    body = res.json()

    assert body["text"] == "You work at Acme [1]."
    assert body["model"] == MODEL
    assert body["usage"]["cost_usd"] == 0.0  # local inference is free
    # The [1] in the text resolves positionally against this list.
    assert body["citations"], "a grounded answer must carry its citations"
    assert "Acme" in body["citations"][0]["memory"]["statement"]


def test_unknown_alias_is_404_not_500(client_and_backend) -> None:
    client, _ = client_and_backend
    res = client.post("/chat", json={"question": "hi", "alias": "no-such-alias"})
    assert res.status_code == 404
    assert "no-such-alias" in res.json()["detail"]


def test_all_deployments_failing_is_503_with_a_reason() -> None:
    backend = _backend()
    chat = _chat(backend, [ConnectionError("ollama is not running")])
    with TestClient(create_app(service=backend, chat=chat)) as client:
        res = client.post("/chat", json={"question": "hi"})

    assert res.status_code == 503
    assert "ollama is not running" in res.json()["detail"]


def test_chat_disabled_answers_503_rather_than_crashing() -> None:
    backend = _backend()
    # chat=None with MEMORY_CHAT_ENABLED unset in this process still builds one, so
    # emulate the disabled state directly on app.state.
    app = create_app(service=backend)
    with TestClient(app) as client:
        app.state.chat = None
        res = client.post("/chat", json={"question": "hi"})

    assert res.status_code == 503
    assert "disabled" in res.json()["detail"]


def test_stream_emits_tokens_then_usage_then_done() -> None:
    backend = _backend()
    chat = _chat(backend, [(["You ", "work ", "at Acme"], ChatUsage(MODEL, 100, 5))])

    with TestClient(create_app(service=backend, chat=chat)) as client:
        with client.stream("POST", "/chat/stream", json={"question": "where?"}) as res:
            assert res.status_code == 200
            assert res.headers["content-type"].startswith("text/event-stream")
            raw = "".join(res.iter_text())

    events = [line[len("event: ") :] for line in raw.splitlines() if line.startswith("event: ")]
    assert events == ["token", "token", "token", "usage", "done"]

    deltas = [
        json.loads(line[len("data: ") :])["delta"]
        for line in raw.splitlines()
        if line.startswith("data: ") and "delta" in line
    ]
    assert "".join(deltas) == "You work at Acme"


def test_stream_reports_failure_as_an_error_event() -> None:
    backend = _backend()
    chat = _chat(backend, [ConnectionError("model vanished")])

    with TestClient(create_app(service=backend, chat=chat)) as client:
        with client.stream("POST", "/chat/stream", json={"question": "where?"}) as res:
            raw = "".join(res.iter_text())

    assert "event: error" in raw
    assert "model vanished" in raw
