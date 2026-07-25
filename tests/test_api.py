"""M9 verification: the REST layer over the acceptance path.

Deterministic and offline: the app is fed an ex_mem backend wired with the same fakes as
the service contract suite (``HashEmbedder`` + a content-addressed ``MappingProvider``),
so these tests exercise the HTTP surface — routing, validation, status codes, temporal
semantics end-to-end — without Ollama, Postgres, or a network. It is the plan's
acceptance script expressed over HTTP:

    POST /remember (Acme) -> POST /remember (Globex, supersedes) -> GET /recall (Globex)
    -> GET /recall?as_of (Acme) -> GET history (chain) -> DELETE (tombstone) -> POST /improve
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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

from tests._fakes import HashEmbedder, MappingProvider

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=30)

WORK_ACME = "I work at Acme as a backend engineer."
WORK_GLOBEX = "Actually I moved to Globex last month."

MAPPING = {
    WORK_ACME: [
        dict(subject="I", predicate="works_at", object="Acme",
             evidence_span="I work at Acme", confidence=0.9),
    ],
    WORK_GLOBEX: [
        dict(subject="I", predicate="works_at", object="Globex",
             evidence_span="moved to Globex", confidence=0.9),
    ],
}


def _fake_backend() -> ExMemBackend:
    embed = HashEmbedder(dim=64)
    store = SQLiteMetadataStore(":memory:")
    retriever = HybridRetriever(embed, store, RetrievalConfig(top_k=10))
    repo = SqliteRepository(":memory:")
    provider = MappingProvider(Stage.LOCAL_SMALL, MAPPING)
    ladder = ExtractionLadder({Stage.LOCAL_SMALL: provider}, offline=True, budget=Budget(0.0))
    return ExMemBackend(repo, retriever, ladder, store, offline=True, embedder=embed)


@pytest.fixture
def client():
    backend = _fake_backend()
    with TestClient(create_app(service=backend)) as c:
        yield c


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# --------------------------------------------------------------------------- #
def test_acceptance_flow(client) -> None:
    # 1. remember a fact
    r = client.post("/remember", json={"content": WORK_ACME, "occurred_at": _iso(T0), "extract": "sync"})
    assert r.status_code == 200, r.text
    acme = r.json()
    assert acme["extracted"] == 1 and len(acme["fact_ids"]) == 1
    acme_id = acme["fact_ids"][0]

    # 2. recall returns it
    r = client.get("/recall", params={"q": "where do I work"})
    assert r.status_code == 200
    assert [h["memory"]["object_key"] for h in r.json()] == ["acme"]

    # 3. a superseding fact
    r = client.post("/remember", json={"content": WORK_GLOBEX, "occurred_at": _iso(T1), "extract": "sync"})
    globex_id = r.json()["fact_ids"][0]

    # 4. current recall shows only the new value
    r = client.get("/recall", params={"q": "where do I work"})
    assert [h["memory"]["object_key"] for h in r.json()] == ["globex"]

    # 5. time-travel recall shows the old value
    r = client.get("/recall", params={"q": "where do I work", "as_of": _iso(T0 + timedelta(days=1))})
    assert [h["memory"]["object_key"] for h in r.json()] == ["acme"]

    # 6. history chains both versions, oldest first
    r = client.get(f"/memories/{acme_id}/history")
    assert [m["object_key"] for m in r.json()] == ["acme", "globex"]

    # 7. tombstone the current fact; recall goes empty but history stays
    r = client.delete(f"/memories/{globex_id}")
    assert r.status_code == 200 and r.json()["hard"] is False
    assert client.get("/recall", params={"q": "where do I work"}).json() == []
    assert [m["object_key"] for m in client.get(f"/memories/{acme_id}/history").json()] == ["acme", "globex"]

    # 8. improve runs and reports
    r = client.post("/improve", json={"ops": ["reindex"]})
    assert r.status_code == 200 and "reindex" in r.json()["ops"]


def test_get_memory_and_404(client) -> None:
    r = client.post("/remember", json={"content": WORK_ACME, "occurred_at": _iso(T0), "extract": "sync"})
    mid = r.json()["fact_ids"][0]

    got = client.get(f"/memories/{mid}")
    assert got.status_code == 200 and got.json()["object_key"] == "acme"

    missing = client.get("/memories/00000000-0000-0000-0000-000000000000")
    assert missing.status_code == 404


def test_recall_include_episodes(client) -> None:
    client.post("/remember", json={"content": WORK_ACME, "occurred_at": _iso(T0), "extract": "never"})
    # No fact extracted -> default recall empty...
    assert client.get("/recall", params={"q": "where do I work"}).json() == []
    # ...but the raw episode is recallable when asked.
    eps = client.get("/recall", params={"q": "where do I work", "include_episodes": "true"}).json()
    assert any(h["kind"] == "episode" and "Acme" in h["episode"]["content"] for h in eps)


def test_remember_validation_rejects_empty(client) -> None:
    r = client.post("/remember", json={"content": ""})
    assert r.status_code == 422  # min_length=1


def test_hard_delete(client) -> None:
    r = client.post("/remember", json={"content": WORK_ACME, "occurred_at": _iso(T0), "extract": "sync"})
    mid = r.json()["fact_ids"][0]
    d = client.delete(f"/memories/{mid}", params={"hard": "true"})
    assert d.status_code == 200 and d.json()["hard"] is True
    assert client.get(f"/memories/{mid}").status_code == 404


def test_health(client) -> None:
    h = client.get("/health").json()
    assert h["ok"] is True and h["backend"] == "exmem" and h["offline"] is True
