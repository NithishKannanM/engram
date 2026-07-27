"""The memory graph: node set, supersede edges, and time travel.

The load-bearing behaviour is that `current` is computed against `as_of` rather than
read off `valid_until`. Without that the time slider is decorative — scrubbing back
would still show the old fact greyed out, which is precisely the thing a temporal
ledger exists to answer.
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

ACME = "I work at Acme as a backend engineer."
GLOBEX = "Actually I moved to Globex last month."

MAPPING = {
    ACME: [
        dict(subject="I", predicate="works_at", object="Acme",
             evidence_span="I work at Acme", confidence=0.9),
    ],
    GLOBEX: [
        dict(subject="I", predicate="works_at", object="Globex",
             evidence_span="moved to Globex", confidence=0.9),
    ],
}


def _backend() -> ExMemBackend:
    embed = HashEmbedder(dim=64)
    store = SQLiteMetadataStore(":memory:")
    retriever = HybridRetriever(embed, store, RetrievalConfig(top_k=10))
    repo = SqliteRepository(":memory:")
    provider = MappingProvider(Stage.LOCAL_SMALL, MAPPING)
    ladder = ExtractionLadder({Stage.LOCAL_SMALL: provider}, offline=True, budget=Budget(0.0))
    return ExMemBackend(repo, retriever, ladder, store, offline=True, embedder=embed)


@pytest.fixture
def client():
    with TestClient(create_app(service=_backend())) as c:
        yield c


def _seed(client: TestClient) -> None:
    """Write Acme at T0, then supersede it with Globex at T1."""
    for content, when in ((ACME, T0), (GLOBEX, T1)):
        res = client.post(
            "/remember",
            json={"content": content, "extract": "sync", "occurred_at": when.isoformat()},
        )
        assert res.status_code == 200, res.text


def test_empty_scope_returns_an_empty_graph(client: TestClient) -> None:
    body = client.get("/graph").json()
    assert body["nodes"] == [] and body["edges"] == []


def test_graph_includes_superseded_rows_and_links_them(client: TestClient) -> None:
    _seed(client)
    body = client.get("/graph").json()

    statements = {n["statement"] for n in body["nodes"]}
    # Recall would drop the superseded row; the graph must keep it — it IS the history.
    assert any("Acme" in s for s in statements)
    assert any("Globex" in s for s in statements)

    supersedes = [e for e in body["edges"] if e["kind"] == "supersedes"]
    assert len(supersedes) == 1

    by_id = {n["id"]: n for n in body["nodes"]}
    edge = supersedes[0]
    assert "Acme" in by_id[edge["source"]]["statement"], "edge points old -> new"
    assert "Globex" in by_id[edge["target"]]["statement"]


def test_current_flag_reflects_today(client: TestClient) -> None:
    _seed(client)
    by_statement = {n["statement"]: n for n in client.get("/graph").json()["nodes"]}

    acme = next(v for k, v in by_statement.items() if "Acme" in k)
    globex = next(v for k, v in by_statement.items() if "Globex" in k)
    assert acme["current"] is False
    assert globex["current"] is True
    assert acme["valid_until"] is not None


def test_as_of_revives_the_older_fact(client: TestClient) -> None:
    """Scrubbing the time slider back must make the old fact valid again."""
    _seed(client)
    midpoint = (T0 + timedelta(days=10)).isoformat()
    body = client.get("/graph", params={"as_of": midpoint}).json()

    by_statement = {n["statement"]: n for n in body["nodes"]}
    acme = next(v for k, v in by_statement.items() if "Acme" in k)
    assert acme["current"] is True, "at T0+10d, Acme was still true"

    # Globex had not happened yet, so it is not current at that instant.
    globex = next((v for k, v in by_statement.items() if "Globex" in k), None)
    if globex is not None:
        assert globex["current"] is False


def test_as_of_keeps_the_supersede_chain_visible(client: TestClient) -> None:
    """Time travel must not delete the edge it exists to animate.

    The repository's own ``as_of`` is a VISIBILITY filter — it returns only rows valid
    at that instant, which drops every superseded row and every supersede edge with it.
    The graph deliberately does not use it that way: it keeps everything already known
    at the instant and moves the `current` flag instead, so the node set (and the force
    layout) stays stable while the highlight travels.
    """
    _seed(client)
    later = (T1 + timedelta(days=1)).isoformat()
    body = client.get("/graph", params={"as_of": later}).json()

    assert len(body["nodes"]) == 2, "both versions stay in the graph"
    assert len([e for e in body["edges"] if e["kind"] == "supersedes"]) == 1

    by_statement = {n["statement"]: n for n in body["nodes"]}
    acme = next(v for k, v in by_statement.items() if "Acme" in k)
    globex = next(v for k, v in by_statement.items() if "Globex" in k)
    assert acme["current"] is False and globex["current"] is True


def test_as_of_hides_facts_not_yet_known(client: TestClient) -> None:
    """A fact recorded after the chosen instant is not shown as history that existed."""
    _seed(client)
    early = (T0 + timedelta(days=1)).isoformat()
    body = client.get("/graph", params={"as_of": early}).json()

    statements = {n["statement"] for n in body["nodes"]}
    assert any("Acme" in s for s in statements)
    assert not any("Globex" in s for s in statements), "Globex had not happened yet"


def test_exmem_reports_related_as_unavailable_rather_than_empty(client: TestClient) -> None:
    """SQLite has no vector column. Saying "unavailable" is honest; returning zero
    related edges would read as "these facts are unrelated", which is a different claim."""
    body = client.get("/graph").json()
    assert body["unavailable"] == ["related"]
    assert not [e for e in body["edges"] if e["kind"] == "related"]


def test_limit_is_bounded(client: TestClient) -> None:
    _seed(client)
    assert client.get("/graph", params={"limit": 1}).status_code == 200
    assert client.get("/graph", params={"limit": 0}).status_code == 422
    assert client.get("/graph", params={"limit": 99_999}).status_code == 422
