"""M11 verification: the benchmark rig's pure pieces, fully offline.

The live gate is ``python -m local_infra.bench.cli --backends exmem,cognee`` (needs
Ollama + Postgres). These tests instead pin down everything that must be correct
*without* infra: seeded-generator determinism, real-vs-local pricing math, the local
IRR judge, the metric helpers, and the runner driving a fake ``MemoryService`` — so a
regression in the harness is caught in milliseconds, not only during a live run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from local_infra.bench.generator import Fact, generate_conversation
from local_infra.bench.judge import CachingJudge, LocalJudge
from local_infra.bench.metrics import answer_present, percentile
from local_infra.bench.pricing import (
    CLOUD_PRICING,
    UnknownModelPricing,
    Usage,
    cost_usd,
    cost_usd_or_none,
    is_local,
    total_cost_usd,
)
from local_infra.bench.runner import BenchmarkRunner
from local_infra.core.models import Episode
from local_infra.core.service import (
    ForgetResult,
    HealthReport,
    ImproveReport,
    MemoryService,
    RecallHit,
    RememberResult,
)


# --------------------------------------------------------------------------- #
# generator                                                                    #
# --------------------------------------------------------------------------- #
def test_generator_is_seed_deterministic():
    a = generate_conversation(n_facts=20, n_turns=60, seed=42)
    b = generate_conversation(n_facts=20, n_turns=60, seed=42)
    assert [f.statement for f in a.facts] == [f.statement for f in b.facts]
    assert [t.content for t in a.turns] == [t.content for t in b.turns]
    # A different seed yields a different conversation.
    c = generate_conversation(n_facts=20, n_turns=60, seed=7)
    assert [f.statement for f in a.facts] != [f.statement for f in c.facts]


def test_generator_shape():
    conv = generate_conversation(n_facts=12, n_turns=40, seed=1)
    assert len(conv.facts) == 12
    assert len(conv.probes) == 12
    # Each fact-plant consumes one loop iteration but appends a user turn AND the
    # assistant's acknowledgement, so total turns = n_turns + (# planted facts).
    planted = [f for f in conv.facts if f.planted_at_turn >= 0]
    assert len(conv.turns) == 40 + len(planted)
    # Every planted fact's answer is embedded verbatim in its statement (so a perfect
    # retriever can score recall@k = 1.0), and each fact was planted at a real turn.
    for f in conv.facts:
        assert f.answer.lower() in f.statement.lower()
        assert 0 <= f.planted_at_turn < 40


# --------------------------------------------------------------------------- #
# pricing                                                                      #
# --------------------------------------------------------------------------- #
def test_local_models_are_free():
    assert is_local("llama3.2:latest")
    assert is_local("nomic-embed-text")
    assert cost_usd("llama3.1:8b", 10_000, 5_000) == 0.0


def test_cloud_pricing_uses_real_numbers():
    # Real Anthropic list price: opus 4.8 is $5/$25 per 1M tokens.
    assert CLOUD_PRICING["claude-opus-4-8"] == (5.00, 25.00)
    assert cost_usd("claude-opus-4-8", 1_000_000, 1_000_000) == 30.0
    total = total_cost_usd([
        Usage("claude-opus-4-8", 1_000_000, 0),   # $5
        Usage("llama3.2:latest", 5_000_000, 0),   # free
        Usage("claude-haiku-4-5", 0, 1_000_000),  # $5
    ])
    assert total == 10.0


# --------------------------------------------------------------------------- #
# judge                                                                        #
# --------------------------------------------------------------------------- #
def _fact(answer: str) -> Fact:
    return Fact(id="fact_000", category="personal", statement="x", probe="q?", answer=answer)


def test_local_judge_exact_and_fuzzy_and_miss():
    j = LocalJudge(threshold=0.75)
    exact = j.judge(_fact("Globex"), "The user works at Globex now.")
    assert exact.recalled and exact.score == 1.0
    miss = j.judge(_fact("PostgreSQL"), "The user deploys to AWS and prefers Rust.")
    assert not miss.recalled
    # Empty ground truth or empty context never counts as recalled.
    assert not j.judge(_fact(""), "anything").recalled
    assert not j.judge(_fact("Globex"), "").recalled


def test_caching_judge_memoizes():
    calls = {"n": 0}

    class Counting:
        def judge(self, fact, answer_text):
            calls["n"] += 1
            return LocalJudge().judge(fact, answer_text)

    cj = CachingJudge(Counting())
    f = _fact("Globex")
    cj.judge(f, "works at Globex")
    cj.judge(f, "works at Globex")  # same key -> served from cache
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# metrics                                                                      #
# --------------------------------------------------------------------------- #
def test_percentile_and_answer_present():
    vals = [10.0, 20.0, 30.0, 40.0]
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0], 0.9) == 5.0
    assert percentile(vals, 0.5) == 25.0
    assert answer_present("PostgreSQL", ["My project uses PostgreSQL as its database."])
    assert not answer_present("PostgreSQL", ["The API runs on port 8080."])
    assert not answer_present("", ["anything"])


# --------------------------------------------------------------------------- #
# runner over a fake MemoryService (offline, deterministic)                    #
# --------------------------------------------------------------------------- #
class _FakeService(MemoryService):
    """Minimal in-memory service: stores each remembered turn and retrieves by
    token-overlap with the probe. Enough to exercise the runner end-to-end without
    a real backend — a planted fact's statement contains its answer, so an overlap
    retriever surfaces it and both recall@k and IRR score it a hit."""

    backend_name = "fake"

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}

    async def remember(self, content, *, scope="global", metadata=None, source="api",
                       occurred_at=None, extract="async") -> RememberResult:
        self._store.setdefault(scope, []).append(content)
        # Count planted facts crudely: a first-person assertion contributes one.
        extracted = 1 if content.lower().startswith(("my ", "i ", "the ", "we ")) or (
            "name is" in content.lower()
        ) else 0
        return RememberResult(episode_id=uuid4(), scope_id=0, extracted=extracted)

    @staticmethod
    def _tok(s: str) -> set[str]:
        stop = {"the", "and", "for", "does", "use", "what", "who", "how", "many",
                "user", "users", "their", "have", "long", "been", "role", "what's"}
        words = "".join(c if c.isalnum() else " " for c in s.lower()).split()
        return {w for w in words if len(w) > 2 and w not in stop}

    async def recall(self, query, *, scope="global", limit=10, memory_types=None,
                     as_of=None, include_invalidated=False, include_episodes=False):
        q = self._tok(query)
        scored = []
        for content in self._store.get(scope, []):
            overlap = len(q & self._tok(content))
            if overlap:
                scored.append((overlap, content))
        scored.sort(key=lambda t: t[0], reverse=True)
        now = datetime.now(timezone.utc)
        return [
            RecallHit(
                score=float(o),
                kind="episode",
                episode=Episode(id=uuid4(), scope_id=0, content=c, source="fake",
                                occurred_at=now, recorded_at=now),
            )
            for o, c in scored[:limit]
        ]

    async def forget(self, memory_id, *, reason="user_forget", hard=False):
        return ForgetResult(memory_id, ok=True, hard=hard, reason=reason)

    async def improve(self, *, scope=None, ops=(), dry_run=False):
        return ImproveReport()

    async def get(self, memory_id):
        return None

    async def history(self, subject_key, predicate_key, *, scope="global"):
        return []

    async def history_for(self, memory_id):
        return []

    async def health(self):
        return HealthReport(backend="fake", ok=True, memories=0, episodes=0,
                            indexed_chunks=0, offline=True)


async def test_runner_drives_service_and_computes_metrics():
    conv = generate_conversation(n_facts=15, n_turns=45, seed=42)
    svc = _FakeService()
    runner = BenchmarkRunner(svc, scope="project:bench_test", k=5)
    result = await runner.run(conv)

    assert result.backend == "fake"
    assert result.facts == 15
    assert len(result.per_fact) == 15
    # Overlap retrieval surfaces the planted statement for most probes.
    assert 0.0 <= result.recall_at_k <= 1.0
    assert result.recall_at_k >= 0.5
    # IRR (fuzzy judge) is at least as generous as exact recall@k.
    assert result.irr >= result.recall_at_k
    # Latency percentiles are populated and ordered; cost is zero for a local run.
    assert result.p95_recall_ms >= result.p50_recall_ms >= 0.0
    assert result.p95_ingest_ms >= result.p50_ingest_ms >= 0.0
    assert result.total_extracted > 0
    assert result.cost_usd == 0.0


# --------------------------------------------------------------------------- #
# Provider-qualified pricing.
#
# The old rule ("anything not in CLOUD_PRICING is local and free") priced
# `anthropic/claude-sonnet-5` at $0.00, which silently falsified the "an
# all-local run is $0.00" claim the moment a router used qualified ids.
# --------------------------------------------------------------------------- #
def test_provider_qualified_cloud_ids_are_not_local():
    assert not is_local("anthropic/claude-sonnet-5")
    assert not is_local("openrouter/anthropic/claude-sonnet-5")
    assert not is_local("vertex_ai/claude-haiku-4-5")


def test_provider_qualified_cloud_ids_price_correctly():
    # Same price as the bare id: the namespace is peeled before lookup.
    assert cost_usd("anthropic/claude-sonnet-5", 1_000_000, 1_000_000) == 18.0
    assert cost_usd("openrouter/anthropic/claude-opus-4-8", 1_000_000, 0) == 5.0


def test_local_namespaces_stay_free():
    assert is_local("ollama/llama3.2:latest")
    assert is_local("ollama_chat/llama3.1:8b")
    assert cost_usd("ollama_chat/llama3.1:8b", 10_000_000, 10_000_000) == 0.0


def test_unknown_model_raises_instead_of_billing_zero():
    with pytest.raises(UnknownModelPricing):
        cost_usd("someprovider/mystery-model", 1_000, 1_000)
    # A typo'd cloud id must not quietly report $0.00 either.
    with pytest.raises(UnknownModelPricing):
        cost_usd("claude-sonet-5", 1_000, 1_000)
    assert cost_usd_or_none("someprovider/mystery-model", 1_000, 1_000) is None
