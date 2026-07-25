"""Benchmark runner — drives a ``MemoryService`` through the public seam only.

The runner never imports a concrete backend; it takes an already-built
:class:`~local_infra.core.service.MemoryService` and exercises exactly the two calls
a real client makes: ``remember`` to ingest each conversation turn, then ``recall`` to
probe. This is what makes the exmem and cognee numbers directly comparable — they run
the identical harness behind the identical interface.

Isolation: every run uses a unique ``scope`` (passed in by the CLI). Recall is
scope-filtered in both backends, so a shared ledger (cognee's Postgres) does not leak
facts from other runs into this run's retrieval — no destructive TRUNCATE needed.

Extraction is forced ``sync`` so facts are committed before the probe phase; the async
worker path is covered by ``test_worker``, not here.
"""

from __future__ import annotations

import time

from local_infra.bench.generator import SyntheticConversation
from local_infra.bench.judge import CachingJudge, Judge, LocalJudge
from local_infra.bench.metrics import (
    BenchmarkResult,
    RecallResult,
    answer_present,
    percentile,
)
from local_infra.core.service import MemoryService


class BenchmarkRunner:
    def __init__(
        self,
        service: MemoryService,
        *,
        scope: str,
        k: int = 5,
        judge: Judge | None = None,
        progress: bool = False,
    ) -> None:
        self._svc = service
        self._scope = scope
        self._k = k
        self._judge = CachingJudge(judge or LocalJudge())
        self._progress = progress

    async def run(self, conv: SyntheticConversation) -> BenchmarkResult:
        ingest_ms: list[float] = []
        recall_ms: list[float] = []
        extracted = 0

        # -- Phase A: replay the conversation (ingest each turn, extract synchronously) --
        for i, turn in enumerate(conv.turns):
            t0 = time.perf_counter()
            res = await self._svc.remember(turn.content, scope=self._scope, extract="sync")
            ingest_ms.append((time.perf_counter() - t0) * 1000.0)
            extracted += res.extracted
            if self._progress:
                print(f"  ingest {i + 1}/{len(conv.turns)}  (+{res.extracted} facts)", end="\r")
        if self._progress:
            print()

        # -- Phase B: probe recall for each planted fact --
        per_fact: list[RecallResult] = []
        for fact in conv.probes:
            t0 = time.perf_counter()
            hits = await self._svc.recall(fact.probe, scope=self._scope, limit=self._k)
            recall_ms.append((time.perf_counter() - t0) * 1000.0)

            statements = [h.text for h in hits]
            hit_at_k = answer_present(fact.answer, statements)
            answer_ctx = "\n".join(statements)
            j = self._judge.judge(fact, answer_ctx)
            per_fact.append(
                RecallResult(
                    fact=fact,
                    hit_at_k=hit_at_k,
                    recalled=j.recalled,
                    judge_score=j.score,
                    top_text=answer_ctx,
                )
            )

        n = len(conv.probes)
        recall_at_k = sum(1 for r in per_fact if r.hit_at_k) / n if n else 0.0
        irr = sum(1 for r in per_fact if r.recalled) / n if n else 0.0

        return BenchmarkResult(
            backend=getattr(self._svc, "backend_name", "unknown"),
            facts=n,
            k=self._k,
            recall_at_k=recall_at_k,
            irr=irr,
            p50_ingest_ms=percentile(ingest_ms, 0.50),
            p95_ingest_ms=percentile(ingest_ms, 0.95),
            p50_recall_ms=percentile(recall_ms, 0.50),
            p95_recall_ms=percentile(recall_ms, 0.95),
            total_extracted=extracted,
            # All-local run: extraction and embeddings are Ollama, which is free.
            # A cloud-escalating configuration would populate this from bench.pricing.
            cost_usd=0.0,
            notes="local Ollama (no cloud escalation)",
            per_fact=per_fact,
        )
