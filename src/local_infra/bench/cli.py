"""Benchmark CLI — ``python -m local_infra.bench.cli``.

Runs the synthetic-conversation benchmark against one or more backends and prints a
comparison table of recall@k, IRR, p50/p95 latency (ingest and recall), and cost.

M11 verification command:

    uv run python -m local_infra.bench.cli --backends exmem,cognee \
        --facts 20 --turns 60 --seed 42

This is a LIVE gate (like scripts/*_smoke): exmem needs a running Ollama; cognee also
needs the Postgres container up. Each backend runs in an isolated scope so a shared
ledger is never polluted, and exmem writes its index/ledger to a throwaway temp dir.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import uuid
from pathlib import Path

from local_infra.bench.generator import generate_conversation
from local_infra.bench.judge import LocalJudge
from local_infra.bench.metrics import BenchmarkResult
from local_infra.bench.runner import BenchmarkRunner
from local_infra.config import Settings, get_settings
from local_infra.core.service import MemoryService


async def _build_backend(name: str, settings: Settings, tmp: Path) -> MemoryService:
    """Construct one backend, isolated for a clean benchmark run.

    Imports are deferred so choosing exmem never imports cognee (and vice-versa),
    mirroring ``core.registry.build_backend``."""
    name = name.lower()
    if name == "exmem":
        from local_infra.backends.exmem_backend import ExMemBackend

        # Throwaway ledger + index so we never touch the user's real state dir.
        return ExMemBackend.from_settings(
            settings,
            ledger_path=str(tmp / "bench_exmem_ledger.db"),
            index_path=str(tmp / "bench_exmem_index.db"),
        )
    if name == "cognee":
        from local_infra.backends.cognee_backend import CogneeBackend

        # Shared Postgres ledger; isolation comes from the unique per-run scope.
        return await CogneeBackend.from_settings(settings)
    raise ValueError(f"unknown backend {name!r} (expected 'exmem' or 'cognee')")


def _print_table(results: list[BenchmarkResult]) -> None:
    if not results:
        print("no results")
        return
    columns = list(results[0].summary_row().keys())
    rows = [r.summary_row() for r in results]
    widths = {c: max(len(c), *(len(row[c]) for row in rows)) for c in columns}

    def fmt(row: dict[str, str]) -> str:
        return "  ".join(row[c].ljust(widths[c]) for c in columns)

    header = {c: c for c in columns}
    print(fmt(header))
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print(fmt(row))


def _print_per_fact(results: list[BenchmarkResult]) -> None:
    for r in results:
        print(f"\n[{r.backend}] per-fact recall (k={r.k}):")
        for rr in r.per_fact:
            k_icon = "✓" if rr.hit_at_k else "✗"
            j_icon = "✓" if rr.recalled else "✗"
            print(
                f"  recall@k {k_icon}  IRR {j_icon} ({rr.judge_score:.2f})  "
                f"[{rr.fact.category}] {rr.fact.probe}  -> {rr.fact.answer!r}"
            )


async def _main_async(args: argparse.Namespace) -> int:
    base = get_settings()
    conv = generate_conversation(
        n_facts=args.facts, n_turns=args.turns, seed=args.seed
    )
    print(
        f"generated {len(conv.facts)} facts across {len(conv.turns)} turns "
        f"(seed={args.seed}); probing top-{args.k}\n"
    )

    results: list[BenchmarkResult] = []
    with tempfile.TemporaryDirectory(prefix="local_infra_bench_") as td:
        tmp = Path(td)
        for name in args.backends:
            # Point settings at this backend and honor --offline for the ladder.
            settings = base.model_copy(update={"backend": name, "offline": args.offline})
            # Unique scope per (backend, run) -> recall is isolated even on shared pg.
            scope = f"project:bench_{args.seed}_{name}_{uuid.uuid4().hex[:8]}"
            print(f"=== {name} (scope {scope}) ===")
            backend = await _build_backend(name, settings, tmp)
            try:
                runner = BenchmarkRunner(
                    backend, scope=scope, k=args.k,
                    judge=LocalJudge(threshold=args.judge_threshold),
                    progress=args.progress,
                )
                result = await runner.run(conv)
                results.append(result)
            finally:
                await backend.close()

    print()
    _print_table(results)
    if args.verbose:
        _print_per_fact(results)
    print(
        "\ncost note: local Ollama inference is free — a local run is $0.00. "
        "Cloud escalation (real Anthropic pricing in bench/pricing.py) is disabled "
        "offline and gated by cloud_budget_usd=0."
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="local_infra memory benchmark rig (M11)")
    p.add_argument(
        "--backends", default="exmem",
        help="comma-separated backends to run (exmem,cognee)",
    )
    p.add_argument("--facts", type=int, default=20, help="number of facts to plant")
    p.add_argument("--turns", type=int, default=60, help="conversation length in turns")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible)")
    p.add_argument("--k", type=int, default=5, help="recall@k / retrieval limit")
    p.add_argument(
        "--judge-threshold", type=float, default=0.75,
        help="local judge fuzzy accept threshold for IRR",
    )
    p.add_argument(
        "--offline", action="store_true",
        help="force MEMORY_OFFLINE behavior (ladder never escalates to cloud)",
    )
    p.add_argument("--progress", action="store_true", help="print per-turn ingest progress")
    p.add_argument("--verbose", action="store_true", help="print per-fact recall breakdown")
    args = p.parse_args()
    args.backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
