"""M6 smoke: run the extraction ladder against real local models and trace it.

    uv run python -m scripts.extract_smoke --file tests/fixtures/convo.md --trace

Offline by default (local rungs only). Prints the gate decision, one row per
provider call (stage/latency/parsed/accepted), and every accepted fact with its
confidence terms — so a regression in grounding or escalation is visible at a glance.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

from local_infra.config import get_settings
from local_infra.extraction.budget import Budget
from local_infra.extraction.gate import gate
from local_infra.extraction.ladder import ExtractionLadder
from local_infra.extraction.registry import build_local_providers, close_providers

REPO = pathlib.Path(__file__).resolve().parent.parent


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="tests/fixtures/convo.md")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--online", action="store_true", help="allow cloud rung (not built yet)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = (REPO / args.file) if not pathlib.Path(args.file).is_absolute() else pathlib.Path(args.file)
    text = path.read_text(encoding="utf-8")

    settings = get_settings()
    providers = build_local_providers(settings)
    ladder = ExtractionLadder(providers, offline=not args.online, budget=Budget(settings.cloud_budget_usd))

    decision = gate(text)
    print(f"\nsource: {path}  ({len(text)} chars)")
    print(f"gate:   should_extract={decision.should_extract} predicted={decision.predicted_facts} "
          f"complex={decision.is_complex}  max_stage={ladder.max_stage().value}\n")

    try:
        result = await ladder.run(text, seed=args.seed)
    finally:
        await close_providers(providers)

    if args.trace:
        print("attempts:")
        print(f"  {'stage':<13} {'ok':<3} {'ms':>7} {'parsed':>7} {'accepted':>9}")
        for a in result.attempts:
            print(f"  {a.stage.value:<13} {str(a.ok):<3} {a.latency_ms:>7} {a.parsed:>7} {a.accepted:>9}")
        print()

    print(f"stages visited: {' -> '.join(s.value for s in result.stages)}")
    print(f"final stage:    {result.final_stage.value}   cost=${result.cost_usd:.4f}\n")

    print(f"accepted facts ({len(result.facts)}):")
    print(f"  {'stage':<12} {'S/P/O':<42} {'grnd':>5} {'strc':>5} {'self':>5} {'comp':>5}")
    for sf in result.facts:
        f, sc = sf.fact, sf.score
        spo = f"{f.subject} / {f.predicate} / {f.object}"[:42]
        print(f"  {sf.stage.value:<12} {spo:<42} {sc.groundedness:>5.2f} {sc.structural:>5.2f} "
              f"{sc.self_report:>5.2f} {sc.composite:>5.2f}")

    if result.deferred:
        print(f"\ndeferred spans ({len(result.deferred)}):")
        for d in result.deferred:
            print(f"  - {d[:80]}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
