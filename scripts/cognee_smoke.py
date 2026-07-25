"""M8 smoke: drive the REAL CogneeBackend end-to-end and trace it.

    uv run python -m scripts.cognee_smoke            # ledger + real extraction + real embed
    uv run python -m scripts.cognee_smoke --cognify  # also build + query the cognee graph

Where the pytest contract suite fakes extraction/embeddings/graph for determinism, this
exercises the live stack: Postgres temporal ledger, Ollama extraction ladder, Ollama
embeddings, and (with --cognify) cognee's graph channel. It runs the plan's acceptance
path in a throwaway scope and prints a trace, so a real-world regression in extraction
grounding, supersession, or the cognee wiring is visible at a glance.

Not a deterministic gate — a local model may phrase a predicate differently, which the
trace shows rather than hides. Postgres + Ollama must be up; --cognify also needs the
cognee stack (see scripts/spike_cognee.py for the load-bearing config).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from local_infra.backends.cognee_backend import CogneeBackend
from local_infra.backends.cognee_channel import CogneeGraphChannel, NullCogneeChannel
from local_infra.backends.pg.migrate import run_migrations
from local_infra.backends.pg.repository import PgRepository
from local_infra.config import get_settings
from local_infra.embed.ollama import OllamaEmbedder
from local_infra.extraction.registry import build_ladder

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=30)


def line(label: str, value: str) -> None:
    print(f"  {label:<22} {value}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cognify", action="store_true", help="build + query the cognee graph channel")
    args = ap.parse_args()

    settings = get_settings()
    await run_migrations(settings)
    repo = await PgRepository(settings.pg_dsn, embedder=OllamaEmbedder()).connect()

    channel = NullCogneeChannel()
    if args.cognify:
        channel = CogneeGraphChannel(
            pg_host=settings.pg_host, pg_port=settings.pg_port, pg_user=settings.pg_user,
            pg_password=settings.pg_password, pg_database=settings.pg_database,
            ollama_base_url=settings.ollama_base_url, llm_model=settings.extract_model_large,
            embed_model=settings.embed_model, embed_dim=settings.embed_dim,
            state_dir=str(settings.state_dir),
        )
    backend = CogneeBackend(repo, build_ladder(settings, offline=True), channel=channel)

    scope = f"session:smoke-{uuid4()}"
    print(f"\n=== CogneeBackend smoke (scope={scope}, channel={channel.name}) ===\n")

    try:
        print("[1] remember 'I work at Acme.' (real extraction)")
        r1 = await backend.remember("I work at Acme.", scope=scope, occurred_at=T0, extract="sync")
        line("episode", str(r1.episode_id))
        line("facts extracted", f"{r1.extracted} (stage={r1.final_stage})")
        for fid in r1.fact_ids:
            m = await backend.get(fid)
            line("fact", f"{m.subject_key} / {m.predicate_key} / {m.object_key}  [{m.attrs.get('cardinality')}]")

        print("\n[2] recall 'where do I work'")
        for h in await backend.recall("where do I work", scope=scope):
            line("hit", f"{h.memory.object_key}  (score={h.score:.3f})")

        print("\n[3] remember 'Actually I moved to Globex.' (contradiction)")
        r2 = await backend.remember(
            "Actually I moved to Globex.", scope=scope, occurred_at=T1, extract="sync"
        )
        line("facts extracted", f"{r2.extracted} (stage={r2.final_stage})")

        print("\n[4] recall 'where do I work' (expect current only)")
        hits = await backend.recall("where do I work", scope=scope)
        for h in hits:
            line("hit", f"{h.memory.object_key}  (valid_until={h.memory.valid_until})")

        print("\n[5] time-travel recall as_of T0+1d (expect the older belief)")
        for h in await backend.recall("where do I work", scope=scope, as_of=T0 + timedelta(days=1)):
            line("hit", f"{h.memory.object_key}")

        # history over the predicate the model actually used for the first fact.
        if r1.fact_ids:
            first = await backend.get(r1.fact_ids[0])
            print(f"\n[6] history({first.subject_key!r}, {first.predicate_key!r})")
            for m in await backend.history(first.subject_key, first.predicate_key, scope=scope):
                arrow = "  (current)" if m.valid_until is None else f"  ->closed {m.valid_until.date()}"
                line("version", f"{m.object_key}{arrow}")

        if args.cognify:
            print("\n[7] improve(reindex) -> cognee cognify, then recall via graph channel")
            rep = await backend.improve(scope=scope, ops=("reindex",))
            line("cognified", str(rep.reindexed))
            snippets = await channel.search(f"scope_{r1.scope_id}", "where does the user work", top_k=5)
            line("cognee snippets", str(len(snippets)))
            for s in snippets[:2]:
                line("snippet", s[:100])

        print("\n=== smoke complete ===")
        return 0
    finally:
        await backend.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
