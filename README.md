# Engram

**A local-first, hybrid graph + vector memory service that persists and evolves across sessions.**

An *engram* is the physical trace a memory leaves behind. This service is that trace for
a local AI: it remembers facts from conversation, keeps them temporally correct as they
change, and recalls them later — all on-machine, with no cloud dependency required.

This repository is **Phase 1** of a planned local-first desktop AI environment. It is the
memory core only — a standalone, independently verifiable component with a REST API. No
UI, no Rust, no Tauri; a later phase consumes this service over HTTP.

> The Python import package is `local_infra` (the internal namespace used throughout the
> code and tests); the project/distribution is named **engram**.

## Why memory first

The hybrid graph+vector memory is the novel, most-likely-to-be-wrong part of the system.
A chat UI built on an unproven memory layer just hides its failures, so Phase 1 builds and
verifies the memory layer in isolation.

## What it does

- **Bitemporal facts.** Every fact carries validity time; nothing is destroyed. Contradictory
  updates *supersede* rather than overwrite, so `recall(..., as_of=<t>)` time-travels and full
  history is replayable. Postgres is the record of truth for temporal facts regardless of backend.
- **Two swappable backends behind one `MemoryService` interface.**
  - `ExMemBackend` — zero-infra: SQLite ledger + in-process FAISS + BM25 (RRF fusion).
  - `CogneeBackend` — Postgres + pgvector + tsvector, with a cognee graph channel as a third
    RRF retrieval channel that only re-ranks ledger rows (it can never inject a non-ledger id).
- **Hybrid extraction ladder.** Local-first, grammar-constrained JSON with groundedness
  validation, escalating stage-by-stage; a strict-offline mode (`MEMORY_OFFLINE=1`) never
  reaches a cloud model.
- **Async ingestion.** `POST /remember` returns immediately with a job id; a background worker
  runs the LLM extraction so a slow local model never blocks the request.
- **Consolidation.** `POST /improve` dedups/merges near-duplicate facts and prunes only the
  rows it folded — no distinct, valid fact is ever lost.

## Stack

Python 3.14 · Ollama (local inference + `nomic-embed-text` embeddings) · Postgres 17 + pgvector
in a rootless Podman container (port 5433) · FastAPI · cognee 1.4 (graph/enrichment) · uv.

## Quickstart

```sh
uv sync                                   # install
make up && make migrate                   # start Postgres (podman-compose) + apply migrations
make test                                 # full suite (pg up: 95 passed)
uv run uvicorn local_infra.api.app:app    # serve the REST API on :8077
```

Choose the backend with `MEMORY_BACKEND=exmem` (default, zero-infra) or `MEMORY_BACKEND=cognee`.
All settings are `MEMORY_*` env vars (or a `.env`); see `src/local_infra/config.py`.

### Acceptance path

```
POST /remember  "I work at Acme as a backend engineer"   → episode + facts
POST /remember  "Actually I moved to Globex last month"  → supersedes works_at
GET  /recall?q="where do I work"                         → Globex only
GET  /recall?q="where do I work"&as_of=<last month>      → Acme
GET  /memories/{id}/history                              → both, chained
DELETE /memories/{id}                                    → tombstoned, episode intact
POST /improve                                            → dedup report, no fact lost
```

## Benchmarks

The benchmark rig drives both backends through the same synthetic-conversation harness and
reports recall@k, IRR (information retention rate), p50/p95 latency, and cost:

```sh
make bench
# uv run python -m local_infra.bench.cli --backends exmem,cognee --facts 20 --turns 60 --seed 42
```

IRR is scored by a deterministic, offline local judge (cached by fact + retrieved text), so
the whole rig runs with no cloud calls. Cost uses real Anthropic pricing for the cloud rung;
an all-local run (Ollama extraction + embeddings) is **$0.00**.

Representative run (seed 42, 20 facts / 60 turns):

| backend | recall@k | IRR | p50 ingest | p50 recall | cost |
|---------|----------|-----|-----------|-----------|------|
| exmem   | 85%      | 85% | ~4.4 s    | ~12 ms    | $0   |
| cognee  | 85%      | 85% | ~4.2 s    | ~47 ms    | $0   |

Accuracy matches because both backends share the same temporal ledger and extraction ladder —
only *recall* diverges (in-process FAISS+BM25 vs. Postgres pgvector+tsvector+graph).

## Layout

```
src/local_infra/
  core/         MemoryService ABC, pure temporal resolver, models, registry
  backends/     exmem (sqlite+faiss) and cognee (pg+graph) over a shared LedgerBackend
  extraction/   the escalation ladder, providers, gate, confidence
  embed/        Ollama / sentence-transformers embedders behind one protocol
  ingest/       job queue + async extraction worker
  consolidate/  dedup / merge / prune for improve()
  api/          FastAPI REST layer over the MemoryService seam
  bench/        the M11 benchmark rig
migrations/     forward-only SQL, applied by a Python (asyncpg) runner — no psql on-box
```
