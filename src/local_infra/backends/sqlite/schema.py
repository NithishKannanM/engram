"""SQLite ledger schema — mirrors the Postgres columns minus vector/tsvector.

Those live in FAISS/BM25 for the ex_mem backend, so the ledger tables here are
column-identical to Postgres for everything the contract test checks. The
single-valued guard is reproduced as a partial UNIQUE index using json_extract,
so the same "two current single-valued rows" bug raises here too.
"""

from __future__ import annotations

SQLITE_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS scopes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,
    key          TEXT NOT NULL,
    offline_only INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    scope_id    INTEGER NOT NULL REFERENCES scopes(id),
    content     TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'api',
    metadata    TEXT NOT NULL DEFAULT '{}',
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id                  TEXT PRIMARY KEY,
    scope_id            INTEGER NOT NULL REFERENCES scopes(id),
    subject             TEXT NOT NULL,
    predicate           TEXT NOT NULL,
    object              TEXT NOT NULL,
    subject_key         TEXT NOT NULL,
    predicate_key       TEXT NOT NULL,
    object_key          TEXT NOT NULL,
    statement           TEXT NOT NULL,
    memory_type         TEXT NOT NULL DEFAULT 'fact',
    confidence          REAL NOT NULL DEFAULT 1.0,
    extraction_stage    TEXT,
    attrs               TEXT NOT NULL DEFAULT '{}',
    embed_model         TEXT,
    valid_from          TEXT NOT NULL,
    valid_until         TEXT,
    recorded_at         TEXT NOT NULL,
    last_seen_at        TEXT NOT NULL,
    seen_count          INTEGER NOT NULL DEFAULT 1,
    supersedes_id       TEXT,
    invalidated_by_id   TEXT,
    invalidation_reason TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_mem_single_current
    ON memories (scope_id, subject_key, predicate_key)
    WHERE valid_until IS NULL AND json_extract(attrs, '$.cardinality') = 'single';

CREATE INDEX IF NOT EXISTS idx_mem_triple_current
    ON memories (scope_id, subject_key, predicate_key)
    WHERE valid_until IS NULL;

CREATE INDEX IF NOT EXISTS idx_mem_asof
    ON memories (scope_id, valid_from, valid_until);

CREATE TABLE IF NOT EXISTS memory_episodes (
    memory_id     TEXT NOT NULL REFERENCES memories(id),
    episode_id    TEXT NOT NULL REFERENCES episodes(id),
    evidence_span TEXT,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (memory_id, episode_id)
);

CREATE TABLE IF NOT EXISTS predicates (
    key         TEXT PRIMARY KEY,
    cardinality TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS memory_revisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id     TEXT,
    revision_type TEXT NOT NULL,
    reason        TEXT,
    decided_by    TEXT,
    observed_at   TEXT,
    payload       TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL
);
"""
