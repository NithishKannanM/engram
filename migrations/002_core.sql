-- 002: core tables — scopes, episodes, the memory ledger, and provenance.
--
-- vector(768) is hardcoded to nomic-embed-text's dimension, confirmed empirically
-- (M0.5 spike). Changing the embedding model means editing this and reindexing —
-- it is intentionally not silent.

-- Scopes: global / project:<key> / session:<id>. `name` is also the cognee dataset.
CREATE TABLE IF NOT EXISTS scopes (
    id         BIGSERIAL PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    kind       TEXT NOT NULL CHECK (kind IN ('global', 'project', 'session')),
    key        TEXT NOT NULL,
    -- A scope may be pinned to never leave the machine, regardless of global mode.
    offline_only BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The single global scope always exists so `remember()` never has to create it.
INSERT INTO scopes (name, kind, key)
VALUES ('global', 'global', 'global')
ON CONFLICT (name) DO NOTHING;

-- Episodes: durable, raw provenance. Written synchronously by remember() BEFORE
-- any extraction runs, so no information is ever lost to a failed LLM call.
CREATE TABLE IF NOT EXISTS episodes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope_id    BIGINT NOT NULL REFERENCES scopes (id) ON DELETE RESTRICT,
    content     TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'api',
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),  -- world time of the utterance
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()   -- system time it was stored
);
CREATE INDEX IF NOT EXISTS idx_episodes_scope ON episodes (scope_id, occurred_at DESC);

-- Memories: the bitemporal fact ledger. Postgres is the record of truth here,
-- regardless of which backend (cognee / ex_mem) is active.
CREATE TABLE IF NOT EXISTS memories (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope_id      BIGINT NOT NULL REFERENCES scopes (id) ON DELETE RESTRICT,

    -- The fact as a subject-predicate-object triple, plus a natural-language
    -- rendering used for embedding / full-text. *_key columns are normalized
    -- (lowercased, trimmed) so contradiction lookups are exact and index-friendly.
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    object        TEXT NOT NULL,
    subject_key   TEXT NOT NULL,
    predicate_key TEXT NOT NULL,
    object_key    TEXT NOT NULL,
    statement     TEXT NOT NULL,

    memory_type   TEXT NOT NULL DEFAULT 'fact',
    confidence    REAL NOT NULL DEFAULT 1.0,
    extraction_stage TEXT,
    -- attrs->>'cardinality' is read by idx_mem_single_current; keep it here.
    attrs         JSONB NOT NULL DEFAULT '{}'::jsonb,

    embedding     VECTOR(768),
    embed_model   TEXT,  -- record which model produced `embedding`; a swap must be detectable.
    tsv           TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', statement)) STORED,

    -- World time (when the fact was true in reality).
    valid_from    TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_until   TIMESTAMPTZ,  -- NULL == currently valid.
    -- System time (when we learned / last saw it).
    recorded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    seen_count    INTEGER NOT NULL DEFAULT 1,

    -- Temporal chain links.
    supersedes_id       UUID REFERENCES memories (id),
    invalidated_by_id   UUID REFERENCES memories (id),
    invalidation_reason TEXT
);

-- m:n provenance. ON DELETE RESTRICT so a careless episode delete fails loudly
-- rather than orphaning the facts derived from it.
CREATE TABLE IF NOT EXISTS memory_episodes (
    memory_id     UUID NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    episode_id    UUID NOT NULL REFERENCES episodes (id) ON DELETE RESTRICT,
    evidence_span TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, episode_id)
);
CREATE INDEX IF NOT EXISTS idx_mem_episodes_episode ON memory_episodes (episode_id);
