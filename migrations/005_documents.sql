-- 005: document ingestion — uploaded files, their chunks, and their own job queue.
--
-- A NEW file rather than an edit to 001–004: the runner checksums every migration and
-- refuses one that changed after being applied, so an already-migrated database can
-- only move forward.
--
-- Bytes do NOT live here. The blob goes to state_dir/documents/<sha[:2]>/<sha>; this
-- table stores the sha256 and metadata. A 25MB PDF in a jsonb/bytea column would be
-- pulled into memory on every metadata SELECT and bloat every backup.

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope_id    BIGINT NOT NULL REFERENCES scopes (id) ON DELETE RESTRICT,
    filename    TEXT NOT NULL,      -- display only; the on-disk name is the sha256
    kind        TEXT NOT NULL CHECK (kind IN ('pdf', 'docx', 'txt', 'md')),
    sha256      TEXT NOT NULL,
    size_bytes  BIGINT NOT NULL,
    pages       INTEGER,            -- NULL for formats without page structure
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    error       TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Re-uploading the same file into the same scope is idempotent: the caller gets
    -- back the existing document instead of a second copy that would double every
    -- chunk's contribution to retrieval. The same bytes in a DIFFERENT scope are a
    -- separate document on purpose — scopes are the isolation boundary.
    UNIQUE (scope_id, sha256)
);
CREATE INDEX IF NOT EXISTS idx_documents_scope ON documents (scope_id, created_at DESC);

-- Chunks are DERIVED data: re-splitting the blob reproduces them exactly. Hence
-- ON DELETE CASCADE — deliberately the opposite of memory_episodes' RESTRICT, where
-- the episode is irreplaceable provenance and a silent orphan would be data loss.
CREATE TABLE IF NOT EXISTS document_chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    scope_id     BIGINT NOT NULL REFERENCES scopes (id) ON DELETE RESTRICT,
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    -- Offsets into the extracted document text, so a citation can highlight the exact
    -- source span rather than a re-joined approximation of it.
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    page_from    INTEGER,
    tokens       INTEGER NOT NULL DEFAULT 0,
    embedding    VECTOR(768),   -- nomic-embed-text, same model as `memories`
    embed_model  TEXT,
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (document_id, chunk_index)
);

-- These indexes are deliberately NOT partial, unlike the `memories` ones. A chunk has
-- no valid_until: it is either present or deleted, so there is no "current rows only"
-- subset to filter on and a WHERE clause would only make the index unusable.
CREATE INDEX IF NOT EXISTS idx_doc_chunks_vec
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_tsv
    ON document_chunks USING GIN (tsv);

-- Retrieval always filters by scope first, then ranks.
CREATE INDEX IF NOT EXISTS idx_doc_chunks_scope ON document_chunks (scope_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_document ON document_chunks (document_id, chunk_index);

-- Documents get their own queue: extraction_jobs.episode_id is NOT NULL REFERENCES
-- episodes, so a document job is structurally unrepresentable there. Same claim
-- protocol (FOR UPDATE SKIP LOCKED), same worker code, different table.
CREATE TABLE IF NOT EXISTS document_jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    scope_id    BIGINT NOT NULL REFERENCES scopes (id) ON DELETE RESTRICT,
    status      TEXT NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'running', 'done', 'deferred', 'failed')),
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    locked_by   TEXT,
    locked_at   TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_document_jobs_claimable
    ON document_jobs (created_at)
    WHERE status IN ('queued', 'deferred');
