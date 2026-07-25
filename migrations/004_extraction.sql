-- 004: extraction ingest queue + telemetry.
--
-- remember() writes an episode, then enqueues an extraction_job. A background
-- worker claims jobs with SELECT ... FOR UPDATE SKIP LOCKED so multiple workers
-- never process the same episode. Every ladder rung logs an extraction_attempt
-- so the confidence weights can be tuned against benchmark ground truth instead
-- of guessed forever.

CREATE TABLE IF NOT EXISTS extraction_jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id  UUID NOT NULL REFERENCES episodes (id) ON DELETE RESTRICT,
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
-- Worker claim path: oldest queued/deferred job first. Partial index keeps it
-- tight as `done` rows accumulate.
CREATE INDEX IF NOT EXISTS idx_jobs_claimable
    ON extraction_jobs (created_at)
    WHERE status IN ('queued', 'deferred');

CREATE TABLE IF NOT EXISTS extraction_attempts (
    id               BIGSERIAL PRIMARY KEY,
    job_id           UUID NOT NULL REFERENCES extraction_jobs (id) ON DELETE CASCADE,
    stage            TEXT NOT NULL,   -- GATE / LOCAL_SMALL / LOCAL_REPAIR / LOCAL_LARGE / CLOUD / DEFER
    provider         TEXT,
    model            TEXT,
    ok               BOOLEAN NOT NULL DEFAULT FALSE,
    -- Composite confidence terms, persisted so the weights are tunable, not fixed.
    groundedness     REAL,
    self_consistency REAL,
    structural       REAL,
    self_report      REAL,
    composite        REAL,
    facts_count      INTEGER,
    latency_ms       INTEGER,
    cost_usd         REAL NOT NULL DEFAULT 0.0,
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_attempts_job ON extraction_attempts (job_id, created_at);
