-- 003: temporal machinery — predicate cardinality registry, the invalidation
-- audit log, and the load-bearing partial indexes on `memories`.

-- Predicate cardinality registry. SINGLE-valued predicates supersede by rule
-- (no LLM) when the object changes; MULTI predicates coexist. Seeding the common
-- single-valued ones handles most preference churn deterministically.
CREATE TABLE IF NOT EXISTS predicates (
    key         TEXT PRIMARY KEY,
    cardinality TEXT NOT NULL CHECK (cardinality IN ('single', 'multi')),
    description TEXT
);

INSERT INTO predicates (key, cardinality, description) VALUES
    ('lives_in',        'single', 'current place of residence'),
    ('works_at',        'single', 'current employer'),
    ('current_role',    'single', 'current job title / role'),
    ('current_project', 'single', 'project actively worked on'),
    ('prefers_editor',  'single', 'preferred code editor'),
    ('prefers_shell',   'single', 'preferred shell'),
    ('prefers_language','single', 'preferred programming language'),
    ('uses_os',         'single', 'operating system in use'),
    ('primary_email',   'single', 'primary email address'),
    ('phone_number',    'single', 'current phone number'),
    ('timezone',        'single', 'current timezone'),
    ('marital_status',  'single', 'marital status'),
    ('employer',        'single', 'current employer (alias of works_at)'),
    ('job_title',       'single', 'current job title'),
    ('city',            'single', 'current city'),
    ('country',         'single', 'current country'),
    ('age',             'single', 'current age'),
    ('birthday',        'single', 'date of birth'),
    ('full_name',       'single', 'full legal name'),
    ('nickname',        'single', 'preferred name to be called'),
    ('default_branch',  'single', 'default git branch preference'),
    ('current_goal',    'single', 'active primary goal'),
    ('owns_device',     'single', 'primary computing device'),
    ('gpu',             'single', 'current GPU'),
    ('salary',          'single', 'current salary'),
    -- A few explicitly MULTI predicates so coexistence is exercised from day one.
    ('knows_language',  'multi',  'a programming/spoken language known'),
    ('likes',           'multi',  'something liked'),
    ('dislikes',        'multi',  'something disliked'),
    ('has_skill',       'multi',  'a skill possessed'),
    ('worked_at',       'multi',  'a past employer (history)')
ON CONFLICT (key) DO NOTHING;

-- Append-only, fully replayable audit of every ledger mutation. A bad
-- contradiction-judge batch can be reversed by replaying these in reverse.
CREATE TABLE IF NOT EXISTS memory_revisions (
    id            BIGSERIAL PRIMARY KEY,
    memory_id     UUID REFERENCES memories (id),
    revision_type TEXT NOT NULL CHECK (
        revision_type IN ('insert', 'close', 'touch', 'reopen', 'hard_delete')
    ),
    reason        TEXT,
    decided_by    TEXT,  -- e.g. 'rule:single_valued', 'judge:local', 'user_forget'
    observed_at   TIMESTAMPTZ,
    payload       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_mem_revisions_memory ON memory_revisions (memory_id, created_at);

-- THE integrity guard: a single-valued predicate may have at most ONE
-- currently-valid value per scope. A bug that forgets to close the old row
-- raises IntegrityError instead of silently storing two contradictory truths.
CREATE UNIQUE INDEX IF NOT EXISTS idx_mem_single_current
    ON memories (scope_id, subject_key, predicate_key)
    WHERE valid_until IS NULL AND (attrs->>'cardinality') = 'single';

-- Recall hot path: partial HNSW over only currently-valid rows.
CREATE INDEX IF NOT EXISTS idx_mem_vec_current
    ON memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE valid_until IS NULL;

-- Contradiction-candidate lookup by triple key (no LLM).
CREATE INDEX IF NOT EXISTS idx_mem_triple_current
    ON memories (scope_id, subject_key, predicate_key)
    WHERE valid_until IS NULL;

-- Sparse full-text channel, current rows only.
CREATE INDEX IF NOT EXISTS idx_mem_tsv_current
    ON memories USING GIN (tsv)
    WHERE valid_until IS NULL;

-- Time-travel range scans (as-of queries).
CREATE INDEX IF NOT EXISTS idx_mem_asof
    ON memories (scope_id, valid_from, valid_until);
