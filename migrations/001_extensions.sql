-- 001: extensions.
-- pgvector supplies the `vector` type and HNSW index used by the recall hot path.
-- gen_random_uuid() is core in pg13+, so no pgcrypto is needed for UUID PKs.
CREATE EXTENSION IF NOT EXISTS vector;
