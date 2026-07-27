"""Document ingestion: validate → read → split → embed → retrieve.

Deliberately separate from `exmem/` (whose chunker is conversation-shaped) and from
`core/repository.py` (whose `Repository` protocol is implemented by both SQLite and
Postgres and asserted against both). Documents are a Postgres-only capability; the
exmem backend answers 501 rather than pretending.
"""
