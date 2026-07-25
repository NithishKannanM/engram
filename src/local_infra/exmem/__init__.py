"""Vendored ex_mem engine — SQLite + FAISS/BM25 hybrid retrieval.

Copied from projects/ex_mem and bug-fixed on port (see the module docstrings):
the chunker now counts the first turn's tokens, the retriever adds incrementally
instead of re-embedding the whole index, and the strategy honors its configured
db_path so state survives a restart. Embedding is injected via the
:class:`~local_infra.embed.base.Embedder` protocol rather than hardcoded to
sentence-transformers.
"""
