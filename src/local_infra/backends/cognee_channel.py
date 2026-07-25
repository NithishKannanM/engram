"""The cognee enrichment + graph-recall channel, behind a small seam.

cognee is an *enrichment* layer and a third retrieval channel — never the record of
truth for temporal facts (Postgres is, always). Keeping it behind :class:`CogneeChannel`
means the contract suite runs against a no-op channel (deterministic, no LLM, no graph
build), while production wires the real graph channel — exactly how extraction and
embeddings are faked in the ex_mem suite.

Datasets are keyed by scope id (``scope_<id>``), which the backend derives identically
on write (add), recall (search), and consolidation (cognify).

The real channel's configuration is the load-bearing, six-times-debugged block from the
M0.5 spike (``scripts/spike_cognee.py``); see that file and the project memory for why
each line is exactly what it is. ``cognee`` is imported lazily, *after* the env vars are
set, because it reads them at import time.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CogneeChannel(Protocol):
    name: str

    async def add(self, dataset: str, content: str) -> None:
        """Ingest raw text into ``dataset`` (cheap; graph build is deferred to cognify)."""
        ...

    async def cognify(self, dataset: str) -> int:
        """Build/refresh the graph for ``dataset``. Returns items processed."""
        ...

    async def search(self, dataset: str, query: str, *, top_k: int = 10) -> list[str]:
        """Return ranked text snippets for ``query`` (a retrieval channel, no completion)."""
        ...

    async def close(self) -> None:
        ...


class NullCogneeChannel:
    """No-op channel: the ledger carries all correctness, cognee adds nothing. Used by
    the contract suite and as the safe fallback when cognee is unconfigured."""

    name = "null"

    async def add(self, dataset: str, content: str) -> None:
        return None

    async def cognify(self, dataset: str) -> int:
        return 0

    async def search(self, dataset: str, query: str, *, top_k: int = 10) -> list[str]:
        return []

    async def close(self) -> None:
        return None


class CogneeGraphChannel:
    """The real channel: cognee 1.4.0 on our Postgres + Ollama, fully offline."""

    name = "cognee"

    def __init__(
        self,
        *,
        pg_host: str,
        pg_port: int,
        pg_user: str,
        pg_password: str,
        pg_database: str,
        ollama_base_url: str,
        llm_model: str,
        embed_model: str,
        embed_dim: int,
        state_dir: str,
    ) -> None:
        import os
        import pathlib

        # Must be set before importing cognee — it reads them at module import time.
        os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
        os.environ.setdefault("CACHING", "false")
        os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

        state = pathlib.Path(state_dir)
        state.mkdir(parents=True, exist_ok=True)

        import cognee  # noqa: E402 — deferred until env is set
        from cognee.modules.search.types import SearchType

        self._cognee = cognee
        self._chunks_search = SearchType.CHUNKS

        cognee.config.system_root_directory(str(state / "system"))
        cognee.config.data_root_directory(str(state / "data"))
        cognee.config.set_relational_db_config(
            {
                "db_provider": "postgres",
                "db_host": pg_host,
                "db_port": str(pg_port),
                "db_username": pg_user,
                "db_password": pg_password,
                "db_name": pg_database,
            }
        )
        # Vector config takes ONLY vector_db_*-prefixed keys (not the relational shape).
        cognee.config.set_vector_db_config(
            {
                "vector_db_provider": "pgvector",
                "vector_db_url": "",
                "vector_db_key": "",
                "vector_db_host": pg_host,
                "vector_db_port": pg_port,
                "vector_db_name": pg_database,
                "vector_db_username": pg_user,
                "vector_db_password": pg_password,
            }
        )
        cognee.config.set_graph_database_provider("kuzu")  # embedded; pg cannot host it
        # LLM/embeddings via Ollama's OpenAI-compatible base; bare model tags.
        cognee.config.set_llm_provider("ollama")
        cognee.config.set_llm_endpoint(f"{ollama_base_url}/v1")
        cognee.config.set_llm_model(llm_model)
        cognee.config.set_llm_api_key("ollama")
        cognee.config.set_embedding_provider("ollama")
        cognee.config.set_embedding_endpoint(f"{ollama_base_url}/api/embed")
        cognee.config.set_embedding_model(embed_model)
        cognee.config.set_embedding_dimensions(embed_dim)
        cognee.config.set_embedding_api_key("ollama")

    async def add(self, dataset: str, content: str) -> None:
        await self._cognee.add(content, dataset_name=dataset)

    async def cognify(self, dataset: str) -> int:
        # Enrichment must never break the service. cognee's internal summarization step
        # uses instructor/json_mode structured output, which a local model can fail
        # (e.g. llama3.1:8b emits a dict/schema for SummarizedContent instead of a
        # string) — a graph that failed to build just means recall falls back to the
        # authoritative pg channels. Report 0 built rather than raise.
        try:
            await self._cognee.cognify(datasets=[dataset])
        except Exception:
            return 0
        return 1

    async def search(self, dataset: str, query: str, *, top_k: int = 10) -> list[str]:
        try:
            results = await self._cognee.search(
                query_text=query,
                query_type=self._chunks_search,
                datasets=[dataset],
                top_k=top_k,
            )
        except Exception:
            # Enrichment must never break recall — the ledger channels stand alone.
            return []
        return [str(r) for r in results]

    async def close(self) -> None:
        return None
