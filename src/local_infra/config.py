"""Runtime configuration for the memory service.

Single source of truth for every knob the service reads. Everything is
overridable via `MEMORY_*` environment variables (or a local `.env`) so the
same code runs identically in tests, the CLI, and the API without editing
Python. Defaults describe *this* machine — rootless Postgres on :5433, a local
Ollama, offline-first — because that is the environment Phase 1 targets.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/local_infra/config.py -> project root is three parents up.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-driven settings; construct via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Postgres (rootless Podman container, host port 5433) ---
    pg_host: str = "127.0.0.1"
    pg_port: int = 5433
    pg_user: str = "local_infra"
    pg_password: str = "local_infra"
    pg_database: str = "local_infra"
    # Pool sizing stays small: this box shares 8.7GB RAM with an 8B model.
    pg_pool_min: int = 1
    pg_pool_max: int = 8

    # --- Ollama (local inference + embeddings) ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    # Extraction ladder models. The 4B (qwen3.5) is deliberately absent: it
    # returns empty content under Ollama's `format` grammar constraint.
    extract_model_small: str = "llama3.2:latest"  # stage 1/2, ~0.3s warm
    extract_model_large: str = "llama3.1:8b"  # stage 3, last offline rung
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768  # nomic-embed-text; a change here means a full reindex.

    # --- Extraction / escalation policy ---
    offline: bool = False  # MEMORY_OFFLINE=1 -> ladder never reaches CLOUD.
    cloud_budget_usd: float = 0.0  # per-process ceiling; 0 disables cloud spend.

    # --- Backend selection + REST layer ---
    backend: str = "exmem"  # "exmem" (zero-infra) | "cognee" (pg ledger + graph)
    api_host: str = "127.0.0.1"
    api_port: int = 8077
    # Browser-origin clients (the Tauri desktop shell) need CORS: its webview is a
    # real origin, so without these headers every fetch fails as an opaque network
    # error. Vite dev serves :1420; a bundled Tauri app on Linux/Windows serves
    # http://tauri.localhost, on macOS tauri://localhost.
    cors_origins: list[str] = [
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "http://tauri.localhost",
        "tauri://localhost",
    ]

    # --- Chat / RAG synthesis (LiteLLM-routed; NOT the extraction ladder) ---
    chat_enabled: bool = True
    # Default to models already resident for extraction. Only ONE chat-sized model
    # fits in 6GB, so introducing a third would make Ollama evict and reload on
    # every alternation — `keep_alive` exists precisely to avoid that ~8.5s cost.
    chat_model_default: str = "ollama_chat/llama3.1:8b"
    chat_model_fast: str = "ollama_chat/llama3.2:latest"
    # Ignored entirely when offline: registry never builds a cloud deployment then.
    chat_model_cloud: str | None = None
    chat_temperature: float = 0.3
    chat_max_tokens: int = 1024
    # Ollama silently defaults to a 2048-token context. A RAG prompt then gets
    # left-truncated and the model answers from nothing — which looks like broken
    # retrieval, not a truncated prompt. 8192 would cost ~1GB more KV cache and
    # spill past the ~5.8GB of free VRAM, so 4096 is the compromise.
    chat_num_ctx: int = 4096
    chat_timeout_s: float = 120.0
    # Separate from cloud_budget_usd on purpose: sharing the ladder's budget would
    # let chat spend starve extraction's cloud rung with no visible cause.
    chat_budget_usd: float = 0.0
    chat_recall_limit: int = 6
    # Keys live server-side. Accepting them per-request would force the router to
    # BUILD a cloud deployment from request data, destroying the structural offline
    # guarantee, and would leak secrets into access logs and tracebacks.
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # --- Document RAG (Postgres backend only; exmem answers 501) ---
    documents_enabled: bool = True
    # Chunk sizing, in the splitter's `len // 4` token estimate. 350 keeps a chunk well
    # inside nomic-embed-text's window while staying large enough to carry a whole
    # argument; 48 tokens of overlap means a fact straddling a boundary is retrievable
    # from either side. `min` must stay below `max` — the splitter clamps it if not.
    doc_chunk_tokens: int = 350
    doc_chunk_min_tokens: int = 80
    doc_chunk_overlap_tokens: int = 48
    # Fused memory+chunk results returned by /rag before synthesis.
    rag_limit: int = 8

    # --- Cognee on-disk state (kept OUT of .venv/site-packages) ---
    state_dir: Path = Field(default=_PROJECT_ROOT / ".cognee_state")

    @property
    def pg_dsn(self) -> str:
        """libpq/asyncpg DSN. No driver prefix — asyncpg.connect wants this shape."""
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Clear the cache in tests to re-read env."""
    return Settings()
