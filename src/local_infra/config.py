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
