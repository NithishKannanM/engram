"""Forward-only SQL migration runner.

This box has no `psql` binary, so migrations run through asyncpg from Python.
Files in ``migrations/`` named ``NNN_name.sql`` are applied in lexical order,
each in its own transaction, and recorded in ``schema_migrations`` with a
checksum. Re-running is a no-op; an edited-after-apply file is refused loudly
rather than silently diverging the schema from its recorded history.

    uv run python -m local_infra.backends.pg.migrate
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

import asyncpg

from local_infra.config import Settings, get_settings

# src/local_infra/backends/pg/migrate.py -> project root is four parents up.
MIGRATIONS_DIR = Path(__file__).resolve().parents[4] / "migrations"

_TRACKING_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    checksum    TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _migration_files() -> list[Path]:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"no *.sql migrations found in {MIGRATIONS_DIR}")
    return files


async def _connect(settings: Settings, *, retries: int = 20, delay: float = 1.5) -> asyncpg.Connection:
    """Connect, retrying while a freshly-started container finishes booting."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await asyncpg.connect(settings.pg_dsn)
        except (OSError, asyncpg.PostgresError) as exc:  # not up yet
            last = exc
            if attempt < retries:
                await asyncio.sleep(delay)
    raise ConnectionError(
        f"could not connect to {settings.pg_host}:{settings.pg_port} "
        f"after {retries} attempts: {last!r}"
    )


async def run_migrations(settings: Settings | None = None) -> list[str]:
    """Apply every pending migration. Returns the filenames newly applied."""
    settings = settings or get_settings()
    conn = await _connect(settings)
    applied: list[str] = []
    try:
        await conn.execute(_TRACKING_DDL)
        recorded = {
            r["filename"]: r["checksum"]
            for r in await conn.fetch("SELECT filename, checksum FROM schema_migrations")
        }
        for path in _migration_files():
            sql = path.read_text(encoding="utf-8")
            checksum = _checksum(sql)
            if path.name in recorded:
                if recorded[path.name] != checksum:
                    raise RuntimeError(
                        f"{path.name} was edited after it was applied "
                        f"(checksum drift). Migrations are forward-only — add a new "
                        f"file instead of editing an applied one."
                    )
                continue
            # Each migration is atomic: DDL + its tracking row commit together.
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename, checksum) VALUES ($1, $2)",
                    path.name,
                    checksum,
                )
            applied.append(path.name)
            print(f"  applied {path.name}")
    finally:
        await conn.close()
    return applied


async def _main() -> int:
    applied = await run_migrations()
    if applied:
        print(f"migrate: {len(applied)} migration(s) applied.")
    else:
        print("migrate: schema already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
