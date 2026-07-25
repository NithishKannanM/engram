"""SQLite metadata store for retrievable chunks (vendored from ex_mem, unchanged).

Already persists when constructed with a file path — the persistence bug was in
the *strategy* passing ``:memory:`` regardless, not here.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    turn_start  INTEGER NOT NULL,
    turn_end    INTEGER NOT NULL,
    role        TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    importance  REAL NOT NULL DEFAULT 1.0,
    metadata    TEXT NOT NULL DEFAULT '{}',
    faiss_id    INTEGER NOT NULL DEFAULT -1
);
CREATE INDEX IF NOT EXISTS idx_turn ON chunks(turn_start);
CREATE INDEX IF NOT EXISTS idx_faiss ON chunks(faiss_id);
"""


@dataclass
class Chunk:
    id: str
    text: str
    turn_start: int
    turn_end: int
    role: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    metadata: dict = field(default_factory=dict)
    faiss_id: int = -1  # row index in the FAISS flat index


class SQLiteMetadataStore:
    def __init__(self, db_path: str = ":memory:") -> None:
        self._path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()

    # -- write --

    def upsert(self, chunk: Chunk) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO chunks
               (id, text, turn_start, turn_end, role, timestamp, importance, metadata, faiss_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                chunk.id, chunk.text, chunk.turn_start, chunk.turn_end,
                chunk.role, chunk.timestamp, chunk.importance,
                json.dumps(chunk.metadata), chunk.faiss_id,
            ),
        )
        self._conn.commit()

    # -- read --

    def get(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE id=?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def all_chunks(self) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT * FROM chunks ORDER BY turn_start"
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # -- util --

    @staticmethod
    def _row_to_chunk(row: tuple) -> Chunk:
        return Chunk(
            id=row[0], text=row[1], turn_start=row[2], turn_end=row[3],
            role=row[4], timestamp=row[5], importance=row[6],
            metadata=json.loads(row[7]), faiss_id=row[8],
        )

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
