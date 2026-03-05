from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLiteKVStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    def set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        cur = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,))
        row = cur.fetchone()
        return None if row is None else str(row[0])

    def close(self) -> None:
        self._conn.close()
