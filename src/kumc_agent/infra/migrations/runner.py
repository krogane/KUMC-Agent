from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from kumc_agent.config.schema import MigrationSection
from kumc_agent.infra.database.postgres import PostgresClient


@dataclass(frozen=True)
class MigrationResult:
    applied: tuple[str, ...]
    skipped: tuple[str, ...]


@dataclass(frozen=True)
class PostgresMigrationRunner:
    client: PostgresClient
    config: MigrationSection

    def _table_name(self) -> str:
        value = (self.config.table_name or "").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(f"Invalid migration table name: {value!r}")
        return value

    def pending_files(self) -> tuple[Path, ...]:
        if not self.config.directory.exists():
            return tuple()
        return tuple(sorted(self.config.directory.glob("*.sql")))

    def apply(self) -> MigrationResult:
        if not self.client.is_configured():
            raise RuntimeError("KUMC_DATABASE_URL is required to run migrations.")

        applied: list[str] = []
        skipped: list[str] = []
        table_name = self._table_name()
        with self.client.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    create table if not exists {table_name} (
                      version text primary key,
                      applied_at timestamptz not null default now()
                    )
                    """
                )
                for path in self.pending_files():
                    version = path.stem
                    cur.execute(
                        f"select 1 from {table_name} where version = %s",
                        (version,),
                    )
                    if cur.fetchone():
                        skipped.append(version)
                        continue
                    cur.execute(path.read_text(encoding="utf-8"))
                    cur.execute(
                        f"insert into {table_name} (version) values (%s)",
                        (version,),
                    )
                    applied.append(version)
            conn.commit()
        return MigrationResult(applied=tuple(applied), skipped=tuple(skipped))
