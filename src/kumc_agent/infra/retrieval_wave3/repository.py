from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.retrieval import AccessContext, RetrievalQuery, ScoredChunk
from kumc_agent.features.retrieval.access import is_chunk_visible
from kumc_agent.infra.database.postgres import PostgresClient


class RetrievalRepository(Protocol):
    def load_chunks(self, *, query: RetrievalQuery) -> list[Chunk]:
        ...

    def load_embeddings(
        self,
        *,
        model: str,
        dimensions: int,
    ) -> dict[str, np.ndarray]:
        ...

    def save_embeddings(
        self,
        *,
        model: str,
        dimensions: int,
        embeddings: dict[str, tuple[np.ndarray, str]],
    ) -> None:
        ...

    def record_search_run(
        self,
        *,
        query: RetrievalQuery,
        results: list[ScoredChunk],
        status: str,
        metadata: dict[str, object],
    ) -> str:
        ...


@dataclass(frozen=True)
class FileRetrievalRepository:
    root_dir: Path

    def load_chunks(self, *, query: RetrievalQuery) -> list[Chunk]:
        path = self.root_dir / "chunks.jsonl"
        if not path.exists():
            return []
        chunks: list[Chunk] = []
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                if not line.strip():
                    continue
                payload = json.loads(line)
                chunk = Chunk(
                    id=str(payload["id"]),
                    document_id=str(payload["document_id"]),
                    text=str(payload["text"]),
                    index=int(payload["chunk_index"]),
                    metadata=dict(payload.get("metadata", {})),
                )
                if _matches_source_filter(chunk, query.source_filter) and is_chunk_visible(
                    chunk,
                    query.access,
                ):
                    chunks.append(chunk)
        return chunks

    def load_embeddings(
        self,
        *,
        model: str,
        dimensions: int,
    ) -> dict[str, np.ndarray]:
        path = self.root_dir / "embeddings.jsonl"
        if not path.exists():
            return {}
        out: dict[str, np.ndarray] = {}
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("model") != model or int(payload.get("dimensions") or 0) != dimensions:
                    continue
                out[str(payload["chunk_id"])] = np.asarray(payload["embedding"], dtype=np.float32)
        return out

    def save_embeddings(
        self,
        *,
        model: str,
        dimensions: int,
        embeddings: dict[str, tuple[np.ndarray, str]],
    ) -> None:
        if not embeddings:
            return
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with (self.root_dir / "embeddings.jsonl").open("a", encoding="utf-8") as fw:
            for chunk_id, (vector, checksum) in embeddings.items():
                fw.write(
                    json.dumps(
                        {
                            "chunk_id": chunk_id,
                            "model": model,
                            "dimensions": dimensions,
                            "embedding": vector.astype(float).tolist(),
                            "checksum": checksum,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def record_search_run(
        self,
        *,
        query: RetrievalQuery,
        results: list[ScoredChunk],
        status: str,
        metadata: dict[str, object],
    ) -> str:
        from uuid import uuid4

        run_id = str(uuid4())
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with (self.root_dir / "search_runs.jsonl").open("a", encoding="utf-8") as fw:
            fw.write(
                json.dumps(
                    {
                        "id": run_id,
                        "query": query.text,
                        "actor_id": query.access.user_id,
                        "guild_id": query.access.guild_id,
                        "source_filter": query.source_filter,
                        "mode": query.mode,
                        "status": status,
                        "metadata": metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        with (self.root_dir / "search_run_results.jsonl").open("a", encoding="utf-8") as fw:
            for result in results:
                fw.write(
                    json.dumps(
                        {
                            "search_run_id": run_id,
                            "chunk_id": result.chunk.id,
                            "rank": result.rank,
                            "score": result.score,
                            "score_breakdown": result.score_breakdown,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return run_id


@dataclass(frozen=True)
class PostgresRetrievalRepository:
    postgres: PostgresClient

    def load_chunks(self, *, query: RetrievalQuery) -> list[Chunk]:
        source_filter = _normalize_source_filter(query.source_filter)
        params: list[object] = []
        where = [
            "c.index_status = 'active'",
            "c.redaction_policy <> 'deny'",
            "si.index_status = 'active'",
            "si.deleted_at is null",
        ]
        if source_filter != "all":
            where.append("si.source_kind = %s")
            params.append(source_filter)
        sql = f"""
            select c.id, c.document_id, c.text, c.chunk_index, c.metadata
            from chunks c
            join source_items si on si.id = c.source_item_id
            where {' and '.join(where)}
        """
        chunks: list[Chunk] = []
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                for row in cur.fetchall():
                    metadata = row[4]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    chunk = Chunk(
                        id=str(row[0]),
                        document_id=str(row[1]),
                        text=str(row[2]),
                        index=int(row[3]),
                        metadata=dict(metadata or {}),
                    )
                    if is_chunk_visible(chunk, query.access):
                        chunks.append(chunk)
        return chunks

    def load_embeddings(
        self,
        *,
        model: str,
        dimensions: int,
    ) -> dict[str, np.ndarray]:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select chunk_id, embedding
                    from embeddings
                    where model = %s and dimensions = %s
                    """,
                    (model, dimensions),
                )
                out: dict[str, np.ndarray] = {}
                for row in cur.fetchall():
                    payload = row[1]
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    out[str(row[0])] = np.asarray(payload, dtype=np.float32)
                return out

    def save_embeddings(
        self,
        *,
        model: str,
        dimensions: int,
        embeddings: dict[str, tuple[np.ndarray, str]],
    ) -> None:
        if not embeddings:
            return
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                for chunk_id, (vector, checksum) in embeddings.items():
                    cur.execute(
                        """
                        insert into embeddings (chunk_id, model, dimensions, embedding, checksum)
                        values (%s, %s, %s, %s::jsonb, %s)
                        on conflict (chunk_id, model, dimensions) do update set
                          embedding = excluded.embedding,
                          checksum = excluded.checksum,
                          created_at = now()
                        """,
                        (
                            chunk_id,
                            model,
                            dimensions,
                            json.dumps(vector.astype(float).tolist()),
                            checksum,
                        ),
                    )
            conn.commit()

    def record_search_run(
        self,
        *,
        query: RetrievalQuery,
        results: list[ScoredChunk],
        status: str,
        metadata: dict[str, object],
    ) -> str:
        from uuid import uuid4

        run_id = str(uuid4())
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into search_runs (
                      id, query, actor_id, guild_id, source_filter, mode, status, metadata
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        run_id,
                        query.text,
                        query.access.user_id,
                        query.access.guild_id or None,
                        query.source_filter,
                        query.mode,
                        status,
                        json.dumps(metadata, ensure_ascii=False, default=str),
                    ),
                )
                for result in results:
                    cur.execute(
                        """
                        insert into search_run_results (
                          search_run_id, chunk_id, rank, score, score_breakdown
                        )
                        values (%s, %s, %s, %s, %s::jsonb)
                        on conflict do nothing
                        """,
                        (
                            run_id,
                            result.chunk.id,
                            result.rank,
                            result.score,
                            json.dumps(result.score_breakdown, ensure_ascii=False),
                        ),
                    )
            conn.commit()
        return run_id


def build_retrieval_repository(
    *,
    postgres: PostgresClient,
    fallback_dir: Path,
) -> RetrievalRepository:
    if postgres.is_configured():
        return PostgresRetrievalRepository(postgres=postgres)
    return FileRetrievalRepository(root_dir=fallback_dir)


def _matches_source_filter(chunk: Chunk, source_filter: str) -> bool:
    normalized = _normalize_source_filter(source_filter)
    if normalized == "all":
        return True
    metadata = dict(chunk.metadata or {})
    return str(metadata.get("source_kind") or metadata.get("source_type") or "") == normalized


def _normalize_source_filter(value: str) -> str:
    normalized = (value or "all").strip().lower()
    aliases = {
        "drive": "google_drive",
        "hatena": "hatenablog",
    }
    return aliases.get(normalized, normalized)
