from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Iterable, Protocol

import numpy as np


@dataclass(frozen=True)
class IndexEmbeddingCacheKey:
    provider: str
    model: str
    dimensions: int
    chunk_id: str
    embedding_text_hash: str


@dataclass(frozen=True)
class IndexEmbeddingRecord:
    provider: str
    model: str
    dimensions: int
    chunk_id: str
    embedding_text_hash: str
    vector: np.ndarray
    chunk_metadata_hash: str = ""
    source_kind: str = ""
    source_item_id: str = ""
    created_at: str = ""

    @property
    def key(self) -> IndexEmbeddingCacheKey:
        return IndexEmbeddingCacheKey(
            provider=self.provider,
            model=self.model,
            dimensions=self.dimensions,
            chunk_id=self.chunk_id,
            embedding_text_hash=self.embedding_text_hash,
        )

    def as_payload(self) -> dict[str, object]:
        vector = np.asarray(self.vector, dtype=np.float32)
        return {
            "provider": self.provider,
            "model": self.model,
            "dimensions": int(self.dimensions),
            "chunk_id": self.chunk_id,
            "embedding_text_hash": self.embedding_text_hash,
            "vector": vector.astype(float).tolist(),
            "chunk_metadata_hash": self.chunk_metadata_hash,
            "source_kind": self.source_kind,
            "source_item_id": self.source_item_id,
            "created_at": self.created_at or datetime.now(UTC).isoformat(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> IndexEmbeddingRecord:
        dimensions = int(payload.get("dimensions") or 0)
        vector = np.asarray(payload.get("vector") or [], dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("cached embedding vector must be 1D")
        if dimensions <= 0 or int(vector.shape[0]) != dimensions:
            raise ValueError("cached embedding vector dimension mismatch")
        chunk_id = str(payload.get("chunk_id") or "").strip()
        embedding_text_hash = str(payload.get("embedding_text_hash") or "").strip()
        provider = str(payload.get("provider") or "").strip()
        model = str(payload.get("model") or "").strip()
        if not chunk_id or not embedding_text_hash or not provider or not model:
            raise ValueError("cached embedding record is missing required key fields")
        return cls(
            provider=provider,
            model=model,
            dimensions=dimensions,
            chunk_id=chunk_id,
            embedding_text_hash=embedding_text_hash,
            vector=vector,
            chunk_metadata_hash=str(payload.get("chunk_metadata_hash") or ""),
            source_kind=str(payload.get("source_kind") or ""),
            source_item_id=str(payload.get("source_item_id") or ""),
            created_at=str(payload.get("created_at") or ""),
        )


@dataclass(frozen=True)
class IndexEmbeddingCacheLoadResult:
    records: dict[IndexEmbeddingCacheKey, IndexEmbeddingRecord]
    invalid_records: int = 0


class IndexEmbeddingCache(Protocol):
    def load(
        self,
        *,
        provider: str,
        model: str,
        dimensions: int,
    ) -> IndexEmbeddingCacheLoadResult:
        ...

    def save(self, records: Iterable[IndexEmbeddingRecord]) -> None:
        ...

    def compact(self, active_keys: Iterable[IndexEmbeddingCacheKey]) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class FileIndexEmbeddingCache:
    cache_dir: Path

    def load(
        self,
        *,
        provider: str,
        model: str,
        dimensions: int,
    ) -> IndexEmbeddingCacheLoadResult:
        path = self._path(provider=provider, model=model, dimensions=dimensions)
        return self._load_path(
            path,
            provider=provider,
            model=model,
            dimensions=dimensions,
        )

    def save(self, records: Iterable[IndexEmbeddingRecord]) -> None:
        grouped: dict[tuple[str, str, int], list[IndexEmbeddingRecord]] = {}
        for record in records:
            grouped.setdefault(
                (record.provider, record.model, int(record.dimensions)),
                [],
            ).append(record)
        if not grouped:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for (provider, model, dimensions), group in grouped.items():
            path = self._path(provider=provider, model=model, dimensions=dimensions)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fw:
                for record in group:
                    fw.write(
                        json.dumps(record.as_payload(), ensure_ascii=False, default=str)
                        + "\n"
                    )

    def compact(self, active_keys: Iterable[IndexEmbeddingCacheKey]) -> dict[str, object]:
        grouped: dict[tuple[str, str, int], set[IndexEmbeddingCacheKey]] = {}
        for key in active_keys:
            grouped.setdefault(
                (key.provider, key.model, int(key.dimensions)),
                set(),
            ).add(key)
        if not grouped:
            return {"status": "skipped", "reason": "no_active_keys"}
        compacted_files = 0
        kept_records = 0
        invalid_records = 0
        for (provider, model, dimensions), keys in grouped.items():
            path = self._path(provider=provider, model=model, dimensions=dimensions)
            loaded = self._load_path(
                path,
                provider=provider,
                model=model,
                dimensions=dimensions,
            )
            invalid_records += loaded.invalid_records
            records = [record for key, record in loaded.records.items() if key in keys]
            if not path.exists() and not records:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as fw:
                for record in records:
                    fw.write(
                        json.dumps(record.as_payload(), ensure_ascii=False, default=str)
                        + "\n"
                    )
            tmp_path.replace(path)
            compacted_files += 1
            kept_records += len(records)
        return {
            "status": "succeeded",
            "compacted_files": compacted_files,
            "kept_records": kept_records,
            "invalid_records": invalid_records,
        }

    def _load_path(
        self,
        path: Path,
        *,
        provider: str,
        model: str,
        dimensions: int,
    ) -> IndexEmbeddingCacheLoadResult:
        if not path.exists():
            return IndexEmbeddingCacheLoadResult(records={})
        records: dict[IndexEmbeddingCacheKey, IndexEmbeddingRecord] = {}
        invalid_records = 0
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("cached embedding record is not an object")
                    record = IndexEmbeddingRecord.from_payload(payload)
                except Exception:
                    invalid_records += 1
                    continue
                if (
                    record.provider != provider
                    or record.model != model
                    or int(record.dimensions) != int(dimensions)
                ):
                    continue
                records[record.key] = record
        return IndexEmbeddingCacheLoadResult(
            records=records,
            invalid_records=invalid_records,
        )

    def _path(self, *, provider: str, model: str, dimensions: int) -> Path:
        provider_part = _safe_path_part(provider)
        model_part = _safe_path_part(model)
        return self.cache_dir / f"{provider_part}-{model_part}-{int(dimensions)}.jsonl"


def _safe_path_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return safe.strip("._-") or "default"
