from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class IndexQualityResult:
    passed: bool
    critical_failures: tuple[str, ...]
    metadata: dict[str, object]


class IndexQualitySmokeChecker:
    def __init__(
        self,
        *,
        min_chunk_ratio: float = 0.5,
        smoke_queries: tuple[str, ...] = tuple(),
    ) -> None:
        self._min_chunk_ratio = max(0.0, float(min_chunk_ratio))
        self._smoke_queries = tuple(query.strip() for query in smoke_queries if query.strip())

    def check(self, *, staging_dir: Path, current_dir: Path) -> IndexQualityResult:
        failures: list[str] = []
        staging_chunks = _read_chunks(staging_dir / "dense_chunks.jsonl")
        current_chunks = _read_chunks(current_dir / "dense_chunks.jsonl")

        for required in ("dense_vectors.npy", "dense_chunks.jsonl", "bm25_tokens.json", "bm25_chunks.jsonl"):
            if not (staging_dir / required).exists():
                failures.append(f"missing_artifact:{required}")
        if not staging_chunks:
            failures.append("chunk_count_zero")
        if current_chunks and len(staging_chunks) / max(1, len(current_chunks)) < self._min_chunk_ratio:
            failures.append("chunk_count_ratio_below_threshold")
        disallowed = [
            str(chunk.get("id") or "")
            for chunk in staging_chunks
            if _index_status(chunk) in {"deleted", "quarantined", "permission_lost"}
        ]
        if disallowed:
            failures.append("disallowed_chunk_status_present")
        smoke_results = self._check_smoke_queries(staging_chunks)
        if any(not item["matched"] for item in smoke_results):
            failures.append("smoke_query_no_match")
        metadata = {
            "staging_dir": str(staging_dir),
            "current_dir": str(current_dir),
            "chunk_count": len(staging_chunks),
            "previous_chunk_count": len(current_chunks),
            "min_chunk_ratio": self._min_chunk_ratio,
            "disallowed_chunk_ids": disallowed[:20],
            "smoke_queries": smoke_results,
        }
        return IndexQualityResult(
            passed=not failures,
            critical_failures=tuple(failures),
            metadata=metadata,
        )

    def _check_smoke_queries(self, chunks: list[dict[str, object]]) -> list[dict[str, object]]:
        if not self._smoke_queries:
            return []
        haystack = "\n".join(str(chunk.get("text") or "") for chunk in chunks).lower()
        results: list[dict[str, object]] = []
        for query in self._smoke_queries:
            tokens = [part for part in query.lower().split() if part]
            matched = bool(query.lower() in haystack or any(token in haystack for token in tokens))
            results.append({"query": query, "matched": matched})
        return results


def _read_chunks(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    chunks: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                chunks.append(payload)
    return chunks


def _index_status(chunk: dict[str, object]) -> str:
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        return "active"
    return str(metadata.get("index_status") or "active")
