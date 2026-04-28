from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever


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
        dense_load = _load_dense_vectors(staging_dir / "dense_vectors.npy")
        if dense_load["status"] != "succeeded":
            failures.append("dense_index_load_failed")
        elif staging_chunks and int(dense_load["rows"]) != len(staging_chunks):
            failures.append("dense_chunk_vector_mismatch")
        sparse_load = _load_sparse_index(staging_dir)
        if sparse_load["status"] != "succeeded":
            failures.append("sparse_index_load_failed")
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
        smoke_results = self._check_smoke_queries(staging_dir=staging_dir, chunks=staging_chunks)
        if any(not item["matched"] for item in smoke_results):
            failures.append("smoke_query_no_match")
        feature_load = _load_feature_indexes(staging_dir)
        failures.extend(feature_load["failures"])
        metadata = {
            "staging_dir": str(staging_dir),
            "current_dir": str(current_dir),
            "chunk_count": len(staging_chunks),
            "previous_chunk_count": len(current_chunks),
            "min_chunk_ratio": self._min_chunk_ratio,
            "disallowed_chunk_ids": disallowed[:20],
            "smoke_queries": smoke_results,
            "dense_load": dense_load,
            "sparse_load": sparse_load,
            "feature_load": feature_load["metadata"],
        }
        return IndexQualityResult(
            passed=not failures,
            critical_failures=tuple(failures),
            metadata=metadata,
        )

    def _check_smoke_queries(
        self,
        *,
        staging_dir: Path,
        chunks: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not self._smoke_queries:
            return []
        haystack = "\n".join(str(chunk.get("text") or "") for chunk in chunks).lower()
        sparse = SudachiBM25Retriever(index_dir=staging_dir)
        results: list[dict[str, object]] = []
        for query in self._smoke_queries:
            tokens = [part for part in query.lower().split() if part]
            sparse_hits = sparse.search(query, top_k=1)
            matched = bool(
                sparse_hits
                or query.lower() in haystack
                or any(token in haystack for token in tokens)
            )
            results.append(
                {
                    "query": query,
                    "matched": matched,
                    "sparse_hits": len(sparse_hits),
                }
            )
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


def _load_dense_vectors(path: Path) -> dict[str, object]:
    try:
        matrix = np.load(path)
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}
    if matrix.ndim != 2:
        return {"status": "failed", "error": "dense_vectors_not_2d"}
    return {"status": "succeeded", "rows": int(matrix.shape[0]), "dimensions": int(matrix.shape[1])}


def _load_sparse_index(staging_dir: Path) -> dict[str, object]:
    try:
        retriever = SudachiBM25Retriever(index_dir=staging_dir)
        chunks = retriever.search("", top_k=1)
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}
    return {"status": "succeeded", "probe_hits": len(chunks)}


def _load_feature_indexes(staging_dir: Path) -> dict[str, object]:
    checks = {
        "image": (
            staging_dir / "image_search" / "image_assets.jsonl",
            staging_dir / "image_search" / "image_text_vectors.npy",
            staging_dir / "image_search" / "image_feature_vectors.npy",
        ),
        "member_profiles": (
            staging_dir / "member_profiles" / "dense_chunks.jsonl",
            staging_dir / "member_profiles" / "dense_vectors.npy",
        ),
        "task_event": (
            staging_dir / "task_event" / "task_event_documents.jsonl",
            staging_dir / "task_event" / "dense_vectors.npy",
        ),
    }
    failures: list[str] = []
    metadata: dict[str, object] = {}
    for name, paths in checks.items():
        existing = [path for path in paths if path.exists()]
        if not existing:
            metadata[name] = {"status": "not_present"}
            continue
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            failures.append(f"{name}_index_incomplete")
            metadata[name] = {"status": "failed", "missing": missing}
            continue
        load_result = {"status": "succeeded"}
        for path in paths:
            if path.suffix == ".npy":
                result = _load_dense_vectors(path)
                if result["status"] != "succeeded":
                    load_result = {"status": "failed", "error": result.get("error", "")}
                    failures.append(f"{name}_index_load_failed")
                    break
        metadata[name] = load_result
    return {"failures": failures, "metadata": metadata}
