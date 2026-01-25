from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: dict[str, object]


def _load_chunk_record(*, obj: dict[str, object], path: Path, line_no: int) -> Chunk:
    text = obj.get("text")
    if not isinstance(text, str):
        raise ValueError(f"Missing/invalid 'text' in {path.name} line {line_no}")

    metadata = obj.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(
            "Missing/invalid 'metadata' (must be dict) in "
            f"{path.name} line {line_no}"
        )

    return Chunk(text=text, metadata=metadata)


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file does not exist: {path}")

    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fr:
        for line_no, line in enumerate(fr, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name} at line {line_no}: {exc}"
                ) from exc
            chunks.append(_load_chunk_record(obj=obj, path=path, line_no=line_no))

    return chunks


def load_chunks_from_dir(chunk_dir: Path) -> list[Chunk]:
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")

    chunks: list[Chunk] = []
    jsonl_files = sorted(chunk_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No .jsonl chunk files found under %s", chunk_dir)
        return chunks

    for path in jsonl_files:
        chunks.extend(load_chunks(path))

    logger.info("Loaded %d chunks from %d chunk files", len(chunks), len(jsonl_files))
    return chunks


def load_chunks_from_dirs(chunk_dirs: Iterable[Path]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for chunk_dir in chunk_dirs:
        chunks.extend(load_chunks_from_dir(chunk_dir))
    return chunks


def write_chunks(path: Path, chunks: Iterable[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as fw:
        for chunk in chunks:
            record = {"text": chunk.text, "metadata": chunk.metadata}
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
