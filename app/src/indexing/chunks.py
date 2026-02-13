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


def chunk_embedding_text(chunk: Chunk) -> str:
    text = (chunk.text or "").strip()
    metadata_lines: list[str] = []
    source_type = str(chunk.metadata.get("source_type") or "").strip().lower()
    if source_type == "hatenablog":
        title = str(chunk.metadata.get("hatenablog_title") or "").strip()
        if title:
            metadata_lines.append(f"hatenablog_title: {title}")
        created_at = str(chunk.metadata.get("hatenablog_created_at") or "").strip()
        if created_at:
            metadata_lines.append(f"hatenablog_created_at: {created_at}")
    if source_type == "vc_transcript":
        meeting_label = str(chunk.metadata.get("meeting_label") or "").strip()
        if meeting_label:
            metadata_lines.append(meeting_label)
    category_name = str(chunk.metadata.get("category_name") or "").strip()
    channel_name = str(chunk.metadata.get("channel_name") or "").strip()
    if channel_name:
        channel_display = (
            f"{category_name} / {channel_name}" if category_name else channel_name
        )
        metadata_lines.append(f"channel_name: {channel_display}")
    drive_path = str(chunk.metadata.get("drive_file_path") or "").strip()
    if drive_path:
        metadata_lines.append(f"drive_file_path: {drive_path}")

    if not metadata_lines:
        return text
    meta_block = "\n".join(metadata_lines)
    if not text:
        return meta_block
    return f"{text}\n{meta_block}"


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


def load_chunks_from_recursive_dir(chunk_dir: Path) -> list[Chunk]:
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")

    chunks: list[Chunk] = []
    jsonl_files = sorted(chunk_dir.rglob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No .jsonl chunk files found under %s", chunk_dir)
        return chunks

    for path in jsonl_files:
        chunks.extend(load_chunks(path))

    logger.info(
        "Loaded %d chunks from %d chunk files under %s",
        len(chunks),
        len(jsonl_files),
        chunk_dir,
    )
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
