from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.document import Document
from kumc_agent.domain.policies.chunk_visibility import is_chunk_allowed_for_answer_context
from kumc_agent.domain.ports.prompts import PromptRepositoryPort
from kumc_agent.domain.ports.storage import StoragePort

FileSignature = tuple[int, int]


class FileSystemStorage(StoragePort):
    def __init__(self, *, chunks_path: Path, index_documents_path: Path) -> None:
        self._chunks_path = chunks_path
        self._index_documents_path = index_documents_path
        self._chunks_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_documents_path.parent.mkdir(parents=True, exist_ok=True)

    def save_documents(self, documents: list[Document]) -> None:
        self._index_documents_path.parent.mkdir(parents=True, exist_ok=True)
        with self._index_documents_path.open("w", encoding="utf-8") as fw:
            for doc in documents:
                fw.write(
                    json.dumps(
                        {
                            "id": doc.id,
                            "text": doc.text,
                            "source_type": doc.source_type,
                            "source_name": doc.source_name,
                            "source_uri": doc.source_uri,
                            "metadata": doc.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def save_chunks(self, chunks: list[Chunk]) -> None:
        self._chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with self._chunks_path.open("w", encoding="utf-8") as fw:
            for chunk in chunks:
                fw.write(
                    json.dumps(
                        {
                            "id": chunk.id,
                            "document_id": chunk.document_id,
                            "text": chunk.text,
                            "index": chunk.index,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def load_chunks(self) -> list[Chunk]:
        if not self._chunks_path.exists():
            return []
        out: list[Chunk] = []
        with self._chunks_path.open("r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                chunk = Chunk(
                    id=str(payload["id"]),
                    document_id=str(payload["document_id"]),
                    text=str(payload["text"]),
                    index=int(payload["index"]),
                    metadata=dict(payload.get("metadata", {})),
                )
                if is_chunk_allowed_for_answer_context(chunk):
                    out.append(chunk)
        return out


class FilePromptRepository(PromptRepositoryPort):
    def __init__(self, prompts_dir: Path) -> None:
        self._prompts_dir = prompts_dir
        self._cache: dict[Path, tuple[FileSignature, str]] = {}

    def get(self, name: str) -> str:
        file_name = name if name.endswith(".md") else f"{name}.md"
        path = self._prompts_dir / file_name
        if not path.exists():
            self._cache.pop(path, None)
            raise FileNotFoundError(f"Prompt not found: {path}")
        signature = self._file_signature(path)
        cached = self._cache.get(path)
        if (
            cached is not None
            and signature is not None
            and cached[0] == signature
        ):
            return cached[1]
        value = path.read_text(encoding="utf-8").strip()
        if signature is not None:
            self._cache[path] = (signature, value)
        return value

    @staticmethod
    def _file_signature(path: Path) -> FileSignature | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return (int(stat.st_mtime_ns), int(stat.st_size))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
