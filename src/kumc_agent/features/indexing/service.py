from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.document import Document
from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever
from kumc_agent.infra.storage.filesystem import FileSystemStorage
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class IndexBuildResult:
    loaded_sources: int
    documents: int
    chunks: int
    index_dir: Path


class IndexingService:
    def __init__(
        self,
        *,
        storage: FileSystemStorage,
        embedder: EmbedderPort,
        faiss_index: FaissLikeIndex,
        bm25_index: SudachiBM25Retriever,
        raw_dir: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._faiss_index = faiss_index
        self._bm25_index = bm25_index
        self._raw_dir = raw_dir
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def build(
        self,
        *,
        loaded_sources: int,
    ) -> IndexBuildResult:
        documents = self._parse_documents_from_raw()
        self._storage.save_documents(documents)
        chunks = self._build_chunks(documents)
        self._storage.save_chunks(chunks)

        embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])
        self._faiss_index.build(chunks=chunks, embeddings=embeddings)
        self._bm25_index.build(chunks)

        return IndexBuildResult(
            loaded_sources=loaded_sources,
            documents=len(documents),
            chunks=len(chunks),
            index_dir=self._faiss_index._index_dir,  # noqa: SLF001
        )

    def update(self, *, loaded_sources: int) -> IndexBuildResult:
        return self.build(loaded_sources=loaded_sources)

    def _parse_documents_from_raw(self) -> list[Document]:
        if not self._raw_dir.exists():
            return []

        documents: list[Document] = []
        for path in sorted(self._raw_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".meta.json", ".mtime.json"}:
                continue
            text = self._read_raw_text(path)
            if not text.strip():
                continue
            source_type = path.parent.name
            source_name = str(path.relative_to(self._raw_dir))
            doc_id = stable_hash(f"{source_type}:{source_name}")
            documents.append(
                Document(
                    id=doc_id,
                    text=text,
                    source_type=source_type,
                    source_name=source_name,
                    source_uri="",
                    metadata={"path": source_name},
                )
            )
        return documents

    def _read_raw_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            texts: list[str] = []
            with path.open("r", encoding="utf-8") as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(payload.get("text") or "").strip()
                    if text:
                        texts.append(text)
            return "\n".join(texts)
        return path.read_text(encoding="utf-8", errors="ignore")

    def _build_chunks(self, documents: list[Document]) -> list[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        chunks: list[Chunk] = []
        for doc in documents:
            pieces = splitter.split_text(doc.text)
            for idx, piece in enumerate(pieces):
                normalized = piece.strip()
                if not normalized:
                    continue
                chunk_id = stable_hash(f"{doc.id}:{idx}:{normalized[:64]}")
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=doc.id,
                        text=normalized,
                        index=idx,
                        metadata={
                            "source_type": doc.source_type,
                            "source_name": doc.source_name,
                        },
                    )
                )
        return chunks
