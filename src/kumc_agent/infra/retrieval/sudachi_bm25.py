from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from kumc_agent.domain.models.chunk import Chunk


class SudachiBM25Retriever:
    def __init__(self, *, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._tokens_path = self._index_dir / "bm25_tokens.json"
        self._chunks_path = self._index_dir / "bm25_chunks.jsonl"
        self._index_dir.mkdir(parents=True, exist_ok=True)

    def build(self, chunks: list[Chunk]) -> None:
        tokenized = [self._tokenize(chunk.text) for chunk in chunks]
        self._tokens_path.write_text(json.dumps(tokenized, ensure_ascii=False), encoding="utf-8")
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

    def search(self, query: str, *, top_k: int) -> list[Chunk]:
        if not self._tokens_path.exists() or not self._chunks_path.exists():
            return []
        tokenized = json.loads(self._tokens_path.read_text(encoding="utf-8"))
        if not tokenized:
            return []
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(self._tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        chunks = self._load_chunks()
        out: list[Chunk] = []
        for idx in order[: max(0, top_k)]:
            if idx < len(chunks):
                out.append(chunks[idx])
        return out

    def _load_chunks(self) -> list[Chunk]:
        out: list[Chunk] = []
        with self._chunks_path.open("r", encoding="utf-8") as fr:
            for line in fr:
                payload = json.loads(line)
                out.append(
                    Chunk(
                        id=str(payload["id"]),
                        document_id=str(payload["document_id"]),
                        text=str(payload["text"]),
                        index=int(payload["index"]),
                        metadata=dict(payload.get("metadata", {})),
                    )
                )
        return out

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in (text or "").lower().split() if token]
