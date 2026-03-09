from __future__ import annotations

import json
import logging
from pathlib import Path
import unicodedata

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - fallback for minimal runtime
    class BM25Okapi:  # type: ignore[no-redef]
        def __init__(
            self,
            corpus: list[list[str]],
            *,
            k1: float = 1.5,
            b: float = 0.75,
        ) -> None:
            self._corpus = corpus
            self._k1 = k1
            self._b = b

        def get_scores(self, query_tokens: list[str]):
            query = set(query_tokens)
            scores: list[float] = []
            for tokens in self._corpus:
                if not tokens:
                    scores.append(0.0)
                    continue
                token_set = set(tokens)
                overlap = float(len(query & token_set))
                scores.append(overlap / float(len(token_set)))
            return scores

logger = logging.getLogger(__name__)

try:
    from sudachipy import dictionary as sudachi_dictionary
    from sudachipy import tokenizer as sudachi_tokenizer
except Exception:  # pragma: no cover - optional dependency at runtime
    sudachi_dictionary = None
    sudachi_tokenizer = None

from kumc_agent.domain.models.chunk import Chunk


class SudachiBM25Retriever:
    def __init__(
        self,
        *,
        index_dir: Path,
        sudachi_mode: str = "B",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        use_normalized_form: bool = True,
        remove_symbols: bool = True,
    ) -> None:
        self._index_dir = index_dir
        self._tokens_path = self._index_dir / "bm25_tokens.json"
        self._chunks_path = self._index_dir / "bm25_chunks.jsonl"
        self._sudachi_mode = (sudachi_mode or "B").upper()
        self._bm25_k1 = float(bm25_k1)
        self._bm25_b = float(bm25_b)
        self._use_normalized_form = bool(use_normalized_form)
        self._remove_symbols = bool(remove_symbols)
        self._sudachi = self._build_sudachi_tokenizer()
        self._split_mode = self._resolve_split_mode(self._sudachi_mode)
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
        return [chunk for chunk, _ in self.search_with_scores(query, top_k=top_k)]

    def search_with_scores(self, query: str, *, top_k: int) -> list[tuple[Chunk, float]]:
        if not self._tokens_path.exists() or not self._chunks_path.exists():
            return []
        tokenized = json.loads(self._tokens_path.read_text(encoding="utf-8"))
        if not tokenized:
            return []
        bm25 = BM25Okapi(tokenized, k1=self._bm25_k1, b=self._bm25_b)
        scores = bm25.get_scores(self._tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        chunks = self._load_chunks()
        out: list[tuple[Chunk, float]] = []
        for idx in order[: max(0, top_k)]:
            if idx < len(chunks):
                out.append((chunks[idx], float(scores[idx])))
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

    def _tokenize(self, text: str) -> list[str]:
        value = (text or "").strip()
        if not value:
            return []
        if self._sudachi is None or self._split_mode is None:
            return self._whitespace_tokenize(value)
        out: list[str] = []
        try:
            for morpheme in self._sudachi.tokenize(value, self._split_mode):
                pos = morpheme.part_of_speech() or []
                if pos and pos[0] == "空白":
                    continue
                normalized = str(morpheme.normalized_form() or "").strip()
                surface = str(morpheme.surface() or "").strip()
                token = normalized if self._use_normalized_form else surface
                if not token:
                    token = surface or normalized
                token = token.lower()
                if self._remove_symbols and self._is_symbol_only(token):
                    continue
                if token:
                    out.append(token)
        except Exception:
            logger.exception(
                "Sudachi tokenization failed. Falling back to whitespace tokenization."
            )
            return self._whitespace_tokenize(value)
        return out

    def _whitespace_tokenize(self, value: str) -> list[str]:
        out: list[str] = []
        for token in value.lower().split():
            item = token.strip()
            if not item:
                continue
            if self._remove_symbols and self._is_symbol_only(item):
                continue
            out.append(item)
        return out

    @staticmethod
    def _resolve_split_mode(mode: str):
        if sudachi_tokenizer is None:
            return None
        if mode == "A":
            return sudachi_tokenizer.Tokenizer.SplitMode.A
        if mode == "C":
            return sudachi_tokenizer.Tokenizer.SplitMode.C
        return sudachi_tokenizer.Tokenizer.SplitMode.B

    @staticmethod
    def _is_symbol_only(text: str) -> bool:
        has_visible = False
        for ch in text:
            if ch.isspace():
                continue
            has_visible = True
            category = unicodedata.category(ch)
            if category.startswith(("L", "N", "M")):
                return False
        return has_visible

    @staticmethod
    def _build_sudachi_tokenizer():
        if sudachi_dictionary is None:
            return None
        try:
            return sudachi_dictionary.Dictionary().create()
        except Exception:
            logger.exception(
                "Sudachi tokenizer initialization failed. Falling back to whitespace tokenization."
            )
            return None
