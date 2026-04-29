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
FileSignature = tuple[str, int, int]

try:
    from sudachipy import dictionary as sudachi_dictionary
    from sudachipy import tokenizer as sudachi_tokenizer
except Exception:  # pragma: no cover - optional dependency at runtime
    sudachi_dictionary = None
    sudachi_tokenizer = None

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.policies.chunk_visibility import is_chunk_allowed_for_answer_context
from kumc_agent.features.indexing.paths import resolve_current_index_dir


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
        self._sudachi_mode = (sudachi_mode or "B").upper()
        self._bm25_k1 = float(bm25_k1)
        self._bm25_b = float(bm25_b)
        self._use_normalized_form = bool(use_normalized_form)
        self._remove_symbols = bool(remove_symbols)
        self._sudachi = self._build_sudachi_tokenizer()
        self._split_mode = self._resolve_split_mode(self._sudachi_mode)
        self._cached_tokens: list[list[str]] | None = None
        self._cached_tokens_signature: FileSignature | None = None
        self._cached_chunks: list[Chunk] | None = None
        self._cached_chunks_signature: FileSignature | None = None
        self._cached_bm25: BM25Okapi | None = None
        self._cached_bm25_signature: FileSignature | None = None
        self._index_dir.mkdir(parents=True, exist_ok=True)

    def build(self, chunks: list[Chunk]) -> None:
        tokenized = [self._tokenize(chunk.text) for chunk in chunks]
        tokens_path = self._write_path("bm25_tokens.json")
        chunks_path = self._write_path("bm25_chunks.jsonl")
        tokens_path.write_text(json.dumps(tokenized, ensure_ascii=False), encoding="utf-8")
        with chunks_path.open("w", encoding="utf-8") as fw:
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
        self._cached_tokens = tokenized
        self._cached_tokens_signature = self._file_signature(tokens_path)
        self._cached_chunks = list(chunks)
        self._cached_chunks_signature = self._file_signature(chunks_path)
        self._cached_bm25 = self._build_bm25(tokenized)
        self._cached_bm25_signature = self._cached_tokens_signature

    def search(self, query: str, *, top_k: int) -> list[Chunk]:
        return [chunk for chunk, _ in self.search_with_scores(query, top_k=top_k)]

    def search_with_scores(
        self,
        query: str,
        *,
        top_k: int,
        query_tokens: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        bm25 = self._load_bm25()
        if bm25 is None:
            return []
        tokens = query_tokens if query_tokens is not None else self._tokenize(query)
        scores = bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        chunks = self._load_chunks()
        out: list[tuple[Chunk, float]] = []
        for idx in order[: max(0, top_k)]:
            if idx < len(chunks):
                chunk = chunks[idx]
                if not is_chunk_allowed_for_answer_context(chunk):
                    continue
                out.append((chunk, float(scores[idx])))
        return out

    def _load_bm25(self) -> BM25Okapi | None:
        tokenized = self._load_tokenized()
        if not tokenized:
            self._cached_bm25 = None
            self._cached_bm25_signature = self._cached_tokens_signature
            return None
        if (
            self._cached_bm25 is not None
            and self._cached_bm25_signature is not None
            and self._cached_bm25_signature == self._cached_tokens_signature
        ):
            return self._cached_bm25
        bm25 = self._build_bm25(tokenized)
        self._cached_bm25 = bm25
        self._cached_bm25_signature = self._cached_tokens_signature
        return bm25

    def _load_tokenized(self) -> list[list[str]]:
        tokens_path = self._read_path("bm25_tokens.json")
        if not tokens_path.exists():
            self._cached_tokens = None
            self._cached_tokens_signature = None
            return []
        current_signature = self._file_signature(tokens_path)
        if (
            self._cached_tokens is not None
            and self._cached_tokens_signature is not None
            and self._cached_tokens_signature == current_signature
        ):
            return self._cached_tokens
        tokenized = json.loads(tokens_path.read_text(encoding="utf-8"))
        if not isinstance(tokenized, list):
            tokenized = []
        normalized: list[list[str]] = []
        for item in tokenized:
            if not isinstance(item, list):
                continue
            normalized.append(
                [str(token).strip() for token in item if str(token).strip()]
            )
        self._cached_tokens = normalized
        self._cached_tokens_signature = current_signature
        return normalized

    def _load_chunks(self) -> list[Chunk]:
        chunks_path = self._read_path("bm25_chunks.jsonl")
        if not chunks_path.exists():
            self._cached_chunks = None
            self._cached_chunks_signature = None
            return []
        current_signature = self._file_signature(chunks_path)
        if (
            self._cached_chunks is not None
            and self._cached_chunks_signature is not None
            and self._cached_chunks_signature == current_signature
        ):
            return self._cached_chunks
        out: list[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as fr:
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
        self._cached_chunks = out
        self._cached_chunks_signature = current_signature
        return out

    def _build_bm25(self, tokenized: list[list[str]]) -> BM25Okapi:
        return BM25Okapi(tokenized, k1=self._bm25_k1, b=self._bm25_b)

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

    def _read_path(self, name: str) -> Path:
        return resolve_current_index_dir(self._index_dir) / name

    def _write_path(self, name: str) -> Path:
        return self._index_dir / name

    @staticmethod
    def _file_signature(path: Path) -> FileSignature | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
