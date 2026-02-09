from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from langchain_core.documents import Document

from indexing.utils import ensure_dir

logger = logging.getLogger(__name__)

KEYWORD_INDEX_SUBDIR = "keyword"
KEYWORD_INDEX_SCHEMA_VERSION = 1
KEYWORD_CORPUS_SPARSE = "sparse"
KEYWORD_CORPUS_SPARSE_SECOND_REC = "sparse_second_rec"
KEYWORD_CORPUS_SECOND_REC_SPARSE = "second_rec_sparse"

_BM25_EPSILON = 0.25


@dataclass(frozen=True)
class _Posting:
    doc_ids: np.ndarray
    term_freqs: np.ndarray
    idf: float


class KeywordInvertedIndex:
    def __init__(
        self,
        *,
        docs: list[Document],
        doc_lengths: Sequence[int],
        postings: dict[str, _Posting],
        avg_doc_len: float,
        k1: float,
        b: float,
    ) -> None:
        self.docs = docs
        self._doc_lengths = np.asarray(doc_lengths, dtype=np.float32)
        self._postings = postings
        self._avg_doc_len = float(avg_doc_len)
        self._k1 = float(k1)
        self._b = float(b)

    @property
    def k1(self) -> float:
        return self._k1

    @property
    def b(self) -> float:
        return self._b

    @classmethod
    def build(
        cls,
        *,
        docs: list[Document],
        tokenize_doc: Callable[[Document], list[str]],
        k1: float,
        b: float,
    ) -> "KeywordInvertedIndex":
        if not docs:
            return cls(
                docs=[],
                doc_lengths=[],
                postings={},
                avg_doc_len=0.0,
                k1=k1,
                b=b,
            )

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_lengths: list[int] = []
        for doc_id, doc in enumerate(docs):
            tokens = tokenize_doc(doc)
            doc_lengths.append(len(tokens))
            if not tokens:
                continue
            frequencies = Counter(tokens)
            for token, freq in frequencies.items():
                postings[token].append((doc_id, freq))

        avg_doc_len = float(sum(doc_lengths)) / float(len(doc_lengths))
        posting_map = _build_posting_map(
            postings=postings,
            doc_lengths=doc_lengths,
            avg_doc_len=avg_doc_len,
        )
        return cls(
            docs=docs,
            doc_lengths=doc_lengths,
            postings=posting_map,
            avg_doc_len=avg_doc_len,
            k1=k1,
            b=b,
        )

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        if not self.docs:
            return np.zeros(0, dtype=np.float32)
        if not query_tokens:
            return np.zeros(len(self.docs), dtype=np.float32)

        scores = np.zeros(len(self.docs), dtype=np.float32)
        for token in query_tokens:
            posting = self._postings.get(token)
            if posting is None or posting.idf == 0.0:
                continue

            lengths = self._doc_lengths[posting.doc_ids]
            tf = posting.term_freqs
            denom = tf + self._k1 * (
                1.0 - self._b + self._b * lengths / self._avg_doc_len
            )
            contribution = posting.idf * (tf * (self._k1 + 1.0) / denom)
            scores[posting.doc_ids] += contribution
        return scores

    def to_payload(self) -> dict[str, object]:
        postings_payload: dict[str, dict[str, object]] = {}
        for token, posting in self._postings.items():
            entries: list[list[int]] = []
            for doc_id, tf in zip(posting.doc_ids.tolist(), posting.term_freqs.tolist()):
                entries.append([int(doc_id), int(tf)])
            postings_payload[token] = {"idf": posting.idf, "entries": entries}

        return {
            "schema_version": KEYWORD_INDEX_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "k1": self._k1,
            "b": self._b,
            "avg_doc_len": self._avg_doc_len,
            "doc_lengths": self._doc_lengths.astype(int).tolist(),
            "docs": [
                {"page_content": doc.page_content, "metadata": doc.metadata or {}}
                for doc in self.docs
            ],
            "postings": postings_payload,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "KeywordInvertedIndex":
        schema_version = payload.get("schema_version")
        if schema_version != KEYWORD_INDEX_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported keyword index schema_version: "
                f"{schema_version} (expected {KEYWORD_INDEX_SCHEMA_VERSION})"
            )

        docs_raw = payload.get("docs")
        if not isinstance(docs_raw, list):
            raise ValueError("Invalid keyword index payload: docs is not a list.")

        docs: list[Document] = []
        for row in docs_raw:
            if not isinstance(row, dict):
                raise ValueError("Invalid keyword index payload: docs entry is not a dict.")
            page_content = row.get("page_content")
            metadata = row.get("metadata")
            if not isinstance(page_content, str):
                raise ValueError(
                    "Invalid keyword index payload: docs.page_content is not str."
                )
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError(
                    "Invalid keyword index payload: docs.metadata is not dict."
                )
            docs.append(Document(page_content=page_content, metadata=metadata))

        doc_lengths_raw = payload.get("doc_lengths")
        if not isinstance(doc_lengths_raw, list):
            raise ValueError(
                "Invalid keyword index payload: doc_lengths is not a list."
            )
        doc_lengths = [int(value) for value in doc_lengths_raw]
        if len(doc_lengths) != len(docs):
            raise ValueError(
                "Invalid keyword index payload: doc_lengths size does not match docs."
            )

        postings_raw = payload.get("postings")
        if not isinstance(postings_raw, dict):
            raise ValueError("Invalid keyword index payload: postings is not a dict.")
        postings: dict[str, _Posting] = {}
        for token, row in postings_raw.items():
            if not isinstance(token, str) or not isinstance(row, dict):
                raise ValueError("Invalid keyword index payload: malformed postings row.")
            entries_raw = row.get("entries")
            if not isinstance(entries_raw, list):
                raise ValueError(
                    "Invalid keyword index payload: postings.entries is not a list."
                )
            idf = float(row.get("idf") or 0.0)
            doc_ids: list[int] = []
            term_freqs: list[float] = []
            for entry in entries_raw:
                if (
                    not isinstance(entry, list)
                    or len(entry) != 2
                ):
                    raise ValueError(
                        "Invalid keyword index payload: posting entry must be [doc_id, tf]."
                    )
                doc_id = int(entry[0])
                tf = int(entry[1])
                if doc_id < 0 or doc_id >= len(docs):
                    raise ValueError(
                        f"Invalid keyword index payload: doc_id out of range ({doc_id})."
                    )
                doc_ids.append(doc_id)
                term_freqs.append(float(tf))
            postings[token] = _Posting(
                doc_ids=np.asarray(doc_ids, dtype=np.int32),
                term_freqs=np.asarray(term_freqs, dtype=np.float32),
                idf=idf,
            )

        k1_raw = payload.get("k1")
        b_raw = payload.get("b")
        avg_doc_len_raw = payload.get("avg_doc_len")
        k1 = float(k1_raw) if k1_raw is not None else 1.5
        b = float(b_raw) if b_raw is not None else 0.75
        avg_doc_len = float(avg_doc_len_raw) if avg_doc_len_raw is not None else 0.0
        return cls(
            docs=docs,
            doc_lengths=doc_lengths,
            postings=postings,
            avg_doc_len=avg_doc_len,
            k1=k1,
            b=b,
        )


def keyword_index_path(*, index_dir: Path, corpus_name: str) -> Path:
    return index_dir / KEYWORD_INDEX_SUBDIR / f"{corpus_name}.json"


def save_keyword_index(
    *,
    index_dir: Path,
    corpus_name: str,
    index: KeywordInvertedIndex,
) -> Path:
    path = keyword_index_path(index_dir=index_dir, corpus_name=corpus_name)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fw:
        json.dump(index.to_payload(), fw, ensure_ascii=False, separators=(",", ":"))
    return path


def load_keyword_index(
    *,
    index_dir: Path,
    corpus_name: str,
) -> KeywordInvertedIndex | None:
    path = keyword_index_path(index_dir=index_dir, corpus_name=corpus_name)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fr:
            payload = json.load(fr)
        if not isinstance(payload, dict):
            raise ValueError("Top-level payload must be a JSON object.")
        return KeywordInvertedIndex.from_payload(payload)
    except Exception:
        logger.warning("Failed to load keyword index from %s", path, exc_info=True)
        return None


def build_and_save_keyword_index(
    *,
    index_dir: Path,
    corpus_name: str,
    docs: list[Document],
    tokenize_doc: Callable[[Document], list[str]],
    k1: float,
    b: float,
) -> Path:
    index = KeywordInvertedIndex.build(
        docs=docs,
        tokenize_doc=tokenize_doc,
        k1=k1,
        b=b,
    )
    path = save_keyword_index(index_dir=index_dir, corpus_name=corpus_name, index=index)
    logger.info(
        "Saved keyword inverted index: corpus=%s docs=%d path=%s",
        corpus_name,
        len(index.docs),
        path,
    )
    return path


def tokenize_sparse_doc(
    doc: Document,
    *,
    sparse_stage: str,
    sudachi_tokenize: Callable[[str], list[str]],
) -> list[str]:
    metadata = doc.metadata or {}
    stage = str(metadata.get("chunk_stage") or "")
    if stage == sparse_stage:
        text = (doc.page_content or "").strip()
        if not text:
            return []
        return [token for token in text.split() if token]
    return sudachi_tokenize(doc.page_content)


def _build_posting_map(
    *,
    postings: dict[str, list[tuple[int, int]]],
    doc_lengths: Sequence[int],
    avg_doc_len: float,
) -> dict[str, _Posting]:
    if not postings:
        return {}

    corpus_size = len(doc_lengths)
    idf: dict[str, float] = {}
    idf_sum = 0.0
    negative_tokens: list[str] = []
    for token, entries in postings.items():
        doc_freq = len(entries)
        value = math.log(corpus_size - doc_freq + 0.5) - math.log(doc_freq + 0.5)
        idf[token] = value
        idf_sum += value
        if value < 0.0:
            negative_tokens.append(token)

    average_idf = idf_sum / float(len(idf)) if idf else 0.0
    eps = _BM25_EPSILON * average_idf
    for token in negative_tokens:
        idf[token] = eps

    posting_map: dict[str, _Posting] = {}
    for token, entries in postings.items():
        doc_ids: list[int] = []
        term_freqs: list[float] = []
        for doc_id, tf in entries:
            if tf <= 0:
                continue
            doc_ids.append(doc_id)
            term_freqs.append(float(tf))
        if not doc_ids:
            continue
        posting_map[token] = _Posting(
            doc_ids=np.asarray(doc_ids, dtype=np.int32),
            term_freqs=np.asarray(term_freqs, dtype=np.float32),
            idf=idf[token],
        )
    if avg_doc_len <= 0:
        return {}
    return posting_map
