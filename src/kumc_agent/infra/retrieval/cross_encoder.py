from __future__ import annotations

from functools import lru_cache
import logging
from threading import Lock

from kumc_agent.domain.models.chunk import Chunk

logger = logging.getLogger(__name__)
_CROSS_ENCODER_CLIENTS: dict[str, object] = {}
_CROSS_ENCODER_CLIENTS_LOCK = Lock()


class CrossEncoderReranker:
    def __init__(self, *, model_name: str) -> None:
        self._model_name = model_name

    def prepare_runtime(self) -> None:
        _preload_torch_runtime()

    def rerank(self, query: str, chunks: list[Chunk], *, top_k: int) -> list[Chunk]:
        limited_top_k = max(0, top_k)
        if limited_top_k <= 0 or not chunks:
            return []
        scored = self.score_documents(query=query, chunks=chunks)
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk for _, _, chunk in scored[:limited_top_k]]

    def score_documents(
        self,
        *,
        query: str,
        chunks: list[Chunk],
    ) -> list[tuple[float, int, Chunk]]:
        if not chunks:
            return []
        self.prepare_runtime()
        if len(chunks) == 1:
            return [(1.0, 0, chunks[0])]
        try:
            cross_encoder = _cross_encoder_client(self._model_name)
            pairs = [(query, chunk.text) for chunk in chunks]
            scores = cross_encoder.predict(
                pairs,
                show_progress_bar=False,
            )
            scored: list[tuple[float, int, Chunk]] = []
            for idx, (chunk, score) in enumerate(zip(chunks, scores)):
                try:
                    raw_value = float(score)
                except (TypeError, ValueError):
                    raw_value = 0.0
                clamped = max(0.0, min(1.0, raw_value))
                scored.append((clamped, idx, chunk))
            return scored
        except Exception:
            logger.exception(
                "Cross-encoder scoring failed. Falling back to lexical overlap rerank."
            )
            scored_chunks = self._rerank_lexical(
                query=query,
                chunks=chunks,
                top_k=len(chunks),
            )
            rank_map = {chunk.id: float(len(chunks) - idx) for idx, chunk in enumerate(scored_chunks)}
            return [
                (rank_map.get(chunk.id, 0.0), idx, chunk)
                for idx, chunk in enumerate(chunks)
            ]

    @staticmethod
    def _rerank_lexical(query: str, chunks: list[Chunk], *, top_k: int) -> list[Chunk]:
        query_tokens = set((query or "").lower().split())
        scored = []
        for idx, chunk in enumerate(chunks):
            chunk_tokens = set((chunk.text or "").lower().split())
            score = float(len(query_tokens & chunk_tokens))
            scored.append((score, idx, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk for _, _, chunk in scored[:top_k]]


@lru_cache(maxsize=1)
def _preload_torch_runtime() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        logger.debug("torch preload for reranker runtime skipped.", exc_info=True)


def _cross_encoder_client(model_name: str):
    if not model_name:
        raise RuntimeError("Cross-encoder model name is required.")
    cached = _CROSS_ENCODER_CLIENTS.get(model_name)
    if cached is not None:
        return cached
    with _CROSS_ENCODER_CLIENTS_LOCK:
        cached = _CROSS_ENCODER_CLIENTS.get(model_name)
        if cached is not None:
            return cached
        client = _build_cross_encoder_client(model_name)
        _CROSS_ENCODER_CLIENTS[model_name] = client
        return client


def _build_cross_encoder_client(model_name: str):
    try:
        return _create_cross_encoder(model_name)
    except NotImplementedError as exc:
        if not _is_meta_tensor_error(exc):
            raise
        logger.warning(
            "Cross-encoder automatic device init failed with meta tensor. Retrying on CPU."
        )
        return _create_cross_encoder(model_name, device="cpu")


def _is_meta_tensor_error(exc: BaseException) -> bool:
    stack: list[BaseException | None] = [exc]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None:
            continue
        ident = id(current)
        if ident in visited:
            continue
        visited.add(ident)
        text = str(current).lower()
        if "meta tensor" in text and "to_empty" in text:
            return True
        stack.append(getattr(current, "__cause__", None))
        stack.append(getattr(current, "__context__", None))
    return False


def _create_cross_encoder(model_name: str, *, device: str | None = None):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for cross-encoder reranking."
        ) from exc
    kwargs: dict[str, object] = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if device:
        kwargs["device"] = device
    return CrossEncoder(
        model_name,
        **kwargs,
    )
