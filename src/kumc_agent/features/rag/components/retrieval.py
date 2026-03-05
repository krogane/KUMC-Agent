from __future__ import annotations

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever
from kumc_agent.utils.hashing import merge_unique


class RetrievalComponent:
    def __init__(
        self,
        *,
        embedder: EmbedderPort,
        dense_index: FaissLikeIndex,
        sparse_index: SudachiBM25Retriever,
    ) -> None:
        self._embedder = embedder
        self._dense_index = dense_index
        self._sparse_index = sparse_index

    def retrieve(
        self,
        query: str,
        *,
        dense_top_k: int,
        sparse_top_k: int,
    ) -> list[Chunk]:
        query_vector = self._embedder.embed_query(query)
        dense = [item.chunk for item in self._dense_index.search(query_vector=query_vector, top_k=dense_top_k)]
        sparse = self._sparse_index.search(query, top_k=sparse_top_k)
        merged = [str(value) for value in merge_unique([chunk.id for chunk in [*dense, *sparse]])]
        by_id = {chunk.id: chunk for chunk in [*dense, *sparse]}
        return [by_id[chunk_id] for chunk_id in merged if chunk_id in by_id]
