from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS

from config import EmbeddingFactory
from indexing.index_metadata import (
    get_embedding_dimension,
    read_index_metadata,
)

logger = logging.getLogger(__name__)


def load_faiss_index(*, index_dir: Path, embedding_factory: EmbeddingFactory) -> FAISS:
    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index directory not found: {index_dir}. "
            "Build the index first (e.g., run your build script)."
        )

    embeddings = embedding_factory.get_embeddings()
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    index_dim = getattr(vectorstore.index, "d", None)
    if not isinstance(index_dim, int):
        index_dim = None
    embedding_dim = get_embedding_dimension(embeddings)
    metadata = read_index_metadata(index_dir)
    meta_model = None
    meta_dim = None
    if metadata:
        meta_model_raw = metadata.get("embedding_model")
        if isinstance(meta_model_raw, str):
            meta_model = meta_model_raw
        meta_dim_raw = metadata.get("embedding_dim")
        if isinstance(meta_dim_raw, int):
            meta_dim = meta_dim_raw

    if index_dim is not None and embedding_dim is not None and index_dim != embedding_dim:
        expected = meta_model or "the model used to build the index"
        raise RuntimeError(
            "FAISS index dimension mismatch: "
            f"index_dim={index_dim}, embedding_dim={embedding_dim}. "
            "The index was likely built with a different embedding model. "
            f"Rebuild the index with EMBEDDING_MODEL={embedding_factory.model_name} "
            f"or switch EMBEDDING_MODEL back to {expected}."
        )

    if meta_model and meta_model != embedding_factory.model_name:
        logger.warning(
            "FAISS index metadata uses a different embedding model: %s (current=%s).",
            meta_model,
            embedding_factory.model_name,
        )
    if meta_dim is not None and index_dim is not None and meta_dim != index_dim:
        logger.warning(
            "FAISS index metadata dim mismatch: meta_dim=%s index_dim=%s",
            meta_dim,
            index_dim,
        )

    return vectorstore
