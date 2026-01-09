from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS

from config import EmbeddingFactory


def load_faiss_index(*, index_dir: Path, embedding_factory: EmbeddingFactory) -> FAISS:
    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index directory not found: {index_dir}. "
            "Build the index first (e.g., run your build script)."
        )

    return FAISS.load_local(
        str(index_dir),
        embedding_factory.get_embeddings(),
        allow_dangerous_deserialization=True,
    )
