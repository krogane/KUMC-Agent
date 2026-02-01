from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import EmbeddingFactory
from indexing.chunks import Chunk, chunk_embedding_text
from indexing.index_metadata import get_embedding_dimension, write_index_metadata
from indexing.utils import ensure_dir

logger = logging.getLogger(__name__)


def build_faiss_index(*, chunks: list[Chunk], model_name: str, index_dir: Path) -> None:
    if not chunks:
        logger.warning("No documents to index. Skipping FAISS build.")
        return

    ensure_dir(index_dir)

    embeddings = EmbeddingFactory(model_name).get_embeddings()

    docs = [
        Document(page_content=chunk_embedding_text(chunk), metadata=chunk.metadata)
        for chunk in chunks
    ]
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local(str(index_dir))
    embedding_dim = get_embedding_dimension(embeddings)
    index_dim = getattr(vectorstore.index, "d", None)
    write_index_metadata(
        index_dir,
        embedding_model=model_name,
        embedding_dim=embedding_dim,
        index_dim=index_dim if isinstance(index_dim, int) else None,
    )
    logger.info("Saved FAISS index to %s", index_dir)
