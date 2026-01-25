from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import EmbeddingFactory
from indexing.chunks import Chunk
from indexing.utils import ensure_dir

logger = logging.getLogger(__name__)


def build_faiss_index(*, chunks: list[Chunk], model_name: str, index_dir: Path) -> None:
    if not chunks:
        logger.warning("No documents to index. Skipping FAISS build.")
        return

    ensure_dir(index_dir)

    embeddings = EmbeddingFactory(model_name).get_embeddings()

    docs = [
        Document(page_content=chunk.text, metadata=chunk.metadata)
        for chunk in chunks
    ]
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local(str(index_dir))
    logger.info("Saved FAISS index to %s", index_dir)
