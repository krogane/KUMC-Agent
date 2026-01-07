from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import AppConfig, EmbeddingFactory

logger = logging.getLogger(__name__)

# コンフィグ
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
MODEL_NAME: str = "intfloat/multilingual-e5-small"

LOG_LEVEL: str = "INFO"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def chunk_markdown_to_jsonl(
    *,
    raw_data_dir: Path,
    chunk_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> None:

    _ensure_dir(chunk_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    md_files = sorted(raw_data_dir.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found under %s", raw_data_dir)
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )

    for f in md_files:
        text = f.read_text(encoding="utf-8")
        docs = splitter.create_documents([text], metadatas=[{"source": f.name}])

        out_path = chunk_dir / f"{f.stem}.jsonl"
        with out_path.open("w", encoding="utf-8") as fw:
            for i, doc in enumerate(docs):
                record = {
                    "text": doc.page_content,
                    "metadata": {
                        **(doc.metadata or {}),
                        "chunk_id": i,
                    },
                }
                fw.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Chunked %s -> %s (%d chunks)", f.name, out_path.name, len(docs))


def load_documents_from_jsonl(chunk_dir: Path) -> list[Document]:
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")

    docs: list[Document] = []
    jsonl_files = sorted(chunk_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No .jsonl chunk files found under %s", chunk_dir)
        return docs

    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as fr:
            for line_no, line in enumerate(fr, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in {path.name} at line {line_no}: {e}"
                    ) from e

                text = obj.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"Missing/invalid 'text' in {path.name} line {line_no}")

                metadata = obj.get("metadata")
                if metadata is None:
                    metadata = {}
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"Missing/invalid 'metadata' (must be dict) in {path.name} line {line_no}"
                    )

                # Persist file context
                metadata.setdefault("chunk_file", path.name)
                metadata.setdefault("line_no", line_no)

                docs.append(Document(page_content=text, metadata=metadata))

    logger.info("Loaded %d documents from %d chunk files", len(docs), len(jsonl_files))
    return docs


def build_faiss_index(*, docs: list[Document], model_name: str, index_dir: Path) -> None:
    if not docs:
        logger.warning("No documents to index. Skipping FAISS build.")
        return

    _ensure_dir(index_dir)

    embeddings = EmbeddingFactory(model_name).get_embeddings()

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local(str(index_dir))
    logger.info("Saved FAISS index to %s", index_dir)


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = AppConfig.from_here(
        embedding_model_name=MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    logger.info("BASE_DIR      : %s", cfg.base_dir)
    logger.info("RAW_DATA_DIR  : %s", cfg.raw_data_dir)
    logger.info("CHUNK_DIR     : %s", cfg.chunk_dir)
    logger.info("INDEX_DIR     : %s", cfg.index_dir)
    logger.info("CHUNK_SIZE    : %d", cfg.chunk_size)
    logger.info("CHUNK_OVERLAP : %d", cfg.chunk_overlap)
    logger.info("MODEL         : %s", cfg.embedding_model_name)

    chunk_markdown_to_jsonl(
        raw_data_dir=cfg.raw_data_dir,
        chunk_dir=cfg.chunk_dir,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    docs = load_documents_from_jsonl(cfg.chunk_dir)
    build_faiss_index(
        docs=docs,
        model_name=cfg.embedding_model_name,
        index_dir=cfg.index_dir,
    )


if __name__ == "__main__":
    main()
