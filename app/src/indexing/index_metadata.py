from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

INDEX_METADATA_FILENAME = "index_metadata.json"


def read_index_metadata(index_dir: Path) -> dict[str, object] | None:
    path = index_dir / INDEX_METADATA_FILENAME
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        logger.warning("Failed to read index metadata from %s", path, exc_info=True)
        return None


def write_index_metadata(
    index_dir: Path,
    *,
    embedding_model: str,
    embedding_dim: int | None,
    index_dim: int | None,
) -> None:
    path = index_dir / INDEX_METADATA_FILENAME
    payload = {
        "schema_version": 1,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "index_dim": index_dim,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=True, indent=2)
    except Exception:
        logger.warning("Failed to write index metadata to %s", path, exc_info=True)


def get_embedding_dimension(embeddings: Embeddings) -> int | None:
    model = getattr(embeddings, "_model", None)
    get_dim = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(get_dim):
        try:
            return int(get_dim())
        except Exception:
            logger.debug("Failed to read embedding dim from model.", exc_info=True)

    try:
        vector = embeddings.embed_query(" ")
    except Exception:
        logger.debug("Failed to compute embedding dim from query.", exc_info=True)
        return None
    return len(vector) if vector else None
