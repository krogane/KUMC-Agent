from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexingConfig:
    raw_dir: Path
    chunks_path: Path
    index_dir: Path
    chunk_size: int = 1000
    chunk_overlap: int = 100
