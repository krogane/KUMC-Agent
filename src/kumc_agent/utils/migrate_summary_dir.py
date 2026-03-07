from __future__ import annotations

import shutil
from pathlib import Path


def migrate_summery_chunk_dir(*, base_dir: Path) -> None:
    chunks_dir = base_dir / "data" / "chunks"
    legacy_dir = chunks_dir / "summery_chunk"
    current_dir = chunks_dir / "summary_chunk"
    if not legacy_dir.exists():
        return
    if current_dir.exists():
        return
    current_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy_dir), str(current_dir))
