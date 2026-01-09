from __future__ import annotations

from pathlib import Path

DEFAULT_EVAL_FILE: Path = Path("app") / "data" / "eval" / "ragas.jsonl"
DEFAULT_RESULT_DIR: Path = Path("app") / "data" / "eval" / "result"
DEFAULT_SAVE_DATASET: Path = DEFAULT_RESULT_DIR / "ragas_dataset.jsonl"
DEFAULT_RESULT_PATH: Path = DEFAULT_RESULT_DIR / "ragas_result.csv"

LOG_LEVEL: str = "INFO"
