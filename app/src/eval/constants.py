from __future__ import annotations

from pathlib import Path

APP_DIR: Path = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR: Path = APP_DIR / "data" / "eval"
DEFAULT_EVAL_FILE: Path = DEFAULT_EVAL_DIR / "ragas.jsonl"
DEFAULT_RESULT_DIR: Path = DEFAULT_EVAL_DIR / "result"
DEFAULT_SAVE_DATASET: Path = DEFAULT_RESULT_DIR / "ragas_dataset.jsonl"
DEFAULT_RESULT_PATH: Path = DEFAULT_RESULT_DIR / "ragas_result.csv"

LOG_LEVEL: str = "INFO"
