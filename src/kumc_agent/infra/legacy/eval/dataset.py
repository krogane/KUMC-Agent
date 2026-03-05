from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalItem:
    question: str
    ground_truths: list[str]


def load_eval_items(path: Path) -> list[EvalItem]:
    if not path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found: {path}. "
            "Create a JSONL file with {question, ground_truth} or {question, ground_truths}."
        )

    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as fr:
        for line_no, line in enumerate(fr, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path} line {line_no}: {e}"
                ) from e

            question = obj.get("question", obj.get("query"))
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Missing/invalid 'question' (or 'query') in {path} line {line_no}"
                )

            ground_truths = obj.get("ground_truths", obj.get("ground_truth"))
            ground_truths = _normalize_ground_truths(ground_truths, path, line_no)

            items.append(
                EvalItem(question=question.strip(), ground_truths=ground_truths)
            )

    return items


def _normalize_ground_truths(value: object, path: Path, line_no: int) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise ValueError(
        "Missing/invalid 'ground_truths' (string or list of strings) in "
        f"{path} line {line_no}"
    )
