from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from eval.dataset import EvalItem
from pipeline.rag_pipeline import RagPipeline


def build_records(
    *,
    items: Iterable[EvalItem],
    rag_pipeline: RagPipeline,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in items:
        answer, contexts = rag_pipeline.answer_with_contexts(item.question)
        records.append(
            {
                "question": item.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truths": item.ground_truths,
                "ground_truth": item.ground_truths[0],
            }
        )
    return records


def save_dataset(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fw:
        for record in records:
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
