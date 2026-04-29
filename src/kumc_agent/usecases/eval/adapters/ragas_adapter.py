from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from kumc_agent.usecases.eval.adapters.base import AdapterRunResult
from kumc_agent.usecases.eval.ragas import EvaluateRagasRequest, EvaluateRagasUsecase
from kumc_agent.usecases.eval.schema import EvalCase


class RagasEvalAdapter:
    def __init__(
        self,
        *,
        target: str,
        ragas_usecase: EvaluateRagasUsecase | None = None,
    ) -> None:
        self.target = target
        self._ragas_usecase = ragas_usecase

    def run_case(self, *, case: EvalCase, request: Any) -> AdapterRunResult:
        actual = case.input.get("adapter_output") or case.input.get("actual")
        if isinstance(actual, dict):
            metadata = dict(actual.get("metadata") or {})
            metadata.setdefault("adapter", "ragas_fixture")
            metadata.setdefault("target", case.target)
            metadata.setdefault("side_effects", [])
            return AdapterRunResult(
                actual={**actual, "metadata": metadata},
                metrics={},
                metadata={"adapter": "ragas_fixture", "fixture_mode": True},
            )
        if self._ragas_usecase is None:
            return AdapterRunResult(
                actual={
                    "answer": "",
                    "citations": [],
                    "contexts": [],
                    "metadata": {
                        "adapter": "ragas",
                        "degraded": True,
                        "skipped_reason": "ragas_usecase_unavailable",
                        "side_effects": [],
                    },
                },
                metadata={
                    "adapter": "ragas",
                    "degraded": True,
                    "skipped_reason": "ragas_usecase_unavailable",
                },
            )
        result = self._run_ragas_case(case=case, request=request)
        metadata = {
            "adapter": "ragas",
            "ragas_metadata": result.ragas_metadata,
            "degraded": bool(result.ragas_metadata.get("skipped_reason")),
            "side_effects": [],
        }
        record = result.records[0] if result.records else {}
        answer = str(record.get("answer") or "")
        citations = _list_of_dicts(record.get("citations"))
        contexts = _list_of_dicts(record.get("contexts"))
        retrieval_trace = _list_of_dicts(record.get("retrieval_trace"))
        return AdapterRunResult(
            actual={
                "answer": answer,
                "citations": citations,
                "sources": _list_of_dicts(record.get("sources")) or citations,
                "contexts": contexts,
                "retrieval_trace": retrieval_trace,
                "metadata": metadata,
            },
            metrics={
                "exact_match": result.exact_match,
                "token_overlap": result.token_overlap,
                **{
                    f"ragas.{key}": value
                    for key, value in result.ragas_metrics.items()
                },
            },
            metadata=metadata,
            status="degraded" if metadata["degraded"] else "completed",
        )

    def _run_ragas_case(self, *, case: EvalCase, request: Any):
        with tempfile.TemporaryDirectory() as tmp:
            eval_file = Path(tmp) / "ragas_case.jsonl"
            record = _case_to_ragas_record(case)
            eval_file.write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            return self._ragas_usecase.execute(
                EvaluateRagasRequest(
                    eval_file=eval_file,
                    limit=1,
                    ragas_batch_size=getattr(request, "ragas_batch_size", None),
                    ragas_max_workers=getattr(request, "ragas_max_workers", None),
                    ragas_timeout_seconds=getattr(request, "ragas_timeout_seconds", None),
                    ragas_max_retries=getattr(request, "ragas_max_retries", None),
                    cancel_event=getattr(request, "cancel_event", None),
                )
            )


def _case_to_ragas_record(case: EvalCase) -> dict[str, Any]:
    question = str(case.input.get("question") or case.input.get("query") or "").strip()
    ground_truths = case.input.get("ground_truths") or case.expected.get("ground_truths")
    if not isinstance(ground_truths, list):
        ground_truth = str(case.input.get("ground_truth") or case.expected.get("ground_truth") or "").strip()
        ground_truths = [ground_truth] if ground_truth else []
    return {
        "question": question,
        "ground_truths": [str(value) for value in ground_truths if str(value)],
        "ground_truth": str(ground_truths[0]) if ground_truths else "",
    }


def _ground_truth_answer(case: EvalCase) -> str:
    ground_truths = case.input.get("ground_truths") or case.expected.get("ground_truths")
    if isinstance(ground_truths, list) and ground_truths:
        return str(ground_truths[0])
    return str(case.input.get("ground_truth") or case.expected.get("ground_truth") or "")


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]
