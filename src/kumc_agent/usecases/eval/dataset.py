from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kumc_agent.usecases.eval.safety import (
    contains_dangerous_metadata_key,
    sanitize_report_payload,
)
from kumc_agent.usecases.eval.schema import EvalAssertion, EvalCase, SEVERITIES


class EvalDatasetError(ValueError):
    pass


def load_eval_set(
    path: Path,
    *,
    target: str | None = None,
    suite: str | None = None,
    limit: int | None = None,
    allow_ragas_compat: bool = True,
) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"EvalSet not found: {path}")
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as fr:
        for line_number, raw_line in enumerate(fr, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalDatasetError(
                    f"{path}:{line_number}: invalid JSONL row"
                ) from exc
            if not isinstance(payload, dict):
                raise EvalDatasetError(f"{path}:{line_number}: row must be object")
            if allow_ragas_compat and _is_ragas_compat(payload):
                payload = _convert_ragas_compat(
                    payload,
                    target=target or "rag_circle",
                    suite=suite or "smoke",
                    line_number=line_number,
                )
            case = parse_eval_case(
                payload,
                path=path,
                line_number=line_number,
            )
            if target and case.target != target:
                raise EvalDatasetError(
                    f"{path}:{line_number}: case {case.id} target={case.target} "
                    f"does not match requested target={target}"
                )
            if suite and case.suite != suite:
                raise EvalDatasetError(
                    f"{path}:{line_number}: case {case.id} suite={case.suite} "
                    f"does not match requested suite={suite}"
                )
            cases.append(case)
            if limit and limit > 0 and len(cases) >= limit:
                break
    return cases


def parse_eval_case(
    payload: dict[str, Any],
    *,
    path: Path | None = None,
    line_number: int = 0,
) -> EvalCase:
    location = _location(path, line_number)
    case_id = str(payload.get("id") or "").strip()
    target = str(payload.get("target") or "").strip()
    suite = str(payload.get("suite") or "").strip()
    if not case_id:
        raise EvalDatasetError(f"{location}: id is required")
    if not target:
        raise EvalDatasetError(f"{location}: case {case_id}: target is required")
    if not suite:
        raise EvalDatasetError(f"{location}: case {case_id}: suite is required")

    input_payload = payload.get("input")
    if not isinstance(input_payload, dict):
        raise EvalDatasetError(f"{location}: case {case_id}: input must be object")
    expected = payload.get("expected", {})
    if expected is None:
        expected = {}
    if not isinstance(expected, dict):
        raise EvalDatasetError(f"{location}: case {case_id}: expected must be object")
    raw_assertions = payload.get("assertions", [])
    if raw_assertions is None:
        raw_assertions = []
    if not isinstance(raw_assertions, list):
        raise EvalDatasetError(
            f"{location}: case {case_id}: assertions must be list"
        )
    assertions: list[EvalAssertion] = []
    for index, item in enumerate(raw_assertions):
        if not isinstance(item, dict):
            raise EvalDatasetError(
                f"{location}: case {case_id}: assertions[{index}] must be object"
            )
        try:
            assertions.append(EvalAssertion.from_payload(item))
        except ValueError as exc:
            raise EvalDatasetError(
                f"{location}: case {case_id}: assertions[{index}]: {exc}"
            ) from exc

    tags = payload.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        raise EvalDatasetError(f"{location}: case {case_id}: tags must be list")
    severity = str(payload.get("severity") or "major").strip() or "major"
    if severity not in SEVERITIES:
        raise EvalDatasetError(
            f"{location}: case {case_id}: invalid severity={severity}"
        )
    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise EvalDatasetError(f"{location}: case {case_id}: metadata must be object")
    metadata_was_sanitized = contains_dangerous_metadata_key(metadata)
    sanitized_metadata = sanitize_report_payload(metadata)
    if isinstance(sanitized_metadata, dict) and metadata_was_sanitized:
        sanitized_metadata = {
            **sanitized_metadata,
            "sanitizer_warnings": ["dangerous_metadata_key_removed"],
        }
    return EvalCase(
        id=case_id,
        target=target,
        suite=suite,
        input=input_payload,
        expected=expected,
        assertions=tuple(assertions),
        tags=tuple(str(item) for item in tags if str(item).strip()),
        severity=severity,
        metadata=dict(sanitized_metadata),
    )


def _is_ragas_compat(payload: dict[str, Any]) -> bool:
    has_question = bool(str(payload.get("question") or payload.get("query") or "").strip())
    has_truth = "ground_truth" in payload or "ground_truths" in payload
    return has_question and has_truth and "input" not in payload


def _convert_ragas_compat(
    payload: dict[str, Any],
    *,
    target: str,
    suite: str,
    line_number: int,
) -> dict[str, Any]:
    question = str(payload.get("question") or payload.get("query") or "").strip()
    ground_truths: list[str]
    if isinstance(payload.get("ground_truths"), list):
        ground_truths = [
            str(value).strip()
            for value in payload["ground_truths"]
            if str(value).strip()
        ]
    else:
        truth = str(payload.get("ground_truth") or "").strip()
        ground_truths = [truth] if truth else []
    case_id = str(payload.get("id") or f"ragas-compat-{line_number:04d}")
    return {
        "id": case_id,
        "target": target,
        "suite": suite,
        "input": {
            "question": question,
            "ground_truths": ground_truths,
        },
        "expected": {
            "answer_contains": ground_truths,
        },
        "assertions": [
            {"type": "answer_contains_any", "values": ground_truths}
        ] if ground_truths else [],
        "tags": ["ragas_compat"],
        "severity": "major",
        "metadata": {
            "compat_format": "ragas",
        },
    }


def _location(path: Path | None, line_number: int) -> str:
    if path is None:
        return "<eval-set>"
    if line_number > 0:
        return f"{path}:{line_number}"
    return str(path)
