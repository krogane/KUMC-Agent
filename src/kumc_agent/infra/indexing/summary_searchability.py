from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUMMARY_SEARCHABILITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SummarySearchabilityDecision:
    searchable: bool
    summary: str
    reason: str = ""
    fallback_used: bool = False
    parse_failed: bool = False

    @classmethod
    def keep(
        cls,
        *,
        summary: str,
        reason: str = "",
        fallback_used: bool = False,
        parse_failed: bool = False,
    ) -> "SummarySearchabilityDecision":
        return cls(
            searchable=True,
            summary=summary,
            reason=reason,
            fallback_used=fallback_used,
            parse_failed=parse_failed,
        )

    @classmethod
    def exclude(
        cls,
        *,
        reason: str = "",
    ) -> "SummarySearchabilityDecision":
        return cls(searchable=False, summary="", reason=reason)


def normalize_summary_parent_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def build_summary_searchability_prompt(prompt: str) -> str:
    return (
        prompt.rstrip()
        + "\n\n"
        + "上記の要約対象本文が、単体で検索結果として利用できる意味のある文章かも判定してください。\n"
        + "見出しだけ、ページ番号だけ、記号列、意図不明な表セル、ナビゲーション断片、OCRノイズ、"
        + "文脈なしの単語列などは searchable=false にしてください。\n"
        + "単体で検索ヒットとして意味を持つ場合だけ "
        + "searchable=true にしてください。\n"
        + "出力は次のJSONオブジェクトのみとし、Markdownや説明文を付けないでください。\n"
        + '{"searchable": true, "summary": "要約文", "reason": "短い判定理由"}'
    )


def parse_summary_searchability_response(
    response: str,
    *,
    fallback_summary: str,
) -> SummarySearchabilityDecision:
    payload = _strip_code_fences(response or "")
    if not payload.strip():
        return SummarySearchabilityDecision.keep(
            summary=fallback_summary,
            reason="empty_response",
            fallback_used=True,
            parse_failed=True,
        )

    leading = payload.lstrip()
    if leading.startswith("{") or leading.startswith("["):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return SummarySearchabilityDecision.keep(
                summary=fallback_summary,
                reason="invalid_json",
                fallback_used=True,
                parse_failed=True,
            )

        if isinstance(data, list):
            summary = _summary_from_list(data) or fallback_summary
            return SummarySearchabilityDecision.keep(
                summary=summary,
                reason="legacy_json_list",
                fallback_used=not bool(_summary_from_list(data)),
                parse_failed=True,
            )

        if isinstance(data, dict):
            searchable = _parse_bool(data.get("searchable"))
            reason = str(data.get("reason") or "").strip()
            summary = str(data.get("summary") or "").strip()
            if searchable is False:
                return SummarySearchabilityDecision.exclude(reason=reason)
            if not summary:
                summary = fallback_summary
            return SummarySearchabilityDecision.keep(
                summary=summary,
                reason=reason if searchable is not None else reason or "missing_searchable",
                fallback_used=not bool(str(data.get("summary") or "").strip()),
                parse_failed=searchable is None,
            )

    if leading.startswith('"') and leading.rstrip().endswith('"'):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, str) and decoded.strip():
            return SummarySearchabilityDecision.keep(
                summary=decoded.strip(),
                reason="legacy_json_string",
                parse_failed=True,
            )

    return SummarySearchabilityDecision.keep(
        summary=payload.strip(),
        reason="legacy_text",
        parse_failed=True,
    )


def summary_decision_sidecar_path(chunk_path: Path) -> Path:
    return chunk_path.with_name(f"{chunk_path.stem}.summary_decisions.json")


def write_summary_searchability_decisions(
    *,
    path: Path,
    decisions: Iterable[tuple[object, SummarySearchabilityDecision]],
) -> None:
    rows: list[dict[str, object]] = []
    for parent_chunk_id, decision in decisions:
        parent_id = normalize_summary_parent_id(parent_chunk_id)
        if not parent_id:
            continue
        rows.append(
            {
                "parent_chunk_id": parent_id,
                "searchable": decision.searchable,
                "summary": decision.summary,
                "reason": decision.reason,
                "fallback_used": decision.fallback_used,
                "parse_failed": decision.parse_failed,
            }
        )

    payload = {
        "schema_version": SUMMARY_SEARCHABILITY_SCHEMA_VERSION,
        "checked_count": len(rows),
        "excluded_count": sum(1 for row in rows if row["searchable"] is False),
        "fallback_count": sum(1 for row in rows if row["fallback_used"] is True),
        "decisions": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_summary_searchability_decisions(
    path: Path,
) -> dict[str, SummarySearchabilityDecision]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    decisions_raw = payload.get("decisions")
    if not isinstance(decisions_raw, list):
        return {}
    decisions: dict[str, SummarySearchabilityDecision] = {}
    for row in decisions_raw:
        if not isinstance(row, dict):
            continue
        parent_id = normalize_summary_parent_id(row.get("parent_chunk_id"))
        if not parent_id:
            continue
        searchable = _parse_bool(row.get("searchable"))
        decisions[parent_id] = SummarySearchabilityDecision(
            searchable=True if searchable is None else searchable,
            summary=str(row.get("summary") or ""),
            reason=str(row.get("reason") or ""),
            fallback_used=bool(row.get("fallback_used", False)),
            parse_failed=bool(row.get("parse_failed", False)),
        )
    return decisions


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


def _summary_from_list(values: list[object]) -> str:
    summaries: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            summaries.append(text)
    return "\n".join(summaries)
