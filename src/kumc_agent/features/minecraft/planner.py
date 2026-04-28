from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from kumc_agent.domain.models.minecraft import ServerOperationPlan
from kumc_agent.domain.ports.llms import LLMPort


class UnsupportedServerOperationError(ValueError):
    pass


@dataclass(frozen=True)
class ServerOperationPlanner:
    default_server_name: str = "default"
    llm: LLMPort | None = None
    prompts_dir: Path | None = None
    prompt_name: str = "server_operation_planner.md"

    def plan(self, text: str) -> tuple[ServerOperationPlan, ...]:
        source = (text or "").strip()
        if not source:
            raise UnsupportedServerOperationError("対応操作を確認してください。")
        labeled = _parse_labeled(source)
        if labeled.get("operation"):
            return (self._plan_from_payload(labeled, sequence_index=0, planner="deterministic_labeled"),)
        if self.llm is None:
            raise UnsupportedServerOperationError("対応操作を確認してください。")
        return self._plan_with_llm(source)

    def _plan_with_llm(self, source: str) -> tuple[ServerOperationPlan, ...]:
        raw = self.llm.generate(
            system_prompt=self._system_prompt(),
            user_prompt=source,
            temperature=0.0,
            max_output_tokens=1200,
        )
        payload = _extract_json(raw)
        operations_payload = _operations_payload(payload)
        if not operations_payload:
            raise UnsupportedServerOperationError("対応操作を確認してください。")
        plans = tuple(
            self._plan_from_payload(
                item,
                sequence_index=index,
                planner="llm",
            )
            for index, item in enumerate(operations_payload)
        )
        if any(plan.operation == "unsupported" for plan in plans):
            raise UnsupportedServerOperationError("対応操作を確認してください。")
        return plans

    def _system_prompt(self) -> str:
        prompt_path = (self.prompts_dir or Path("assets/prompts")) / self.prompt_name
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return _DEFAULT_SYSTEM_PROMPT

    def _plan_from_payload(
        self,
        payload: dict[str, Any],
        *,
        sequence_index: int,
        planner: str,
    ) -> ServerOperationPlan:
        data = _validate_plan_payload(payload)
        operation = _normalize_operation(data.get("operation", "unsupported"))
        depends_on = data.get("depends_on", "")
        if not depends_on and sequence_index > 0:
            depends_on = "previous"
        metadata = {
            "planner": planner,
            "sequence_index": sequence_index,
            "depends_on": depends_on,
        }
        if data.get("unsupported_reason"):
            metadata["unsupported_reason"] = data["unsupported_reason"]
        return ServerOperationPlan(
            operation=operation,
            server_name=data.get("server_name") or self.default_server_name,
            service_name=data.get("service_name", ""),
            server_dir=data.get("server_dir", ""),
            path=data.get("path", ""),
            query=data.get("query", ""),
            player_name=data.get("player_name", ""),
            whitelist_action=data.get("whitelist_action", ""),
            reason=data.get("reason", ""),
            confidence=data.get("confidence", "medium"),
            metadata=metadata,
        )


def _parse_labeled(text: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    patterns = {
        "operation": ("operation", "op", "操作", "action"),
        "server_name": ("server", "server_name", "サーバー", "対象サーバー"),
        "service_name": ("service", "service_name", "サービス"),
        "server_dir": ("server_dir",),
        "path": ("path", "dir", "directory", "パス", "ディレクトリ"),
        "query": ("query", "検索", "search"),
        "player_name": ("player", "player_name", "mcid", "プレイヤー"),
        "whitelist_action": ("whitelist_action", "whitelist_action_mode", "mode"),
        "reason": ("reason", "理由"),
        "confidence": ("confidence", "信頼度"),
    }
    for key, labels in patterns.items():
        value = _extract_labeled_value(text, labels)
        if value:
            payload[key] = value
    return payload


def _extract_labeled_value(text: str, labels: tuple[str, ...]) -> str:
    for label in labels:
        match = re.search(rf"{re.escape(label)}[:：=]\s*([^\s,、。]+)", text, re.I)
        if match:
            return match.group(1).strip()
    return ""


def _extract_json(text: str) -> object:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    start_obj = stripped.find("{")
    start_list = stripped.find("[")
    starts = [index for index in (start_obj, start_list) if index >= 0]
    if not starts:
        raise UnsupportedServerOperationError("対応操作を確認してください。")
    start = min(starts)
    end = stripped.rfind("}" if stripped[start] == "{" else "]")
    if end < start:
        raise UnsupportedServerOperationError("対応操作を確認してください。")
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise UnsupportedServerOperationError("対応操作を確認してください。") from exc


def _operations_payload(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("operations"), list):
        items = payload["operations"]
    elif isinstance(payload, dict) and payload.get("operation"):
        items = [payload]
    else:
        return []
    return [item for item in items if isinstance(item, dict)]


def _validate_plan_payload(payload: dict[str, Any]) -> dict[str, str]:
    allowed = {
        "operation",
        "server_name",
        "service_name",
        "server_dir",
        "path",
        "query",
        "player_name",
        "whitelist_action",
        "reason",
        "confidence",
        "depends_on",
        "unsupported_reason",
    }
    data: dict[str, str] = {}
    for key, value in payload.items():
        if key not in allowed:
            continue
        if value is None:
            data[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            data[key] = str(value).strip()
        else:
            raise UnsupportedServerOperationError("対応操作を確認してください。")
    operation = data.get("operation", "").strip()
    if not operation:
        raise UnsupportedServerOperationError("対応操作を確認してください。")
    if data.get("confidence") not in {"", "low", "medium", "high"}:
        data["confidence"] = "medium"
    if not data.get("confidence"):
        data["confidence"] = "medium"
    return data


def _normalize_operation(value: str) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    aliases = {
        "mc_status": "status",
        "docker": "docker_ps",
        "ps": "docker_ps",
        "docker_ps_a": "docker_ps",
        "compose": "compose_up",
        "up": "compose_up",
        "down": "compose_down",
        "restart_mc_server": "restart",
        "server_restart": "restart",
        "whitelist": "whitelist_update",
        "backup": "backup_create",
        "create_backup": "backup_create",
    }
    return aliases.get(normalized, normalized)


_DEFAULT_SYSTEM_PROMPT = """\
You are a server-operation planner for KUMC-Agent.
Return only JSON. Do not return shell commands.
Schema:
{"operations":[{"operation":"docker_ps|file_search|compose_up|compose_restart|restart|whitelist_update|compose_down|backup_create|unsupported","server_name":"","service_name":"","path":"","query":"","player_name":"","whitelist_action":"add|remove","reason":"","confidence":"low|medium|high","depends_on":"","unsupported_reason":""}]}
Use unsupported when the request is not one of the listed operations or required intent is unclear.
"""
