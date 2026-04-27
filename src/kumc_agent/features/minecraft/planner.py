from __future__ import annotations

from dataclasses import dataclass
import re

from kumc_agent.domain.models.minecraft import ServerOperationPlan


@dataclass(frozen=True)
class ServerOperationPlanner:
    default_server_name: str = "default"

    def plan(self, text: str) -> tuple[ServerOperationPlan, ...]:
        source = (text or "").strip()
        if not source:
            return (
                ServerOperationPlan(
                    operation="docker_ps",
                    server_name=self.default_server_name,
                    confidence="low",
                    metadata={"planner": "deterministic", "reason": "empty_input"},
                ),
            )
        labeled = _parse_labeled(source)
        if labeled.get("operation"):
            return (self._plan_from_payload(labeled, source, sequence_index=0),)

        plans: list[ServerOperationPlan] = []
        lowered = source.lower()
        if "docker ps" in lowered or "container" in lowered or "コンテナ" in source:
            plans.append(self._plan_from_payload({"operation": "docker_ps"}, source, len(plans)))
        if "compose down" in lowered or "停止" in source:
            plans.append(self._plan_from_payload({"operation": "compose_down"}, source, len(plans)))
        elif "compose restart" in lowered:
            plans.append(self._plan_from_payload({"operation": "compose_restart"}, source, len(plans)))
        elif "restart" in lowered or "再起動" in source:
            plans.append(self._plan_from_payload({"operation": "restart"}, source, len(plans)))
        elif "compose up" in lowered or "起動" in source:
            plans.append(self._plan_from_payload({"operation": "compose_up"}, source, len(plans)))
        if "whitelist" in lowered or "ホワイトリスト" in source:
            payload = {
                "operation": "whitelist_update",
                "player_name": _infer_player_name(source),
                "whitelist_action": _infer_whitelist_action(source),
            }
            plans.append(self._plan_from_payload(payload, source, len(plans)))
        if "file" in lowered or "ファイル" in source or "検索" in source:
            payload = {
                "operation": "file_search",
                "path": _extract_labeled_value(source, ("path", "dir", "directory", "パス")),
                "query": _extract_labeled_value(source, ("query", "検索", "search")),
            }
            plans.append(self._plan_from_payload(payload, source, len(plans)))
        if not plans:
            plans.append(self._plan_from_payload({"operation": "docker_ps"}, source, 0))
        return tuple(plans)

    def _plan_from_payload(
        self,
        payload: dict[str, str],
        source: str,
        sequence_index: int,
    ) -> ServerOperationPlan:
        operation = payload.get("operation") or _infer_operation(source)
        server_name = payload.get("server_name") or _extract_labeled_value(
            source,
            ("server", "server_name", "サーバー", "対象サーバー"),
        )
        service_name = payload.get("service_name") or _extract_labeled_value(
            source,
            ("service", "service_name", "サービス"),
        )
        depends_on = ""
        if sequence_index > 0:
            depends_on = "previous"
        return ServerOperationPlan(
            operation=operation,
            server_name=server_name or self.default_server_name,
            service_name=service_name or "",
            server_dir=payload.get("server_dir", ""),
            path=payload.get("path", ""),
            query=payload.get("query", ""),
            player_name=payload.get("player_name", ""),
            whitelist_action=payload.get("whitelist_action", ""),
            reason=payload.get("reason", ""),
            confidence="high" if payload.get("operation") else "medium",
            metadata={
                "planner": "deterministic",
                "sequence_index": sequence_index,
                "depends_on": depends_on,
            },
        )


def _parse_labeled(text: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    patterns = {
        "operation": ("operation", "op", "操作", "action"),
        "server_name": ("server", "server_name", "サーバー", "対象サーバー"),
        "service_name": ("service", "service_name", "サービス"),
        "path": ("path", "dir", "directory", "パス", "ディレクトリ"),
        "query": ("query", "検索", "search"),
        "player_name": ("player", "player_name", "mcid", "プレイヤー"),
        "whitelist_action": ("whitelist_action", "whitelist", "mode"),
        "reason": ("reason", "理由"),
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


def _infer_operation(text: str) -> str:
    lowered = text.lower()
    if "compose down" in lowered or "停止" in text:
        return "compose_down"
    if "compose restart" in lowered:
        return "compose_restart"
    if "restart" in lowered or "再起動" in text:
        return "restart"
    if "compose up" in lowered or "起動" in text:
        return "compose_up"
    if "file" in lowered or "ファイル" in text or "検索" in text:
        return "file_search"
    if "whitelist" in lowered or "ホワイトリスト" in text:
        return "whitelist_update"
    return "docker_ps"


def _infer_player_name(text: str) -> str:
    labeled = _extract_labeled_value(text, ("player", "player_name", "mcid", "プレイヤー"))
    if labeled:
        return labeled
    match = re.search(r"(?:whitelist|ホワイトリスト)(?:に|へ| add|追加)?\s*([A-Za-z0-9_]{3,16})", text, re.I)
    return match.group(1) if match else ""


def _infer_whitelist_action(text: str) -> str:
    lowered = text.lower()
    if "remove" in lowered or "delete" in lowered or "削除" in text or "外" in text:
        return "remove"
    if "add" in lowered or "追加" in text or "入" in text:
        return "add"
    return ""
