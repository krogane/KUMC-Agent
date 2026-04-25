from __future__ import annotations

from dataclasses import dataclass
import re

from kumc_agent.domain.models.minecraft import (
    ActionSpec,
    MinecraftDryRun,
    ServerOperation,
)
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.features.foundation.feature_flags import FeatureFlagService
from kumc_agent.features.minecraft.actions import MinecraftActionSpecRegistry
from kumc_agent.infra.minecraft.repository import ServerOperationRepository
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class MinecraftSupportResult:
    text: str
    detail_markdown: str
    operation: ServerOperation | None = None
    warnings: tuple[str, ...] = tuple()


class MinecraftSupportService:
    def __init__(
        self,
        *,
        repository: ServerOperationRepository,
        feature_flags: FeatureFlagService,
        registry: MinecraftActionSpecRegistry | None = None,
    ) -> None:
        self.repository = repository
        self.feature_flags = feature_flags
        self.registry = registry or MinecraftActionSpecRegistry()

    def status(self, *, access: AccessContext) -> MinecraftSupportResult:
        mode = self.feature_flags.mode_for("minecraft_server_ops")
        pending = self.repository.list(status="waiting_approval")
        detail = "\n".join(
            [
                "# Minecraft support status",
                "",
                f"- feature_mode: `{mode}`",
                f"- execution: `disabled in Wave 6`",
                f"- waiting_approval: {len(pending)}",
                "",
                "## Allowed action specs",
                *[self._spec_line(spec) for spec in self.registry.list()],
                "",
                "## Safety",
                "- 任意 shell command は受け付けません。",
                "- この Wave では docker / compose / whitelist 操作を実行しません。",
                "- `mc_request` は dry-run と ServerOperation 保存だけを行います。",
            ]
        )
        return MinecraftSupportResult(
            text=f"Minecraft support: mode={mode}, execution=dry-run-only, waiting_approval={len(pending)}",
            detail_markdown=detail,
        )

    def request(
        self,
        *,
        instruction: str,
        target: str,
        access: AccessContext,
    ) -> MinecraftSupportResult:
        payload = _parse_request(_join_text(instruction, target))
        operation = payload.pop("operation", "docker_ps")
        if not self.registry.has(operation):
            raise ValueError(f"Unsupported Minecraft operation: {operation}")
        spec = self.registry.get(operation)
        missing = [name for name in spec.required_args if not payload.get(name)]
        if missing:
            raise ValueError(f"Missing required Minecraft operation args: {', '.join(missing)}")
        mode = self.feature_flags.mode_for("minecraft_server_ops")
        dry_run = self._dry_run(spec=spec, args=payload, mode=mode)
        status = "disabled" if mode == "disabled" else "waiting_approval"
        server_operation = ServerOperation(
            id=stable_hash(
                f"server-operation:{dry_run.server_name}:{dry_run.operation}:{payload}:{access.user_id}"
            )[:32],
            server_name=dry_run.server_name,
            operation=dry_run.operation,
            requested_by_user_id=access.user_id,
            status=status,
            risk_level=dry_run.risk_level,
            dry_run=dry_run,
            metadata={
                "feature_mode": mode,
                "wave": "6",
                "execution": "not_executed",
                "requires_admin": spec.approval_policy != "self",
            },
        )
        stored = self.repository.save(server_operation)
        detail = self._operation_detail(stored)
        if status == "disabled":
            text = f"Minecraft operation は disabled です。dry-run だけ保存しました: {stored.id}"
        else:
            text = f"Minecraft operation dry-run を保存しました: {stored.id} / approval required"
        return MinecraftSupportResult(
            text=text,
            detail_markdown=detail,
            operation=stored,
            warnings=dry_run.warnings,
        )

    def _dry_run(
        self,
        *,
        spec: ActionSpec,
        args: dict[str, str],
        mode: str,
    ) -> MinecraftDryRun:
        server_name = args.get("server_name") or "default"
        warnings = [
            "This is a dry-run. No command was executed.",
            "Approval is required before any future executor can run this operation.",
        ]
        if mode == "disabled":
            warnings.append("minecraft_server_ops feature flag is disabled.")
        if spec.risk_level in {"high", "critical"}:
            warnings.append("High-risk Minecraft operation. Review downtime and rollback before approval.")
        return MinecraftDryRun(
            operation=spec.operation,
            server_name=server_name,
            args={key: value for key, value in args.items() if key != "server_name"},
            risk_level=spec.risk_level,
            approval_policy=spec.approval_policy,
            impact=_impact(spec),
            expected_downtime=_downtime(spec),
            rollback=_rollback(spec),
            command_preview=_command_preview(spec, args),
            warnings=tuple(warnings),
            execution_allowed=False,
        )

    def _operation_detail(self, operation: ServerOperation) -> str:
        dry_run = operation.dry_run
        if dry_run is None:
            return f"# Minecraft ServerOperation\n\n- id: `{operation.id}`"
        args = [f"- {key}: `{value}`" for key, value in dry_run.args.items()] or ["- none"]
        return "\n".join(
            [
                "# Minecraft ServerOperation dry-run",
                "",
                f"- id: `{operation.id}`",
                f"- status: `{operation.status}`",
                f"- server: `{operation.server_name}`",
                f"- operation: `{operation.operation}`",
                f"- risk: `{operation.risk_level}`",
                f"- approval_policy: `{dry_run.approval_policy}`",
                f"- execution_allowed: `{dry_run.execution_allowed}`",
                "",
                "## Args",
                *args,
                "",
                "## Impact",
                f"- {dry_run.impact}",
                f"- expected_downtime: {dry_run.expected_downtime}",
                "",
                "## Rollback",
                f"- {dry_run.rollback}",
                "",
                "## Command preview",
                *[f"- `{line}`" for line in dry_run.command_preview],
                "",
                "## Warnings",
                *[f"- {warning}" for warning in dry_run.warnings],
            ]
        )

    def _spec_line(self, spec: ActionSpec) -> str:
        return (
            f"- `{spec.operation}` risk={spec.risk_level} "
            f"approval={spec.approval_policy} read_only={spec.read_only}"
        )


def _parse_request(text: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    patterns = {
        "operation": ("operation", "op", "操作", "action"),
        "server_name": ("server", "server_name", "サーバー", "対象サーバー"),
        "service_name": ("service", "service_name", "サービス"),
        "path": ("path", "dir", "directory", "パス", "ディレクトリ"),
        "query": ("query", "検索", "search"),
        "player_name": ("player", "player_name", "mcid", "プレイヤー"),
    }
    for key, labels in patterns.items():
        value = _extract_labeled_value(text, labels)
        if value:
            payload[key] = value
    if "operation" not in payload:
        payload["operation"] = _infer_operation(text)
    return payload


def _extract_labeled_value(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        match = re.search(rf"{re.escape(label)}[:：=]\s*([^\s,、。]+)", text, re.I)
        if match:
            return match.group(1).strip()
    return None


def _infer_operation(text: str) -> str:
    lowered = text.lower()
    if "compose down" in lowered or "停止" in text:
        return "compose_down"
    if "restart" in lowered or "再起動" in text:
        return "restart"
    if "compose up" in lowered or "起動" in text:
        return "compose_up"
    if "file" in lowered or "ファイル" in text or "検索" in text:
        return "file_search"
    if "whitelist" in lowered or "ホワイトリスト" in text:
        return "whitelist_update"
    return "docker_ps"


def _impact(spec: ActionSpec) -> str:
    impacts = {
        "status": "No server impact. Reports support configuration only.",
        "docker_ps": "Read-only container status inspection.",
        "file_search": "Read-only file search if executed by a future isolated executor.",
        "compose_up": "May start containers and change server availability.",
        "compose_restart": "Restarts service containers and disconnects active players.",
        "restart": "Restarts the Minecraft server and disconnects active players.",
        "whitelist_update": "Would change player access if approved and executed later.",
        "compose_down": "Stops server containers and causes service outage.",
    }
    return impacts.get(spec.operation, "Unknown impact. Review manually.")


def _downtime(spec: ActionSpec) -> str:
    if spec.operation == "compose_down":
        return "until manually started again"
    if spec.operation in {"compose_restart", "restart"}:
        return "short outage expected"
    if spec.operation == "compose_up":
        return "startup delay possible"
    return "none"


def _rollback(spec: ActionSpec) -> str:
    rollbacks = {
        "compose_up": "Run compose down for the same service after approval if startup causes issues.",
        "compose_restart": "Review logs and restart the previous stable service after approval.",
        "restart": "Review server logs and restore latest known-good backup if needed.",
        "whitelist_update": "Apply the inverse whitelist operation after approval.",
        "compose_down": "Run compose up for the same service after approval.",
    }
    return rollbacks.get(spec.operation, "No rollback needed for read-only dry-run.")


def _command_preview(spec: ActionSpec, args: dict[str, str]) -> tuple[str, ...]:
    service = args.get("service_name", "<service>")
    path = args.get("path", "<path>")
    query = args.get("query", "<query>")
    player = args.get("player_name", "<player>")
    previews = {
        "status": ("kumc-agent mc status",),
        "docker_ps": ("docker ps --filter name=<minecraft>",),
        "file_search": (f"search_files path={path} query={query}",),
        "compose_up": (f"docker compose up -d {service}",),
        "compose_restart": (f"docker compose restart {service}",),
        "restart": (f"docker compose restart {service}",),
        "whitelist_update": (f"minecraft whitelist update {player}",),
        "compose_down": (f"docker compose down {service}",),
    }
    return previews.get(spec.operation, ("<no command preview>",))


def _join_text(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())
