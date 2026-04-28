from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.domain.models.minecraft import (
    ActionSpec,
    MinecraftDryRun,
    ServerOperation,
    ServerOperationExecutionResult,
    ServerOperationPlan,
)
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.features.foundation.feature_flags import FeatureFlagService
from kumc_agent.features.minecraft.access import ServerManagementAccessPolicy
from kumc_agent.features.minecraft.actions import MinecraftActionSpecRegistry
from kumc_agent.features.minecraft.config import (
    ServerBackupSettings,
    DockerPsSettings,
    ServerDefinition,
    ServerExecutionSettings,
    ServerManagementSettings,
)
from kumc_agent.features.minecraft.planner import (
    ServerOperationPlanner,
    UnsupportedServerOperationError,
)
from kumc_agent.infra.minecraft.repository import ServerOperationRepository
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class MinecraftSupportResult:
    text: str
    detail_markdown: str
    operation: ServerOperation | None = None
    operations: tuple[ServerOperation, ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] | None = None


class MinecraftSupportService:
    def __init__(
        self,
        *,
        repository: ServerOperationRepository,
        feature_flags: FeatureFlagService,
        registry: MinecraftActionSpecRegistry | None = None,
        access_policy: ServerManagementAccessPolicy | None = None,
        settings: ServerManagementSettings | None = None,
        executor: Any | None = None,
        llm: LLMPort | None = None,
        prompts_dir: Path | None = None,
    ) -> None:
        self.repository = repository
        self.feature_flags = feature_flags
        self.registry = registry or MinecraftActionSpecRegistry()
        self.access_policy = access_policy or ServerManagementAccessPolicy()
        self.settings = settings or ServerManagementSettings()
        self.planner = ServerOperationPlanner(
            default_server_name=self.settings.default_server_name,
            llm=llm,
            prompts_dir=prompts_dir,
        )
        self.executor = executor

    def status(self, *, access: AccessContext) -> MinecraftSupportResult:
        if not self.access_policy.is_admin(access):
            return MinecraftSupportResult(
                text=self.access_policy.forbidden_text(),
                detail_markdown="",
                metadata=self.access_policy.forbidden_metadata(),
            )
        mode = self.feature_flags.mode_for("minecraft_server_ops")
        pending = self.repository.list(status="waiting_approval")
        detail = "\n".join(
            [
                "# Minecraft support status",
                "",
                f"- feature_mode: `{mode}`",
                f"- execution: `{_execution_mode_text(mode, self.executor is not None)}`",
                f"- waiting_approval: {len(pending)}",
                "",
                "## Allowed action specs",
                *[self._spec_line(spec) for spec in self.registry.list()],
                "",
                "## Safety",
                "- 任意 shell command は受け付けません。",
                "- 副作用操作は承認前に実行しません。",
                "- executor は登録済み ActionSpec だけを実行します。",
            ]
        )
        return MinecraftSupportResult(
            text=(
                f"Minecraft support: mode={mode}, "
                f"execution={_execution_mode_text(mode, self.executor is not None)}, "
                f"waiting_approval={len(pending)}"
            ),
            detail_markdown=detail,
        )

    def request(
        self,
        *,
        instruction: str,
        target: str,
        access: AccessContext,
    ) -> MinecraftSupportResult:
        if not self.access_policy.is_admin(access):
            return MinecraftSupportResult(
                text=self.access_policy.forbidden_text(),
                detail_markdown="",
                metadata=self.access_policy.forbidden_metadata(),
            )
        request_text = _join_text(instruction, target)
        if _looks_like_shell(request_text):
            raise ValueError("Unsupported Minecraft operation: shell fragments are not accepted")
        try:
            plans = self.planner.plan(request_text)
        except UnsupportedServerOperationError:
            return MinecraftSupportResult(
                text="対応操作を確認してください。",
                detail_markdown="ServerOperation は作成していません。",
                metadata={"candidate_created": False, "unsupported": True},
            )
        mode = self.feature_flags.mode_for("minecraft_server_ops")
        stored_operations: list[ServerOperation] = []
        for index, plan in enumerate(plans):
            operation = _normalize_operation(plan.operation)
            if operation == "unsupported":
                return MinecraftSupportResult(
                    text="対応操作を確認してください。",
                    detail_markdown="ServerOperation は作成していません。",
                    metadata={"candidate_created": False, "unsupported": True},
                )
            if not self.registry.has(operation):
                return MinecraftSupportResult(
                    text="対応操作を確認してください。",
                    detail_markdown="ServerOperation は作成していません。",
                    metadata={
                        "candidate_created": False,
                        "unsupported": True,
                        "operation": operation,
                    },
                )
            spec = self.registry.get(operation)
            args = self._validated_args(plan=plan, spec=spec)
            missing = [name for name in spec.required_args if not args.get(name)]
            if missing:
                raise ValueError(
                    f"Missing required Minecraft operation args: {', '.join(missing)}"
                )
            dry_run = self._dry_run(spec=spec, args=args, mode=mode)
            status = _initial_status(mode=mode, spec=spec, args=args)
            server_operation = ServerOperation(
                id=stable_hash(
                    f"server-operation:{dry_run.server_name}:{dry_run.operation}:{args}:{access.user_id}:{index}"
                )[:32],
                server_name=dry_run.server_name,
                operation=dry_run.operation,
                requested_by_user_id=access.user_id,
                status=status,
                risk_level=dry_run.risk_level,
                dry_run=dry_run,
                metadata={
                    "feature_mode": mode,
                    "execution": "not_executed",
                    "requires_admin": True,
                    "planner": {
                        "confidence": plan.confidence,
                        **dict(plan.metadata),
                    },
                    "sequence_index": index,
                    "depends_on": plan.metadata.get("depends_on", ""),
                    "executor_name": spec.executor_name,
                },
            )
            stored = self.repository.save(server_operation)
            if _can_execute_without_approval(spec) and mode != "disabled":
                stored = self.execute(operation_id=stored.id, access=access).operation or stored
            stored_operations.append(stored)
        detail = "\n\n".join(self._operation_detail(operation) for operation in stored_operations)
        first = stored_operations[0] if stored_operations else None
        if any(operation.status == "disabled" for operation in stored_operations):
            text = f"Minecraft operation は disabled です。dry-run だけ保存しました: {len(stored_operations)} 件"
        elif any(operation.status in {"succeeded", "failed"} for operation in stored_operations):
            text = f"Minecraft read-only operation を実行しました: {len(stored_operations)} 件"
        else:
            text = f"Minecraft operation dry-run を保存しました: {len(stored_operations)} 件 / approval required"
        return MinecraftSupportResult(
            text=text,
            detail_markdown=detail,
            operation=first,
            operations=tuple(stored_operations),
            warnings=tuple(
                warning
                for operation in stored_operations
                for warning in (operation.dry_run.warnings if operation.dry_run else tuple())
            ),
            metadata={"operation_count": len(stored_operations)},
        )

    def list_pending(self, *, access: AccessContext) -> MinecraftSupportResult:
        if not self.access_policy.is_admin(access):
            return MinecraftSupportResult(
                text=self.access_policy.forbidden_text(),
                detail_markdown="",
                metadata=self.access_policy.forbidden_metadata(),
            )
        operations = tuple(self.repository.list_pending_for_approval())
        return MinecraftSupportResult(
            text=f"承認待ち ServerOperation は {len(operations)} 件です。",
            detail_markdown=self._operations_list_detail(operations),
            operation=operations[0] if operations else None,
            operations=operations,
        )

    def show(self, *, operation_id: str, access: AccessContext) -> MinecraftSupportResult:
        if not self.access_policy.is_admin(access):
            return MinecraftSupportResult(
                text=self.access_policy.forbidden_text(),
                detail_markdown="",
                metadata=self.access_policy.forbidden_metadata(),
            )
        operation = self.repository.get(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        return MinecraftSupportResult(
            text=f"ServerOperation を表示します: {operation.id}",
            detail_markdown=self._operation_detail(operation),
            operation=operation,
            operations=(operation,),
        )

    def approve(
        self,
        *,
        operation_id: str,
        access: AccessContext,
    ) -> MinecraftSupportResult:
        if not self.access_policy.is_admin(access):
            return MinecraftSupportResult(
                text=self.access_policy.forbidden_text(),
                detail_markdown="",
                metadata=self.access_policy.forbidden_metadata(),
            )
        operation = self.repository.get(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        if operation.status not in {"waiting_approval", "approved"}:
            raise ValueError(f"ServerOperation is not approvable: {operation.status}")
        approved = self.repository.add_approval(operation_id, access.user_id)
        next_status = self._approval_status(approved)
        approved = self.repository.update_status(
            operation_id,
            next_status,
            {"approval_policy_result": next_status, "last_approved_by": access.user_id},
        )
        return MinecraftSupportResult(
            text=f"ServerOperation approval を反映しました: {approved.id} / status={approved.status}",
            detail_markdown=self._operation_detail(approved),
            operation=approved,
            operations=(approved,),
        )

    def reject(
        self,
        *,
        operation_id: str,
        access: AccessContext,
        comment: str = "",
    ) -> MinecraftSupportResult:
        if not self.access_policy.is_admin(access):
            return MinecraftSupportResult(
                text=self.access_policy.forbidden_text(),
                detail_markdown="",
                metadata=self.access_policy.forbidden_metadata(),
            )
        operation = self.repository.update_status(
            operation_id,
            "rejected",
            {"rejected_by": access.user_id, "rejection_comment": comment},
        )
        return MinecraftSupportResult(
            text=f"ServerOperation を却下しました: {operation.id}",
            detail_markdown=self._operation_detail(operation),
            operation=operation,
            operations=(operation,),
        )

    def execute(self, *, operation_id: str, access: AccessContext) -> MinecraftSupportResult:
        if not self.access_policy.is_admin(access):
            return MinecraftSupportResult(
                text=self.access_policy.forbidden_text(),
                detail_markdown="",
                metadata=self.access_policy.forbidden_metadata(),
            )
        operation = self.repository.get(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        spec = self.registry.get(operation.operation)
        if self.feature_flags.mode_for("minecraft_server_ops") == "disabled":
            raise ValueError("minecraft_server_ops is disabled.")
        if operation.status in {"rejected", "disabled", "cancelled", "failed", "succeeded"}:
            raise ValueError(f"ServerOperation is not executable: {operation.status}")
        if _operation_requires_approval(spec) and operation.status != "approved":
            raise ValueError("ServerOperation must be approved before execution.")
        if spec.risk_level == "critical" and len(operation.approved_by_user_ids) < 2:
            raise ValueError("Critical ServerOperation requires two approvals.")
        self.repository.update_status(
            operation_id,
            "running",
            {"executed_by": access.user_id},
        )
        if self.executor is None:
            result = ServerOperationExecutionResult(
                action_run_id=stable_hash(f"noop:{operation.id}")[:32],
                status="failed",
                stderr="server operation executor is not configured",
                summary="executor not configured",
                metadata={"executor": "none"},
            )
        else:
            try:
                result = self.executor.execute(operation)
            except Exception as exc:
                result = ServerOperationExecutionResult(
                    action_run_id=stable_hash(f"failed:{operation.id}:{type(exc).__name__}")[:32],
                    status="failed",
                    stderr=str(exc),
                    summary=f"{operation.operation} failed before completion",
                    metadata={
                        "executor": spec.executor_name or operation.operation,
                        "exception_type": type(exc).__name__,
                    },
                )
        stored = self.repository.save_execution_result(operation_id, result)
        return MinecraftSupportResult(
            text=f"ServerOperation を実行しました: {stored.id} / status={stored.status}",
            detail_markdown=self._operation_detail(stored),
            operation=stored,
            operations=(stored,),
            metadata={"action_run_id": stored.action_run_id or ""},
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

    def _validated_args(self, *, plan: ServerOperationPlan, spec: ActionSpec) -> dict[str, str]:
        if _looks_like_shell(plan.operation) or any(
            _looks_like_shell(value)
            for value in (
                plan.server_name,
                plan.service_name,
                plan.path,
                plan.query,
                plan.player_name,
            )
        ):
            raise ValueError("Unsupported Minecraft operation: shell fragments are not accepted")
        server_name = plan.server_name or self.settings.default_server_name
        server = self.settings.server(server_name)
        if _requires_configured_server(spec) and server is None:
            raise ValueError("server_name is not allowed.")
        if self.settings.has_server_allow_list() and server is None:
            raise ValueError("server_name is not allowed.")
        args: dict[str, str] = {"server_name": server_name}
        if plan.service_name:
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", plan.service_name):
                raise ValueError("service_name contains invalid characters.")
            if _requires_service_allow_list(spec) and (server is None or not server.services):
                raise ValueError("service allow list is not configured for this server.")
            if server and server.services and plan.service_name not in server.services:
                raise ValueError("service_name is not allowed for this server.")
            args["service_name"] = plan.service_name
        elif spec.operation == "restart" and server and server.services:
            args["service_name"] = server.services[0]
        elif _requires_service_arg(spec) and "service_name" not in args:
            raise ValueError("Missing required Minecraft operation args: service_name")
        elif _requires_service_allow_list(spec) and (server is None or not server.services):
            raise ValueError("service allow list is not configured for this server.")
        if spec.operation in {"compose_up", "compose_restart", "restart", "whitelist_update", "backup_create", "compose_down"}:
            if server is None or server.compose_dir is None:
                raise ValueError("compose_dir is not configured for this server.")
        if plan.path:
            if server is None or not server.allow_file_search_paths:
                raise ValueError("file search path is not configured for this server.")
            if not _path_is_allowed(plan.path, server.allow_file_search_paths):
                raise ValueError("path is outside the allowed search roots.")
            args["path"] = plan.path
        elif spec.operation == "file_search":
            if server is None or not server.allow_file_search_paths:
                raise ValueError("file search path is not configured for this server.")
        if plan.query:
            args["query"] = plan.query
        if plan.player_name:
            if not re.fullmatch(r"[A-Za-z0-9_]{3,16}", plan.player_name):
                raise ValueError("player_name must be a valid Minecraft player name.")
            args["player_name"] = plan.player_name
        if plan.whitelist_action:
            action = plan.whitelist_action.lower()
            if action not in {"add", "remove"}:
                raise ValueError("whitelist_action must be add or remove.")
            args["whitelist_action"] = action
        elif spec.operation == "whitelist_update":
            args["whitelist_action"] = "add"
        if spec.operation == "compose_down" and server and not server.critical_operations_enabled:
            args["critical_operations_enabled"] = "false"
        return args

    def _approval_status(self, operation: ServerOperation) -> str:
        spec = self.registry.get(operation.operation)
        if spec.risk_level == "critical":
            server = self.settings.server(operation.server_name)
            if server and not server.critical_operations_enabled:
                return "disabled"
            return "approved" if len(operation.approved_by_user_ids) >= 2 else "waiting_approval"
        return "approved"

    def _operation_detail(self, operation: ServerOperation) -> str:
        dry_run = operation.dry_run
        if dry_run is None:
            return f"# Minecraft ServerOperation\n\n- id: `{operation.id}`"
        args = [
            f"- {key}: `{_display_arg(key, value)}`"
            for key, value in dry_run.args.items()
        ] or ["- none"]
        return "\n".join(
            [
                "# Minecraft ServerOperation dry-run",
                "",
                f"- id: `{operation.id}`",
                f"- status: `{operation.status}`",
                f"- server: `{operation.server_name}`",
                f"- operation: `{operation.operation}`",
                f"- risk: `{operation.risk_level}`",
                f"- approved_by: `{', '.join(operation.approved_by_user_ids) or 'none'}`",
                f"- action_run_id: `{operation.action_run_id or ''}`",
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

    def _operations_list_detail(self, operations: tuple[ServerOperation, ...]) -> str:
        if not operations:
            return "ServerOperation はありません。"
        return "\n".join(
            [
                "# ServerOperation",
                *[
                    (
                        f"- `{operation.id}` server={operation.server_name} "
                        f"operation={operation.operation} risk={operation.risk_level} "
                        f"status={operation.status}"
                    )
                    for operation in operations
                ],
            ]
        )

    def _spec_line(self, spec: ActionSpec) -> str:
        return (
            f"- `{spec.operation}` risk={spec.risk_level} "
            f"approval={spec.approval_policy} read_only={spec.read_only}"
        )


def _impact(spec: ActionSpec) -> str:
    impacts = {
        "status": "No server impact. Reports support configuration only.",
        "docker_ps": "Read-only container status inspection.",
        "file_search": "Read-only file search if executed by a future isolated executor.",
        "compose_up": "May start containers and change server availability.",
        "compose_restart": "Restarts service containers and disconnects active players.",
        "restart": "Restarts the Minecraft server and disconnects active players.",
        "whitelist_update": "Would change player access if approved and executed later.",
        "backup_create": "Creates a local backup archive for the configured server directory.",
        "compose_down": "Stops server containers and causes service outage.",
    }
    return impacts.get(spec.operation, "Unknown impact. Review manually.")


def _downtime(spec: ActionSpec) -> str:
    if spec.operation == "compose_down":
        return "until manually started again"
    if spec.operation == "backup_create":
        return "none unless disk pressure affects the host"
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
        "backup_create": "Delete the generated backup archive if it is invalid or causes storage pressure.",
        "compose_down": "Run compose up for the same service after approval.",
    }
    return rollbacks.get(spec.operation, "No rollback needed for read-only dry-run.")


def _command_preview(spec: ActionSpec, args: dict[str, str]) -> tuple[str, ...]:
    service = args.get("service_name", "<service>")
    query = _display_arg("query", args.get("query", "<query>"))
    player = _display_arg("player_name", args.get("player_name", "<player>"))
    previews = {
        "status": ("kumc-agent mc status",),
        "docker_ps": ("docker ps --filter name=<minecraft>",),
        "file_search": (f"search_files path=<allowed-path> query={query}",),
        "compose_up": (f"docker compose up -d {service}",),
        "compose_restart": (f"docker compose restart {service}",),
        "restart": (f"docker compose restart {service}",),
        "whitelist_update": (f"minecraft whitelist update {player}",),
        "backup_create": ("create_backup server=<configured>",),
        "compose_down": ("docker compose down",),
    }
    return previews.get(spec.operation, ("<no command preview>",))


def _join_text(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _normalize_operation(value: str) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    aliases = {
        "mc_status": "status",
        "docker": "docker_ps",
        "ps": "docker_ps",
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


def _initial_status(*, mode: str, spec: ActionSpec, args: dict[str, str]) -> str:
    if mode == "disabled":
        return "disabled"
    if spec.operation == "compose_down" and args.get("critical_operations_enabled") == "false":
        return "disabled"
    if _can_execute_without_approval(spec):
        return "running"
    return "waiting_approval"


def _looks_like_shell(value: str) -> bool:
    return bool(re.search(r"(\brm\s+-rf\b|[;&|`$<>]|\bsh\s+-c\b|\bbash\s+-c\b)", value or ""))


def _execution_mode_text(mode: str, executor_configured: bool) -> str:
    if mode == "disabled":
        return "disabled"
    if not executor_configured:
        return "executor_not_configured"
    return "read_only_immediate_and_approval_gated_writes"


def _can_execute_without_approval(spec: ActionSpec) -> bool:
    return spec.read_only and spec.approval_policy in {"self", "admin"}


def _operation_requires_approval(spec: ActionSpec) -> bool:
    if spec.risk_level in {"high", "critical"}:
        return True
    return spec.approval_policy in {
        "admin_dry_run",
        "admin_approval",
        "two_person_or_disabled",
    }


def _requires_configured_server(spec: ActionSpec) -> bool:
    return spec.operation not in {"status", "docker_ps"}


def _requires_service_allow_list(spec: ActionSpec) -> bool:
    return spec.operation in {
        "compose_up",
        "compose_restart",
        "restart",
        "whitelist_update",
    }


def _requires_service_arg(spec: ActionSpec) -> bool:
    return "service_name" in spec.required_args and spec.operation != "restart"


def _path_is_allowed(path: str, allowed_roots: tuple[Path, ...]) -> bool:
    requested = Path(path)
    if not requested.is_absolute() and ".." in requested.parts:
        return False
    for root in allowed_roots:
        root_resolved = root.expanduser().resolve()
        candidate = (
            requested.expanduser().resolve()
            if requested.is_absolute()
            else root_resolved
            if requested == Path(root.name) or str(requested) in {"", "."}
            else (root_resolved / requested).resolve()
        )
        if candidate == root_resolved or root_resolved in candidate.parents:
            return True
    return False


def _display_arg(key: str, value: object) -> str:
    text = _mask_sensitive_text(str(value))
    if key in {"path", "server_dir", "compose_dir", "backup_path", "backup_dir"}:
        if text.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", text):
            return "<configured-path>"
    return text


def _mask_sensitive_text(value: str) -> str:
    text = re.sub(
        r"(?i)(api[_-]?key|token|secret|password|pin)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        value or "",
    )
    text = re.sub(
        r"\b(?:10|172\.(?:1[6-9]|2\d|3[0-1])|192\.168)\.\d{1,3}\.\d{1,3}\b",
        "[internal-ip]",
        text,
    )
    return re.sub(
        r"(?i)(network[_-]?key|unlock(?:ing)?[_ -]?steps?)\s*[:=]\s*[^\n]+",
        r"\1=[REDACTED]",
        text,
    )


def settings_from_runtime(config: object) -> ServerManagementSettings:
    return ServerManagementSettings(
        default_server_name=str(getattr(config, "default_server_name", "default")),
        docker_ps=DockerPsSettings(
            container_name_prefixes=tuple(
                str(value)
                for value in getattr(getattr(config, "docker_ps", object()), "container_name_prefixes", [])
            ),
        ),
        servers=tuple(
            ServerDefinition(
                name=str(getattr(server, "name")),
                compose_dir=getattr(server, "compose_dir", None),
                services=tuple(str(value) for value in getattr(server, "services", [])),
                allow_file_search_paths=tuple(
                    getattr(server, "allow_file_search_paths", [])
                ),
                critical_operations_enabled=bool(
                    getattr(server, "critical_operations_enabled", False)
                ),
            )
            for server in getattr(config, "servers", [])
        ),
        execution=ServerExecutionSettings(
            timeout_seconds=int(
                getattr(getattr(config, "execution", object()), "timeout_seconds", 120)
            ),
            stdout_char_limit=int(
                getattr(getattr(config, "execution", object()), "stdout_char_limit", 4000)
            ),
            stderr_char_limit=int(
                getattr(getattr(config, "execution", object()), "stderr_char_limit", 4000)
            ),
        ),
        backup=ServerBackupSettings(
            backup_dir=Path(
                getattr(
                    getattr(config, "backup", object()),
                    "backup_dir",
                    Path("data/minecraft/backups"),
                )
            ),
            max_backups=int(
                getattr(getattr(config, "backup", object()), "max_backups", 10)
            ),
        ),
    )
