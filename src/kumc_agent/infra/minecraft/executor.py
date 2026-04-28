from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
import tarfile
from typing import Protocol
from uuid import uuid4

from kumc_agent.domain.models.minecraft import (
    ServerOperation,
    ServerOperationExecutionResult,
)
from kumc_agent.features.minecraft.config import ServerManagementSettings


class CommandRunner(Protocol):
    def run(
        self,
        args: list[str],
        *,
        cwd: Path | None,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        ...


class ServerOperationExecutor(Protocol):
    def execute(self, operation: ServerOperation) -> ServerOperationExecutionResult:
        ...


@dataclass(frozen=True)
class SubprocessCommandRunner:
    def run(
        self,
        args: list[str],
        *,
        cwd: Path | None,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            shell=False,
            check=False,
            text=True,
            capture_output=True,
        )


@dataclass(frozen=True)
class ServerOperationExecutorRegistry:
    config: ServerManagementSettings
    runner: CommandRunner = SubprocessCommandRunner()

    def execute(self, operation: ServerOperation) -> ServerOperationExecutionResult:
        if operation.operation == "docker_ps":
            return DockerPsExecutor(config=self.config, runner=self.runner).execute(operation)
        if operation.operation in {"compose_up", "compose_restart", "restart", "compose_down"}:
            return ComposeExecutor(config=self.config, runner=self.runner).execute(operation)
        if operation.operation == "whitelist_update":
            return WhitelistExecutor(config=self.config, runner=self.runner).execute(operation)
        if operation.operation == "file_search":
            return FileSearchExecutor(config=self.config).execute(operation)
        if operation.operation == "backup_create":
            return BackupExecutor(config=self.config).execute(operation)
        raise ValueError(f"No executor registered for server operation: {operation.operation}")


@dataclass(frozen=True)
class DockerPsExecutor:
    config: ServerManagementSettings
    runner: CommandRunner

    def execute(self, operation: ServerOperation) -> ServerOperationExecutionResult:
        completed = self.runner.run(
            ["docker", "ps", "-a", "--format", "{{json .}}"],
            cwd=None,
            timeout=self.config.execution.timeout_seconds,
        )
        stderr = _sanitize_output(completed.stderr, self.config.execution.stderr_char_limit)
        containers = _parse_docker_ps(completed.stdout, self.config.docker_ps.container_name_prefixes)
        status = "succeeded" if completed.returncode == 0 else "failed"
        stdout = _sanitize_output(
            json.dumps(containers, ensure_ascii=False),
            self.config.execution.stdout_char_limit,
        )
        return ServerOperationExecutionResult(
            action_run_id=str(uuid4()),
            status=status,
            stdout=stdout,
            stderr=stderr,
            summary=_docker_ps_summary(containers) if status == "succeeded" else "docker ps failed",
            container_state_after={"containers": containers},
            metadata={"returncode": completed.returncode, "executor": "docker_ps"},
        )


@dataclass(frozen=True)
class ComposeExecutor:
    config: ServerManagementSettings
    runner: CommandRunner

    def execute(self, operation: ServerOperation) -> ServerOperationExecutionResult:
        server = _require_server(self.config, operation.server_name)
        if server.compose_dir is None:
            raise ValueError("compose_dir is not configured for this server.")
        args = dict(operation.dry_run.args) if operation.dry_run else {}
        service = args.get("service_name") or (server.services[0] if server.services else "")
        if operation.operation != "compose_down":
            _validate_service(server.services, service)
        command = _compose_command(operation.operation, service)
        before = _state_snapshot(
            server.compose_dir,
            service,
            runner=self.runner,
            timeout=self.config.execution.timeout_seconds,
        )
        completed = self.runner.run(
            command,
            cwd=server.compose_dir,
            timeout=self.config.execution.timeout_seconds,
        )
        after = _state_snapshot(
            server.compose_dir,
            service,
            runner=self.runner,
            timeout=self.config.execution.timeout_seconds,
        )
        stdout = _sanitize_output(completed.stdout, self.config.execution.stdout_char_limit)
        stderr = _sanitize_output(completed.stderr, self.config.execution.stderr_char_limit)
        status = "succeeded" if completed.returncode == 0 else "failed"
        return ServerOperationExecutionResult(
            action_run_id=str(uuid4()),
            status=status,
            stdout=stdout,
            stderr=stderr,
            summary=f"{operation.operation} {status}",
            server_state_before=before,
            server_state_after=after,
            container_state_before=before,
            container_state_after=after,
            metadata={
                "returncode": completed.returncode,
                "executor": "compose",
                "argv": _mask_argv(command),
            },
        )


@dataclass(frozen=True)
class WhitelistExecutor:
    config: ServerManagementSettings
    runner: CommandRunner

    def execute(self, operation: ServerOperation) -> ServerOperationExecutionResult:
        server = _require_server(self.config, operation.server_name)
        if server.compose_dir is None:
            raise ValueError("compose_dir is not configured for this server.")
        args = dict(operation.dry_run.args) if operation.dry_run else {}
        service = args.get("service_name") or (server.services[0] if server.services else "minecraft")
        player = args.get("player_name", "")
        action = args.get("whitelist_action") or "add"
        _validate_service(server.services, service)
        _validate_player_name(player)
        if action not in {"add", "remove"}:
            raise ValueError("whitelist_action must be add or remove.")
        command = ["docker", "compose", "exec", "-T", service, "whitelist", action, player]
        before = _state_snapshot(
            server.compose_dir,
            service,
            runner=self.runner,
            timeout=self.config.execution.timeout_seconds,
        )
        completed = self.runner.run(
            command,
            cwd=server.compose_dir,
            timeout=self.config.execution.timeout_seconds,
        )
        after = _state_snapshot(
            server.compose_dir,
            service,
            runner=self.runner,
            timeout=self.config.execution.timeout_seconds,
        )
        stdout = _sanitize_output(completed.stdout, self.config.execution.stdout_char_limit)
        stderr = _sanitize_output(completed.stderr, self.config.execution.stderr_char_limit)
        status = "succeeded" if completed.returncode == 0 else "failed"
        return ServerOperationExecutionResult(
            action_run_id=str(uuid4()),
            status=status,
            stdout=stdout,
            stderr=stderr,
            summary=f"whitelist {action} {status}",
            server_state_before=before,
            server_state_after=after,
            container_state_before=before,
            container_state_after=after,
            metadata={"returncode": completed.returncode, "executor": "whitelist"},
        )


@dataclass(frozen=True)
class FileSearchExecutor:
    config: ServerManagementSettings

    def execute(self, operation: ServerOperation) -> ServerOperationExecutionResult:
        server = _require_server(self.config, operation.server_name)
        args = dict(operation.dry_run.args) if operation.dry_run else {}
        relative_path = args.get("path", "")
        query = args.get("query", "")
        root = _allowed_search_root(server.allow_file_search_paths, relative_path)
        results: list[str] = []
        if root.exists():
            pattern = query.lower()
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                try:
                    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if pattern and pattern not in line.lower():
                            continue
                        excerpt = _sanitize_output(line, 240)
                        results.append(f"{path.name}:{line_no}: {excerpt}")
                        if len(results) >= 20:
                            break
                except OSError:
                    continue
                if len(results) >= 20:
                    break
        stdout = "\n".join(results)
        return ServerOperationExecutionResult(
            action_run_id=str(uuid4()),
            status="succeeded",
            stdout=stdout,
            summary=f"file search rows: {len(results)}",
            metadata={"executor": "file_search", "result_count": len(results)},
        )


@dataclass(frozen=True)
class BackupExecutor:
    config: ServerManagementSettings

    def execute(self, operation: ServerOperation) -> ServerOperationExecutionResult:
        server = _require_server(self.config, operation.server_name)
        if server.compose_dir is None:
            raise ValueError("compose_dir is not configured for this server.")
        source_dir = server.compose_dir.expanduser().resolve()
        if not source_dir.exists() or not source_dir.is_dir():
            raise ValueError("compose_dir does not exist or is not a directory.")
        backup_root = self.config.backup.backup_dir.expanduser().resolve() / server.name
        backup_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archive_path = backup_root / f"{server.name}-{timestamp}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in sorted(source_dir.rglob("*")):
                resolved = path.resolve()
                if backup_root == resolved or backup_root in resolved.parents:
                    continue
                archive.add(resolved, arcname=resolved.relative_to(source_dir))
        _prune_backups(backup_root, self.config.backup.max_backups)
        stdout = _sanitize_output(
            json.dumps(
                {
                    "backup_path": "<configured-backup>",
                    "bytes": archive_path.stat().st_size,
                },
                ensure_ascii=False,
            ),
            self.config.execution.stdout_char_limit,
        )
        return ServerOperationExecutionResult(
            action_run_id=str(uuid4()),
            status="succeeded",
            stdout=stdout,
            summary="backup archive created",
            server_state_before={"compose_dir": "<configured>"},
            server_state_after={
                "backup_path": "<configured-backup>",
                "bytes": archive_path.stat().st_size,
            },
            metadata={
                "executor": "backup",
                "backup_path": str(archive_path),
                "backup_bytes": archive_path.stat().st_size,
            },
        )


def _compose_command(operation: str, service: str) -> list[str]:
    if operation == "compose_up":
        return ["docker", "compose", "up", "-d", service]
    if operation in {"compose_restart", "restart"}:
        return ["docker", "compose", "restart", service]
    if operation == "compose_down":
        return ["docker", "compose", "down"]
    raise ValueError(f"Unsupported compose operation: {operation}")


def _require_server(config: ServerManagementSettings, name: str):
    server = config.server(name)
    if server is None:
        raise ValueError("Configured server was not found.")
    return server


def _validate_service(allowed: tuple[str, ...], service: str) -> None:
    if allowed and service not in allowed:
        raise ValueError("service_name is not allowed for this server.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", service):
        raise ValueError("service_name contains invalid characters.")


def _validate_player_name(player: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]{3,16}", player or ""):
        raise ValueError("player_name must be a valid Minecraft player name.")


def _allowed_search_root(allowed: tuple[Path, ...], relative_path: str) -> Path:
    if not allowed:
        raise ValueError("file search paths are not configured.")
    requested = Path(relative_path)
    if not requested.is_absolute() and ".." in requested.parts:
        raise ValueError("path is outside the allowed search roots.")
    for root in allowed:
        root_resolved = root.resolve()
        candidate = (
            requested.expanduser().resolve()
            if requested.is_absolute()
            else root_resolved
            if requested == Path(root.name) or str(requested) in {"", "."}
            else (root_resolved / requested).resolve()
        )
        if candidate == root_resolved or root_resolved in candidate.parents:
            return candidate
    raise ValueError("path is outside the allowed search roots.")


def _state_snapshot(
    compose_dir: Path,
    service: str,
    *,
    runner: CommandRunner,
    timeout: int,
) -> dict[str, object]:
    command = ["docker", "compose", "ps", "--format", "json"]
    if service:
        command.append(service)
    try:
        completed = runner.run(command, cwd=compose_dir, timeout=timeout)
    except Exception as exc:
        return {
            "compose_dir": "<configured>",
            "service_name": service,
            "snapshot_error": type(exc).__name__,
        }
    return {
        "compose_dir": "<configured>",
        "service_name": service,
        "returncode": completed.returncode,
        "containers": _parse_compose_ps(completed.stdout),
        "stderr_excerpt": _sanitize_output(completed.stderr, 800),
    }


def _parse_docker_ps(stdout: str, prefixes: tuple[str, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = str(item.get("Names") or item.get("Name") or "")
        if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        rows.append(
            {
                "id": str(item.get("ID") or "")[:12],
                "name": name,
                "image": _public_image_name(str(item.get("Image") or "")),
                "status": str(item.get("Status") or ""),
                "ports": _sanitize_ports(str(item.get("Ports") or "")),
                "service": _compose_service_label(item),
            }
        )
    return rows


def _parse_compose_ps(stdout: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    payload = (stdout or "").strip()
    if not payload:
        return rows
    items: list[object]
    try:
        parsed = json.loads(payload)
        items = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        items = []
        for line in payload.splitlines():
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    for raw in items:
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "id": str(raw.get("ID") or raw.get("Id") or "")[:12],
                "name": str(raw.get("Name") or raw.get("Names") or ""),
                "service": str(raw.get("Service") or _compose_service_label(raw)),
                "state": str(raw.get("State") or ""),
                "status": str(raw.get("Status") or ""),
                "health": str(raw.get("Health") or ""),
            }
        )
    return rows


def _compose_service_label(item: dict[str, object]) -> str:
    labels = item.get("Labels")
    if isinstance(labels, dict):
        return str(labels.get("com.docker.compose.service") or "")
    label_text = str(labels or "")
    for part in label_text.split(","):
        if part.startswith("com.docker.compose.service="):
            return part.split("=", 1)[1]
    return ""


def _docker_ps_summary(containers: list[dict[str, str]]) -> str:
    running = sum(1 for item in containers if "up" in item.get("status", "").lower())
    stopped = len(containers) - running
    return f"container rows: {len(containers)} / running: {running} / non_running: {stopped}"


def _public_image_name(image: str) -> str:
    return image.split("@", 1)[0][:160]


def _sanitize_ports(ports: str) -> str:
    return re.sub(r"\b(?:10|172\.(?:1[6-9]|2\d|3[0-1])|192\.168)\.\d{1,3}\.\d{1,3}:", "[internal]:", ports)


def _sanitize_output(value: str, limit: int) -> str:
    text = value or ""
    text = re.sub(r"(?i)(api[_-]?key|token|secret|password|pin)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", text)
    text = re.sub(r"\b(?:10|172\.(?:1[6-9]|2\d|3[0-1])|192\.168)\.\d{1,3}\.\d{1,3}\b", "[internal-ip]", text)
    text = re.sub(r"(?i)(network[_-]?key|unlock(?:ing)?[_ -]?steps?)\s*[:=]\s*[^\n]+", r"\1=[REDACTED]", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _mask_argv(args: list[str]) -> list[str]:
    return [_sanitize_output(arg, 240) for arg in args]


def _prune_backups(backup_root: Path, max_backups: int) -> None:
    if max_backups <= 0:
        return
    archives = sorted(
        backup_root.glob("*.tar.gz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in archives[max_backups:]:
        try:
            path.unlink()
        except OSError:
            continue
