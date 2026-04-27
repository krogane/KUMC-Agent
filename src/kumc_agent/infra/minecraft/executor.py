from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
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
        stdout = _sanitize_output(completed.stdout, self.config.execution.stdout_char_limit)
        stderr = _sanitize_output(completed.stderr, self.config.execution.stderr_char_limit)
        containers = _parse_docker_ps(stdout, self.config.docker_ps.container_name_prefixes)
        status = "succeeded" if completed.returncode == 0 else "failed"
        return ServerOperationExecutionResult(
            action_run_id=str(uuid4()),
            status=status,
            stdout=json.dumps(containers, ensure_ascii=False),
            stderr=stderr,
            summary=f"container rows: {len(containers)}" if status == "succeeded" else "docker ps failed",
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
        service = args.get("service_name") or "minecraft"
        _validate_service(server.services, service)
        command = _compose_command(operation.operation, service)
        before = _state_snapshot(server.compose_dir, service)
        completed = self.runner.run(
            command,
            cwd=server.compose_dir,
            timeout=self.config.execution.timeout_seconds,
        )
        after = _state_snapshot(server.compose_dir, service)
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
        before = _state_snapshot(server.compose_dir, service)
        completed = self.runner.run(
            command,
            cwd=server.compose_dir,
            timeout=self.config.execution.timeout_seconds,
        )
        after = _state_snapshot(server.compose_dir, service)
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


def _compose_command(operation: str, service: str) -> list[str]:
    if operation == "compose_up":
        return ["docker", "compose", "up", "-d", service]
    if operation in {"compose_restart", "restart"}:
        return ["docker", "compose", "restart", service]
    if operation == "compose_down":
        return ["docker", "compose", "stop", service]
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
    if requested.is_absolute() or ".." in requested.parts:
        raise ValueError("path is outside the allowed search roots.")
    for root in allowed:
        candidate = (root / requested).resolve() if requested != Path(root.name) else root.resolve()
        root_resolved = root.resolve()
        if candidate == root_resolved or root_resolved in candidate.parents:
            return candidate
    raise ValueError("path is outside the allowed search roots.")


def _state_snapshot(compose_dir: Path, service: str) -> dict[str, str]:
    return {"compose_dir": "<configured>", "service_name": service, "exists": str(compose_dir.exists())}


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
            }
        )
    return rows


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
