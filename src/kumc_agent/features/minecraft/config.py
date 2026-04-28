from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ServerDefinition:
    name: str
    compose_dir: Path | None = None
    services: tuple[str, ...] = tuple()
    allow_file_search_paths: tuple[Path, ...] = tuple()
    critical_operations_enabled: bool = False


@dataclass(frozen=True)
class DockerPsSettings:
    container_name_prefixes: tuple[str, ...] = tuple()


@dataclass(frozen=True)
class ServerExecutionSettings:
    timeout_seconds: int = 120
    stdout_char_limit: int = 4000
    stderr_char_limit: int = 4000


@dataclass(frozen=True)
class ServerBackupSettings:
    backup_dir: Path = Path("data/minecraft/backups")
    max_backups: int = 10


@dataclass(frozen=True)
class ServerManagementSettings:
    default_server_name: str = "default"
    docker_ps: DockerPsSettings = field(default_factory=DockerPsSettings)
    servers: tuple[ServerDefinition, ...] = tuple()
    execution: ServerExecutionSettings = field(default_factory=ServerExecutionSettings)
    backup: ServerBackupSettings = field(default_factory=ServerBackupSettings)

    def server(self, name: str) -> ServerDefinition | None:
        normalized = (name or self.default_server_name).strip()
        for server in self.servers:
            if server.name == normalized:
                return server
        return None

    def has_server_allow_list(self) -> bool:
        return bool(self.servers)
