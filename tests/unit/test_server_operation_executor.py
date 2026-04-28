from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.minecraft import MinecraftDryRun, ServerOperation
from kumc_agent.features.minecraft.config import (
    DockerPsSettings,
    ServerBackupSettings,
    ServerDefinition,
    ServerExecutionSettings,
    ServerManagementSettings,
)
from kumc_agent.infra.minecraft.executor import (
    BackupExecutor,
    ComposeExecutor,
    DockerPsExecutor,
    FileSearchExecutor,
)


@dataclass
class FakeRunner:
    responses: list[subprocess.CompletedProcess[str]]
    calls: list[tuple[list[str], Path | None, int]]

    def run(
        self,
        args: list[str],
        *,
        cwd: Path | None,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, cwd, timeout))
        if self.responses:
            return self.responses.pop(0)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="[]", stderr="")


def _operation(
    *,
    operation: str,
    server_name: str = "survival",
    args: dict[str, str] | None = None,
) -> ServerOperation:
    return ServerOperation(
        id="op1",
        server_name=server_name,
        operation=operation,
        requested_by_user_id="admin",
        dry_run=MinecraftDryRun(
            operation=operation,
            server_name=server_name,
            args=args or {},
        ),
    )


class ServerOperationExecutorTests(unittest.TestCase):
    def test_docker_ps_parses_before_output_truncation_and_keeps_service_label(self) -> None:
        stdout = "\n".join(
            [
                '{"ID":"abcdef1234567890","Names":"mc-survival","Image":"repo/image:latest","Status":"Up 1 minute","Ports":"192.168.1.10:25565->25565/tcp","Labels":"com.docker.compose.service=minecraft"}',
                '{"ID":"deadbeef00000000","Names":"other","Image":"private@sha256:abc","Status":"Exited","Ports":"","Labels":""}',
            ]
        )
        runner = FakeRunner(
            responses=[
                subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),
            ],
            calls=[],
        )
        executor = DockerPsExecutor(
            config=ServerManagementSettings(
                docker_ps=DockerPsSettings(container_name_prefixes=("mc-",)),
                execution=ServerExecutionSettings(stdout_char_limit=20),
            ),
            runner=runner,
        )

        result = executor.execute(_operation(operation="docker_ps"))

        self.assertEqual(result.status, "succeeded")
        self.assertIn("service", result.container_state_after["containers"][0])
        self.assertEqual(result.container_state_after["containers"][0]["service"], "minecraft")
        self.assertIn("[internal]", result.container_state_after["containers"][0]["ports"])

    def test_compose_down_uses_docker_compose_down_and_captures_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compose_dir = Path(tmp) / "compose"
            compose_dir.mkdir()
            runner = FakeRunner(
                responses=[
                    subprocess.CompletedProcess(args=[], returncode=0, stdout='[{"Name":"mc","Service":"minecraft","State":"running"}]', stderr=""),
                    subprocess.CompletedProcess(args=[], returncode=0, stdout="done", stderr=""),
                    subprocess.CompletedProcess(args=[], returncode=0, stdout='[{"Name":"mc","Service":"minecraft","State":"exited"}]', stderr=""),
                ],
                calls=[],
            )
            executor = ComposeExecutor(
                config=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
                            compose_dir=compose_dir,
                            services=("minecraft",),
                        ),
                    ),
                ),
                runner=runner,
            )

            result = executor.execute(_operation(operation="compose_down"))

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(runner.calls[1][0], ["docker", "compose", "down"])
        self.assertEqual(result.container_state_before["containers"][0]["state"], "running")
        self.assertEqual(result.container_state_after["containers"][0]["state"], "exited")

    def test_file_search_allows_absolute_paths_under_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            root.mkdir()
            log = root / "latest.log"
            log.write_text("token=abc 10.0.0.2 error\n", encoding="utf-8")
            executor = FileSearchExecutor(
                config=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
                            allow_file_search_paths=(root,),
                        ),
                    ),
                )
            )

            result = executor.execute(
                _operation(
                    operation="file_search",
                    args={"path": str(root), "query": "error"},
                )
            )

        self.assertEqual(result.status, "succeeded")
        self.assertIn("[REDACTED]", result.stdout)
        self.assertIn("[internal-ip]", result.stdout)

    def test_backup_executor_creates_archive_and_prunes_old_backups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            compose_dir = base / "compose"
            compose_dir.mkdir()
            (compose_dir / "server.properties").write_text("motd=test\n", encoding="utf-8")
            backup_dir = base / "backups"
            old_server_dir = backup_dir / "survival"
            old_server_dir.mkdir(parents=True)
            for index in range(3):
                (old_server_dir / f"old-{index}.tar.gz").write_text("old", encoding="utf-8")
            executor = BackupExecutor(
                config=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
                            compose_dir=compose_dir,
                            services=("minecraft",),
                        ),
                    ),
                    backup=ServerBackupSettings(backup_dir=backup_dir, max_backups=2),
                )
            )

            result = executor.execute(_operation(operation="backup_create"))

            archives = sorted((backup_dir / "survival").glob("*.tar.gz"))
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(len(archives), 2)
        self.assertIn("backup archive created", result.summary)


if __name__ == "__main__":
    unittest.main()
