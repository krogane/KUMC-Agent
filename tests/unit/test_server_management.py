from __future__ import annotations

from dataclasses import dataclass
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.config.schema import RiskFeatureFlagsSection
from kumc_agent.domain.models.minecraft import (
    ServerOperation,
    ServerOperationExecutionResult,
)
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.workflow import WorkRequest
from kumc_agent.features.foundation.feature_flags import FeatureFlagService
from kumc_agent.features.minecraft.access import ServerManagementAccessPolicy
from kumc_agent.features.minecraft.config import (
    ServerDefinition,
    ServerManagementSettings,
)
from kumc_agent.features.minecraft.planner import ServerOperationPlanner
from kumc_agent.features.minecraft.service import MinecraftSupportService
from kumc_agent.features.workflow import WorkflowService
from kumc_agent.infra.audit import FileAuditLogRepository
from kumc_agent.infra.minecraft import FileServerOperationRepository
from kumc_agent.infra.workflow import FileWorkflowRepository


def _flags(mode: str = "approval_required") -> FeatureFlagService:
    return FeatureFlagService(
        RiskFeatureFlagsSection(
            action_execution="approval_required",
            external_posting="approval_required",
            minecraft_server_ops=mode,
            accounting_finalize="approval_required",
            auto_reply="approval_required",
            automation_auto_run="disabled",
            vc_recording="disabled",
            image_generation="approval_required",
        )
    )


@dataclass
class FakeExecutor:
    calls: list[str]
    status: str = "succeeded"

    def execute(self, operation: ServerOperation) -> ServerOperationExecutionResult:
        self.calls.append(operation.operation)
        return ServerOperationExecutionResult(
            action_run_id=f"run-{operation.id}",
            status=self.status,
            stdout="token=secret 10.0.0.1 ok",
            stderr="",
            summary="fake",
            metadata={"executor": "fake"},
        )


@dataclass
class FailingExecutor:
    def execute(self, operation: ServerOperation) -> ServerOperationExecutionResult:
        raise TimeoutError("timeout token=secret 192.168.1.1")


@dataclass
class FakePlannerLLM:
    output: str

    def generate(self, **kwargs: object) -> str:
        return self.output


class ServerManagementTests(unittest.TestCase):
    def test_non_admin_is_denied_without_internal_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=Path(tmp) / "minecraft"),
                feature_flags=_flags(),
                access_policy=ServerManagementAccessPolicy(admin_user_ids=("admin-id",)),
            )

            result = service.request(
                instruction="operation: compose_restart server: survival service: minecraft",
                target="",
                access=AccessContext(user_id="user"),
            )

            self.assertIn("権限がありません", result.text)
            self.assertEqual(result.operations, tuple())
            self.assertNotIn("survival", result.text)
            self.assertNotIn("minecraft", result.text)
            self.assertEqual(service.repository.list(), [])

    def test_planner_extracts_multiple_operations(self) -> None:
        planner = ServerOperationPlanner(
            default_server_name="survival",
            llm=FakePlannerLLM(
                output=(
                    '{"operations":['
                    '{"operation":"restart","server_name":"survival","confidence":"high"},'
                    '{"operation":"whitelist_update","server_name":"survival","player_name":"Steve","whitelist_action":"add","confidence":"high","depends_on":"previous"}'
                    ']}'
                )
            ),
        )

        plans = planner.plan("survivalを再起動して、その後whitelistにSteveを追加")

        self.assertEqual([plan.operation for plan in plans], ["restart", "whitelist_update"])
        self.assertEqual(plans[1].player_name, "Steve")

    def test_allow_list_rejects_unknown_service_without_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compose_dir = Path(tmp) / "compose"
            compose_dir.mkdir()
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=Path(tmp) / "minecraft"),
                feature_flags=_flags(),
                settings=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
                            compose_dir=compose_dir,
                            services=("minecraft",),
                        ),
                    ),
                ),
            )

            with self.assertRaises(ValueError):
                service.request(
                    instruction="operation: compose_restart server: survival service: db",
                    target="",
                    access=AccessContext(user_id="admin", is_admin=True),
                )

            self.assertEqual(service.repository.list(), [])

    def test_shell_fragment_is_rejected_without_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=Path(tmp) / "minecraft"),
                feature_flags=_flags(),
            )

            with self.assertRaises(ValueError):
                service.request(
                    instruction="rm -rf /",
                    target="",
                    access=AccessContext(user_id="admin", is_admin=True),
                )

            self.assertEqual(service.repository.list(), [])

    def test_approval_updates_server_operation_status_and_critical_needs_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose_dir = root / "compose"
            compose_dir.mkdir()
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=root / "minecraft"),
                feature_flags=_flags(),
                settings=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
                            compose_dir=compose_dir,
                            services=("minecraft",),
                            critical_operations_enabled=True,
                        ),
                    ),
                ),
            )
            workflow = WorkflowService(
                repository=FileWorkflowRepository(root_dir=root / "workflow"),
                minecraft=service,
            )
            request = workflow.run(
                WorkRequest(
                    work_type="mc_request",
                    instruction="operation: compose_down server: survival service: minecraft",
                    access=AccessContext(user_id="admin1", is_admin=True),
                )
            )
            operation_id = request.server_operations[0].id

            first = workflow.approval(
                action="approve",
                target_type="server_operation",
                target_id=operation_id,
                access=AccessContext(user_id="admin1", is_admin=True),
            )
            second = workflow.approval(
                action="approve",
                target_type="server_operation",
                target_id=operation_id,
                access=AccessContext(user_id="admin2", is_admin=True),
            )

            self.assertEqual(first.server_operations[0].status, "waiting_approval")
            self.assertEqual(second.server_operations[0].status, "approved")

    def test_execute_requires_approval_and_uses_registered_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compose_dir = Path(tmp) / "compose"
            compose_dir.mkdir()
            fake = FakeExecutor(calls=[])
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=Path(tmp) / "minecraft"),
                feature_flags=_flags(),
                settings=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
                            compose_dir=compose_dir,
                            services=("minecraft",),
                        ),
                    ),
                ),
                executor=fake,
            )
            result = service.request(
                instruction="operation: compose_restart server: survival service: minecraft",
                target="",
                access=AccessContext(user_id="admin", is_admin=True),
            )
            operation_id = result.operation.id

            with self.assertRaises(ValueError):
                service.execute(
                    operation_id=operation_id,
                    access=AccessContext(user_id="admin", is_admin=True),
                )

            service.approve(
                operation_id=operation_id,
                access=AccessContext(user_id="admin", is_admin=True),
            )
            executed = service.execute(
                operation_id=operation_id,
                access=AccessContext(user_id="admin", is_admin=True),
            )

            self.assertEqual(fake.calls, ["compose_restart"])
            self.assertEqual(executed.operation.status, "succeeded")
            self.assertIn("[internal-ip]", executed.operation.metadata["stdout_excerpt"])
            self.assertIn("[REDACTED]", executed.operation.metadata["stdout_excerpt"])

    def test_workflow_execute_audit_includes_execution_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose_dir = root / "compose"
            compose_dir.mkdir()
            fake = FakeExecutor(calls=[])
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=root / "minecraft"),
                feature_flags=_flags(),
                settings=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
                            compose_dir=compose_dir,
                            services=("minecraft",),
                        ),
                    ),
                ),
                executor=fake,
            )
            audit_path = root / "audit.jsonl"
            workflow = WorkflowService(
                repository=FileWorkflowRepository(root_dir=root / "workflow"),
                audit_log=FileAuditLogRepository(path=audit_path),
                minecraft=service,
            )
            request = workflow.run(
                WorkRequest(
                    work_type="mc_request",
                    instruction="operation: compose_restart server: survival service: minecraft",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )
            operation_id = request.server_operations[0].id
            workflow.approval(
                action="approve",
                target_type="server_operation",
                target_id=operation_id,
                access=AccessContext(user_id="admin", is_admin=True),
            )
            workflow.run(
                WorkRequest(
                    work_type="server_operation_execute",
                    target=operation_id,
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )

            events = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            execute_event = events[-1]
            metadata = execute_event["metadata"]
            self.assertEqual(execute_event["action"], "workflow.server_operation.execute")
            self.assertEqual(metadata["operation_id"], operation_id)
            self.assertEqual(metadata["approved_by_user_ids"], ["admin"])
            self.assertIn("[REDACTED]", metadata["stdout_excerpt"])
            self.assertIn("[internal-ip]", metadata["stdout_excerpt"])
            self.assertEqual(metadata["executor_summary"], "fake")

    def test_unknown_natural_language_returns_confirmation_without_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=Path(tmp) / "minecraft"),
                feature_flags=_flags(),
            )

            result = service.request(
                instruction="いい感じにサーバーを調整して",
                target="",
                access=AccessContext(user_id="admin", is_admin=True),
            )

            self.assertIn("対応操作を確認してください", result.text)
            self.assertEqual(result.operations, tuple())
            self.assertEqual(service.repository.list(), [])

    def test_unknown_labeled_operation_returns_confirmation_without_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=Path(tmp) / "minecraft"),
                feature_flags=_flags(),
            )

            result = service.request(
                instruction="operation: nginx_reload server: survival",
                target="",
                access=AccessContext(user_id="admin", is_admin=True),
            )

            self.assertIn("対応操作を確認してください", result.text)
            self.assertEqual(result.operations, tuple())
            self.assertEqual(service.repository.list(), [])

    def test_file_search_requires_approval_even_though_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search_root = root / "logs"
            search_root.mkdir()
            fake = FakeExecutor(calls=[])
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=root / "minecraft"),
                feature_flags=_flags(),
                settings=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
                            compose_dir=root / "compose",
                            services=("minecraft",),
                            allow_file_search_paths=(search_root,),
                        ),
                    ),
                ),
                executor=fake,
            )
            result = service.request(
                instruction="operation: file_search server: survival path: logs query: error",
                target="",
                access=AccessContext(user_id="admin", is_admin=True),
            )

            with self.assertRaises(ValueError):
                service.execute(
                    operation_id=result.operation.id,
                    access=AccessContext(user_id="admin", is_admin=True),
                )

            service.approve(
                operation_id=result.operation.id,
                access=AccessContext(user_id="admin", is_admin=True),
            )
            executed = service.execute(
                operation_id=result.operation.id,
                access=AccessContext(user_id="admin", is_admin=True),
            )

            self.assertEqual(executed.operation.status, "succeeded")

    def test_file_search_absolute_path_is_masked_in_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search_root = root / "logs"
            search_root.mkdir()
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=root / "minecraft"),
                feature_flags=_flags(),
                settings=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
                            compose_dir=root / "compose",
                            services=("minecraft",),
                            allow_file_search_paths=(search_root,),
                        ),
                    ),
                ),
            )
            result = service.request(
                instruction=(
                    "operation: file_search server: survival "
                    f"path: {search_root} query: token=abc"
                ),
                target="",
                access=AccessContext(user_id="admin", is_admin=True),
            )

            self.assertIn("<configured-path>", result.detail_markdown)
            self.assertIn("[REDACTED]", result.detail_markdown)
            self.assertNotIn(str(search_root), result.detail_markdown)

    def test_executor_exception_is_saved_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose_dir = root / "compose"
            compose_dir.mkdir()
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=root / "minecraft"),
                feature_flags=_flags(),
                settings=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
                            compose_dir=compose_dir,
                            services=("minecraft",),
                        ),
                    ),
                ),
                executor=FailingExecutor(),
            )
            result = service.request(
                instruction="operation: compose_restart server: survival service: minecraft",
                target="",
                access=AccessContext(user_id="admin", is_admin=True),
            )
            service.approve(
                operation_id=result.operation.id,
                access=AccessContext(user_id="admin", is_admin=True),
            )

            executed = service.execute(
                operation_id=result.operation.id,
                access=AccessContext(user_id="admin", is_admin=True),
            )

            self.assertEqual(executed.operation.status, "failed")
            self.assertIn("[internal-ip]", executed.operation.metadata["stderr_excerpt"])
            self.assertIn("[REDACTED]", executed.operation.metadata["stderr_excerpt"])


if __name__ == "__main__":
    unittest.main()
