from __future__ import annotations

from dataclasses import dataclass
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
        planner = ServerOperationPlanner(default_server_name="survival")

        plans = planner.plan("survivalを再起動して、その後whitelistにSteveを追加")

        self.assertEqual([plan.operation for plan in plans], ["restart", "whitelist_update"])
        self.assertEqual(plans[1].player_name, "Steve")

    def test_allow_list_rejects_unknown_service_without_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=Path(tmp) / "minecraft"),
                feature_flags=_flags(),
                settings=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
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
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=root / "minecraft"),
                feature_flags=_flags(),
                settings=ServerManagementSettings(
                    default_server_name="survival",
                    servers=(
                        ServerDefinition(
                            name="survival",
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
            fake = FakeExecutor(calls=[])
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=Path(tmp) / "minecraft"),
                feature_flags=_flags(),
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


if __name__ == "__main__":
    unittest.main()
