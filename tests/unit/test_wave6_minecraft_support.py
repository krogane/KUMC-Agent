from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.config.schema import RiskFeatureFlagsSection
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.workflow import WorkRequest
from kumc_agent.features.foundation.feature_flags import FeatureFlagService
from kumc_agent.features.minecraft import MinecraftSupportService
from kumc_agent.features.minecraft.actions import MinecraftActionSpecRegistry
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


class Wave6MinecraftSupportTests(unittest.TestCase):
    def test_registry_contains_only_known_minecraft_actions(self) -> None:
        registry = MinecraftActionSpecRegistry()

        names = {spec.operation for spec in registry.list()}

        self.assertIn("docker_ps", names)
        self.assertIn("compose_restart", names)
        self.assertIn("compose_down", names)
        self.assertFalse(registry.has("rm -rf /"))

    def test_mc_request_saves_dry_run_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=root / "minecraft"),
                feature_flags=_flags(),
            )

            result = service.request(
                instruction=(
                    "operation: compose_restart server: survival "
                    "service: minecraft"
                ),
                target="",
                access=AccessContext(user_id="admin", is_admin=True),
            )

            self.assertIsNotNone(result.operation)
            self.assertEqual(result.operation.status, "waiting_approval")
            self.assertEqual(result.operation.risk_level, "high")
            self.assertFalse(result.operation.dry_run.execution_allowed)
            self.assertIn("dry-run", result.text)
            saved = service.repository.get(result.operation.id)
            self.assertEqual(saved.operation, "compose_restart")

    def test_disabled_feature_still_only_stores_disabled_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=Path(tmp) / "minecraft"),
                feature_flags=_flags("disabled"),
            )

            result = service.request(
                instruction="operation: docker_ps server: survival",
                target="",
                access=AccessContext(user_id="admin", is_admin=True),
            )

            self.assertEqual(result.operation.status, "disabled")
            self.assertFalse(result.operation.dry_run.execution_allowed)
            self.assertIn("disabled", result.text)

    def test_mc_request_rejects_missing_required_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=Path(tmp) / "minecraft"),
                feature_flags=_flags(),
            )

            with self.assertRaises(ValueError):
                service.request(
                    instruction="operation: compose_restart server: survival",
                    target="",
                    access=AccessContext(user_id="admin", is_admin=True),
                )

    def test_workflow_mc_status_and_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            minecraft = MinecraftSupportService(
                repository=FileServerOperationRepository(root_dir=root / "minecraft"),
                feature_flags=_flags(),
            )
            workflow = WorkflowService(
                repository=FileWorkflowRepository(root_dir=root / "workflow"),
                minecraft=minecraft,
            )

            status = workflow.run(WorkRequest(work_type="mc_status"))
            request = workflow.run(
                WorkRequest(
                    work_type="mc_request",
                    instruction="operation: whitelist_update player: steve server: survival",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )

            self.assertIn("dry-run-only", status.text)
            self.assertEqual(len(request.server_operations), 1)
            self.assertEqual(request.metadata["execution_allowed"], False)

    def test_migration_contains_server_operations_table(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "006_wave6_minecraft_support.sql"
        ).read_text(encoding="utf-8")

        self.assertIn("create table if not exists server_operations", sql)
        self.assertIn("dry_run jsonb", sql)


if __name__ == "__main__":
    unittest.main()
