from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.operations import Asset, MemberProfile
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.workflow import WorkRequest
from kumc_agent.features.workflow import WorkflowService
from kumc_agent.infra.operations import FileOperationsRepository
from kumc_agent.infra.workflow import FileWorkflowRepository


class DesignGapFoundationTests(unittest.TestCase):
    def test_file_operations_repository_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = FileOperationsRepository(root_dir=Path(tmp) / "operations")
            asset = repository.save_asset(
                Asset(
                    id="asset-1",
                    source_kind="discord",
                    title="新歓画像",
                    description="2026 新歓",
                    uri="discord://asset-1",
                    contains_people=True,
                )
            )
            profile = repository.save_member_profile(
                MemberProfile(
                    id="member-1",
                    display_name="Alice",
                    discord_user_id="alice",
                    roles=("organizer",),
                    skills=("poster",),
                )
            )

            self.assertEqual(repository.get_asset(asset.id).title, "新歓画像")
            self.assertEqual(repository.list_assets(query="新歓")[0].id, "asset-1")
            self.assertEqual(repository.search_member_profiles(query="poster")[0].id, profile.id)

    def test_image_usage_and_member_search_are_safe_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operations = FileOperationsRepository(root_dir=root / "operations")
            operations.save_asset(
                Asset(id="asset-1", title="広報写真", contains_people=True, rights_status="unknown")
            )
            service = WorkflowService(
                repository=FileWorkflowRepository(root_dir=root / "workflow"),
                operations=operations,
            )

            image = service.run(
                WorkRequest(work_type="image_search", instruction="広報")
            )
            usage = service.run(
                WorkRequest(
                    work_type="image_usage_request",
                    target="asset-1",
                    instruction="purpose: X告知 medium: X",
                    access=AccessContext(user_id="organizer", is_admin=True),
                )
            )
            denied_member = service.run(
                WorkRequest(
                    work_type="member_search",
                    instruction="poster",
                    access=AccessContext(user_id="member"),
                )
            )

            self.assertEqual(len(image.assets), 1)
            self.assertEqual(len(usage.asset_usage_requests), 1)
            self.assertTrue(usage.asset_usage_requests[0].needs_people_check)
            self.assertIn("権限", denied_member.text)

    def test_generic_approval_records_no_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = WorkflowService(
                repository=FileWorkflowRepository(root_dir=Path(tmp) / "workflow"),
                operations=FileOperationsRepository(root_dir=Path(tmp) / "operations"),
            )

            response = service.approval(
                action="approve",
                target_type="asset_usage",
                target_id="asset-usage-1",
                comment="ok",
                access=AccessContext(user_id="admin", is_admin=True),
            )

            self.assertEqual(len(response.approvals), 1)
            self.assertEqual(response.approvals[0].after["side_effects"], "none")


if __name__ == "__main__":
    unittest.main()
