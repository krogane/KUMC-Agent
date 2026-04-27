from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.config.schema import (
    DatabaseSection,
    ObjectStorageSection,
    RedisSection,
    RiskFeatureFlagsSection,
)
from kumc_agent.features.foundation.feature_flags import FeatureFlagService
from kumc_agent.features.foundation.health import FoundationHealthService
from kumc_agent.infra.audit.repository import FileAuditLogRepository
from kumc_agent.infra.cache.redis_client import RedisClient
from kumc_agent.infra.database.postgres import PostgresClient
from kumc_agent.infra.object_storage.s3 import S3ObjectStorageClient


class FoundationServicesTests(unittest.TestCase):
    def _flags(self) -> FeatureFlagService:
        return FeatureFlagService(
            RiskFeatureFlagsSection(
                action_execution="approval_required",
                external_posting="approval_required",
                minecraft_server_ops="approval_required",
                accounting_finalize="approval_required",
                auto_reply="approval_required",
                automation_auto_run="disabled",
                vc_recording="disabled",
                image_generation="approval_required",
            )
        )

    def test_feature_flags_block_high_risk_defaults(self) -> None:
        flags = self._flags()
        self.assertTrue(flags.requires_approval("action_execution"))
        self.assertTrue(flags.is_disabled("vc_recording"))
        self.assertIn("vc_recording", flags.disabled_flags())

    def test_health_check_writes_audit_log_with_unconfigured_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.jsonl"
            health = FoundationHealthService(
                postgres=PostgresClient(
                    DatabaseSection(url="", connect_timeout_seconds=1.0, application_name="test")
                ),
                redis=RedisClient(RedisSection(url="", socket_timeout_seconds=1.0)),
                object_storage=S3ObjectStorageClient(
                    ObjectStorageSection(
                        endpoint_url="",
                        bucket="",
                        region="ap-northeast-1",
                        access_key_id="",
                        secret_access_key="",
                        prefix="test",
                        use_ssl=True,
                    )
                ),
                audit_log=FileAuditLogRepository(path=audit_path),
                feature_flags=self._flags(),
            )

            report = health.check(actor_id="tester", actor_type="unit")

            self.assertEqual(report.status, "degraded")
            self.assertTrue(audit_path.exists())
            event = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(event["action"], "admin.health")
            self.assertEqual(event["actor_id"], "tester")

if __name__ == "__main__":
    unittest.main()
