from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.cli import _workflow_response_payload
from kumc_agent.domain.models.minecraft import MinecraftDryRun, ServerOperation
from kumc_agent.domain.models.workflow import WorkResponse


class CliServerManagementPayloadTests(unittest.TestCase):
    def test_server_operation_payload_masks_sensitive_metadata(self) -> None:
        response = WorkResponse(
            text="done",
            server_operations=(
                ServerOperation(
                    id="op1",
                    server_name="survival",
                    operation="docker_ps",
                    requested_by_user_id="admin",
                    dry_run=MinecraftDryRun(
                        operation="docker_ps",
                        server_name="survival",
                        args={"path": "/srv/private/logs", "query": "token=abc"},
                    ),
                    metadata={
                        "stdout_excerpt": "token=abc 192.168.1.10",
                        "server_state_before": {"compose_dir": "/srv/private"},
                        "trace_id": "trace-1",
                    },
                ),
            ),
            metadata={"routing_decision": "server", "secret": "value"},
        )

        payload = _workflow_response_payload(response)

        operation = payload["server_operations"][0]
        self.assertEqual(payload["metadata"], {"routing_decision": "server"})
        self.assertIn("[REDACTED]", operation["metadata"]["stdout_excerpt"])
        self.assertIn("[internal-ip]", operation["metadata"]["stdout_excerpt"])
        self.assertNotIn("server_state_before", operation["metadata"])
        self.assertEqual(operation["dry_run"]["args"]["path"], "<configured-path>")
        self.assertIn("[REDACTED]", operation["dry_run"]["args"]["query"])
        self.assertIsInstance(operation["dry_run"], dict)


if __name__ == "__main__":
    unittest.main()
