from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class LayerRuleTests(unittest.TestCase):
    def test_domain_has_no_external_sdk_imports(self) -> None:
        domain_dir = SRC / "kumc_agent" / "domain"
        forbidden = ("discord", "google", "langchain", "faiss", "llama_cpp")

        for path in domain_dir.rglob("*.py"):
            content = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotRegex(
                    content,
                    re.compile(rf"(^|\n)\s*(from|import)\s+{re.escape(token)}"),
                    msg=f"Forbidden import '{token}' in {path}",
                )

    def test_frontends_do_not_import_infra(self) -> None:
        frontends_dir = SRC / "kumc_agent" / "frontends"
        for path in frontends_dir.rglob("*.py"):
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("kumc_agent.infra", content, msg=f"Frontend imports infra: {path}")

    def test_frontends_do_not_build_app_contexts(self) -> None:
        frontends_dir = SRC / "kumc_agent" / "frontends"
        forbidden = (
            "build_runtime_context",
            "build_foundation_app_context",
            "build_retrieval_app_context",
            "build_agentic_app_context",
            "build_workflow_app_context",
            "build_automation_app_context",
            "build_ingestion_app_context",
        )
        for path in frontends_dir.rglob("*.py"):
            content = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(
                    token,
                    content,
                    msg=f"Frontend builds app/runtime context: {path}",
                )


if __name__ == "__main__":
    unittest.main()
