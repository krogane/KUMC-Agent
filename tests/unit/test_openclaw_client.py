from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.infra.openclaw.client import OpenClawClient


class OpenClawClientTests(unittest.TestCase):
    def test_run_turn_success_with_json_line(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='INFO line\n{"text":"hello","route":"openclaw"}\n',
            stderr="",
        )
        with patch("kumc_agent.infra.openclaw.client.subprocess.run", return_value=completed):
            response = client.run_turn(
                query="status",
                session_id="guild:1",
                user_context={"question_author": "tester"},
            )

        self.assertTrue(response.ok)
        self.assertIsNotNone(response.result)
        assert response.result is not None
        self.assertEqual(response.result.text, "hello")
        self.assertEqual(response.result.payload.get("route"), "openclaw")

    def test_run_turn_returns_command_not_found(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        with patch(
            "kumc_agent.infra.openclaw.client.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            response = client.run_turn(query="status", session_id="guild:1")
        self.assertFalse(response.ok)
        self.assertEqual(response.failure.reason if response.failure else "", "command_not_found")

    def test_run_turn_returns_non_zero_exit(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=2,
            stdout="",
            stderr="boom",
        )
        with patch("kumc_agent.infra.openclaw.client.subprocess.run", return_value=completed):
            response = client.run_turn(query="status", session_id="guild:1")
        self.assertFalse(response.ok)
        self.assertEqual(response.failure.reason if response.failure else "", "non_zero_exit")

    def test_run_turn_retries_without_agent_when_unknown_agent(self) -> None:
        client = OpenClawClient(enabled=True, agent="assistant")
        unknown_agent = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=1,
            stdout="",
            stderr='Error: Unknown agent id "assistant"',
        )
        success = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"text":"hello from default"}\n',
            stderr="",
        )
        with patch(
            "kumc_agent.infra.openclaw.client.subprocess.run",
            side_effect=[unknown_agent, success],
        ) as run_mock:
            response = client.run_turn(query="status", session_id="guild:1")

        self.assertTrue(response.ok)
        self.assertEqual(run_mock.call_count, 2)
        first_cmd = run_mock.call_args_list[0].args[0]
        second_cmd = run_mock.call_args_list[1].args[0]
        self.assertIn("--agent", first_cmd)
        self.assertNotIn("--agent", second_cmd)

    def test_run_turn_retries_with_local_on_gateway_unavailable(self) -> None:
        client = OpenClawClient(enabled=True, agent="main")
        gateway_error = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=1,
            stdout="",
            stderr="Gateway target: ws://127.0.0.1:18789 connect ECONNREFUSED",
        )
        success = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"text":"hello from local"}\n',
            stderr="",
        )
        with patch(
            "kumc_agent.infra.openclaw.client.subprocess.run",
            side_effect=[gateway_error, success],
        ) as run_mock:
            response = client.run_turn(query="status", session_id="guild:1")

        self.assertTrue(response.ok)
        self.assertEqual(run_mock.call_count, 2)
        first_cmd = run_mock.call_args_list[0].args[0]
        second_cmd = run_mock.call_args_list[1].args[0]
        self.assertNotIn("--local", first_cmd)
        self.assertIn("--local", second_cmd)

    def test_run_turn_retries_with_local_when_gateway_warning_and_empty_response_on_zero_exit(self) -> None:
        client = OpenClawClient(enabled=True, agent="main")
        gateway_warning = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"route":"openclaw"}\n',
            stderr="Gateway agent failed; falling back to embedded: Error: gateway closed",
        )
        success = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"text":"hello from local"}\n',
            stderr="",
        )
        with patch(
            "kumc_agent.infra.openclaw.client.subprocess.run",
            side_effect=[gateway_warning, success],
        ) as run_mock:
            response = client.run_turn(query="status", session_id="guild:1")

        self.assertTrue(response.ok)
        self.assertEqual(run_mock.call_count, 2)
        first_cmd = run_mock.call_args_list[0].args[0]
        second_cmd = run_mock.call_args_list[1].args[0]
        self.assertNotIn("--local", first_cmd)
        self.assertIn("--local", second_cmd)

    def test_run_turn_does_not_retry_when_gateway_warning_has_answer(self) -> None:
        client = OpenClawClient(enabled=True, agent="main")
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"text":"answer despite warning","route":"openclaw"}\n',
            stderr="Gateway target: ws://127.0.0.1:18789",
        )
        with patch("kumc_agent.infra.openclaw.client.subprocess.run", return_value=completed) as run_mock:
            response = client.run_turn(query="status", session_id="guild:1")

        self.assertTrue(response.ok)
        self.assertEqual(run_mock.call_count, 1)

    def test_run_turn_uses_valid_default_session_id(self) -> None:
        client = OpenClawClient(enabled=True, agent="main")
        success = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"text":"ok"}\n',
            stderr="",
        )
        with patch("kumc_agent.infra.openclaw.client.subprocess.run", return_value=success) as run_mock:
            response = client.run_turn(query="status", session_id="")
        self.assertTrue(response.ok)
        cmd = run_mock.call_args.args[0]
        self.assertIn("--session-id", cmd)
        session_value = cmd[cmd.index("--session-id") + 1]
        self.assertEqual(session_value, "default")

    def test_run_turn_returns_invalid_json(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout="not-json",
            stderr="",
        )
        with patch("kumc_agent.infra.openclaw.client.subprocess.run", return_value=completed):
            response = client.run_turn(query="status", session_id="guild:1")
        self.assertFalse(response.ok)
        self.assertEqual(response.failure.reason if response.failure else "", "invalid_json")

    def test_run_turn_extracts_text_from_nested_response_content_list(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"response":{"content":[{"type":"text","text":"hello nested"}]}}\n',
            stderr="Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY",
        )
        with patch("kumc_agent.infra.openclaw.client.subprocess.run", return_value=completed):
            response = client.run_turn(query="status", session_id="guild:1")
        self.assertTrue(response.ok)
        assert response.result is not None
        self.assertEqual(response.result.text, "hello nested")

    def test_run_turn_extracts_text_from_assistant_messages(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":[{"type":"output_text","text":"assistant reply"}]}]}\n',
            stderr="",
        )
        with patch("kumc_agent.infra.openclaw.client.subprocess.run", return_value=completed):
            response = client.run_turn(query="status", session_id="guild:1")
        self.assertTrue(response.ok)
        assert response.result is not None
        self.assertEqual(response.result.text, "assistant reply")

    def test_run_turn_extracts_text_from_top_level_payloads(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"payloads":[{"text":"hello from payload"}],"meta":{"aborted":false}}\n',
            stderr="Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY",
        )
        with patch("kumc_agent.infra.openclaw.client.subprocess.run", return_value=completed):
            response = client.run_turn(query="status", session_id="guild:1")
        self.assertTrue(response.ok)
        assert response.result is not None
        self.assertEqual(response.result.text, "hello from payload")

    def test_run_turn_extracts_text_from_nested_result_payloads(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"ok":true,"result":{"payloads":[{"text":"hello from nested payload"}]}}\n',
            stderr="",
        )
        with patch("kumc_agent.infra.openclaw.client.subprocess.run", return_value=completed):
            response = client.run_turn(query="status", session_id="guild:1")
        self.assertTrue(response.ok)
        assert response.result is not None
        self.assertEqual(response.result.text, "hello from nested payload")

    def test_run_turn_extracts_text_from_embedded_json_payload(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        embedded = {
            "text": "hello from embedded json",
            "route": "rag",
            "sources": [{"id": "s1", "label": "source", "uri": "https://example.com"}],
            "routing_decision": {"target_model": "rag"},
            "fast_mode": True,
            "metadata": {"rag_query": "KUMC 活動内容"},
        }
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout=json.dumps({"payloads": [{"text": json.dumps(embedded, ensure_ascii=False)}]}, ensure_ascii=False)
            + "\n",
            stderr="",
        )
        with patch("kumc_agent.infra.openclaw.client.subprocess.run", return_value=completed):
            response = client.run_turn(query="status", session_id="guild:1")
        self.assertTrue(response.ok)
        assert response.result is not None
        self.assertEqual(response.result.text, "hello from embedded json")
        self.assertEqual(response.result.payload.get("route"), "rag")
        self.assertEqual(response.result.payload.get("sources"), embedded["sources"])
        self.assertEqual(response.result.payload.get("routing_decision"), embedded["routing_decision"])
        self.assertEqual(response.result.payload.get("fast_mode"), True)
        self.assertEqual(response.result.payload.get("metadata"), embedded["metadata"])

    def test_run_turn_writes_trace_log_and_masks_sensitive_values_when_debug_enabled(self) -> None:
        payload = {
            "text": "hello",
            "route": "openclaw",
            "intent": "search",
            "tool_selection_reason": "needs web lookup",
            "prompt_text": "reach user@example.com with sk-1234567890ABCDEFGHIJKLMNOP",
            "metadata": {
                "routing_decision": "rag",
                "api_key": "secret-value",
                "user_email": "user@example.com",
            },
        }
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout=f"{json.dumps(payload, ensure_ascii=False)}\n",
            stderr="",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "openclaw_trace.jsonl"
            with patch.dict(
                os.environ,
                {
                    "DEBUG": "true",
                    "KUMC_OPENCLAW_TRACE_LOG_PATH": str(trace_path),
                },
                clear=False,
            ):
                client = OpenClawClient(enabled=True, agent="ops")
                with patch(
                    "kumc_agent.infra.openclaw.client.subprocess.run",
                    return_value=completed,
                ):
                    response = client.run_turn(
                        query="contact user@example.com",
                        session_id="guild:1",
                        user_context={"access_token": "abc123"},
                    )

            self.assertTrue(response.ok)
            self.assertTrue(trace_path.exists())
            lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
            records = [json.loads(line) for line in lines]
            events = {str(record.get("event")) for record in records}
            self.assertIn("run_turn_start", events)
            self.assertIn("message_built", events)
            self.assertIn("run_turn_success", events)

            trace_text = trace_path.read_text(encoding="utf-8")
            self.assertNotIn("user@example.com", trace_text)
            self.assertNotIn("secret-value", trace_text)
            self.assertIn("<redacted-email>", trace_text)
            self.assertIn("***REDACTED***", trace_text)

    def test_run_turn_sets_configured_model_before_agent_turn(self) -> None:
        client = OpenClawClient(
            enabled=True,
            agent="ops",
            model="gemini/gemini-3-flash-preview",
        )
        set_model = subprocess.CompletedProcess(
            args=["openclaw", "models", "set"],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )
        turn = subprocess.CompletedProcess(
            args=["openclaw", "agent"],
            returncode=0,
            stdout='{"text":"hello"}\n',
            stderr="",
        )
        with patch(
            "kumc_agent.infra.openclaw.client.subprocess.run",
            side_effect=[set_model, turn],
        ) as run_mock:
            response = client.run_turn(query="status", session_id="guild:1")

        self.assertTrue(response.ok)
        self.assertEqual(run_mock.call_count, 2)
        first_cmd = run_mock.call_args_list[0].args[0]
        self.assertTrue(str(first_cmd[0]).endswith("openclaw"))
        self.assertEqual(
            first_cmd[1:],
            ["models", "set", "google/gemini-3-flash-preview"],
        )

    def test_run_turn_returns_failure_when_model_configuration_fails(self) -> None:
        client = OpenClawClient(
            enabled=True,
            agent="ops",
            model="gemini/gemini-3-flash-preview",
        )
        set_model_failure = subprocess.CompletedProcess(
            args=["openclaw", "models", "set"],
            returncode=1,
            stdout="",
            stderr="boom",
        )
        with patch(
            "kumc_agent.infra.openclaw.client.subprocess.run",
            return_value=set_model_failure,
        ):
            response = client.run_turn(query="status", session_id="guild:1")

        self.assertFalse(response.ok)
        self.assertEqual(
            response.failure.reason if response.failure else "",
            "model_configuration_failed",
        )

    def test_run_turn_bridges_kumc_gemini_api_key_to_openclaw_env(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"text":"ok"}\n',
            stderr="",
        )
        with patch.dict(
            "kumc_agent.infra.openclaw.client.os.environ",
            {"KUMC_GEMINI_API_KEY": "test-key"},
            clear=True,
        ):
            with patch(
                "kumc_agent.infra.openclaw.client.subprocess.run",
                return_value=completed,
            ) as run_mock:
                response = client.run_turn(query="status", session_id="guild:1")

        self.assertTrue(response.ok)
        env = run_mock.call_args.kwargs.get("env", {})
        self.assertEqual(env.get("GEMINI_API_KEY"), "test-key")

    def test_run_turn_bridges_google_api_key_to_gemini_api_key_when_missing(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"text":"ok"}\n',
            stderr="",
        )
        with patch.dict(
            "kumc_agent.infra.openclaw.client.os.environ",
            {"GOOGLE_API_KEY": "google-only-key"},
            clear=True,
        ):
            with patch(
                "kumc_agent.infra.openclaw.client.subprocess.run",
                return_value=completed,
            ) as run_mock:
                response = client.run_turn(query="status", session_id="guild:1")

        self.assertTrue(response.ok)
        env = run_mock.call_args.kwargs.get("env", {})
        self.assertEqual(env.get("GEMINI_API_KEY"), "google-only-key")
        self.assertEqual(env.get("GOOGLE_API_KEY"), "google-only-key")

    def test_run_turn_sets_project_root_and_src_in_environment(self) -> None:
        client = OpenClawClient(enabled=True, agent="ops")
        completed = subprocess.CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"text":"ok"}\n',
            stderr="",
        )
        with patch(
            "kumc_agent.infra.openclaw.client.subprocess.run",
            return_value=completed,
        ) as run_mock:
            response = client.run_turn(query="status", session_id="guild:1")

        self.assertTrue(response.ok)
        env = run_mock.call_args.kwargs.get("env", {})
        root = Path(env.get("KUMC_AGENT_PROJECT_ROOT", ""))
        src = Path(env.get("KUMC_AGENT_PROJECT_SRC", ""))
        self.assertTrue(root.is_dir())
        self.assertTrue(src.is_dir())
        self.assertEqual(src, root / "src")
        pythonpath = str(env.get("PYTHONPATH", ""))
        self.assertIn(str(src), pythonpath.split(os.pathsep))

    def test_run_turn_syncs_bootstrap_files_from_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir).resolve()
            config_dir = tmp_root / "openclaw-config"
            workspace_dir = tmp_root / "workspace"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "AGENTS.md").write_text("# Config Agents\n", encoding="utf-8")
            (config_dir / "SOUL.md").write_text("# Config Soul\n", encoding="utf-8")

            client = OpenClawClient(enabled=True, agent="ops", config_dir=config_dir)
            agents_list = subprocess.CompletedProcess(
                args=["openclaw", "agents", "list", "--json"],
                returncode=0,
                stdout=json.dumps(
                    [{"id": "ops", "workspace": str(workspace_dir), "isDefault": False}]
                ),
                stderr="",
            )
            turn = subprocess.CompletedProcess(
                args=["openclaw"],
                returncode=0,
                stdout='{"text":"ok"}\n',
                stderr="",
            )
            with patch(
                "kumc_agent.infra.openclaw.client.subprocess.run",
                side_effect=[agents_list, turn],
            ) as run_mock:
                response = client.run_turn(query="status", session_id="guild:1")
            self.assertTrue(response.ok)
            self.assertEqual(run_mock.call_count, 2)
            first_cmd = run_mock.call_args_list[0].args[0]
            self.assertEqual(first_cmd[1:], ["agents", "list", "--json"])
            self.assertEqual(
                (workspace_dir / "AGENTS.md").read_text(encoding="utf-8"),
                "# Config Agents\n",
            )
            self.assertEqual(
                (workspace_dir / "SOUL.md").read_text(encoding="utf-8"),
                "# Config Soul\n",
            )

    def test_run_turn_falls_back_when_config_dir_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_dir = Path(tmp_dir) / "missing"
            client = OpenClawClient(enabled=True, agent="ops", config_dir=missing_dir)
            completed = subprocess.CompletedProcess(
                args=["openclaw"],
                returncode=0,
                stdout='{"text":"ok"}\n',
                stderr="",
            )
            with patch(
                "kumc_agent.infra.openclaw.client.subprocess.run",
                return_value=completed,
            ) as run_mock:
                response = client.run_turn(query="status", session_id="guild:1")

        self.assertTrue(response.ok)
        self.assertEqual(run_mock.call_count, 1)
        cmd = run_mock.call_args.args[0]
        self.assertIn("agent", cmd)


if __name__ == "__main__":
    unittest.main()
