from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.infra.llm.gemini import GeminiLLM
from kumc_agent.infra.indexing import llm_client as indexing_llm_client
from kumc_agent.infra.vc import llm_client as vc_llm_client


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.kwargs = kwargs


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_content(self, *, model: str, contents, config):  # type: ignore[no-untyped-def]
        kwargs = dict(getattr(config, "kwargs", {}))
        self.calls.append(
            {
                "model": model,
                "contents": list(contents),
                "config_kwargs": kwargs,
            }
        )
        return types.SimpleNamespace(text="ok")


class _RaiseDeveloperInstructionErrorOnceModels(_FakeModels):
    def generate_content(self, *, model: str, contents, config):  # type: ignore[no-untyped-def]
        kwargs = dict(getattr(config, "kwargs", {}))
        self.calls.append(
            {
                "model": model,
                "contents": list(contents),
                "config_kwargs": kwargs,
            }
        )
        if len(self.calls) == 1:
            raise RuntimeError("Developer instruction is not enabled for models/gemma-3-27b-it")
        return types.SimpleNamespace(text="ok")


def _fake_google_modules(*, models: _FakeModels) -> dict[str, object]:
    genai_module = types.ModuleType("google.genai")
    genai_module.types = types.SimpleNamespace(
        GenerateContentConfig=_FakeGenerateContentConfig,
    )
    genai_module.Client = lambda *, api_key: types.SimpleNamespace(models=models)

    google_module = types.ModuleType("google")
    google_module.genai = genai_module
    return {"google": google_module, "google.genai": genai_module}


class GeminiThinkingFallbackTests(unittest.TestCase):
    def test_gemini_llm_sends_request_without_thinking_config(self) -> None:
        models = _FakeModels()
        llm = GeminiLLM(
            api_key="dummy",
            model="gemini-2.5-flash-lite",
            requests_per_minute=60,
        )

        with patch.dict(sys.modules, _fake_google_modules(models=models), clear=False):
            with patch("kumc_agent.infra.llm.gemini.wait_for_gemini_rate_limit"):
                text = llm.generate(
                    system_prompt="system",
                    user_prompt="user",
                    temperature=0.1,
                    max_output_tokens=128,
                )

        self.assertEqual(text, "ok")
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(
            models.calls[0]["contents"],
            [{"role": "user", "parts": [{"text": "user"}]}],
        )
        self.assertEqual(models.calls[0]["config_kwargs"]["system_instruction"], "system")
        self.assertNotIn("thinking_config", models.calls[0]["config_kwargs"])

    def test_gemini_llm_uses_custom_limiter_name(self) -> None:
        models = _FakeModels()
        llm = GeminiLLM(
            api_key="dummy",
            model="gemini-2.5-flash-lite",
            requests_per_minute=60,
            limiter_name="index_summary",
        )

        with patch.dict(sys.modules, _fake_google_modules(models=models), clear=False):
            with patch(
                "kumc_agent.infra.llm.gemini.wait_for_gemini_rate_limit"
            ) as wait_mock:
                text = llm.generate(
                    system_prompt="system",
                    user_prompt="user",
                    temperature=0.1,
                    max_output_tokens=128,
                )

        self.assertEqual(text, "ok")
        self.assertEqual(wait_mock.call_count, 1)
        self.assertEqual(
            wait_mock.call_args.kwargs["max_requests_per_minute"],
            60,
        )
        self.assertEqual(wait_mock.call_args.kwargs["limiter_name"], "index_summary")

    def test_gemini_llm_retries_without_system_instruction_when_disabled(self) -> None:
        models = _RaiseDeveloperInstructionErrorOnceModels()
        llm = GeminiLLM(
            api_key="dummy",
            model="models/gemma-3-27b-it",
            requests_per_minute=60,
        )

        with patch.dict(sys.modules, _fake_google_modules(models=models), clear=False):
            with patch(
                "kumc_agent.infra.llm.gemini.wait_for_gemini_rate_limit"
            ) as wait_mock:
                text = llm.generate(
                    system_prompt="system",
                    user_prompt="user",
                    temperature=0.1,
                    max_output_tokens=128,
                )

        self.assertEqual(text, "ok")
        self.assertEqual(wait_mock.call_count, 2)
        self.assertEqual(len(models.calls), 2)
        self.assertEqual(models.calls[0]["config_kwargs"]["system_instruction"], "system")
        self.assertNotIn("system_instruction", models.calls[1]["config_kwargs"])
        retry_prompt = models.calls[1]["contents"][0]["parts"][0]["text"]  # type: ignore[index]
        self.assertIn("system", retry_prompt)
        self.assertIn("user", retry_prompt)

    def test_vc_gemini_request_has_no_thinking_config(self) -> None:
        models = _FakeModels()
        vc_llm_client._genai_client.cache_clear()

        with patch.dict(sys.modules, _fake_google_modules(models=models), clear=False):
            with patch("kumc_agent.infra.vc.llm_client.wait_for_gemini_rate_limit"):
                with patch(
                    "kumc_agent.infra.vc.llm_client._genai_client",
                    return_value=types.SimpleNamespace(models=models),
                ):
                    text = vc_llm_client.generate_text(
                        provider="gemini",
                        api_key="dummy",
                        prompt="prompt",
                        model="gemini-2.5-flash",
                        system_prompt="system",
                        llama_model_path="",
                        llama_ctx_size=4096,
                        temperature=0.2,
                        max_output_tokens=256,
                        llama_threads=4,
                        llama_gpu_layers=0,
                        response_mime_type="text/plain",
                        gemini_requests_per_minute=60,
                    )

        self.assertEqual(text, "ok")
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(
            models.calls[0]["contents"],
            [{"role": "user", "parts": [{"text": "prompt"}]}],
        )
        self.assertEqual(models.calls[0]["config_kwargs"]["system_instruction"], "system")
        self.assertNotIn("thinking_config", models.calls[0]["config_kwargs"])

    def test_indexing_gemini_request_has_no_thinking_config(self) -> None:
        models = _FakeModels()
        indexing_llm_client._genai_client.cache_clear()

        with patch.dict(sys.modules, _fake_google_modules(models=models), clear=False):
            with patch("kumc_agent.infra.indexing.llm_client.wait_for_gemini_rate_limit"):
                with patch(
                    "kumc_agent.infra.indexing.llm_client._genai_client",
                    return_value=types.SimpleNamespace(models=models),
                ):
                    text = indexing_llm_client.generate_text(
                        provider="gemini",
                        api_key="dummy",
                        prompt="prompt",
                        model="gemini-2.5-flash",
                        system_prompt="system",
                        llama_model_path="",
                        llama_ctx_size=4096,
                        temperature=0.2,
                        max_output_tokens=256,
                        llama_threads=4,
                        llama_gpu_layers=0,
                        response_mime_type="text/plain",
                        gemini_requests_per_minute=60,
                        gemini_rate_limiter_name="index_summary",
                    )

        self.assertEqual(text, "ok")
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(
            models.calls[0]["contents"],
            [{"role": "user", "parts": [{"text": "prompt"}]}],
        )
        self.assertEqual(models.calls[0]["config_kwargs"]["system_instruction"], "system")
        self.assertNotIn("thinking_config", models.calls[0]["config_kwargs"])


if __name__ == "__main__":
    unittest.main()
