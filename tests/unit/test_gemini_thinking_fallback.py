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
from kumc_agent.infra.vc import llm_client as vc_llm_client


class _FakeThinkingConfig:
    def __init__(self, *, thinking_level: str) -> None:
        self.thinking_level = thinking_level


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
        if "thinking_config" in kwargs:
            raise RuntimeError("Thinking level is not supported for this model.")
        return types.SimpleNamespace(text="ok")


def _fake_google_modules(*, models: _FakeModels) -> dict[str, object]:
    genai_module = types.ModuleType("google.genai")
    genai_module.types = types.SimpleNamespace(
        ThinkingConfig=_FakeThinkingConfig,
        GenerateContentConfig=_FakeGenerateContentConfig,
    )
    genai_module.Client = lambda *, api_key: types.SimpleNamespace(models=models)

    google_module = types.ModuleType("google")
    google_module.genai = genai_module
    return {"google": google_module, "google.genai": genai_module}


class GeminiThinkingFallbackTests(unittest.TestCase):
    def test_gemini_llm_skips_thinking_for_unsupported_model(self) -> None:
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
                    thinking_level="minimal",
                )

        self.assertEqual(text, "ok")
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(
            models.calls[0]["contents"],
            [{"role": "user", "parts": [{"text": "user"}]}],
        )
        self.assertEqual(models.calls[0]["config_kwargs"]["system_instruction"], "system")
        self.assertNotIn("thinking_config", models.calls[0]["config_kwargs"])

    def test_gemini_llm_retries_without_thinking_on_error(self) -> None:
        models = _FakeModels()
        llm = GeminiLLM(
            api_key="dummy",
            model="gemini-2.5-flash",
            requests_per_minute=60,
        )

        with patch.dict(sys.modules, _fake_google_modules(models=models), clear=False):
            with patch("kumc_agent.infra.llm.gemini.wait_for_gemini_rate_limit"):
                text = llm.generate(
                    system_prompt="system",
                    user_prompt="user",
                    temperature=0.1,
                    max_output_tokens=128,
                    thinking_level="minimal",
                )

        self.assertEqual(text, "ok")
        self.assertEqual(len(models.calls), 2)
        self.assertEqual(
            models.calls[0]["contents"],
            [{"role": "user", "parts": [{"text": "user"}]}],
        )
        self.assertEqual(
            models.calls[1]["contents"],
            [{"role": "user", "parts": [{"text": "user"}]}],
        )
        self.assertEqual(models.calls[0]["config_kwargs"]["system_instruction"], "system")
        self.assertEqual(models.calls[1]["config_kwargs"]["system_instruction"], "system")
        self.assertIn("thinking_config", models.calls[0]["config_kwargs"])
        self.assertNotIn("thinking_config", models.calls[1]["config_kwargs"])

    def test_vc_gemini_retries_without_thinking_on_error(self) -> None:
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
                        thinking_level="minimal",
                        llama_threads=4,
                        llama_gpu_layers=0,
                        response_mime_type="text/plain",
                        gemini_requests_per_minute=60,
                    )

        self.assertEqual(text, "ok")
        self.assertEqual(len(models.calls), 2)
        self.assertEqual(
            models.calls[0]["contents"],
            [{"role": "user", "parts": [{"text": "prompt"}]}],
        )
        self.assertEqual(
            models.calls[1]["contents"],
            [{"role": "user", "parts": [{"text": "prompt"}]}],
        )
        self.assertEqual(models.calls[0]["config_kwargs"]["system_instruction"], "system")
        self.assertEqual(models.calls[1]["config_kwargs"]["system_instruction"], "system")
        self.assertIn("thinking_config", models.calls[0]["config_kwargs"])
        self.assertNotIn("thinking_config", models.calls[1]["config_kwargs"])


if __name__ == "__main__":
    unittest.main()
