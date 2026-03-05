from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.config.load import ConfigLoadError, load_runtime_config


class ConfigLoadingTests(unittest.TestCase):
    def _prepare_base(self, base: Path) -> None:
        (base / "configs" / "ops").mkdir(parents=True)
        (base / "configs" / "experiments" / "rag").mkdir(parents=True)

        (base / "configs" / "ops" / "app.yaml").write_text(
            dedent(
                """
                app:
                  command_prefix: "/ai"
                  index_command_prefix: "/ai build-index"
                  max_input_characters: 1024
                  log_level: "INFO"
                  data_dir: "data"
                  raw_dir: "data/raw"
                  chunks_path: "data/chunks/chunks.jsonl"
                  index_dir: "data/index"
                  eval_dir: "data/eval"
                  cache_dir: "data/cache"
                  answer_record_log_path: "logs/answer_records.jsonl"
                  source_max_count: 3
                integrations:
                  discord:
                    bot_token: ""
                  drive:
                    folder_id: ""
                    google_application_credentials: ""
                    max_files: 0
                  crafters_colony:
                    author_url: ""
                    max_pages: 100
                    max_articles: 0
                  gemini_api_key: ""
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (base / "configs" / "ops" / "providers.yaml").write_text(
            dedent(
                """
                providers:
                  llm:
                    provider: "gemini"
                    gemini_model: "gemini-x"
                    llama_model_path: "model.gguf"
                    temperature: 0.0
                    max_output_tokens: 128
                    thinking_level: "minimal"
                    threads: 4
                    gpu_layers: 0
                  embeddings:
                    provider: "local"
                    model: "e5"
                    dimensions: 64
                  reranker:
                    model: "cross"
                    enabled: true
                  function_call:
                    enabled: true
                    provider: "gemini"
                    gemini_model: "gemini-x"
                    llama_model_path: "fc.gguf"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (base / "configs" / "ops" / "security.yaml").write_text(
            "security:\n  maintenance_command_author_ids: []\n  discord_guild_allow_list: []\n  refusal_keywords: ['秘密']\n",
            encoding="utf-8",
        )
        (base / "configs" / "ops" / "scheduler.yaml").write_text(
            "scheduler:\n  auto_index_enabled: false\n  auto_index_time: '03:00'\n  auto_index_weekdays: [0,1,2]\n",
            encoding="utf-8",
        )
        (base / "configs" / "ops" / "features.yaml").write_text(
            dedent(
                """
                features:
                  rag: true
                  indexing: true
                  eval: true
                  summarization: true
                  vc: false
                  docgen: false
                  http: false
                  recency_mode: "off"
                  sources:
                    drive: false
                    discord: false
                    hatenablog: false
                    crafters_colony: false
                  retrieval:
                    top_k: 5
                    dense_top_k: 5
                    sparse_top_k: 5
                    rerank_pool_size: 10
                    mmr_lambda: 0.5
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (base / "configs" / "ops" / "model.yaml").write_text(
            dedent(
                """
                model:
                  root_dir: "model"
                  llm_dir: "model/llm"
                  embedding_dir: "model/embedding"
                  cross_encoder_dir: "model/cross-encoder"
                  whisper_dir: "model/whisper"
                  ocr_dir: "model/ocr"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (base / "configs" / "ops" / "vc.yaml").write_text(
            dedent(
                """
                vc:
                  feature_enabled: false
                  auto_join_enabled: false
                  auto_join_weekdays: [5]
                  auto_join_time: "20:00"
                  auto_join_duration_minutes: 30
                  target_voice_channel_name: "例会"
                  auto_join_min_participants: 3
                  participant_check_interval_seconds: 10
                  summary_transcribe_interval_seconds: 300
                  transcribe_model: "model/whisper/openai/whisper-large-v3-turbo"
                  transcribe_device: "auto"
                  transcribe_torch_dtype: "auto"
                  transcribe_language: "ja"
                  auto_quit_enabled: true
                  final_summary_enabled: true
                  summary_previous_max: 2
                  summary_target_characters: 100
                  summary_llm_provider: "gemini"
                  summary_gemini_model: "gemini-x"
                  summary_llama_model_path: "model/llm/model.gguf"
                  summary_llama_ctx_size: 4096
                  summary_temperature: 0.2
                  summary_max_output_tokens: 256
                  summary_thinking_level: "minimal"
                  minutes_enabled: true
                  minutes_drive_dir: "議事録"
                  minutes_fetch_max_retries: 2
                  minutes_apply_max_retries: 2
                  minutes_llm_max_retries: 2
                  minutes_history_summary_max: 2
                  minutes_image_batch_size: 10
                  minutes_edit_llm_provider: "gemini"
                  minutes_edit_gemini_model: "gemini-x"
                  minutes_edit_llama_model_path: "model/llm/model.gguf"
                  minutes_edit_llama_ctx_size: 4096
                  minutes_edit_temperature: 0.2
                  minutes_edit_max_output_tokens: 1024
                  minutes_edit_thinking_level: "minimal"
                  final_summary_llm_provider: "gemini"
                  final_summary_gemini_model: "gemini-x"
                  final_summary_llama_model_path: "model/llm/model.gguf"
                  final_summary_llama_ctx_size: 4096
                  final_summary_temperature: 0.0
                  final_summary_max_output_tokens: 1024
                  final_summary_thinking_level: "minimal"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (base / "configs" / "experiments" / "rag" / "baseline.yaml").write_text(
            "features:\n  retrieval:\n    top_k: 7\n",
            encoding="utf-8",
        )

    def test_priority_ops_env_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._prepare_base(base)
            with patch.dict(
                os.environ,
                {
                    "KUMC_DISCORD_BOT_TOKEN": "token",
                    "KUMC_GEMINI_API_KEY": "key",
                    "KUMC_DRIVE_FOLDER_ID": "folder",
                    "KUMC_RETRIEVAL_TOP_K": "9",
                    "KUMC_EXPERIMENT_PROFILE": "rag/baseline",
                },
                clear=False,
            ):
                config = load_runtime_config(base_dir=base)

            self.assertEqual(config.features.retrieval.top_k, 7)
            self.assertEqual(config.integrations.discord.bot_token, "token")

    def test_unknown_key_in_experiment_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._prepare_base(base)
            (base / "configs" / "experiments" / "rag" / "baseline.yaml").write_text(
                "features:\n  unknown_key: true\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "KUMC_DISCORD_BOT_TOKEN": "token",
                    "KUMC_GEMINI_API_KEY": "key",
                    "KUMC_DRIVE_FOLDER_ID": "folder",
                    "KUMC_EXPERIMENT_PROFILE": "rag/baseline",
                },
                clear=False,
            ):
                with self.assertRaises(ConfigLoadError):
                    load_runtime_config(base_dir=base)


if __name__ == "__main__":
    unittest.main()
