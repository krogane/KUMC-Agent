from __future__ import annotations

import json
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

from kumc_agent.runtime.container import build_runtime_context
from kumc_agent.usecases.chat.answer import ChatRequest
from kumc_agent.usecases.eval.ragas import EvaluateRagasRequest
from kumc_agent.usecases.indexing.build import BuildIndexRequest


class ChatIndexEvalIntegrationTests(unittest.TestCase):
    def _write_minimal_project(self, base: Path) -> None:
        (base / "configs" / "main").mkdir(parents=True)
        (base / "assets" / "prompts").mkdir(parents=True)
        (base / "data" / "raw" / "docs").mkdir(parents=True)
        (base / "data" / "eval").mkdir(parents=True)

        (base / "assets" / "prompts" / "answer_rag.md").write_text(
            '{"answer": "...", "sources": ["1"]}', encoding="utf-8"
        )
        (base / "assets" / "prompts" / "answer_no_rag.md").write_text(
            '{"answer": "...", "sources": []}', encoding="utf-8"
        )
        (base / "assets" / "prompts" / "answer_idea.md").write_text(
            '{"answer": "...", "sources": ["1"]}', encoding="utf-8"
        )
        (base / "assets" / "prompts" / "routing.md").write_text("routing", encoding="utf-8")
        (base / "assets" / "prompts" / "summarization.md").write_text("sum", encoding="utf-8")

        (base / "configs" / "main" / "app.yaml").write_text(
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
                rag:
                  generation:
                    rag:
                      provider: "gemini"
                      gemini_model: "gemini-x"
                      temperature: 0.0
                      max_output_tokens: 128
                      thinking_level: "minimal"
                      prompt_name: "answer_rag"
                    no_rag:
                      provider: "gemini"
                      gemini_model: "gemini-x"
                      temperature: 0.0
                      max_output_tokens: 128
                      thinking_level: "minimal"
                      prompt_name: "answer_no_rag"
                    idea_generation:
                      prompt_name: "answer_idea"
                      temperature: 0.0
                integrations:
                  discord: {bot_token: ""}
                  drive: {folder_id: "", google_application_credentials: "", max_files: 0}
                  crafters_colony: {author_url: "", max_pages: 100, max_articles: 0}
                  gemini_api_key: ""
                  gemini_requests_per_minute: 60
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (base / "configs" / "main" / "providers.yaml").write_text(
            dedent(
                """
                providers:
                  llm:
                    provider: "gemini"
                    gemini_model: "gemini-x"
                    temperature: 0.0
                    max_output_tokens: 128
                    thinking_level: "minimal"
                  embeddings:
                    provider: "local"
                    model: "e5"
                    dimensions: 64
                  reranker:
                    model: "cross"
                    enabled: false
                  function_call:
                    enabled: false
                    provider: "gemini"
                    gemini_model: "gemini-x"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (base / "configs" / "main" / "security.yaml").write_text(
            (
                "security:\n"
                "  maintenance_command_author_ids: []\n"
                "  discord_guild_allow_list: []\n"
                "  discord_member_profile_guild_ids: []\n"
            ),
            encoding="utf-8",
        )
        (base / "configs" / "main" / "scheduler.yaml").write_text(
            "scheduler:\n  auto_index_enabled: false\n  auto_index_time: '03:00'\n  auto_index_weekdays: [0]\n",
            encoding="utf-8",
        )
        (base / "configs" / "main" / "features.yaml").write_text(
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
                    recency_weight_soft: 0.2
                    recency_weight_hard: 0.5
                    recency_half_life_days: 30.0
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (base / "configs" / "main" / "model.yaml").write_text(
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
        (base / "configs" / "main" / "vc.yaml").write_text(
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
                  minutes_edit_temperature: 0.2
                  minutes_edit_max_output_tokens: 1024
                  minutes_edit_thinking_level: "minimal"
                  final_summary_llm_provider: "gemini"
                  final_summary_gemini_model: "gemini-x"
                  final_summary_temperature: 0.0
                  final_summary_max_output_tokens: 1024
                  final_summary_thinking_level: "minimal"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        for file_name in (
            "infrastructure.yaml",
            "rag.yaml",
            "indexing.yaml",
            "evaluation.yaml",
            "integrations.yaml",
            "summarization.yaml",
        ):
            (base / "configs" / "main" / file_name).write_text("", encoding="utf-8")

        (base / "data" / "raw" / "docs" / "sample.md").write_text(
            "KUMCは京都大学のMinecraftサークルです。例会は土曜日です。",
            encoding="utf-8",
        )

        (base / "data" / "eval" / "ragas.jsonl").write_text(
            json.dumps({"question": "KUMCは何のサークルですか", "ground_truth": "Minecraft"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    def test_end_to_end_build_chat_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_minimal_project(base)
            with patch.dict(
                os.environ,
                {
                    "KUMC_DISCORD_BOT_TOKEN": "dummy",
                    "KUMC_GEMINI_API_KEY": "dummy",
                    "KUMC_DRIVE_FOLDER_ID": "dummy",
                },
                clear=False,
            ):
                context = build_runtime_context(base_dir=base)
                build_result = context.build_index.execute(BuildIndexRequest(refresh_sources=False))
                self.assertGreater(build_result.documents, 0)
                self.assertGreater(build_result.chunks, 0)

                answer = context.chat_answer.execute(ChatRequest(query="KUMCは何のサークルですか"))
                self.assertTrue(answer.text)

                eval_result = context.eval_ragas.execute(
                    EvaluateRagasRequest(eval_file=base / "data" / "eval" / "ragas.jsonl")
                )
                self.assertEqual(eval_result.total, 1)


if __name__ == "__main__":
    unittest.main()
