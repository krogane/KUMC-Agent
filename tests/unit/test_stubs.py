from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.features.docgen.service import DocGenService
from kumc_agent.features.vc.config import VCManagerConfig
from kumc_agent.features.vc.service import VCService
from kumc_agent.frontends.http.app import main as http_main


class StubTests(unittest.TestCase):
    def test_vc_service_unbound_noop(self) -> None:
        config = VCManagerConfig(
            raw_data_dir=Path("data/raw"),
            summery_chunk_dir=Path("data/chunks/summery_chunk"),
            discord_guild_allow_list=tuple(),
            drive_folder_id="",
            google_application_credentials="",
            gemini_api_key="",
            gemini_requests_per_minute=60,
            llama_threads=4,
            llama_gpu_layers=0,
            vc_feature_enabled=False,
            vc_auto_join_enabled=False,
            vc_auto_join_weekdays=(5,),
            vc_auto_join_start_hour=20,
            vc_auto_join_start_minute=0,
            vc_auto_join_duration_minutes=30,
            vc_target_voice_channel_name="例会",
            vc_auto_join_min_participants=3,
            vc_participant_check_interval_seconds=10,
            vc_summary_transcribe_interval_seconds=300,
            vc_transcribe_model="",
            vc_transcribe_device="auto",
            vc_transcribe_torch_dtype="auto",
            vc_transcribe_language="ja",
            vc_auto_quit_enabled=True,
            vc_final_summary_enabled=True,
            vc_summary_previous_max=2,
            vc_summary_target_characters=100,
            vc_summary_llm_provider="gemini",
            vc_summary_gemini_model="gemini-3.1-flash-lite-preview",
            vc_summary_llama_model="",
            vc_summary_llama_model_path="",
            vc_summary_llama_ctx_size=4096,
            vc_summary_temperature=0.2,
            vc_summary_max_output_tokens=256,
            vc_summary_thinking_level="minimal",
            vc_minutes_enabled=True,
            vc_minutes_drive_dir="議事録",
            vc_minutes_fetch_max_retries=2,
            vc_minutes_apply_max_retries=2,
            vc_minutes_llm_max_retries=2,
            vc_minutes_history_summary_max=2,
            vc_minutes_image_batch_size=10,
            vc_minutes_edit_llm_provider="gemini",
            vc_minutes_edit_gemini_model="gemini-3.1-flash-lite-preview",
            vc_minutes_edit_llama_model="",
            vc_minutes_edit_llama_model_path="",
            vc_minutes_edit_llama_ctx_size=4096,
            vc_minutes_edit_temperature=0.2,
            vc_minutes_edit_max_output_tokens=1024,
            vc_minutes_edit_thinking_level="minimal",
            vc_final_summary_llm_provider="gemini",
            vc_final_summary_gemini_model="gemini-3.1-flash-lite-preview",
            vc_final_summary_llama_model="",
            vc_final_summary_llama_model_path="",
            vc_final_summary_llama_ctx_size=4096,
            vc_final_summary_temperature=0.0,
            vc_final_summary_max_output_tokens=1024,
            vc_final_summary_thinking_level="minimal",
        )
        service = VCService(config=config)
        self.assertFalse(service.has_active_session())
        self.assertFalse(service.has_model_activity())
        self.assertFalse(service.should_use_fast_model_for_query())

    def test_docgen_stub(self) -> None:
        with self.assertRaises(NotImplementedError):
            DocGenService().run()

    def test_http_stub(self) -> None:
        with self.assertRaises(NotImplementedError):
            http_main()


if __name__ == "__main__":
    unittest.main()
