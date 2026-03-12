from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.features.rag.components.generation import GenerationComponent
from kumc_agent.features.rag.config import RagPromptTextSettings


class _RecordingLLM:
    def __init__(self, response_text: str = '{"answer":"ok","sources":[]}') -> None:
        self.last_system_prompt = ""
        self.last_user_prompt = ""
        self.response_text = response_text

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response_text


class _DictPromptRepo:
    def __init__(self, payload: dict[str, str]) -> None:
        self._payload = payload

    def get(self, name: str) -> str:
        if name not in self._payload:
            raise FileNotFoundError(name)
        return self._payload[name]


class GenerationComponentTests(unittest.TestCase):
    def _component(
        self,
        prompt_payload: dict[str, str],
        *,
        prompt_texts: RagPromptTextSettings | None = None,
        llm_response: str = '{"answer":"ok","sources":[]}',
    ) -> tuple[GenerationComponent, _RecordingLLM]:
        llm = _RecordingLLM(response_text=llm_response)
        component = GenerationComponent(
            llm=llm,
            no_rag_llm=llm,
            refusal_llm=llm,
            prompts=_DictPromptRepo(prompt_payload),
            source_max_count=3,
            prompt_texts=prompt_texts,
        )
        return component, llm

    def test_rag_prompt_includes_circle_basic_info(self) -> None:
        component, llm = self._component(
            {
                "answer_rag": '{"answer":"...", "sources":[1]}',
                "system_rules": "あなたはKUMC Agentです。今日は{today_label}です。",
                "circle_basic_info": "KUMCはMinecraftサークルです。",
            }
        )
        component.generate_rag_answer(
            query="活動内容を教えて",
            chunks=[
                Chunk(
                    id="1",
                    document_id="doc-1",
                    text="KUMCは大学サークルです。",
                    index=0,
                    metadata={},
                )
            ],
            history=None,
            include_capabilities_info=False,
            temperature=0.0,
            max_output_tokens=128,
        )
        self.assertIn("# サークルの基本情報\nKUMCはMinecraftサークルです。", llm.last_user_prompt)

    def test_no_rag_prompt_does_not_include_circle_basic_info(self) -> None:
        component, llm = self._component(
            {
                "answer_no_rag": '{"answer":"...", "sources":[1]}',
                "system_rules": "あなたはKUMC Agentです。今日は{today_label}です。",
                "circle_basic_info": "KUMCはMinecraftサークルです。",
            }
        )
        component.generate_no_rag(
            query="今日の天気は？",
            history=None,
            include_capabilities_info=False,
            temperature=0.0,
            max_output_tokens=128,
        )
        self.assertNotIn("# サークルの基本情報", llm.last_user_prompt)

    def test_history_prompt_texts_are_configurable(self) -> None:
        component, llm = self._component(
            {
                "answer_rag": '{"answer":"...", "sources":[1]}',
                "system_rules": "あなたはKUMC Agentです。今日は{today_label}です。",
            },
            prompt_texts=RagPromptTextSettings(
                empty_context="EMPTY_CONTEXT",
                empty_history="EMPTY_HISTORY",
                history_user_prefix="U> ",
                history_assistant_prefix="A> ",
                history_sources_label="SRC:",
                gemini_header_chat_history="CHAT",
                gemini_header_retry_history="RETRY",
                gemini_header_circle_info="CIRCLE",
                gemini_header_capabilities="CAP",
                gemini_header_context="CTX",
                gemini_header_output_format="OUT",
                gemini_header_instructions="INS",
                gemini_header_question="Q",
                llama_header_question="LQ",
                llama_header_previous_attempt="LPREV",
                llama_header_circle_info="LCIRCLE",
                llama_header_capabilities="LCAP",
                llama_header_context="LCTX",
                llama_header_output_format="LOUT",
                llama_header_instructions="LINS",
            ),
        )
        component.generate_rag_answer(
            query="活動内容を教えて",
            chunks=[
                Chunk(
                    id="1",
                    document_id="doc-1",
                    text="KUMCは大学サークルです。",
                    index=0,
                    metadata={},
                )
            ],
            history=[("質問", "回答", ["docs/guide.md"])],
            include_capabilities_info=False,
            temperature=0.0,
            max_output_tokens=128,
        )
        self.assertIn("CHAT\nU> 質問\nA> 回答\nSRC: docs/guide.md", llm.last_user_prompt)

    def test_no_rag_parses_code_fenced_json(self) -> None:
        component, _ = self._component(
            {
                "answer_no_rag": '{"answer":"...", "sources":[1]}',
                "system_rules": "あなたはKUMC Agentです。今日は{today_label}です。",
            },
            llm_response='```json\n{"answer":"フェンス回答","sources":[]}\n```',
        )
        answer = component.generate_no_rag(
            query="質問",
            history=None,
            include_capabilities_info=False,
            temperature=0.0,
            max_output_tokens=128,
        )
        self.assertEqual(answer.text, "フェンス回答")

    def test_no_rag_parses_json_embedded_in_text(self) -> None:
        component, _ = self._component(
            {
                "answer_no_rag": '{"answer":"...", "sources":[1]}',
                "system_rules": "あなたはKUMC Agentです。今日は{today_label}です。",
            },
            llm_response='補足です。\n{"answer":"埋め込み回答","sources":[]}\n以上です。',
        )
        answer = component.generate_no_rag(
            query="質問",
            history=None,
            include_capabilities_info=False,
            temperature=0.0,
            max_output_tokens=128,
        )
        self.assertEqual(answer.text, "埋め込み回答")

    def test_no_rag_returns_raw_text_when_json_is_invalid(self) -> None:
        component, _ = self._component(
            {
                "answer_no_rag": '{"answer":"...", "sources":[1]}',
                "system_rules": "あなたはKUMC Agentです。今日は{today_label}です。",
            },
            llm_response='{"answer":"壊れたJSON",}',
        )
        answer = component.generate_no_rag(
            query="質問",
            history=None,
            include_capabilities_info=False,
            temperature=0.0,
            max_output_tokens=128,
        )
        self.assertEqual(answer.text, '{"answer":"壊れたJSON",}')

    def test_rag_parses_hyphenated_source_selection(self) -> None:
        component, _ = self._component(
            {
                "answer_rag": '{"answer":"...", "sources":[1]}',
                "system_rules": "あなたはKUMC Agentです。今日は{today_label}です。",
            },
            llm_response='{"answer":"参照あり","sources":["1-2"]}',
        )
        answer = component.generate_rag_answer(
            query="質問",
            chunks=[
                Chunk(
                    id="1",
                    document_id="doc-1",
                    text="本文",
                    index=0,
                    metadata={"source_type": "docs", "drive_file_id": "file-123"},
                )
            ],
            history=None,
            include_capabilities_info=False,
            temperature=0.0,
            max_output_tokens=128,
            append_sources_to_response=False,
        )
        self.assertEqual(answer.text, "参照あり")
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].label, "https://docs.google.com/document/d/file-123/")

    def test_discord_context_annotation_keeps_date_line_plain(self) -> None:
        component, llm = self._component(
            {
                "answer_rag": '{"answer":"...", "sources":[1]}',
                "system_rules": "あなたはKUMC Agentです。今日は{today_label}です。",
            }
        )
        component.generate_rag_answer(
            query="質問",
            chunks=[
                Chunk(
                    id="1",
                    document_id="doc-1",
                    text="2025/01/01\nalice: hello",
                    index=0,
                    metadata={
                        "source_type": "messages",
                        "guild_name": "KUMC",
                        "category_name": "雑談",
                        "channel_name": "general",
                        "first_message_date": "2025-01-01",
                    },
                )
            ],
            history=None,
            include_capabilities_info=False,
            temperature=0.0,
            max_output_tokens=128,
            append_sources_to_response=False,
        )
        self.assertIn(
            "channel_name: KUMC / 雑談 / general\nfirst_message_date: 2025-01-01",
            llm.last_user_prompt,
        )
        self.assertIn("2025/01/01\n[1-1] alice: hello", llm.last_user_prompt)
        self.assertNotIn("[1-1] 2025/01/01", llm.last_user_prompt)


if __name__ == "__main__":
    unittest.main()
