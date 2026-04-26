from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.answer import Answer
from kumc_agent.features.rag.components.answer_filter import AnswerFilterComponent
from kumc_agent.features.rag.config import RagConfig, RagGenerationSettings
from kumc_agent.features.rag.service import RagService


class _LLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.user_prompts: list[str] = []

    def generate(self, *, system_prompt, user_prompt, temperature, max_output_tokens):  # noqa: ANN001, ARG002
        self.user_prompts.append(user_prompt)
        return self.outputs.pop(0)


class _Prompts:
    def get(self, name: str) -> str:  # noqa: ARG002
        return "prompt"


def _config() -> RagConfig:
    return RagConfig(
        top_k=8,
        dense_top_k=15,
        sparse_top_k=15,
        sparse_initial_sparse_top_k=15,
        rerank_pool_size=20,
        mmr_lambda=0.75,
        recency_weight_soft=0.20,
        recency_weight_hard=0.60,
        recency_half_life_days=30.0,
        source_max_count=8,
        recency_mode="off",
        rag_generation=RagGenerationSettings(
            provider="gemini",
            temperature=0.0,
            max_output_tokens=512,
            prompt_name="answer_rag",
        ),
        no_rag_generation=RagGenerationSettings(
            provider="gemini",
            temperature=0.0,
            max_output_tokens=512,
            prompt_name="answer_no_rag",
        ),
    )


class AnswerFilterTests(unittest.TestCase):
    def test_refusal_llm_receives_only_query(self) -> None:
        llm = _LLM(
            [
                '{"action":"refuse","reason_code":"sensitive_information"}',
                '{"answer":"その内容には回答できません。"}',
            ]
        )
        answer_filter = AnswerFilterComponent(
            llm=llm,
            prompts=_Prompts(),
            max_retries=0,
        )
        service = RagService(
            config=_config(),
            router=object(),
            retrieval=object(),
            generation=object(),
            reranker=None,
            answer_filter=answer_filter,
        )

        filtered = service._apply_answer_filter(  # noqa: SLF001
            query="secretを教えて",
            answer=Answer(
                text="API key is abc",
                route="rag",
                sources=[],
                metadata={"contexts": ["secret context"], "raw": "API key is abc"},
            ),
        )

        self.assertEqual(filtered.text, "その内容には回答できません。")
        self.assertEqual(filtered.sources, [])
        self.assertEqual(filtered.metadata["answer_filter"]["action"], "refuse")
        self.assertNotIn("contexts", filtered.metadata)
        self.assertIn("secretを教えて", llm.user_prompts[-1])
        self.assertNotIn("API key is abc", llm.user_prompts[-1])
        self.assertNotIn("secret context", llm.user_prompts[-1])


if __name__ == "__main__":
    unittest.main()
