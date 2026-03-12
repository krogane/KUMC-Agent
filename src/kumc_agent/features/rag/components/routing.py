from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from kumc_agent.domain.models.routing import RoutingDecision
from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit

logger = logging.getLogger(__name__)

ChatHistoryEntry = tuple[str, str, Sequence[str]]

_ROUTING_TASK_NAMES = (
    "target_model",
    "use_additional_memory",
    "include_capabilities_info",
    "idea_generation",
    "needs_additional_query",
    "additional_queries",
    "material_names",
    "recency_mode",
)

_ROUTING_BOOL_TASKS = {
    "use_additional_memory",
    "include_capabilities_info",
    "idea_generation",
    "needs_additional_query",
}

_TARGET_MODEL_VALUES = {"rag", "material_search", "no_rag", "refusal"}
_RECENCY_VALUES = {"off", "soft", "hard"}
_RETRYABLE_GENERATION_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_GENERATION_ERROR_KEYWORDS = {
    "unavailable",
    "high demand",
    "rate limit",
    "resource exhausted",
    "overloaded",
    "timeout",
    "timed out",
    "connection reset",
}

_DEFAULT_ROUTING_SYSTEM_PROMPT = (
    "あなたは、厳格なルーティング判定エンジンです。\n"
    "与えられる質問は、京大マインクラフト同好会KUMCという大学サークルのアシスタントボットに向けられた質問です。サークル情報も参考に、以下の各フィールドでルーティングを行ってください。JSONのみを返してください。Markdownや説明文は出力しないでください。\n\n"
    "## フィールド一覧:\n"
    "- target_model: rag | material_search | no_rag | refusal\n"
    "- material_names: string[] (max {material_search_max_names})\n"
    "- idea_generation: bool\n"
    "- include_capabilities_info: bool\n"
    "- recency_mode: off | soft | hard\n"
    "- use_additional_memory: bool\n"
    "- needs_additional_query: bool\n"
    "- additional_queries: string[] (max 3)\n\n"
    "## 各フィールド・選択肢の説明:\n"
    "- target_model(refusal): 質問が機微な個人情報（住所、電話番号、パスワード、口座情報など）に関する場合は target_model=refusal とする。また、質問が契約内容に関する場合も target_model=refusal とする。\n"
    "- target_model(no_rag): 質問に「一般的な知識のみで回答できる」または「サークル関連情報は不要」と判断した場合は target_model=no_rag とする。ただし、質問が上記のrefusalに少しでも該当する場合は target_model=refusal とする。\n"
    "- target_model(rag): 質問に「一般的な知識のみでは回答できない」かつ「サークル関連情報が必要」と判断した場合は target_model=rag とする。ただし、質問が上記のrefusalに少しでも該当する場合は target_model=refusal とする。\n"
    "- target_model(material_search): 質問が特定の資料名に言及している場合は target_model=material_search とし、material_names に資料名を最大 {material_search_max_names} 件入れる。資料名を抽出できない場合は material_names=[] のまま返す。\n"
    "- idea_generation: 質問がアイデア（案や計画を含む）の作成を要求するものである場合は idea_generation=true とする。ただし、target_model=no_rag の場合は idea_generation=false を強制する。\n"
    "- include_capabilities_info: 質問に「アシスタントの情報（機能や能力など）」が必要と判断した場合は include_capabilities_info=true とする。\n"
    "- recency_mode: 最新情報の重視度。通常は soft。最新の情報が重要な質問は hard。時系列を考慮しなくても良い・過去の資料・出来事について質問している場合は off。target_model=no_rag/refusal の場合は off を選ぶ。\n"
    "- use_additional_memory: 回答に追加のチャット履歴があると望ましい場合（例: 指示語が含まれる・文脈が曖昧・過去のチャットに関連する）は true とする。\n"
    "- needs_additional_query: 「質問文にRAG検索に必要最低限の語句が全く含まれていない場合」または「質問への回答に多段階検索が必須な場合」にのみ true とする。\n"
    "- additional_queries: needs_additional_query=true の場合のみ出力する。重複を避けた追加クエリを1件、必要最小限の場合にのみ2件まで出力する。needs_additional_query=false の場合は [] とする。\n\n"
    "## サークル情報\n"
    "- 主な活動内容: 週1回（土曜20:00〜）のオンライン例会・メンバー同士のマルチプレイ（サバイバルやHypixelなど）・マップ制作（京大RPGやミニゲーム）・Minecraftサーバー運営・NFなどのイベント出展・新歓の開催・外部団体とのコラボ（コラボ先はStardy・エンドラRTA軍団・北田さんなど）・対面でのご飯会・プログラミング関連（AtCoderやハッカソンへの参加）\n\n"
    "## 現在の日付\n"
    "{today_label}"
)


@dataclass(frozen=True)
class RoutingTaskConfig:
    provider: str
    gemini_model: str
    llama_model_path: str
    prompt_name: str = "routing"


@dataclass(frozen=True)
class _RoutingPromptSections:
    intro: str
    field_lines: list[str]
    rule_lines: list[str]
    circle_info: str
    today: str


class QueryRouter:
    def __init__(
        self,
        *,
        refusal_keywords: list[str],
        routing_enabled: bool,
        provider: str,
        gemini_model: str,
        llama_model_path: str,
        prompt_name: str = "routing",
        temperature: float,
        max_new_tokens: int,
        max_retries: int,
        log_enabled: bool,
        material_search_max_names: int,
        llm_threads: int,
        llm_gpu_layers: int,
        llm_ctx_size: int,
        gemini_api_key: str,
        gemini_requests_per_minute: int,
        task_configs: dict[str, RoutingTaskConfig] | None = None,
    ) -> None:
        self._refusal_keywords = list(refusal_keywords)
        self._routing_enabled = routing_enabled
        self._provider = provider
        self._gemini_model = gemini_model
        self._llama_model_path = llama_model_path
        self._prompt_name = str(prompt_name or "routing").strip() or "routing"
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._max_retries = max_retries
        self._log_enabled = log_enabled
        self._material_search_max_names = max(1, material_search_max_names)
        self._llm_threads = llm_threads
        self._llm_gpu_layers = llm_gpu_layers
        self._llm_ctx_size = llm_ctx_size
        self._gemini_api_key = gemini_api_key
        self._gemini_requests_per_minute = max(0, int(gemini_requests_per_minute))
        self._routing_prompt_template_cache: dict[str, str] = {}
        self._task_configs = self._resolve_task_configs(task_configs)

    def route(
        self,
        query: str,
        *,
        question_author: str | None = None,
        history: Sequence[ChatHistoryEntry] | None = None,
    ) -> RoutingDecision:
        if self._routing_enabled:
            try:
                return self._route_with_task_llms(
                    query=query,
                    question_author=question_author,
                    history=history,
                )
            except Exception:
                logger.exception("Routing failed. Defaulting to safe routing.")
                return self._default_decision()
        return self._default_decision()

    def _resolve_task_configs(
        self,
        task_configs: dict[str, RoutingTaskConfig] | None,
    ) -> dict[str, RoutingTaskConfig]:
        defaults = RoutingTaskConfig(
            provider=self._provider,
            gemini_model=self._gemini_model,
            llama_model_path=self._llama_model_path,
            prompt_name=self._prompt_name,
        )
        resolved: dict[str, RoutingTaskConfig] = {}
        for task_name in _ROUTING_TASK_NAMES:
            candidate = task_configs.get(task_name) if task_configs else None
            if isinstance(candidate, dict):
                candidate = RoutingTaskConfig(
                    provider=str(candidate.get("provider", "")),
                    gemini_model=str(candidate.get("gemini_model", "")),
                    llama_model_path=str(candidate.get("llama_model_path", "")),
                    prompt_name=str(candidate.get("prompt_name", "")),
                )
            if not isinstance(candidate, RoutingTaskConfig):
                candidate = defaults
            provider = str(candidate.provider or "").strip() or defaults.provider
            gemini_model = (
                str(candidate.gemini_model or "").strip() or defaults.gemini_model
            )
            llama_model_path = (
                str(candidate.llama_model_path or "").strip()
                or defaults.llama_model_path
            )
            prompt_name = str(candidate.prompt_name or "").strip() or defaults.prompt_name
            resolved[task_name] = RoutingTaskConfig(
                provider=provider,
                gemini_model=gemini_model,
                llama_model_path=llama_model_path,
                prompt_name=prompt_name,
            )
        return resolved

    @staticmethod
    def _default_decision() -> RoutingDecision:
        return RoutingDecision(
            target_model="rag",
            recency_mode="off",
            material_names=[],
            idea_generation=False,
            include_capabilities_info=False,
            use_additional_memory=False,
            needs_additional_query=False,
            additional_queries=[],
        )

    def _route_with_task_llms(
        self,
        *,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
    ) -> RoutingDecision:
        phase_one = self._run_tasks_parallel(
            task_names=("target_model", "use_additional_memory"),
            query=query,
            question_author=question_author,
            history=history,
            context={},
        )
        target_model = str(phase_one.get("target_model") or "rag").strip().lower()
        if target_model not in _TARGET_MODEL_VALUES:
            target_model = "rag"
        use_additional_memory = bool(phase_one.get("use_additional_memory", False))

        if target_model == "refusal":
            return RoutingDecision(
                target_model="refusal",
                recency_mode="off",
                use_additional_memory=use_additional_memory,
            )

        phase_two_names: list[str] = [
            "include_capabilities_info",
            "idea_generation",
            "needs_additional_query",
        ]
        if target_model == "material_search":
            phase_two_names.append("material_names")
        if target_model in {"rag", "material_search"}:
            phase_two_names.append("recency_mode")

        phase_two = self._run_tasks_parallel(
            task_names=tuple(phase_two_names),
            query=query,
            question_author=question_author,
            history=history,
            context={"target_model": target_model},
        )

        include_capabilities_info = bool(
            phase_two.get("include_capabilities_info", False)
        )
        needs_additional_query = bool(phase_two.get("needs_additional_query", False))

        additional_queries: list[str] = []
        if needs_additional_query:
            additional_queries = self._run_task_with_retries(
                task_name="additional_queries",
                query=query,
                question_author=question_author,
                history=history,
                context={
                    "target_model": target_model,
                    "needs_additional_query": True,
                },
            )
            if not additional_queries:
                needs_additional_query = False

        if target_model == "no_rag":
            return RoutingDecision(
                target_model="no_rag",
                recency_mode="off",
                include_capabilities_info=include_capabilities_info,
                use_additional_memory=use_additional_memory,
            )

        if target_model == "material_search":
            material_names = phase_two.get("material_names")
            if not isinstance(material_names, list):
                material_names = []
            recency_mode = str(phase_two.get("recency_mode") or "off").strip().lower()
            if recency_mode not in _RECENCY_VALUES:
                recency_mode = "off"
            return RoutingDecision(
                target_model="material_search",
                recency_mode=recency_mode,
                material_names=material_names,
                include_capabilities_info=include_capabilities_info,
                use_additional_memory=use_additional_memory,
            )

        recency_mode = str(phase_two.get("recency_mode") or "off").strip().lower()
        if recency_mode not in _RECENCY_VALUES:
            recency_mode = "off"

        return RoutingDecision(
            target_model="rag",
            recency_mode=recency_mode,
            idea_generation=bool(phase_two.get("idea_generation", False)),
            include_capabilities_info=include_capabilities_info,
            use_additional_memory=use_additional_memory,
            needs_additional_query=needs_additional_query,
            additional_queries=additional_queries if needs_additional_query else [],
        )

    def _run_tasks_parallel(
        self,
        *,
        task_names: Sequence[str],
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
        context: dict[str, object],
    ) -> dict[str, object]:
        filtered = [name for name in task_names if name in _ROUTING_TASK_NAMES]
        if not filtered:
            return {}
        if len(filtered) == 1:
            name = filtered[0]
            return {
                name: self._run_task_with_retries(
                    task_name=name,
                    query=query,
                    question_author=question_author,
                    history=history,
                    context=context,
                )
            }

        max_workers = max(1, len(filtered))
        results: dict[str, object] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_task_with_retries,
                    task_name=name,
                    query=query,
                    question_author=question_author,
                    history=history,
                    context=context,
                ): name
                for name in filtered
            }
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                except Exception:
                    logger.exception("Routing task failed: %s", task_name)
                    results[task_name] = self._default_task_value(task_name)
        return results

    def _run_task_with_retries(
        self,
        *,
        task_name: str,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
        context: dict[str, object],
    ) -> object:
        retries = max(0, int(self._max_retries))
        last_raw = ""
        for attempt in range(retries + 1):
            try:
                raw = self._generate_task_payload(
                    task_name=task_name,
                    query=query,
                    question_author=question_author,
                    history=history,
                    context=context,
                )
            except Exception as exc:
                is_retryable = self._is_retryable_generation_error(exc)
                if attempt >= retries:
                    logger.exception("Routing task generation failed: %s", task_name)
                    return self._default_task_value(task_name)
                if is_retryable:
                    wait_seconds = self._retry_backoff_seconds(attempt=attempt)
                    logger.warning(
                        (
                            "Routing task generation transient error. "
                            "task=%s retry=%d/%d wait=%.2fs error=%s"
                        ),
                        task_name,
                        attempt + 1,
                        retries,
                        wait_seconds,
                        str(exc),
                    )
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                else:
                    logger.exception("Routing task generation failed: %s", task_name)
                continue

            last_raw = raw
            if self._log_enabled:
                logger.info("Routing task %s raw output: %s", task_name, raw)
            parsed = self._parse_task_payload(task_name=task_name, text=raw)
            if parsed is not None:
                if self._log_enabled:
                    logger.info("Routing task %s parsed output: %s", task_name, parsed)
                return parsed
            if attempt < retries:
                logger.info(
                    "Invalid routing task output. task=%s retry=%s/%s",
                    task_name,
                    attempt + 1,
                    retries,
                )

        logger.warning(
            "Routing task payload could not be parsed. task=%s raw=%s",
            task_name,
            last_raw,
        )
        return self._default_task_value(task_name)

    @staticmethod
    def _retry_backoff_seconds(*, attempt: int) -> float:
        return min(8.0, float(2 ** max(0, int(attempt))))

    @staticmethod
    def _is_retryable_generation_error(exc: Exception) -> bool:
        for item in QueryRouter._exception_chain(exc):
            status_code = QueryRouter._extract_status_code(item)
            if status_code in _RETRYABLE_GENERATION_STATUS_CODES:
                return True

        message = " ".join(str(item).lower() for item in QueryRouter._exception_chain(exc))
        return any(token in message for token in _RETRYABLE_GENERATION_ERROR_KEYWORDS)

    @staticmethod
    def _exception_chain(exc: Exception) -> list[BaseException]:
        chain: list[BaseException] = [exc]
        current: BaseException | None = exc
        while current is not None and len(chain) < 8:
            next_exc = current.__cause__ or current.__context__
            if next_exc is None or next_exc in chain:
                break
            chain.append(next_exc)
            current = next_exc
        return chain

    @staticmethod
    def _extract_status_code(exc: BaseException) -> int | None:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        if isinstance(status, str) and status.isdigit():
            return int(status)

        message = str(exc)
        match = re.search(r"\b([1-5]\d{2})\b", message)
        if not match:
            return None
        return int(match.group(1))

    def _generate_task_payload(
        self,
        *,
        task_name: str,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
        context: dict[str, object],
    ) -> str:
        task_config = self._task_configs.get(task_name)
        if task_config is None:
            raise ValueError(f"Unknown routing task: {task_name}")

        provider = str(task_config.provider or "").strip().lower().replace(".", "_")
        if provider == "gemini":
            return self._generate_task_payload_gemini(
                task_name=task_name,
                task_config=task_config,
                query=query,
                question_author=question_author,
                history=history,
                context=context,
            )
        if provider in {"llama", "llama_cpp"}:
            return self._generate_task_payload_llama(
                task_name=task_name,
                task_config=task_config,
                query=query,
                question_author=question_author,
                history=history,
                context=context,
            )
        raise ValueError(
            "Unsupported routing provider: "
            f"{task_config.provider}. Use 'gemini' or 'llama_cpp'."
        )

    def _generate_task_payload_gemini(
        self,
        *,
        task_name: str,
        task_config: RoutingTaskConfig,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
        context: dict[str, object],
    ) -> str:
        if not self._gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Please set it in .env")
        if not task_config.gemini_model:
            raise RuntimeError(
                f"Gemini model is not set for routing task: {task_name}"
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is required for Gemini access.") from exc

        client = _genai_client(self._gemini_api_key)
        system_instruction = self._routing_system_prompt(
            task_name=task_name,
            prompt_name=task_config.prompt_name,
        )
        user_prompt = self._routing_task_user_prompt(
            task_name=task_name,
            query=query,
            question_author=question_author,
            history=history,
            context=context,
        )

        request_config: dict[str, object] = {
            "temperature": float(self._temperature),
            "max_output_tokens": max(1, int(self._max_new_tokens)),
            "response_mime_type": "application/json",
            "system_instruction": system_instruction,
        }
        wait_for_gemini_rate_limit(
            max_requests_per_minute=self._gemini_requests_per_minute
        )
        response = client.models.generate_content(
            model=task_config.gemini_model,
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                },
            ],
            config=genai.types.GenerateContentConfig(**request_config),
        )
        return (response.text or "").strip()

    def _generate_task_payload_llama(
        self,
        *,
        task_name: str,
        task_config: RoutingTaskConfig,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
        context: dict[str, object],
    ) -> str:
        if not task_config.llama_model_path:
            raise RuntimeError(
                f"Routing llama_model_path is not set for task: {task_name}"
            )

        llama = _llama_client(
            model_path=task_config.llama_model_path,
            ctx_size=self._llm_ctx_size,
            threads=self._llm_threads,
            gpu_layers=self._llm_gpu_layers,
        )
        grammar = _llama_grammar_from_schema(self._task_schema(task_name=task_name))
        result = llama.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self._routing_system_prompt(
                        task_name=task_name,
                        prompt_name=task_config.prompt_name,
                    ),
                },
                {
                    "role": "user",
                    "content": self._routing_task_user_prompt(
                        task_name=task_name,
                        query=query,
                        question_author=question_author,
                        history=history,
                        context=context,
                    ),
                },
            ],
            max_tokens=max(1, int(self._max_new_tokens)),
            temperature=float(self._temperature),
            grammar=grammar,
        )
        return (
            (result.get("choices", [{}])[0].get("message", {}) or {}).get("content")
            or ""
        ).strip()

    def _routing_system_prompt(self, *, task_name: str, prompt_name: str) -> str:
        rendered_template = self._render_routing_template(
            self._load_routing_prompt_template(prompt_name)
        )
        sections = self._extract_prompt_sections(rendered_template)

        fallback_sections = self._extract_prompt_sections(
            self._render_routing_template(_DEFAULT_ROUTING_SYSTEM_PROMPT)
        )

        intro = sections.intro or fallback_sections.intro
        field_line = self._select_field_line(task_name, sections.field_lines)
        if not field_line:
            field_line = self._select_field_line(task_name, fallback_sections.field_lines)

        rule_lines = self._select_rule_lines(task_name, sections.rule_lines)
        if not rule_lines:
            rule_lines = self._select_rule_lines(task_name, fallback_sections.rule_lines)

        circle_info = sections.circle_info or fallback_sections.circle_info
        today = sections.today or fallback_sections.today

        return (
            f"{intro}\n\n"
            "次の1項目だけを判定してください。JSONのみを返し、Markdownや説明文は出力しないでください。\n\n"
            "## フィールド一覧:\n"
            f"{field_line}\n\n"
            "## 各フィールド・選択肢の説明:\n"
            f"{'\\n'.join(rule_lines)}\n\n"
            "## サークル情報\n"
            f"{circle_info}\n\n"
            "## 現在の日付\n"
            f"{today}"
        )

    def _render_routing_template(self, template: str) -> str:
        today = datetime.now(ZoneInfo("Asia/Tokyo"))
        weekday = ["月", "火", "水", "木", "金", "土", "日"][today.weekday()]
        today_label = today.strftime("%Y年%m月%d日") + f"（{weekday}）"
        material_limit = str(max(1, int(self._material_search_max_names)))
        return (template or "").replace("{today_label}", today_label).replace(
            "{material_search_max_names}",
            material_limit,
        )

    def _load_routing_prompt_template(self, prompt_name: str) -> str:
        key = str(prompt_name or "routing").strip() or "routing"
        cached = self._routing_prompt_template_cache.get(key)
        if cached is not None:
            return cached

        value = ""
        path = self._routing_prompt_path(key)
        if path is not None and path.exists():
            try:
                value = path.read_text(encoding="utf-8").strip()
            except OSError:
                logger.exception("Failed to load routing prompt file: %s", path)
                value = ""

        if not value:
            value = _DEFAULT_ROUTING_SYSTEM_PROMPT
        self._routing_prompt_template_cache[key] = value
        return value

    @staticmethod
    def _routing_prompt_path(prompt_name: str) -> Path | None:
        resolved = Path(__file__).resolve()
        if len(resolved.parents) <= 5:
            return None
        prompts_dir = resolved.parents[5] / "assets" / "prompts"
        file_name = str(prompt_name or "routing").strip()
        if not file_name:
            file_name = "routing"
        candidate = Path(file_name)
        if candidate.suffix != ".md":
            candidate = candidate.with_suffix(".md")
        if candidate.is_absolute():
            return candidate
        return prompts_dir / candidate

    def _extract_prompt_sections(self, template: str) -> _RoutingPromptSections:
        text = (template or "").strip()
        intro = text
        marker = "## フィールド一覧"
        if marker in text:
            intro = text.split(marker, 1)[0].strip()

        field_section = self._extract_first_markdown_section(
            text,
            titles=("フィールド一覧:", "フィールド一覧"),
        )
        rule_section = self._extract_first_markdown_section(
            text,
            titles=("各フィールド・選択肢の説明:", "各フィールド・選択肢の説明"),
        )
        circle_info = self._extract_first_markdown_section(
            text,
            titles=("サークル情報",),
        )
        today = self._extract_first_markdown_section(
            text,
            titles=("現在の日付",),
        )
        return _RoutingPromptSections(
            intro=intro,
            field_lines=self._extract_bullet_lines(field_section),
            rule_lines=self._extract_bullet_lines(rule_section),
            circle_info=circle_info.strip(),
            today=today.strip(),
        )

    @staticmethod
    def _extract_first_markdown_section(text: str, *, titles: Sequence[str]) -> str:
        for title in titles:
            section = QueryRouter._extract_markdown_section(text, title=title)
            if section:
                return section
        return ""

    @staticmethod
    def _extract_markdown_section(text: str, *, title: str) -> str:
        pattern = re.compile(
            rf"^##\s*{re.escape(title)}\s*\n(?P<body>.*?)(?=^##\s+|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            return ""
        return match.group("body").strip()

    @staticmethod
    def _extract_bullet_lines(section: str) -> list[str]:
        lines: list[str] = []
        for raw in (section or "").splitlines():
            line = raw.strip()
            if not line.startswith("- "):
                continue
            lines.append(line)
        return lines

    @staticmethod
    def _bullet_key(line: str) -> str:
        content = line[2:].strip() if line.startswith("- ") else line.strip()
        if ":" not in content:
            return ""
        return content.split(":", 1)[0].strip()

    def _select_field_line(self, task_name: str, field_lines: Sequence[str]) -> str:
        target_key = task_name
        for line in field_lines:
            if self._bullet_key(line) == target_key:
                return line
        if task_name == "material_names":
            return f"- material_names: string[] (max {self._material_search_max_names})"
        if task_name == "additional_queries":
            return "- additional_queries: string[] (max 3)"
        if task_name == "target_model":
            return "- target_model: rag | material_search | no_rag | refusal"
        if task_name == "recency_mode":
            return "- recency_mode: off | soft | hard"
        return f"- {task_name}: bool"

    def _select_rule_lines(
        self,
        task_name: str,
        rule_lines: Sequence[str],
    ) -> list[str]:
        selected: list[str] = []
        for line in rule_lines:
            key = self._bullet_key(line)
            if not key:
                continue
            if task_name == "target_model" and key.startswith("target_model("):
                selected.append(line)
                continue
            if task_name == "material_names" and key in {
                "material_names",
                "target_model(material_search)",
            }:
                selected.append(line)
                continue
            if key == task_name:
                selected.append(line)

        if selected:
            return selected

        fallback = {
            "target_model": [
                "- target_model: rag / material_search / no_rag / refusal から1つ選択する。"
            ],
            "material_names": [
                "- material_names: material_searchに必要な資料名を重複なく抽出する。"
            ],
            "recency_mode": [
                "- recency_mode: off / soft / hard から1つ選択する。"
            ],
            "additional_queries": [
                "- additional_queries: 重複しない追加クエリを必要最小限で返す。"
            ],
        }
        if task_name in fallback:
            return fallback[task_name]
        return [f"- {task_name}: true または false を返す。"]

    @staticmethod
    def _format_routing_history(history: Sequence[ChatHistoryEntry] | None) -> str:
        if not history:
            return "（履歴なし）"
        lines: list[str] = []
        for user_text, assistant_text, _ in history:
            user_value = (user_text or "").strip()
            assistant_value = (assistant_text or "").strip()
            if user_value:
                lines.append(f"ユーザー: {user_value}")
            if assistant_value:
                lines.append(f"アシスタント: {assistant_value}")
        return "\n".join(lines) if lines else "（履歴なし）"

    def _routing_task_user_prompt(
        self,
        *,
        task_name: str,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
        context: dict[str, object],
    ) -> str:
        history_text = self._format_routing_history(history)
        author_value = " ".join(
            segment.strip()
            for segment in str(question_author or "").splitlines()
            if segment.strip()
        )
        question_block = (query or "").strip()
        if author_value:
            question_block = f"author: {author_value}\n{question_block}"

        context_lines: list[str] = []
        for key, value in context.items():
            context_lines.append(
                f"- {key}: {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
            )
        context_block = "\n".join(context_lines) if context_lines else "（なし）"

        return (
            "## それまでのチャット履歴\n"
            f"{history_text}\n\n"
            "## 今回の質問\n"
            f"{question_block}\n\n"
            "## 判定対象フィールド\n"
            f"{task_name}\n\n"
            "## 既知の判定結果\n"
            f"{context_block}\n\n"
            "## 出力要件\n"
            f"- JSON objectとして `{task_name}` キーを必ず含めること。"
        )

    def _parse_task_payload(self, *, task_name: str, text: str) -> object | None:
        payload = self._load_json_payload((text or "").strip())
        if not isinstance(payload, dict):
            return None
        if task_name not in payload:
            return None

        value = payload.get(task_name)
        if task_name in _ROUTING_BOOL_TASKS:
            return self._coerce_bool(value)

        if task_name == "target_model":
            target_model = str(value or "").strip().lower()
            if target_model in _TARGET_MODEL_VALUES:
                return target_model
            return None

        if task_name == "recency_mode":
            recency_mode = str(value or "").strip().lower()
            if recency_mode in _RECENCY_VALUES:
                return recency_mode
            return None

        if task_name == "material_names":
            return self._normalize_material_names(
                value,
                max_items=self._material_search_max_names,
            )

        if task_name == "additional_queries":
            return self._normalize_queries(value)

        return None

    def _default_task_value(self, task_name: str) -> object:
        if task_name in _ROUTING_BOOL_TASKS:
            return False
        if task_name == "target_model":
            return "rag"
        if task_name == "recency_mode":
            return "off"
        return []

    def _task_schema(self, *, task_name: str) -> dict[str, object]:
        property_schema: dict[str, object]
        if task_name in _ROUTING_BOOL_TASKS:
            property_schema = {"type": "boolean"}
        elif task_name == "target_model":
            property_schema = {
                "type": "string",
                "enum": ["rag", "material_search", "no_rag", "refusal"],
            }
        elif task_name == "recency_mode":
            property_schema = {
                "type": "string",
                "enum": ["off", "soft", "hard"],
            }
        elif task_name == "material_names":
            property_schema = {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": max(1, int(self._material_search_max_names)),
            }
        elif task_name == "additional_queries":
            property_schema = {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
            }
        else:
            raise ValueError(f"Unsupported routing task for schema: {task_name}")

        return {
            "type": "object",
            "properties": {task_name: property_schema},
            "required": [task_name],
            "additionalProperties": False,
        }

    @staticmethod
    def _load_json_payload(text: str) -> dict[str, object] | None:
        cleaned = QueryRouter._strip_code_fence(text).strip()
        if not cleaned:
            return None
        parsed = QueryRouter._load_json_object(cleaned)
        if parsed is not None:
            return parsed
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        return QueryRouter._load_json_object(cleaned[start : end + 1])

    @staticmethod
    def _load_json_object(text: str) -> dict[str, object] | None:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return text
        lines = stripped.splitlines()
        if len(lines) < 2:
            return text
        if not lines[-1].strip().startswith("```"):
            return text
        return "\n".join(lines[1:-1]).strip()

    @staticmethod
    def _coerce_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return False

    @staticmethod
    def _normalize_queries(raw: object) -> list[str]:
        values: list[str] = []
        if isinstance(raw, str):
            candidate = raw.strip()
            if candidate:
                values = [candidate]
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, str):
                    continue
                candidate = item.strip()
                if not candidate:
                    continue
                values.append(candidate)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in values:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= 3:
                break
        return deduped

    @staticmethod
    def _normalize_material_names(raw: object, *, max_items: int) -> list[str]:
        values: list[str] = []
        if isinstance(raw, str):
            candidate = raw.strip()
            if candidate:
                values = [candidate]
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, str):
                    continue
                candidate = item.strip()
                if not candidate:
                    continue
                values.append(candidate)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in values:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max(1, int(max_items)):
                break
        return deduped


@lru_cache(maxsize=4)
def _genai_client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is required for Gemini access.") from exc
    return genai.Client(api_key=api_key)


@lru_cache(maxsize=16)
def _llama_client(
    *,
    model_path: str,
    ctx_size: int,
    threads: int,
    gpu_layers: int,
):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Please install it to use llama.cpp."
        ) from exc
    return Llama(
        model_path=model_path,
        n_ctx=max(1, int(ctx_size)),
        n_threads=max(1, int(threads)),
        n_gpu_layers=int(gpu_layers),
    )


def _llama_grammar_from_schema(schema: dict[str, object]):
    try:
        from llama_cpp import LlamaGrammar
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is required for llama.cpp JSON schema grammar."
        ) from exc
    return LlamaGrammar.from_json_schema(
        json.dumps(schema, ensure_ascii=False),
        verbose=False,
    )
