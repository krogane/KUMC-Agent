from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
from typing import Sequence
from zoneinfo import ZoneInfo

from kumc_agent.domain.models.routing import RoutingDecision
from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit
from kumc_agent.infra.llm.gemini_thinking import run_with_optional_thinking

logger = logging.getLogger(__name__)

ChatHistoryEntry = tuple[str, str, Sequence[str]]

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


class QueryRouter:
    def __init__(
        self,
        *,
        refusal_keywords: list[str],
        routing_enabled: bool,
        provider: str,
        gemini_model: str,
        llama_model_path: str,
        temperature: float,
        max_new_tokens: int,
        max_retries: int,
        log_enabled: bool,
        material_search_max_names: int,
        llm_thinking_level: str,
        llm_threads: int,
        llm_gpu_layers: int,
        llm_ctx_size: int,
        gemini_api_key: str,
        gemini_requests_per_minute: int,
    ) -> None:
        self._refusal_keywords = list(refusal_keywords)
        self._routing_enabled = routing_enabled
        self._provider = provider
        self._gemini_model = gemini_model
        self._llama_model_path = llama_model_path
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._max_retries = max_retries
        self._log_enabled = log_enabled
        self._material_search_max_names = max(1, material_search_max_names)
        self._llm_thinking_level = llm_thinking_level
        self._llm_threads = llm_threads
        self._llm_gpu_layers = llm_gpu_layers
        self._llm_ctx_size = llm_ctx_size
        self._gemini_api_key = gemini_api_key
        self._gemini_requests_per_minute = max(0, int(gemini_requests_per_minute))
        self._routing_prompt_template: str | None = None

    def route(
        self,
        query: str,
        *,
        question_author: str | None = None,
        history: Sequence[ChatHistoryEntry] | None = None,
    ) -> RoutingDecision:
        if self._routing_enabled:
            try:
                return self._route_with_function_calling(
                    query=query,
                    question_author=question_author,
                    history=history,
                )
            except Exception:
                logger.exception(
                    "Function-call routing failed. Defaulting to safe routing."
                )
                return self._default_decision()
        return self._default_decision()

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

    def _route_with_function_calling(
        self,
        *,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
    ) -> RoutingDecision:
        retries = max(0, int(self._max_retries))
        last_raw = ""
        for attempt in range(retries + 1):
            raw = self._generate_routing_payload(
                query=query,
                question_author=question_author,
                history=history,
            )
            last_raw = raw
            if self._log_enabled:
                logger.info("Routing raw output: %s", raw)
            decision = self._parse_routing_payload(raw)
            if decision is not None:
                if self._log_enabled:
                    logger.info("Routing parsed decision: %s", decision)
                return decision
            if attempt < retries:
                logger.info(
                    "Invalid routing output. Retrying %s/%s",
                    attempt + 1,
                    retries,
                )
        logger.warning(
            "Routing payload could not be parsed. Defaulting to safe routing. raw=%s",
            last_raw,
        )
        return self._default_decision()

    def _generate_routing_payload(
        self,
        *,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
    ) -> str:
        provider = str(self._provider or "").strip().lower().replace(".", "_")
        if provider == "gemini":
            return self._generate_routing_payload_gemini(
                query=query,
                question_author=question_author,
                history=history,
            )
        if provider in {"llama", "llama_cpp"}:
            return self._generate_routing_payload_llama(
                query=query,
                question_author=question_author,
                history=history,
            )
        raise ValueError(
            "Unsupported routing provider: "
            f"{self._provider}. Use 'gemini' or 'llama_cpp'."
        )

    def _generate_routing_payload_gemini(
        self,
        *,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
    ) -> str:
        if not self._gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Please set it in .env")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is required for Gemini access.") from exc

        client = _genai_client(self._gemini_api_key)
        system_instruction = self._routing_system_prompt()
        user_prompt = self._routing_user_prompt(
            query=query,
            question_author=question_author,
            history=history,
        )

        def _request(include_thinking: bool):
            request_config: dict[str, object] = {
                "temperature": float(self._temperature),
                "max_output_tokens": max(1, int(self._max_new_tokens)),
                "response_mime_type": "application/json",
                "system_instruction": system_instruction,
            }
            if include_thinking:
                request_config["thinking_config"] = genai.types.ThinkingConfig(
                    thinking_level=self._llm_thinking_level
                )
            wait_for_gemini_rate_limit(
                max_requests_per_minute=self._gemini_requests_per_minute
            )
            return client.models.generate_content(
                model=self._gemini_model,
                contents=[
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}],
                    },
                ],
                config=genai.types.GenerateContentConfig(**request_config),
            )

        response = run_with_optional_thinking(
            model_name=self._gemini_model,
            request_label="Routing Gemini generation",
            run_request=_request,
        )
        return (response.text or "").strip()

    def _generate_routing_payload_llama(
        self,
        *,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
    ) -> str:
        if not self._llama_model_path:
            raise RuntimeError(
                "ROUTING_LLAMA_MODEL_PATH is not set. Please set it to a gguf model path."
            )
        llama = _llama_client(
            model_path=self._llama_model_path,
            ctx_size=self._llm_ctx_size,
            threads=self._llm_threads,
            gpu_layers=self._llm_gpu_layers,
        )
        schema = self._routing_schema(max_material_names=self._material_search_max_names)
        grammar = _llama_grammar_from_schema(schema)
        result = llama.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self._routing_system_prompt(),
                },
                {
                    "role": "user",
                    "content": self._routing_user_prompt(
                        query=query,
                        question_author=question_author,
                        history=history,
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

    def _routing_system_prompt(self) -> str:
        template = self._load_routing_prompt_template()
        today = datetime.now(ZoneInfo("Asia/Tokyo"))
        weekday = ["月", "火", "水", "木", "金", "土", "日"][today.weekday()]
        today_label = today.strftime("%Y年%m月%d日") + f"（{weekday}）"
        material_limit = str(max(1, int(self._material_search_max_names)))
        return template.replace("{today_label}", today_label).replace(
            "{material_search_max_names}",
            material_limit,
        )

    def _load_routing_prompt_template(self) -> str:
        if self._routing_prompt_template is not None:
            return self._routing_prompt_template
        path = self._routing_prompt_path()
        value = ""
        if path is not None and path.exists():
            try:
                value = path.read_text(encoding="utf-8").strip()
            except OSError:
                logger.exception("Failed to load routing prompt file: %s", path)
                value = ""
        if not value:
            value = _DEFAULT_ROUTING_SYSTEM_PROMPT
        self._routing_prompt_template = value
        return value

    @staticmethod
    def _routing_prompt_path() -> Path | None:
        resolved = Path(__file__).resolve()
        if len(resolved.parents) <= 5:
            return None
        return resolved.parents[5] / "assets" / "prompts" / "routing.md"

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

    def _routing_user_prompt(
        self,
        *,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
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
        return (
            "## それまでのチャット履歴\n"
            f"{history_text}\n\n"
            "## 今回の質問\n"
            f"{question_block}"
        )

    def _parse_routing_payload(self, text: str) -> RoutingDecision | None:
        payload = self._load_json_payload((text or "").strip())
        if not isinstance(payload, dict):
            return None
        target_model = str(payload.get("target_model") or "").strip().lower()
        if target_model not in {"rag", "material_search", "no_rag", "refusal"}:
            return None

        recency_mode = str(payload.get("recency_mode") or "").strip().lower()
        if recency_mode not in {"off", "soft", "hard"}:
            return None

        use_additional_memory = self._coerce_bool(payload.get("use_additional_memory"))
        include_capabilities_info = self._coerce_bool(
            payload.get("include_capabilities_info")
        )
        idea_generation = self._coerce_bool(payload.get("idea_generation"))
        needs_additional_query = self._coerce_bool(payload.get("needs_additional_query"))
        material_names = self._normalize_material_names(
            payload.get("material_names"),
            max_items=self._material_search_max_names,
        )
        additional_queries = self._normalize_queries(payload.get("additional_queries"))

        if target_model == "refusal":
            return RoutingDecision(
                target_model="refusal",
                recency_mode="off",
                use_additional_memory=use_additional_memory,
            )

        if target_model == "no_rag":
            return RoutingDecision(
                target_model="no_rag",
                recency_mode="off",
                include_capabilities_info=include_capabilities_info,
                use_additional_memory=use_additional_memory,
            )

        if target_model == "material_search":
            return RoutingDecision(
                target_model="material_search",
                recency_mode=recency_mode,
                material_names=material_names,
                include_capabilities_info=include_capabilities_info,
                use_additional_memory=use_additional_memory,
            )

        if not needs_additional_query:
            additional_queries = []
        elif not additional_queries:
            needs_additional_query = False
        return RoutingDecision(
            target_model="rag",
            recency_mode=recency_mode,
            idea_generation=idea_generation,
            include_capabilities_info=include_capabilities_info,
            use_additional_memory=use_additional_memory,
            needs_additional_query=needs_additional_query,
            additional_queries=additional_queries,
        )

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

    def _heuristic_route(self, query: str) -> RoutingDecision:
        text = (query or "").strip()
        lowered = text.lower()
        recency_mode = self._heuristic_recency_mode(lowered)
        material_names = self._extract_material_names(text)
        use_additional_memory = self._uses_additional_memory_hint(text)
        if material_names and "資料" in text:
            return RoutingDecision(
                target_model="material_search",
                recency_mode=recency_mode,
                material_names=material_names[: self._material_search_max_names],
                include_capabilities_info=False,
                use_additional_memory=use_additional_memory,
            )
        is_general = not any(
            token in lowered
            for token in (
                "kumc",
                "京大",
                "サークル",
                "例会",
                "minecraft",
                "マイクラ",
                "同好会",
            )
        )
        if is_general:
            return RoutingDecision(
                target_model="no_rag",
                recency_mode="off",
                use_additional_memory=use_additional_memory,
            )
        return RoutingDecision(
            target_model="rag",
            recency_mode=recency_mode,
            use_additional_memory=use_additional_memory,
            needs_additional_query=False,
            additional_queries=[],
        )

    @staticmethod
    def _heuristic_recency_mode(lowered: str) -> str:
        if any(token in lowered for token in ("最新", "今日", "きょう", "直近", "今週")):
            return "hard"
        if any(token in lowered for token in ("最近", "今月", "近況")):
            return "soft"
        return "off"

    @staticmethod
    def _uses_additional_memory_hint(text: str) -> bool:
        return any(
            token in text
            for token in ("それ", "これ", "前回", "さっき", "先ほど", "この件", "その件")
        )

    @staticmethod
    def _extract_material_names(text: str) -> list[str]:
        names: list[str] = []
        for pattern in (r"「([^」]+)」", r"\"([^\"]+)\""):
            for match in re.findall(pattern, text):
                value = str(match).strip()
                if value:
                    names.append(value)
        if not names and "資料" in text:
            tail = text.split("資料", 1)[-1].strip(" :：")
            if tail:
                names.append(tail)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in names:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _routing_schema(*, max_material_names: int) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "target_model": {
                    "type": "string",
                    "enum": ["rag", "material_search", "no_rag", "refusal"],
                },
                "material_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": max(1, int(max_material_names)),
                },
                "idea_generation": {"type": "boolean"},
                "include_capabilities_info": {"type": "boolean"},
                "recency_mode": {
                    "type": "string",
                    "enum": ["off", "soft", "hard"],
                },
                "use_additional_memory": {"type": "boolean"},
                "needs_additional_query": {"type": "boolean"},
                "additional_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
            },
            "required": [
                "target_model",
                "material_names",
                "idea_generation",
                "include_capabilities_info",
                "recency_mode",
                "use_additional_memory",
                "needs_additional_query",
                "additional_queries",
            ],
            "additionalProperties": False,
        }


@lru_cache(maxsize=4)
def _genai_client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is required for Gemini access.") from exc
    return genai.Client(api_key=api_key)


@lru_cache(maxsize=4)
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
