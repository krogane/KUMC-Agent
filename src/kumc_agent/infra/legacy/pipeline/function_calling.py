from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from kumc_agent.infra.legacy.config import AppConfig
from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit

logger = logging.getLogger(__name__)


_DEFAULT_ROUTING_SYSTEM_PROMPT = (
    "あなたは、厳格なルーティング判定エンジンです。\n"
    "与えられる質問は、京大マインクラフト同好会KUMCという大学サークルのアシスタントボットに向けられた質問です。サークル情報も参考に、以下の各フィールドでルーティングを行ってください。"
    "JSONのみを返してください。Markdownや説明文は出力しないでください。\n\n"
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
    "- target_model(refusal): 質問が機微な個人情報（住所、電話番号、パスワード、口座情報など）に関する場合は、target_model=refusal とする。また、質問が契約内容に関する場合も、target_model=refusal とする。\n"
    "- target_model(no_rag): 質問に「一般的な知識のみで回答できる」または「サークル関連情報は不要」と判断した場合は、target_model=no_rag とする。ただし、質問が上記のrefusalに少しでも該当する場合は、target_model=refusal とする。\n"
    "- target_model(rag): 質問に「一般的な知識のみでは回答できない」かつ「サークル関連情報が必要」と判断した場合は、target_model=rag とする。ただし、質問が上記のrefusalに少しでも該当する場合は、target_model=refusal とする。\n"
    "- target_model(material_search): 質問が特定の資料名に言及している場合はtarget_model=material_searchとし、material_namesに資料名を最大{material_search_max_names}件入れる。資料名を抽出できない場合はmaterial_names=[]のまま返す。\n"
    "- idea_generation: 質問がアイデア（案や計画を含む）の作成を要求するものである場合は、idea_generation=true とする。ただし、target_model=no_rag の場合、idea_generation=false を強制する。\n"
    "- include_capabilities_info: 質問に「アシスタントの情報（機能や能力など）」が必要と判断した場合は、include_capabilities_info=true とする。\n"
    "- recency_mode: 最新情報の重視度。通常はsoft。最新の情報が重要な質問はhard。時系列を考慮しなくても良い・過去の資料・出来事について質問している場合はoff。target_model=no_rag/refusal の場合は off を選ぶ。\n"
    "- use_additional_memory: 質問に対する回答に追加のチャット履歴があると望ましい場合（例: 質問に指示語が含まれている・質問の文脈が曖昧・質問が過去のチャットに関連する）は use_additional_memory=true とする。\n"
    "- needs_additional_query: 「質問文に、RAG検索に必要最低限の語句が全く含まれていない場合」または「質問への回答に多段階の検索が必須である場合」にのみ、needs_additional_query=true とする。ただし、サークル名などの文脈や、単なるキーワードの抜き出しは追加クエリには不必要である点に留意する。\n"
    "- additional_queries: needs_additional_query=true の場合にのみ、 重複を避けたadditional_queries （文章または空白区切りのキーワード群）を1件出力する。ただし、1件では質問に対して最低限の回答が不可能な場合にのみ2件出力する。クエリを生成する際は、下記の現在の日付も参考にする。needs_additional_query=falseの場合は、additional_queries=[] とする。\n\n"
    "## サークル情報\n"
    "- 主な活動内容: 週1回（土曜20:00〜）のオンライン例会・メンバー同士のマルチプレイ（サバイバルやHypixelなど）・マップ制作（京大RPGやミニゲーム）・Minecraftサーバー運営・NFなどのイベント出展・新歓の開催・外部団体とのコラボ（コラボ先はStardy・エンドラRTA軍団・北田さんなど）・対面でのご飯会・プログラミング関連（AtCoderやハッカソンへの参加）\n"
    "## 現在の日付\n"
    "{today_label}\n"
)


@dataclass(frozen=True)
class FunctionRoutingDecision:
    target_model: str
    material_names: list[str]
    idea_generation: bool
    include_capabilities_info: bool
    recency_mode: str
    use_additional_memory: bool
    needs_additional_query: bool
    additional_queries: list[str]


ChatHistoryEntry = tuple[str, str, Sequence[str]]


def _default_decision() -> FunctionRoutingDecision:
    return FunctionRoutingDecision(
        target_model="rag",
        material_names=[],
        idea_generation=False,
        include_capabilities_info=False,
        recency_mode="off",
        use_additional_memory=False,
        needs_additional_query=False,
        additional_queries=[],
    )


def decide_tools(
    *,
    query: str,
    question_author: str | None = None,
    config: AppConfig,
    history: Sequence[ChatHistoryEntry] | None = None,
) -> FunctionRoutingDecision:
    max_retries = max(0, config.function_call_max_retries)
    last_raw = ""
    for attempt in range(max_retries + 1):
        raw = _generate_routing_payload(
            query=query,
            question_author=question_author,
            config=config,
            history=history,
        )
        last_raw = raw
        if config.function_call_log_enabled:
            logger.info("Function-calling raw output: %s", raw)
        decision = _parse_routing_payload(
            raw,
            material_search_max_names=max(1, config.material_search_max_names),
        )
        if decision is not None:
            if config.function_call_log_enabled:
                logger.info("Function-calling parsed decision: %s", decision)
            return decision
        if attempt < max_retries:
            logger.info(
                "Invalid routing output from function-calling LLM. Retrying %s/%s",
                attempt + 1,
                max_retries,
            )

    logger.warning(
        "Function-calling LLM output could not be parsed. Defaulting to safe routing. raw=%s",
        last_raw,
    )
    return _default_decision()


def _generate_routing_payload(
    *,
    query: str,
    question_author: str | None = None,
    config: AppConfig,
    history: Sequence[ChatHistoryEntry] | None = None,
) -> str:
    provider = (config.function_call_provider or "").lower()
    if provider == "gemini":
        return _generate_routing_payload_gemini(
            query=query,
            question_author=question_author,
            config=config,
            history=history,
        )
    raise ValueError(
        "Unsupported FUNCTION_CALL_PROVIDER: "
        f"{config.function_call_provider}. Use 'gemini'."
    )


def _routing_system_prompt(*, material_search_max_names: int) -> str:
    today = datetime.now(ZoneInfo("Asia/Tokyo"))
    weekday = ["月", "火", "水", "木", "金", "土", "日"][today.weekday()]
    today_label = today.strftime("%Y年%m月%d日") + f"（{weekday}）"
    limit = str(max(1, int(material_search_max_names)))
    template = _routing_system_prompt_template()
    return (
        template.replace("{today_label}", today_label).replace(
            "{material_search_max_names}", limit
        )
    )


@lru_cache(maxsize=1)
def _routing_system_prompt_template() -> str:
    prompt_path = _routing_prompt_path()
    if prompt_path is None:
        return _DEFAULT_ROUTING_SYSTEM_PROMPT
    try:
        if prompt_path.exists():
            value = prompt_path.read_text(encoding="utf-8").strip()
            if value:
                return value
    except OSError:
        logger.exception("Failed to load routing prompt file: %s", prompt_path)
    return _DEFAULT_ROUTING_SYSTEM_PROMPT


def _routing_prompt_path() -> Path | None:
    resolved = Path(__file__).resolve()
    if len(resolved.parents) <= 5:
        return None
    return resolved.parents[5] / "assets" / "prompts" / "routing.md"


def _format_routing_history(
    history: Sequence[ChatHistoryEntry] | None,
) -> str:
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
    *,
    query: str,
    question_author: str | None = None,
    history: Sequence[ChatHistoryEntry] | None,
) -> str:
    history_text = _format_routing_history(history)
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


def _generate_routing_payload_gemini(
    *,
    query: str,
    question_author: str | None = None,
    config: AppConfig,
    history: Sequence[ChatHistoryEntry] | None = None,
) -> str:
    if not config.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Please set it in .env")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is required for Gemini access.") from exc

    client = _genai_client(config.gemini_api_key)
    include_thinking = _routing_gemini_model_supports_thinking(
        config.function_call_gemini_model
    )
    if not include_thinking:
        logger.info(
            "Routing Gemini model %s does not support thinking_level. "
            "Send request without thinking_config.",
            config.function_call_gemini_model,
        )
    try:
        response = _generate_routing_payload_gemini_response(
            client=client,
            genai_module=genai,
            config=config,
            query=query,
            question_author=question_author,
            history=history,
            include_thinking=include_thinking,
        )
    except Exception as exc:
        if include_thinking and _is_unsupported_gemini_thinking_error(exc):
            logger.info(
                "Routing Gemini model %s rejected thinking_level. "
                "Retrying without thinking_config.",
                config.function_call_gemini_model,
            )
            response = _generate_routing_payload_gemini_response(
                client=client,
                genai_module=genai,
                config=config,
                query=query,
                question_author=question_author,
                history=history,
                include_thinking=False,
            )
        else:
            raise
    return (response.text or "").strip()


def _generate_routing_payload_gemini_response(
    *,
    client,
    genai_module,
    config: AppConfig,
    query: str,
    question_author: str | None,
    history: Sequence[ChatHistoryEntry] | None,
    include_thinking: bool,
):
    system_instruction = _routing_system_prompt(
        material_search_max_names=config.material_search_max_names
    )
    user_prompt = _routing_user_prompt(
        query=query,
        question_author=question_author,
        history=history,
    )
    request_config: dict[str, object] = {
        "temperature": config.function_call_temperature,
        "max_output_tokens": max(1, int(config.function_call_max_new_tokens)),
        "response_mime_type": "application/json",
        "system_instruction": system_instruction,
    }
    if include_thinking:
        request_config["thinking_config"] = genai_module.types.ThinkingConfig(
            thinking_level=config.thinking_level
        )
    wait_for_gemini_rate_limit(
        max_requests_per_minute=getattr(
            config,
            "gemini_requests_per_minute",
            60,
        )
    )
    return client.models.generate_content(
        model=config.function_call_gemini_model,
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": user_prompt
                    }
                ],
            },
        ],
        config=genai_module.types.GenerateContentConfig(**request_config),
    )


def _routing_gemini_model_supports_thinking(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    unsupported_prefixes = ("gemini-2.5-flash-lite",)
    return not any(normalized.startswith(prefix) for prefix in unsupported_prefixes)


def _is_unsupported_gemini_thinking_error(exc: Exception) -> bool:
    return "thinking level is not supported for this model" in str(exc).lower()


def _parse_routing_payload(
    text: str,
    *,
    material_search_max_names: int,
) -> FunctionRoutingDecision | None:
    payload = _load_json_payload((text or "").strip())
    if not isinstance(payload, dict):
        return None

    target_model = str(payload.get("target_model") or "").strip().lower()
    if target_model not in {"rag", "material_search", "no_rag", "refusal"}:
        return None
    material_names = _normalize_material_names(
        payload.get("material_names"),
        max_items=max(1, material_search_max_names),
    )

    idea_generation = _coerce_bool(payload.get("idea_generation"))
    include_capabilities_info = _coerce_bool(
        payload.get("include_capabilities_info")
    )
    recency_mode = str(payload.get("recency_mode") or "").strip().lower()
    if recency_mode not in {"off", "soft", "hard"}:
        return None
    use_additional_memory = _coerce_bool(payload.get("use_additional_memory"))
    needs_additional_query = _coerce_bool(payload.get("needs_additional_query"))

    additional_queries = _normalize_queries(payload.get("additional_queries"))

    if target_model == "refusal":
        return FunctionRoutingDecision(
            target_model="refusal",
            material_names=[],
            idea_generation=False,
            include_capabilities_info=False,
            recency_mode="off",
            use_additional_memory=use_additional_memory,
            needs_additional_query=False,
            additional_queries=[],
        )

    if target_model == "no_rag":
        return FunctionRoutingDecision(
            target_model="no_rag",
            material_names=[],
            idea_generation=False,
            include_capabilities_info=include_capabilities_info,
            recency_mode="off",
            use_additional_memory=use_additional_memory,
            needs_additional_query=False,
            additional_queries=[],
        )

    if target_model == "material_search":
        return FunctionRoutingDecision(
            target_model="material_search",
            material_names=material_names,
            idea_generation=False,
            include_capabilities_info=include_capabilities_info,
            recency_mode=recency_mode,
            use_additional_memory=use_additional_memory,
            needs_additional_query=False,
            additional_queries=[],
        )

    if not needs_additional_query:
        additional_queries = []
    elif not additional_queries:
        needs_additional_query = False

    return FunctionRoutingDecision(
        target_model="rag",
        material_names=[],
        idea_generation=idea_generation,
        include_capabilities_info=include_capabilities_info,
        recency_mode=recency_mode,
        use_additional_memory=use_additional_memory,
        needs_additional_query=needs_additional_query,
        additional_queries=additional_queries,
    )


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
        if len(deduped) >= max_items:
            break
    return deduped


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
                "maxItems": max_material_names,
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


def _load_json_payload(text: str) -> dict[str, object] | None:
    cleaned = _strip_code_fence(text).strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


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


@lru_cache(maxsize=1)
def _genai_client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is required for Gemini access.") from exc
    return genai.Client(api_key=api_key)
