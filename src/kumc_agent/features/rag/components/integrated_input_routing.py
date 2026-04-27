from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any

from kumc_agent.domain.models.integrated_input import IntegratedInputDecision, IntegratedRoute

_DEFAULT_PROMPT = "integrated_input_routing"
_VALID_ROUTES = {
    "circle_rag",
    "minecraft_wiki_rag",
    "member_search",
    "image_search",
    "task_management",
    "event_management",
    "server_management",
    "comprehensive_agent",
    "clarify",
    "deny",
}
_VALID_RISKS = {"read_only", "candidate_only", "approval_required", "admin_only"}
_VALID_INTENTS = {
    "question",
    "search",
    "create_candidate",
    "update_candidate",
    "delete_candidate",
    "approval",
    "admin_operation",
    "compose",
    "extract",
    "list",
    "notify",
    "complete",
    "unknown",
}


class IntegratedInputRouter:
    def __init__(
        self,
        *,
        provider: str = "none",
        gemini_model: str = "",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        max_retries: int = 1,
        gemini_api_key: str = "",
        gemini_requests_per_minute: int = 0,
        prompts_dir: Path | None = None,
        prompt_name: str = _DEFAULT_PROMPT,
        log_enabled: bool = False,
    ) -> None:
        self.provider = provider
        self.gemini_model = gemini_model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retries = max(1, max_retries)
        self.gemini_api_key = gemini_api_key
        self.gemini_requests_per_minute = gemini_requests_per_minute
        self.prompts_dir = prompts_dir or Path("assets") / "prompts"
        self.prompt_name = prompt_name
        self.log_enabled = log_enabled

    def decide(
        self,
        text: str,
        *,
        source: str = "all",
        mode: str = "answer",
        depth: str = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> IntegratedInputDecision:
        text = str(text or "").strip()
        if not text:
            return IntegratedInputDecision(
                route="clarify",
                intent="unknown",
                needs_clarification=True,
                clarification_question="入力内容を指定してください。",
                reason="empty_input",
                metadata={"fallback": True, "fallback_reason": "empty_input"},
            )
        if self.provider.lower() == "gemini" and self.gemini_api_key:
            errors: list[str] = []
            for attempt in range(self.max_retries):
                try:
                    payload = self._generate(text, source=source, mode=mode, depth=depth)
                    parsed = self._parse_payload(payload)
                    if parsed is not None:
                        return replace(
                            parsed,
                            metadata={
                                **parsed.metadata,
                                "classifier": "gemini",
                                "model": self.gemini_model,
                                "attempt": attempt + 1,
                            },
                        )
                    errors.append("invalid_payload")
                except Exception as exc:  # pragma: no cover - provider dependent
                    errors.append(str(exc))
            return self._heuristic_decision(
                text,
                source=source,
                metadata={
                    "fallback": True,
                    "fallback_reason": "classifier_failed",
                    "classifier_errors": errors[-3:],
                },
            )
        return self._heuristic_decision(
            text,
            source=source,
            metadata={"fallback": True, "fallback_reason": "classifier_unavailable"},
        )

    def _generate(self, text: str, *, source: str, mode: str, depth: str) -> str:
        try:
            import google.generativeai as genai  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("google-generativeai is required for Gemini routing") from exc
        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel(self.gemini_model)
        prompt = self._prompt()
        response = model.generate_content(
            [
                prompt,
                json.dumps(
                    {
                        "text": text,
                        "source": source,
                        "mode": mode,
                        "depth": depth,
                    },
                    ensure_ascii=False,
                ),
            ],
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_new_tokens,
            },
        )
        return str(getattr(response, "text", "") or "")

    def _prompt(self) -> str:
        path = self.prompts_dir / f"{self.prompt_name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "Return integrated input routing JSON only."

    def _parse_payload(self, text: str) -> IntegratedInputDecision | None:
        parsed = self._load_json_object(self._strip_code_fence(text).strip())
        if parsed is None:
            return None
        route = str(parsed.get("route") or "").strip()
        if route not in _VALID_ROUTES:
            return None
        intent = str(parsed.get("intent") or "question").strip()
        if intent not in _VALID_INTENTS:
            intent = "unknown"
        risk = str(parsed.get("risk") or "read_only").strip()
        if risk not in _VALID_RISKS:
            risk = "read_only"
        required_features = parsed.get("required_features") or tuple()
        if isinstance(required_features, str):
            required_features = [required_features]
        source_filters = parsed.get("source_filters") or tuple()
        if isinstance(source_filters, str):
            source_filters = [source_filters]
        attribute_filters = parsed.get("attribute_filters") or {}
        return IntegratedInputDecision(
            route=route,  # type: ignore[arg-type]
            intent=intent,  # type: ignore[arg-type]
            required_features=tuple(str(item) for item in required_features),
            source_filters=tuple(str(item) for item in source_filters),
            attribute_filters=dict(attribute_filters) if isinstance(attribute_filters, dict) else {},
            risk=risk,  # type: ignore[arg-type]
            freshness_required=bool(parsed.get("freshness_required")),
            needs_clarification=bool(parsed.get("needs_clarification")),
            clarification_question=str(parsed.get("clarification_question") or ""),
            reason=str(parsed.get("reason") or ""),
            metadata={"classifier_payload": parsed},
        )

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
        return match.group(1) if match else stripped

    @staticmethod
    def _load_json_object(text: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None

    def _heuristic_decision(
        self,
        text: str,
        *,
        source: str,
        metadata: dict[str, Any],
    ) -> IntegratedInputDecision:
        features = _detect_required_features(text, source)
        route: IntegratedRoute = "circle_rag"
        intent = _detect_intent(text)
        risk = _risk_for_intent(intent, text)
        if source == "minecraft_wiki":
            route = "minecraft_wiki_rag"
        elif source == "member":
            route = "member_search"
        elif source == "image":
            route = "image_search"
        elif source == "task" or "task_management" in features:
            route = "task_management"
        elif source == "event" or "event_management" in features:
            route = "event_management"
        elif source == "server":
            route = "server_management"
            risk = "approval_required"
        elif "server_management" in features:
            route = "server_management"
            risk = "approval_required"
        elif "minecraft_wiki" in features:
            route = "minecraft_wiki_rag"
        if len(features) >= 2:
            route = "comprehensive_agent"
        return IntegratedInputDecision(
            route=route,
            intent=intent,  # type: ignore[arg-type]
            required_features=features,
            source_filters=(source,) if source not in {"member", "image", "task", "event"} else tuple(),
            risk=risk,  # type: ignore[arg-type]
            freshness_required=route in {"task_management", "event_management", "server_management"},
            reason="heuristic",
            metadata=metadata,
        )


class IntegratedRoutingPolicy:
    def apply(
        self,
        decision: IntegratedInputDecision,
        *,
        text: str,
        source: str,
        access_is_admin: bool = False,
    ) -> IntegratedInputDecision:
        route = decision.route
        risk = decision.risk
        required = tuple(dict.fromkeys(decision.required_features or _detect_required_features(text, source)))
        if source == "minecraft_wiki":
            route = "minecraft_wiki_rag"
            required = ("minecraft_wiki",)
        elif source == "member":
            route = "member_search"
            required = ("member_search",)
        elif source == "image":
            route = "image_search"
            required = ("image_search",)
        elif source == "task":
            route = "task_management"
            required = ("task_management",)
        elif source == "event":
            route = "event_management"
            required = ("event_management",)
        elif source == "server":
            route = "server_management"
            risk = "approval_required"
            required = ("server_management",)
        if _has_server_management(text):
            route = "server_management"
            risk = "approval_required"
            required = _merge_required(required, "server_management")
        if len(required) >= 2:
            route = "comprehensive_agent"
        if risk == "admin_only" and not access_is_admin:
            route = "deny"
        if decision.needs_clarification:
            route = "clarify"
        return replace(
            decision,
            route=route,
            risk=risk,
            required_features=required,
            metadata={
                **decision.metadata,
                "policy_decision": {
                    "route": route,
                    "risk": risk,
                    "required_features": list(required),
                },
            },
        )


def _detect_required_features(query: str, source_filter: str = "all") -> tuple[str, ...]:
    text = query.lower()
    features: list[str] = []
    if source_filter == "minecraft_wiki" or any(
        token in text for token in ("minecraft", "マイクラ", "redstone", "レッドストーン")
    ):
        features.append("minecraft_wiki")
    if source_filter == "member" or any(token in query for token in ("メンバー", "担当候補", "誰", "スキル", "得意")):
        features.append("member_search")
    if source_filter == "image" or any(token in query for token in ("画像", "写真", "素材", "asset", "サムネ")):
        features.append("image_search")
    if source_filter == "task" or any(token in query for token in ("タスク", "todo", "ToDo", "担当タスク", "やること")):
        features.append("task_management")
    if source_filter == "event" or any(token in query for token in ("イベント", "予定", "新歓", "日時", "開催")):
        features.append("event_management")
    if source_filter == "server" or _has_server_management(query):
        features.append("server_management")
    if source_filter in {"all", "drive", "discord", "notion", "hatena", "x", "crafters_colony"}:
        if any(token in query for token in ("資料", "過去", "根拠", "確認", "調べ", "KUMC", "サークル")):
            features.insert(0, "circle_rag")
    if not features:
        features.append("circle_rag")
    return tuple(dict.fromkeys(features))


def _detect_intent(text: str) -> str:
    lowered = text.lower()
    if any(token in text for token in ("一覧", "リスト", "list")):
        return "list"
    if any(token in text for token in ("抽出", "extract")):
        return "extract"
    if any(token in text for token in ("完了", "done", "complete")):
        return "complete"
    if any(token in text for token in ("通知", "notify")):
        return "notify"
    if any(token in text for token in ("削除", "delete")):
        return "delete_candidate"
    if any(token in text for token in ("更新", "変更", "update")):
        return "update_candidate"
    if any(token in text for token in ("追加", "作成", "候補", "申請", "add", "create")):
        return "create_candidate"
    if any(token in lowered for token in ("approve", "reject")) or any(token in text for token in ("承認", "却下")):
        return "approval"
    return "question"


def _risk_for_intent(intent: str, text: str) -> str:
    if _has_server_management(text):
        return "approval_required"
    if intent in {"create_candidate", "update_candidate", "delete_candidate", "extract", "notify", "complete"}:
        return "candidate_only"
    if intent == "admin_operation":
        return "admin_only"
    return "read_only"


def _has_server_management(text: str) -> bool:
    return any(
        token in text
        for token in ("再起動", "停止", "起動", "バックアップ", "ホワイトリスト", "サーバー操作", "server")
    )


def _merge_required(required: tuple[str, ...], feature: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*required, feature]))
