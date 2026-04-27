from __future__ import annotations

from dataclasses import asdict, is_dataclass
import re
from typing import Any

DROP_KEYS = {
    "contexts",
    "context",
    "llm_prompt",
    "raw",
    "secret",
    "executor_args",
    "server_state_before",
    "server_state_after",
    "container_state_before",
    "container_state_after",
    "downloaded_image_path",
    "original_image_ref",
}

SECRET_KEYS = {
    "password",
    "token",
    "api_key",
    "apikey",
    "secret",
    "access_token",
    "refresh_token",
}

TEXT_LIMITS = {
    "ocr_text": 800,
    "surrounding_text": 1200,
    "search_context": 1200,
    "retrieval_context": 1200,
    "prompt": 1200,
}


def sanitize_payload(value: object, *, string_limit: int = 4000) -> object:
    if is_dataclass(value):
        return sanitize_payload(asdict(value), string_limit=string_limit)
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if key_lower in DROP_KEYS or key_lower in SECRET_KEYS:
                continue
            limit = TEXT_LIMITS.get(key_lower, string_limit)
            sanitized[key_text] = sanitize_payload(item, string_limit=limit)
        return sanitized
    if isinstance(value, tuple):
        return [sanitize_payload(item, string_limit=string_limit) for item in value]
    if isinstance(value, list):
        return [sanitize_payload(item, string_limit=string_limit) for item in value]
    if isinstance(value, str):
        return compact_payload_text(mask_payload_secret(value), string_limit)
    return value


def sanitize_payload_metadata(value: object) -> dict[str, object]:
    sanitized = sanitize_payload(value, string_limit=1200)
    return sanitized if isinstance(sanitized, dict) else {}


def compact_payload_text(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def mask_payload_secret(text: str) -> str:
    masked = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )
    masked = re.sub(
        r"\b(?:10|172\.(?:1[6-9]|2\d|3[0-1])|192\.168)\.\d{1,3}\.\d{1,3}\b",
        "[internal-ip]",
        masked,
    )
    return re.sub(
        r"(?i)(network[_-]?key|pin|unlock(?:ing)?[_ -]?steps?)\s*[:=]\s*[^\n]+",
        r"\1=[REDACTED]",
        masked,
    )
