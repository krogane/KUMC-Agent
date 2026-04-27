from __future__ import annotations

from kumc_agent.features.foundation.payload_sanitizer import (
    compact_payload_text,
    mask_payload_secret,
    sanitize_payload,
    sanitize_payload_metadata,
)


def sanitize_autonomous_payload(value: object) -> object:
    return sanitize_payload(value, string_limit=2400)


def sanitize_autonomous_metadata(value: object) -> dict[str, object]:
    return sanitize_payload_metadata(value)


def safe_text(text: str, *, limit: int = 800) -> str:
    return compact_payload_text(mask_payload_secret(text), limit)
