from __future__ import annotations

import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_UNSUPPORTED_PREFIXES = ("gemini-2.5-flash-lite",)
_UNSUPPORTED_THINKING_ERROR = "thinking level is not supported for this model"


def run_with_optional_thinking(
    *,
    model_name: str,
    request_label: str,
    run_request: Callable[[bool], _T],
) -> _T:
    include_thinking = gemini_model_supports_thinking(model_name)
    if not include_thinking:
        logger.info(
            "%s Gemini model %s does not support thinking_level. "
            "Sending request without thinking_config.",
            request_label,
            model_name,
        )
    try:
        return run_request(include_thinking)
    except Exception as exc:
        if include_thinking and is_unsupported_thinking_error(exc):
            logger.info(
                "%s Gemini model %s rejected thinking_level. "
                "Retrying without thinking_config.",
                request_label,
                model_name,
            )
            return run_request(False)
        raise


def gemini_model_supports_thinking(model_name: str) -> bool:
    normalized = _normalize_model_name(model_name)
    return not any(normalized.startswith(prefix) for prefix in _UNSUPPORTED_PREFIXES)


def is_unsupported_thinking_error(exc: Exception) -> bool:
    return _UNSUPPORTED_THINKING_ERROR in str(exc).lower()


def _normalize_model_name(model_name: str) -> str:
    normalized = (model_name or "").strip().lower()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    return normalized
