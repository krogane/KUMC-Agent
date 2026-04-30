from __future__ import annotations

import html
import re

from kumc_agent.infra.indexing.summary_searchability import (
    SummarySearchabilityDecision,
)

_HTML_BLOCK_CLOSE_RE = re.compile(
    r"(?is)</\s*(p|div|section|article|h[1-6]|li|tr|table|ul|ol|figure|figcaption)\s*>"
)
_HTML_BR_RE = re.compile(r"(?is)<\s*br\s*/?\s*>")
_HTML_SCRIPT_STYLE_RE = re.compile(r"(?is)<\s*(script|style)\b[^>]*>.*?</\s*\1\s*>")
_HTML_TAG_RE = re.compile(r"(?s)<[^>]+>")
_CTA_TERMS = ("コメント", "Discord", "BOOTH")


def sanitize_summary_text(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    cleaned = _HTML_SCRIPT_STYLE_RE.sub(" ", raw)
    cleaned = _HTML_BR_RE.sub("\n", cleaned)
    cleaned = _HTML_BLOCK_CLOSE_RE.sub("\n", cleaned)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = html.unescape(cleaned).replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def summary_quality_metadata(
    *,
    source_text: str,
    summary_text: str,
    decision: SummarySearchabilityDecision,
) -> dict[str, object]:
    source_terms = _matched_cta_terms(source_text)
    summary_terms = _matched_cta_terms(summary_text)
    origins = {
        term: "source_text" if term in source_terms else "summary_only"
        for term in summary_terms
    }
    if not origins:
        cta_origin = "none"
    elif all(value == "source_text" for value in origins.values()):
        cta_origin = "source_text"
    elif all(value == "summary_only" for value in origins.values()):
        cta_origin = "summary_only"
    else:
        cta_origin = "mixed"
    return {
        "summary_fallback_used": bool(decision.fallback_used),
        "summary_parse_failed": bool(decision.parse_failed),
        "summary_decision_reason": decision.reason,
        "summary_cta_terms": summary_terms,
        "summary_source_cta_terms": source_terms,
        "summary_cta_origins": origins,
        "summary_cta_origin": cta_origin,
    }


def _matched_cta_terms(text: str) -> list[str]:
    normalized = str(text or "")
    casefolded = normalized.casefold()
    out: list[str] = []
    for term in _CTA_TERMS:
        haystack = casefolded if term.isascii() else normalized
        needle = term.casefold() if term.isascii() else term
        if needle in haystack:
            out.append(term)
    return out
