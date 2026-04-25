from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.domain.models.retrieval import AskResponse, Citation, ContextPack, RetrievalQuery
from kumc_agent.features.retrieval.citation import CitationValidator


@dataclass(frozen=True)
class AnswerFormattingConfig:
    discord_summary_chars: int = 1800
    evidence_bullets: int = 3
    source_bullets: int = 5


class ExtractiveAnswerBuilder:
    def __init__(
        self,
        *,
        citation_validator: CitationValidator,
        config: AnswerFormattingConfig | None = None,
    ) -> None:
        self._validator = citation_validator
        self._config = config or AnswerFormattingConfig()

    def build(self, *, query: RetrievalQuery, context: ContextPack) -> AskResponse:
        validation = self._validator.validate(context=context, access=query.access)
        if not validation.citations:
            text = (
                "確認できる範囲では、回答に使える根拠を見つけられませんでした。\n\n"
                "次に試すとよいこと:\n"
                "- 資料名や時期を指定する\n"
                "- 対象チャンネルや source を指定する"
            )
            return AskResponse(
                text=text,
                detail_markdown=text,
                citations=tuple(),
                confidence="low",
                warnings=validation.warnings,
                metadata={"route": "no_answer"},
            )

        evidence = _evidence_lines(validation.citations, limit=self._config.evidence_bullets)
        sources = _source_lines(validation.citations, limit=self._config.source_bullets)
        conclusion = _conclusion(query.text, validation.citations)
        text = (
            f"結論:\n{conclusion}\n\n"
            f"根拠の要約:\n{evidence}\n\n"
            f"主な情報源:\n{sources}"
        )
        if validation.warnings:
            text += "\n\n注意:\n" + "\n".join(f"- {warning}" for warning in validation.warnings[:3])
        if len(text) > self._config.discord_summary_chars:
            text = text[: self._config.discord_summary_chars - 3].rstrip() + "..."
        detail = _detail_markdown(query=query, context=context, citations=validation.citations)
        return AskResponse(
            text=text,
            detail_markdown=detail,
            citations=validation.citations,
            confidence=validation.confidence,
            warnings=validation.warnings,
            metadata={"route": "rag", "detail_truncated": len(detail) > len(text)},
        )


def _conclusion(query: str, citations: tuple[Citation, ...]) -> str:
    lead = citations[0].quote or citations[0].label
    return (
        "関連する根拠は見つかりました。"
        f"最も関連度が高い根拠では「{lead}」が確認できます。"
    )


def _evidence_lines(citations: tuple[Citation, ...], *, limit: int) -> str:
    lines = []
    for citation in citations[:limit]:
        quote = citation.quote or citation.label
        lines.append(f"- {quote}")
    return "\n".join(lines)


def _source_lines(citations: tuple[Citation, ...], *, limit: int) -> str:
    lines = []
    seen: set[str] = set()
    for citation in citations:
        key = citation.url or citation.label
        if key in seen:
            continue
        seen.add(key)
        label = citation.label
        if citation.url:
            lines.append(f"- [{label}]({citation.url})")
        else:
            lines.append(f"- {label}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _detail_markdown(
    *,
    query: RetrievalQuery,
    context: ContextPack,
    citations: tuple[Citation, ...],
) -> str:
    lines = [
        f"# KUMC-Agent Retrieval Detail",
        "",
        f"Query: {query.text}",
        f"Source: {query.source_filter}",
        f"Confidence: {'low' if not citations else 'with citations'}",
        "",
        "## Citations",
    ]
    for idx, citation in enumerate(citations, start=1):
        lines.append(f"{idx}. {citation.label} ({citation.chunk_id})")
        if citation.url:
            lines.append(f"   - URL: {citation.url}")
        if citation.quote:
            lines.append(f"   - Quote: {citation.quote}")
    lines.extend(["", "## Packed Context", "", context.text])
    return "\n".join(lines).strip() + "\n"
