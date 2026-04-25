from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.domain.models.retrieval import AccessContext, Citation, ContextPack
from kumc_agent.features.retrieval.access import is_chunk_visible


@dataclass(frozen=True)
class CitationValidationResult:
    citations: tuple[Citation, ...]
    warnings: tuple[str, ...]
    confidence: str


class CitationValidator:
    def validate(self, *, context: ContextPack, access: AccessContext) -> CitationValidationResult:
        warnings: list[str] = list(context.warnings)
        valid: list[Citation] = []
        chunk_by_id = {item.chunk.id: item.chunk for item in context.chunks}
        for citation in context.citations:
            chunk = chunk_by_id.get(citation.chunk_id)
            if chunk is None:
                warnings.append(f"citation chunk missing: {citation.chunk_id}")
                continue
            if not is_chunk_visible(chunk, access):
                warnings.append(f"unauthorized citation removed: {citation.chunk_id}")
                continue
            if not citation.url:
                warnings.append(f"citation url missing: {citation.chunk_id}")
            valid.append(citation)
        if not valid:
            confidence = "low"
        elif warnings:
            confidence = "medium"
        else:
            confidence = "high"
        return CitationValidationResult(
            citations=tuple(valid),
            warnings=tuple(warnings),
            confidence=confidence,
        )
