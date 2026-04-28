from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.domain.models.retrieval import Citation, ContextPack, ScoredChunk


@dataclass(frozen=True)
class ContextPackingConfig:
    max_context_characters: int = 8000
    max_quote_characters: int = 180
    max_citations: int = 8


class ContextPacker:
    def __init__(self, config: ContextPackingConfig | None = None) -> None:
        self._config = config or ContextPackingConfig()

    def pack(self, chunks: list[ScoredChunk]) -> ContextPack:
        texts: list[str] = []
        citations: list[Citation] = []
        warnings: list[str] = []
        total = 0
        packed: list[ScoredChunk] = []
        for item in chunks:
            metadata = dict(item.chunk.metadata or {})
            access_scope = metadata.get("access_scope")
            citation_access_scope = dict(access_scope) if isinstance(access_scope, dict) else {}
            policy = str(metadata.get("redaction_policy") or "quote_allowed")
            if policy == "deny":
                warnings.append(f"deny chunk excluded: {item.chunk.id}")
                continue
            text = item.chunk.text.strip()
            if policy in {"summary_only", "admin_only"}:
                text = _summary_only_text(text)
            block = (
                f"[{len(packed) + 1}] {metadata.get('source_title') or metadata.get('source_kind') or 'source'}\n"
                "<untrusted_retrieved_data>\n"
                f"{text}"
                "\n</untrusted_retrieved_data>"
            )
            if total + len(block) > self._config.max_context_characters:
                break
            texts.append(block)
            total += len(block)
            packed.append(item)
            if len(citations) < self._config.max_citations:
                citations.append(
                    Citation(
                        source_item_id=str(metadata.get("source_item_id") or ""),
                        chunk_id=item.chunk.id,
                        label=str(
                            metadata.get("source_title")
                            or metadata.get("source_kind")
                            or item.chunk.document_id
                        ),
                        url=str(
                            metadata.get("canonical_url")
                            or metadata.get("notion_url")
                            or metadata.get("hatenablog_url")
                            or metadata.get("crafters_colony_article_url")
                            or ""
                        ),
                        quote=_quote(text, self._config.max_quote_characters),
                        score=item.score,
                        access_scope=citation_access_scope,
                        metadata={
                            "source_type": str(
                                metadata.get("source_type")
                                or metadata.get("source_kind")
                                or ""
                            ),
                            "redaction_policy": str(metadata.get("redaction_policy") or ""),
                            "index_status": str(metadata.get("index_status") or ""),
                        },
                    )
                )
        return ContextPack(
            chunks=tuple(packed),
            text="\n\n".join(texts),
            citations=tuple(citations),
            warnings=tuple(warnings),
            metadata={"packed_characters": total},
        )


def _summary_only_text(text: str) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= 220:
        return cleaned
    return cleaned[:220].rstrip() + "..."


def _quote(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."
