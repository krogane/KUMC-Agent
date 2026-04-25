from __future__ import annotations

from dataclasses import dataclass
import re

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.source import NormalizedDocument
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class ChunkingSettings:
    max_characters: int = 1800
    overlap_characters: int = 120


class IngestionChunker:
    def __init__(self, settings: ChunkingSettings | None = None) -> None:
        self._settings = settings or ChunkingSettings()

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        text = (document.normalized_text or "").strip()
        if not text:
            return []
        sections = self._sections(document=document, text=text)
        chunks: list[Chunk] = []
        for section_title, section_text in sections:
            for piece in self._split_text(section_text):
                index = len(chunks)
                chunk_kind = self._chunk_kind(document=document, section_title=section_title)
                chunk_id = stable_hash(
                    f"chunk:{document.id}:{index}:{chunk_kind}:{piece[:256]}"
                )
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document.id,
                        text=piece,
                        index=index,
                        metadata={
                            "source_item_id": document.source_item_id,
                            "source_kind": document.source_kind,
                            "source_type": document.source_kind,
                            "external_id": document.external_id,
                            "source_title": document.title,
                            "chunk_kind": chunk_kind,
                            "heading_path": [section_title] if section_title else [],
                            "access_scope": document.access_scope.as_dict(),
                            "checksum": stable_hash(piece),
                            "token_count": _token_count(piece),
                            **document.metadata,
                        },
                    )
                )
        return chunks

    def _sections(
        self,
        *,
        document: NormalizedDocument,
        text: str,
    ) -> list[tuple[str, str]]:
        if document.source_kind == "discord":
            return [("", text)]
        matches = list(re.finditer(r"(?m)^(#{1,4})\s+(.+)$", text))
        if not matches:
            return [("", text)]
        sections: list[tuple[str, str]] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            title = match.group(2).strip()
            sections.append((title, text[start:end].strip()))
        return sections or [("", text)]

    def _split_text(self, text: str) -> list[str]:
        max_chars = max(200, self._settings.max_characters)
        overlap = max(0, min(self._settings.overlap_characters, max_chars // 2))
        cleaned = text.strip()
        if len(cleaned) <= max_chars:
            return [cleaned]
        chunks: list[str] = []
        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + max_chars)
            slice_text = cleaned[start:end]
            if end < len(cleaned):
                split_at = max(slice_text.rfind("\n\n"), slice_text.rfind("\n"), slice_text.rfind("。"))
                if split_at > max_chars // 2:
                    end = start + split_at + 1
                    slice_text = cleaned[start:end]
            chunks.append(slice_text.strip())
            if end >= len(cleaned):
                break
            start = max(0, end - overlap)
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _chunk_kind(*, document: NormalizedDocument, section_title: str) -> str:
        if document.source_kind == "discord":
            return "message_window"
        if document.source_kind == "x":
            return "tweet"
        if document.source_kind == "minecraft_wiki":
            return "wiki_section" if section_title else "wiki_article"
        if document.normalized_format == "csv_as_text":
            return "table"
        return "heading" if section_title else "body"


def _token_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", text.strip()) if token])
