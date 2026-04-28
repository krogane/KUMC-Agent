from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kumc_agent.domain.models.chunk import Chunk


@dataclass(frozen=True)
class AccessContext:
    user_id: str = ""
    guild_id: str = ""
    role_ids: tuple[str, ...] = tuple()
    is_admin: bool = False


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    source_filter: str = "all"
    mode: str = "answer"
    depth: str = "normal"
    access: AccessContext = field(default_factory=AccessContext)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    rank: int
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    source_item_id: str
    chunk_id: str
    label: str
    url: str = ""
    quote: str = ""
    score: float | None = None
    access_scope: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextPack:
    chunks: tuple[ScoredChunk, ...]
    text: str
    citations: tuple[Citation, ...]
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AskResponse:
    text: str
    detail_markdown: str
    citations: tuple[Citation, ...]
    confidence: str
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)
