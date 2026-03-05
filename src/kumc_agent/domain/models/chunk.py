from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    index: int
    metadata: dict[str, object] = field(default_factory=dict)
