from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source_type: str
    source_name: str
    source_uri: str = ""
    updated_at: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)
