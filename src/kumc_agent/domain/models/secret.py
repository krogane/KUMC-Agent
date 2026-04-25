from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class SecretFinding:
    source_item_id: str
    secret_type: str
    severity: str
    redaction_policy: str
    detected_span_hash: str
    chunk_id: str | None = None
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
