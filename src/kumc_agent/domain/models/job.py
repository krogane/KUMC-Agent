from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class JobRecord:
    job_type: str
    status: str
    job_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
