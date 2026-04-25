from __future__ import annotations

from kumc_agent.domain.models.chunk import Chunk


def is_chunk_allowed_for_answer_context(chunk: Chunk) -> bool:
    metadata = dict(chunk.metadata or {})
    index_status = str(metadata.get("index_status") or "active").strip().lower()
    if index_status in {"deleted", "quarantined", "permission_lost"}:
        return False
    redaction_policy = str(metadata.get("redaction_policy") or "quote_allowed").strip().lower()
    return redaction_policy != "deny"
