from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from zoneinfo import ZoneInfo

from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class AutonomousIdempotencyInput:
    slot: str
    scopes: tuple[str, ...]
    timezone: str = "Asia/Tokyo"
    guild_id: str = ""
    channel_id: str = ""
    lookahead_days: dict[str, int] | None = None
    now: datetime | None = None


def build_autonomous_idempotency_key(value: AutonomousIdempotencyInput) -> str:
    tz = ZoneInfo(value.timezone or "Asia/Tokyo")
    now = value.now or datetime.now(tz)
    local_now = now.astimezone(tz) if now.tzinfo else now.replace(tzinfo=tz)
    scope_payload = {
        "scopes": sorted(str(scope) for scope in value.scopes),
        "guild_id": value.guild_id,
        "channel_id": value.channel_id,
        "lookahead_days": dict(value.lookahead_days or {}),
    }
    scope_hash = stable_hash(json.dumps(scope_payload, sort_keys=True, ensure_ascii=False))[:16]
    return f"autonomous-agent:{local_now.date().isoformat()}:{value.slot or 'manual'}:{scope_hash}"
