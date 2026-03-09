from __future__ import annotations

import asyncio
from pathlib import Path


class DiscordLoader:
    def __init__(
        self,
        *,
        bot_token: str,
        raw_dir: Path,
        allow_guild_ids: list[int],
    ) -> None:
        self._bot_token = bot_token
        self._raw_dir = raw_dir
        self._allow_guild_ids = allow_guild_ids

    def load(self) -> int:
        if not self._bot_token:
            return 0
        from kumc_agent.infra.loaders.discord_impl import (
            download_discord_messages,
        )

        output_dir = self._raw_dir / "messages"
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = asyncio.run(
            download_discord_messages(
                token=self._bot_token,
                output_dir=output_dir,
                allowed_guild_ids=set(self._allow_guild_ids) if self._allow_guild_ids else None,
            )
        )
        return int(stats.messages)
