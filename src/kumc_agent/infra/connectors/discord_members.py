from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timezone
import logging

import discord

from kumc_agent.features.member_search import DiscordMemberRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscordMemberDirectoryConnector:
    bot_token: str
    allowed_guild_ids: tuple[str, ...] = tuple()

    def list_members(self, *, guild_id: str) -> list[DiscordMemberRecord]:
        if not self.bot_token:
            return []
        if self.allowed_guild_ids and str(guild_id) not in self.allowed_guild_ids:
            return []
        return asyncio.run(_fetch_members(token=self.bot_token, guild_id=str(guild_id)))


async def _fetch_members(*, token: str, guild_id: str) -> list[DiscordMemberRecord]:
    intents = discord.Intents.default()
    intents.guilds = True
    intents.members = True
    client = _MemberDirectoryClient(guild_id=guild_id, intents=intents)
    await client.start(token)
    if client.error is not None:
        raise client.error
    return client.members


class _MemberDirectoryClient(discord.Client):
    def __init__(self, *, guild_id: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._guild_id = str(guild_id)
        self.members: list[DiscordMemberRecord] = []
        self.error: Exception | None = None

    async def on_ready(self) -> None:
        try:
            guild = self.get_guild(int(self._guild_id))
            if guild is None:
                guild = await self.fetch_guild(int(self._guild_id))
            self.members = await self._collect_guild_members(guild)
        except Exception as exc:
            self.error = exc
        finally:
            await self.close()

    async def _collect_guild_members(self, guild: discord.Guild) -> list[DiscordMemberRecord]:
        out: list[DiscordMemberRecord] = []
        async for member in guild.fetch_members(limit=None):
            roles = [
                role.name
                for role in getattr(member, "roles", [])
                if getattr(role, "name", "") and role.name != "@everyone"
            ]
            role_ids = [
                str(role.id)
                for role in getattr(member, "roles", [])
                if getattr(role, "name", "") and role.name != "@everyone"
            ]
            joined_at = getattr(member, "joined_at", None)
            if joined_at is not None and joined_at.tzinfo is None:
                joined_at = joined_at.replace(tzinfo=timezone.utc)
            out.append(
                DiscordMemberRecord(
                    guild_id=str(guild.id),
                    user_id=str(member.id),
                    display_name=str(getattr(member, "display_name", "") or getattr(member, "name", "")),
                    username=str(getattr(member, "name", "") or ""),
                    roles=tuple(roles),
                    role_ids=tuple(role_ids),
                    joined_at=joined_at,
                    is_bot=bool(getattr(member, "bot", False)),
                    is_active=True,
                    metadata={"guild_name": guild.name},
                )
            )
        return out
