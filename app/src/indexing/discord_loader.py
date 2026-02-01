from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import discord

from indexing.utils import ensure_dir

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_USER_MENTION_RE = re.compile(r"<@!?(\d+)>")


@dataclass
class ChannelState:
    last_message_id: int | None
    last_message_timestamp: str | None


@dataclass
class FetchStats:
    guilds: int = 0
    channels: int = 0
    messages: int = 0


def _strip_urls(text: str) -> str:
    if not text:
        return ""
    cleaned = _URL_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _state_path(base_dir: Path, guild_id: int, channel_id: int) -> Path:
    return base_dir / str(guild_id) / f"{channel_id}.state.json"


def _messages_path(base_dir: Path, guild_id: int, channel_id: int) -> Path:
    return base_dir / str(guild_id) / f"{channel_id}.jsonl"


def _load_channel_state(path: Path) -> ChannelState:
    if not path.exists():
        return ChannelState(last_message_id=None, last_message_timestamp=None)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read state file %s: %s", path.name, exc)
        return ChannelState(last_message_id=None, last_message_timestamp=None)
    if not isinstance(data, dict):
        return ChannelState(last_message_id=None, last_message_timestamp=None)

    last_id = data.get("last_message_id")
    if isinstance(last_id, str) and last_id.isdigit():
        last_id = int(last_id)
    elif not isinstance(last_id, int):
        last_id = None

    last_ts = data.get("last_message_timestamp")
    if not isinstance(last_ts, str):
        last_ts = None

    return ChannelState(last_message_id=last_id, last_message_timestamp=last_ts)


def _cleanup_channel_duplicates(
    base_dir: Path,
    guild_id: int,
    channel_id: int,
) -> None:
    channel_dir = base_dir / str(guild_id)
    if not channel_dir.exists():
        return
    channel_prefix = f"{channel_id}"
    keep_messages = f"{channel_id}.jsonl"
    keep_state = f"{channel_id}.state.json"
    for path in channel_dir.glob(f"{channel_prefix}*.jsonl"):
        if path.name == keep_messages:
            continue
        try:
            path.unlink()
            logger.info("Removed duplicate message file: %s", path.name)
        except Exception as exc:
            logger.warning("Failed to remove duplicate message file %s: %s", path.name, exc)
    for path in channel_dir.glob(f"{channel_prefix}.state*.json"):
        if path.name == keep_state:
            continue
        try:
            path.unlink()
            logger.info("Removed duplicate state file: %s", path.name)
        except Exception as exc:
            logger.warning("Failed to remove duplicate state file %s: %s", path.name, exc)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _build_after(state: ChannelState) -> discord.abc.Snowflake | datetime | None:
    if state.last_message_id:
        return discord.Object(id=state.last_message_id)
    timestamp = _parse_timestamp(state.last_message_timestamp)
    if timestamp:
        return timestamp
    return None


def _write_channel_state(path: Path, *, last_message_id: int, last_timestamp: str) -> None:
    payload = {
        "last_message_id": str(last_message_id),
        "last_message_timestamp": last_timestamp,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _is_human_message(message: discord.Message) -> bool:
    author = message.author
    if author is None:
        return False
    if getattr(author, "bot", False):
        return False
    if message.webhook_id is not None:
        return False
    try:
        if message.is_system():
            return False
    except AttributeError:
        pass
    message_type = getattr(message, "type", None)
    if message_type is not None:
        allowed_types = {discord.MessageType.default}
        if hasattr(discord.MessageType, "reply"):
            allowed_types.add(discord.MessageType.reply)
        if message_type not in allowed_types:
            return False
    return True


def _channel_metadata(
    *,
    guild: discord.Guild,
    channel: discord.abc.GuildChannel | discord.Thread,
) -> dict[str, object]:
    return {
        "guild_id": str(guild.id),
        "guild_name": guild.name,
        "channel_id": str(channel.id),
        "channel_name": channel.name,
        "source_type": "discord_message",
        "source_file_name": f"discord/{guild.id}/{channel.id}",
    }


def _message_record(
    *,
    text: str,
    message: discord.Message,
    base_metadata: dict[str, object],
) -> dict[str, object]:
    author = message.author
    author_name = getattr(author, "display_name", getattr(author, "name", ""))
    timestamp = message.created_at.replace(tzinfo=timezone.utc).isoformat()
    metadata = dict(base_metadata)
    metadata.update(
        {
            "chunk_stage": "discord_message",
            "chunk_id": int(message.id),
            "message_id": str(message.id),
            "message_timestamp": timestamp,
            "author_id": str(getattr(author, "id", "")),
            "author_name": author_name,
        }
    )
    return {"text": text, "metadata": metadata}


def _has_read_permissions(
    channel: discord.abc.GuildChannel | discord.Thread,
    member: discord.Member,
) -> bool:
    perms = channel.permissions_for(member)
    return bool(perms.view_channel and perms.read_message_history)


class _DiscordMessageCollector(discord.Client):
    def __init__(
        self,
        *,
        output_dir: Path,
        allowed_guild_ids: set[int] | None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._output_dir = output_dir
        self._allowed_guild_ids = allowed_guild_ids
        self._stats = FetchStats()
        self._error: Exception | None = None
        self._author_name_map: dict[int, dict[int, tuple[str, datetime]]] = {}

    @property
    def stats(self) -> FetchStats:
        return self._stats

    @property
    def error(self) -> Exception | None:
        return self._error

    def _author_map_for_guild(self, guild_id: int) -> dict[int, tuple[str, datetime]]:
        return self._author_name_map.setdefault(guild_id, {})

    def _get_cached_author_name(self, *, guild_id: int, author_id: int) -> str | None:
        cached = self._author_map_for_guild(guild_id).get(author_id)
        if cached is None:
            return None
        return cached[0]

    def _update_author_name(
        self,
        *,
        guild_id: int,
        author_id: int,
        author_name: str,
        observed_at: datetime | None,
    ) -> None:
        name = (author_name or "").strip()
        if not name:
            return
        timestamp = observed_at or datetime.now(timezone.utc)
        author_map = self._author_map_for_guild(guild_id)
        existing = author_map.get(author_id)
        if existing is not None:
            _, existing_ts = existing
            if existing_ts and timestamp <= existing_ts:
                return
        author_map[author_id] = (name, timestamp)

    def _cache_member_name(
        self,
        *,
        guild_id: int,
        member: discord.User | discord.Member,
        observed_at: datetime | None,
    ) -> None:
        author_id = getattr(member, "id", None)
        if author_id is None:
            return
        author_name = getattr(member, "display_name", None) or getattr(member, "name", "")
        self._update_author_name(
            guild_id=guild_id,
            author_id=int(author_id),
            author_name=str(author_name),
            observed_at=observed_at,
        )

    async def _fetch_member_name(
        self,
        *,
        guild: discord.Guild,
        author_id: int,
        observed_at: datetime | None,
    ) -> str | None:
        member = guild.get_member(author_id)
        if member is None:
            try:
                member = await guild.fetch_member(author_id)
            except Exception:
                member = None
        if member is None:
            try:
                member = await self.fetch_user(author_id)
            except Exception:
                member = None
        if member is None:
            return None
        author_name = getattr(member, "display_name", None) or getattr(member, "name", "")
        author_name = str(author_name).strip()
        if not author_name:
            return None
        self._update_author_name(
            guild_id=guild.id,
            author_id=author_id,
            author_name=author_name,
            observed_at=observed_at,
        )
        return author_name

    async def _replace_user_mentions(
        self,
        *,
        guild: discord.Guild,
        text: str,
        observed_at: datetime | None,
    ) -> str:
        if not text:
            return text
        mention_ids = _USER_MENTION_RE.findall(text)
        if not mention_ids:
            return text

        unique_ids = list(dict.fromkeys(mention_ids))
        replacements: dict[str, str] = {}
        for raw_id in unique_ids:
            try:
                author_id = int(raw_id)
            except ValueError:
                continue
            cached = self._get_cached_author_name(
                guild_id=guild.id,
                author_id=author_id,
            )
            if cached:
                replacements[raw_id] = cached
                continue
            fetched = await self._fetch_member_name(
                guild=guild,
                author_id=author_id,
                observed_at=observed_at,
            )
            if fetched:
                replacements[raw_id] = fetched

        if not replacements:
            return text

        def _sub(match: re.Match[str]) -> str:
            raw_id = match.group(1)
            replacement = replacements.get(raw_id)
            if not replacement:
                return match.group(0)
            return replacement

        return _USER_MENTION_RE.sub(_sub, text)

    async def on_ready(self) -> None:
        try:
            await self._collect()
        except Exception as exc:
            logger.exception("Failed to collect Discord messages")
            self._error = exc
        finally:
            await self.close()

    async def _collect(self) -> None:
        ensure_dir(self._output_dir)
        for guild in self.guilds:
            if self._allowed_guild_ids is not None:
                if guild.id not in self._allowed_guild_ids:
                    continue
            member = await self._resolve_member(guild)
            if member is None:
                logger.warning("Skipping guild %s: bot member not available", guild.id)
                continue
            self._stats.guilds += 1
            await self._collect_guild(guild, member)

    async def _resolve_member(self, guild: discord.Guild) -> discord.Member | None:
        member = guild.me or guild.get_member(self.user.id if self.user else 0)
        if member is not None:
            return member
        if not self.user:
            return None
        try:
            return await guild.fetch_member(self.user.id)
        except Exception as exc:
            logger.warning("Failed to fetch bot member for guild %s: %s", guild.id, exc)
            return None

    async def _collect_guild(
        self,
        guild: discord.Guild,
        member: discord.Member,
    ) -> None:
        seen_channels: set[int] = set()

        for channel in guild.text_channels:
            if not _has_read_permissions(channel, member):
                continue
            await self._collect_channel(guild, channel, member)
            seen_channels.add(channel.id)
            await self._collect_threads(guild, channel, member, seen_channels)

        for thread in guild.threads:
            if thread.id in seen_channels:
                continue
            if not _has_read_permissions(thread, member):
                continue
            await self._collect_channel(guild, thread, member)
            seen_channels.add(thread.id)

    async def _collect_threads(
        self,
        guild: discord.Guild,
        channel: discord.TextChannel,
        member: discord.Member,
        seen_channels: set[int],
    ) -> None:
        for thread in channel.threads:
            if thread.id in seen_channels:
                continue
            if not _has_read_permissions(thread, member):
                continue
            await self._collect_channel(guild, thread, member)
            seen_channels.add(thread.id)

        for private in (False, True):
            try:
                iterator = channel.archived_threads(limit=None, private=private)
            except TypeError:
                if private:
                    continue
                iterator = channel.archived_threads(limit=None)
            try:
                async for thread in iterator:
                    if thread.id in seen_channels:
                        continue
                    if not _has_read_permissions(thread, member):
                        continue
                    await self._collect_channel(guild, thread, member)
                    seen_channels.add(thread.id)
            except discord.Forbidden:
                continue
            except Exception as exc:
                logger.warning(
                    "Failed to list archived threads for channel %s: %s",
                    channel.id,
                    exc,
                )

    async def _collect_channel(
        self,
        guild: discord.Guild,
        channel: discord.abc.GuildChannel | discord.Thread,
        member: discord.Member,
    ) -> None:
        if not _has_read_permissions(channel, member):
            return
        _cleanup_channel_duplicates(self._output_dir, guild.id, channel.id)

        state_path = _state_path(self._output_dir, guild.id, channel.id)
        state = _load_channel_state(state_path)
        after = _build_after(state)

        records: list[tuple[dict[str, object], datetime]] = []
        last_seen_id: int | None = None
        last_seen_ts: str | None = None
        base_metadata = _channel_metadata(guild=guild, channel=channel)

        try:
            async for message in channel.history(
                limit=None,
                after=after,
                oldest_first=True,
            ):
                last_seen_id = message.id
                observed_at = message.created_at
                if observed_at.tzinfo is None:
                    observed_at = observed_at.replace(tzinfo=timezone.utc)
                else:
                    observed_at = observed_at.astimezone(timezone.utc)
                last_seen_ts = observed_at.isoformat()
                if not _is_human_message(message):
                    continue
                if message.author is not None:
                    self._cache_member_name(
                        guild_id=guild.id,
                        member=message.author,
                        observed_at=observed_at,
                    )
                for mentioned in message.mentions:
                    self._cache_member_name(
                        guild_id=guild.id,
                        member=mentioned,
                        observed_at=observed_at,
                    )
                cleaned = _strip_urls(message.content or "")
                if not cleaned:
                    continue
                record = _message_record(
                    text=cleaned,
                    message=message,
                    base_metadata=base_metadata,
                )
                records.append((record, observed_at))
        except discord.Forbidden:
            return
        except Exception as exc:
            logger.warning("Failed to read channel %s: %s", channel.id, exc)
            return

        if last_seen_id is None or last_seen_ts is None:
            return

        ensure_dir(state_path.parent)
        if records:
            for record, observed_at in records:
                text_value = record.get("text")
                if isinstance(text_value, str) and text_value:
                    record["text"] = await self._replace_user_mentions(
                        guild=guild,
                        text=text_value,
                        observed_at=observed_at,
                    )
            messages_path = _messages_path(self._output_dir, guild.id, channel.id)
            with messages_path.open("a", encoding="utf-8") as fw:
                for record, _ in records:
                    fw.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._stats.messages += len(records)
        _write_channel_state(
            state_path,
            last_message_id=last_seen_id,
            last_timestamp=last_seen_ts,
        )
        self._stats.channels += 1


async def download_discord_messages(
    *,
    token: str,
    output_dir: Path,
    allowed_guild_ids: set[int] | None = None,
) -> FetchStats:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    client = _DiscordMessageCollector(
        output_dir=output_dir,
        allowed_guild_ids=allowed_guild_ids,
        intents=intents,
    )
    await client.start(token)
    if client.error:
        raise client.error
    return client.stats
