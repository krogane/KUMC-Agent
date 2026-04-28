from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import threading
from typing import Protocol


@dataclass(frozen=True)
class EventNotificationMessage:
    kind: str
    event_id: str
    title: str
    channel_id: str
    content: str
    custom_id: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EventNotificationDelivery:
    status: str
    channel: str = ""
    message_id: str = ""
    error: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


class EventNotificationSender(Protocol):
    def send(self, message: EventNotificationMessage) -> EventNotificationDelivery:
        ...


class DiscordEventNotificationSender:
    def __init__(self, *, bot_token: str) -> None:
        self._bot_token = bot_token.strip()

    def send(self, message: EventNotificationMessage) -> EventNotificationDelivery:
        if not self._bot_token:
            return EventNotificationDelivery(status="not_configured", channel="discord")
        if not message.channel_id:
            return EventNotificationDelivery(status="not_configured", channel="discord")
        return _run_async_blocking(self._send_async(message))

    async def _send_async(self, message: EventNotificationMessage) -> EventNotificationDelivery:
        try:
            import discord
        except ImportError as exc:  # pragma: no cover - deployment dependency
            return EventNotificationDelivery(
                status="failed",
                channel="discord",
                error=str(exc)[:500],
            )

        intents = discord.Intents.none()
        client = discord.Client(intents=intents)
        result: dict[str, EventNotificationDelivery] = {}

        @client.event
        async def on_ready() -> None:  # pragma: no cover - network path
            try:
                channel = await client.fetch_channel(int(message.channel_id))
                view = _event_view_from_message(discord, message)
                sent = await channel.send(
                    content=message.content[:1900],
                    view=view,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
                result["delivery"] = EventNotificationDelivery(
                    status="sent",
                    channel="discord",
                    message_id=str(sent.id),
                    metadata={"kind": message.kind, "event_id": message.event_id},
                )
            except Exception as exc:
                result["delivery"] = EventNotificationDelivery(
                    status="failed",
                    channel="discord",
                    error=str(exc)[:500],
                )
            finally:
                await client.close()

        try:
            await client.start(self._bot_token)
        except Exception as exc:  # pragma: no cover - network path
            return EventNotificationDelivery(
                status="failed",
                channel="discord",
                error=str(exc)[:500],
            )
        return result.get(
            "delivery",
            EventNotificationDelivery(status="failed", channel="discord", error="send_not_completed"),
        )


class NullEventNotificationSender:
    def send(self, message: EventNotificationMessage) -> EventNotificationDelivery:
        return EventNotificationDelivery(
            status="not_configured",
            channel="none",
            metadata={"kind": message.kind, "event_id": message.event_id},
        )


def _event_view_from_message(discord, message: EventNotificationMessage):
    buttons = message.metadata.get("buttons")
    if not isinstance(buttons, list) and message.custom_id:
        buttons = [
            {
                "label": "完了",
                "style": 3,
                "custom_id": message.custom_id,
            }
        ]
    if not isinstance(buttons, list) or not buttons:
        return None
    view = discord.ui.View(timeout=None)
    for raw in buttons[:25]:
        if not isinstance(raw, dict):
            continue
        custom_id = str(raw.get("custom_id") or "")[:100]
        if not custom_id:
            continue
        view.add_item(
            discord.ui.Button(
                label=str(raw.get("label") or "Open")[:80],
                style=_button_style(discord, int(raw.get("style") or 2)),
                custom_id=custom_id,
            )
        )
    return view


def _button_style(discord, style: int):
    return {
        1: discord.ButtonStyle.primary,
        2: discord.ButtonStyle.secondary,
        3: discord.ButtonStyle.success,
        4: discord.ButtonStyle.danger,
    }.get(style, discord.ButtonStyle.secondary)


def _run_async_blocking(coro) -> EventNotificationDelivery:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, EventNotificationDelivery] = {}

    def _runner() -> None:
        result["delivery"] = asyncio.run(coro)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    return result.get(
        "delivery",
        EventNotificationDelivery(status="failed", channel="discord", error="thread_send_failed"),
    )
