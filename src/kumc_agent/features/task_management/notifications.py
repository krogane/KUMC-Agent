from __future__ import annotations

from dataclasses import dataclass, field
import json
import urllib.error
import urllib.request
from typing import Protocol


@dataclass(frozen=True)
class TaskNotificationMessage:
    kind: str
    task_id: str
    title: str
    channel_id: str
    content: str
    custom_id: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskNotificationDelivery:
    status: str
    channel: str = ""
    message_id: str = ""
    error: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


class TaskNotificationSender(Protocol):
    def send(self, message: TaskNotificationMessage) -> TaskNotificationDelivery:
        ...


class DiscordTaskNotificationSender:
    def __init__(self, *, bot_token: str, api_base_url: str = "https://discord.com/api/v10") -> None:
        self._bot_token = bot_token.strip()
        self._api_base_url = api_base_url.rstrip("/")

    def send(self, message: TaskNotificationMessage) -> TaskNotificationDelivery:
        if not self._bot_token:
            return TaskNotificationDelivery(status="not_configured", channel="discord")
        if not message.channel_id:
            return TaskNotificationDelivery(status="not_configured", channel="discord")
        payload: dict[str, object] = {
            "content": message.content[:1900],
            "allowed_mentions": {"parse": []},
        }
        raw_components = message.metadata.get("components")
        if isinstance(raw_components, list):
            payload["components"] = raw_components
        elif message.custom_id:
            payload["components"] = [
                {
                    "type": 1,
                    "components": [
                        {
                            "type": 2,
                            "style": 3,
                            "label": "完了",
                            "custom_id": message.custom_id[:100],
                        }
                    ],
                }
            ]
        request = urllib.request.Request(
            f"{self._api_base_url}/channels/{message.channel_id}/messages",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bot {self._bot_token}",
                "Content-Type": "application/json",
                "User-Agent": "KUMC-Agent task notification",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                body = json.loads(response.read().decode("utf-8") or "{}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return TaskNotificationDelivery(
                status="failed",
                channel="discord",
                error=str(exc)[:500],
            )
        return TaskNotificationDelivery(
            status="sent",
            channel="discord",
            message_id=str(body.get("id") or ""),
            metadata={"kind": message.kind, "task_id": message.task_id},
        )


class NullTaskNotificationSender:
    def send(self, message: TaskNotificationMessage) -> TaskNotificationDelivery:
        return TaskNotificationDelivery(
            status="not_configured",
            channel="none",
            metadata={"kind": message.kind, "task_id": message.task_id},
        )
