from __future__ import annotations

from typing import Any, Callable

from kumc_agent.features.vc.config import VCManagerConfig


class VCService:
    def __init__(self, *, config: VCManagerConfig) -> None:
        self._config = config
        self._manager: Any | None = None

    def bind_discord_client(
        self,
        *,
        discord_client: object,
        is_indexing_active: Callable[[], bool],
    ) -> None:
        if self._manager is not None:
            return
        from kumc_agent.infra.vc.manager import VoiceMeetingManager

        self._manager = VoiceMeetingManager(
            discord_client=discord_client,
            config=self._config,
            is_indexing_active=is_indexing_active,
        )

    async def start(self) -> None:
        manager = self._manager
        if manager is None:
            return
        await manager.start()

    async def stop(self) -> None:
        manager = self._manager
        if manager is None:
            return
        await manager.stop()

    async def on_voice_state_update(self, member: object, before: object, after: object) -> None:
        manager = self._manager
        if manager is None:
            return
        await manager.on_voice_state_update(member, before, after)

    async def capture_voice_chat_message(self, message: object) -> None:
        manager = self._manager
        if manager is None:
            return
        await manager.capture_voice_chat_message(message)

    async def maybe_join_from_command(self, message: object) -> bool:
        manager = self._manager
        if manager is None:
            return False
        return bool(await manager.maybe_join_from_command(message))

    async def maybe_quit_from_command(self, message: object) -> bool:
        manager = self._manager
        if manager is None:
            return False
        return bool(await manager.maybe_quit_from_command(message))

    def has_active_session(self) -> bool:
        manager = self._manager
        if manager is None:
            return False
        return bool(manager.has_active_session())

    def has_model_activity(self) -> bool:
        manager = self._manager
        if manager is None:
            return False
        return bool(manager.has_model_activity())

    def is_voice_chat_channel(self, channel: object) -> bool:
        manager = self._manager
        if manager is None:
            return False
        return bool(manager.is_voice_chat_channel(channel))

    def should_use_fast_model_for_query(self) -> bool:
        manager = self._manager
        if manager is None:
            return False
        return bool(manager.should_use_fast_model_for_query())

    def notify_rag_started(self) -> None:
        manager = self._manager
        if manager is None:
            return
        manager.notify_rag_started()

    def notify_rag_finished(self) -> None:
        manager = self._manager
        if manager is None:
            return
        manager.notify_rag_finished()
