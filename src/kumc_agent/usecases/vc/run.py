from __future__ import annotations

from typing import Callable

from kumc_agent.features.vc.service import VCService


class VCUsecase:
    def __init__(self, *, service: VCService) -> None:
        self._service = service

    def bind_discord_client(
        self,
        *,
        discord_client: object,
        is_indexing_active: Callable[[], bool],
    ) -> None:
        self._service.bind_discord_client(
            discord_client=discord_client,
            is_indexing_active=is_indexing_active,
        )

    async def start(self) -> None:
        await self._service.start()

    async def stop(self) -> None:
        await self._service.stop()

    async def on_voice_state_update(self, member: object, before: object, after: object) -> None:
        await self._service.on_voice_state_update(member, before, after)

    async def capture_voice_chat_message(self, message: object) -> None:
        await self._service.capture_voice_chat_message(message)

    async def maybe_join_from_command(self, message: object) -> bool:
        return await self._service.maybe_join_from_command(message)

    async def maybe_quit_from_command(self, message: object) -> bool:
        return await self._service.maybe_quit_from_command(message)

    def has_active_session(self) -> bool:
        return self._service.has_active_session()

    def has_model_activity(self) -> bool:
        return self._service.has_model_activity()

    def is_voice_chat_channel(self, channel: object) -> bool:
        return self._service.is_voice_chat_channel(channel)

    def should_use_fast_model_for_query(self) -> bool:
        return self._service.should_use_fast_model_for_query()

    def notify_rag_started(self) -> None:
        self._service.notify_rag_started()

    def notify_rag_finished(self) -> None:
        self._service.notify_rag_finished()
