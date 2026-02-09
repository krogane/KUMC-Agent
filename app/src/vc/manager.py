from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import discord
import numpy as np

from config import AppConfig
from indexing.llm_client import generate_text
from vc.prompts import (
    build_end_judgement_prompt,
    build_final_summary_prompt,
    build_summary_prompt,
)

logger = logging.getLogger(__name__)

try:
    from discord.ext import voice_recv
except Exception:  # pragma: no cover - optional dependency
    voice_recv = None

try:
    import torch
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline as hf_pipeline,
    )
except Exception:  # pragma: no cover - optional dependency
    torch = None
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    hf_pipeline = None


_JST = ZoneInfo("Asia/Tokyo")
_PCM_SAMPLE_RATE = 48_000
_PCM_CHANNELS = 2
_PCM_SAMPLE_WIDTH = 2
_PCM_FRAME_BYTES = _PCM_CHANNELS * _PCM_SAMPLE_WIDTH
_PCM_BYTES_PER_SECOND = _PCM_SAMPLE_RATE * _PCM_CHANNELS * _PCM_SAMPLE_WIDTH
_MAX_TRANSCRIBE_GAP_SECONDS = 1.0
_MIN_AUDIO_BLOCK_SECONDS = 0.35
_FINAL_SUMMARY_CHANNEL = "kumc-agent"
_SUMMARY_SYSTEM_PROMPT = "You are a concise meeting summarization assistant."
_END_JUDGEMENT_SYSTEM_PROMPT = "You are a meeting end-judgement assistant."
_FINAL_SUMMARY_SYSTEM_PROMPT = "You are a concise meeting minutes assistant."
_ANSWER_DISCLAIMER = (
    "※回答は必ずしも正しいとは限りません。重要な情報は確認するようにしてください。"
)


class _PauseRequested(RuntimeError):
    pass


@dataclass
class _AudioEvent:
    user_id: int
    user_name: str
    offset_seconds: float
    pcm: bytes
    duration_seconds: float


@dataclass
class _ChatEvent:
    user_name: str
    offset_seconds: float
    text: str


@dataclass
class _TranscriptLine:
    offset_seconds: float
    user_name: str
    text: str


@dataclass
class _PostProcessJob:
    meeting_key: str
    transcript_index: int
    transcript_path: Path
    chunk_start_offset_seconds: float = 0.0
    chunk_end_offset_seconds: float = 0.0
    is_final_chunk: bool = False
    marker_only: bool = False


@dataclass
class _LLMConfig:
    provider: str
    model: str
    llama_model_path: str
    llama_ctx_size: int
    temperature: float
    max_output_tokens: int
    thinking_level: str


@dataclass
class _MeetingArchive:
    meeting_key: str
    guild_id: int
    meeting_date: str
    meeting_label: str
    summary_chunk_path: Path
    pending_summary_texts: list[str] = field(default_factory=list)
    pending_summary_seconds: float = 0.0
    pending_summary_last_index: int = 0
    pending_end_judge_texts: list[str] = field(default_factory=list)
    pending_end_judge_seconds: float = 0.0


@dataclass
class _MeetingSession:
    meeting_key: str
    guild_id: int
    guild_name: str
    meeting_date: str
    meeting_label: str
    voice_channel: discord.VoiceChannel
    voice_client: discord.VoiceClient
    sink: object
    started_at: datetime
    start_monotonic: float
    meeting_dir: Path
    summary_chunk_path: Path
    transcript_interval_seconds: int
    next_transcript_index: int = 1
    last_flush_offset: float = 0.0
    active: bool = True
    ending: bool = False
    intentional_disconnect: bool = False
    transcribe_task: asyncio.Task[None] | None = None
    event_lock: threading.Lock = field(default_factory=threading.Lock)
    audio_events: dict[int, deque[_AudioEvent]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    chat_events: deque[_ChatEvent] = field(default_factory=deque)


class _FallbackAudioSink:
    def cleanup(self) -> None:
        return


if voice_recv is not None:

    class _VoiceReceiveSink(voice_recv.AudioSink):
        def __init__(self, on_audio_frame: Callable[[int, str, bytes], None]) -> None:
            super().__init__()
            self._on_audio_frame = on_audio_frame
            self._opus_decoders: dict[int, discord.opus.Decoder] = {}
            self._decoder_lock = threading.Lock()

        def wants_opus(self) -> bool:
            return True

        def _decoder_for_stream(self, stream_key: int) -> discord.opus.Decoder:
            with self._decoder_lock:
                decoder = self._opus_decoders.get(stream_key)
                if decoder is None:
                    decoder = discord.opus.Decoder()
                    self._opus_decoders[stream_key] = decoder
                return decoder

        def write(self, user, data) -> None:
            if user is None:
                return
            if getattr(user, "bot", False):
                return
            opus = getattr(data, "opus", None)
            if not isinstance(opus, (bytes, bytearray)):
                return
            if not opus:
                return

            user_id = getattr(user, "id", None)
            if user_id is None:
                return

            stream_key = int(getattr(getattr(data, "packet", None), "ssrc", user_id) or user_id)
            try:
                decoder = self._decoder_for_stream(stream_key)
                pcm = decoder.decode(bytes(opus), fec=False)
            except discord.opus.OpusError:
                return
            except Exception:
                logger.exception("Failed to decode opus packet: stream=%s user=%s", stream_key, user_id)
                return
            if not pcm:
                return

            user_name = (
                str(getattr(user, "display_name", "") or "").strip()
                or str(getattr(user, "name", "") or "").strip()
                or str(user_id)
            )
            self._on_audio_frame(int(user_id), user_name, pcm)

        def cleanup(self) -> None:
            with self._decoder_lock:
                self._opus_decoders.clear()
            try:
                super().cleanup()
            except Exception:
                return

else:

    class _VoiceReceiveSink(_FallbackAudioSink):
        def __init__(self, on_audio_frame: Callable[[int, str, bytes], None]) -> None:
            self._on_audio_frame = on_audio_frame


class VoiceMeetingManager:
    def __init__(
        self,
        *,
        discord_client: discord.Client,
        config: AppConfig,
        is_indexing_active: Callable[[], bool],
    ) -> None:
        self._discord_client = discord_client
        self._config = config
        self._is_indexing_active = is_indexing_active

        self._sessions_by_guild: dict[int, _MeetingSession] = {}
        self._sessions_by_channel: dict[int, _MeetingSession] = {}
        self._archives: dict[str, _MeetingArchive] = {}
        self._autojoin_blocked_date: dict[tuple[int, int], str] = {}

        self._monitor_task: asyncio.Task[None] | None = None
        self._post_worker_task: asyncio.Task[None] | None = None

        self._post_jobs: deque[_PostProcessJob] = deque()
        self._post_jobs_cond = asyncio.Condition()
        self._rag_pause_count = 0
        self._post_worker_current: _PostProcessJob | None = None

        self._asr_pipeline = None
        self._asr_pipeline_lock = threading.Lock()
        self._asr_inference_lock = threading.Lock()

        self._summary_llm_cfg = _LLMConfig(
            provider=(self._config.vc_summary_llm_provider or "").lower(),
            model=self._select_llm_model(
                provider=(self._config.vc_summary_llm_provider or "").lower(),
                gemini_model=self._config.vc_summary_gemini_model,
                llama_model=self._config.vc_summary_llama_model,
            ),
            llama_model_path=self._config.vc_summary_llama_model_path,
            llama_ctx_size=self._config.vc_summary_llama_ctx_size,
            temperature=self._config.vc_summary_temperature,
            max_output_tokens=self._config.vc_summary_max_output_tokens,
            thinking_level=self._config.vc_summary_thinking_level,
        )
        self._end_llm_cfg = _LLMConfig(
            provider=(self._config.vc_end_judge_llm_provider or "").lower(),
            model=self._select_llm_model(
                provider=(self._config.vc_end_judge_llm_provider or "").lower(),
                gemini_model=self._config.vc_end_judge_gemini_model,
                llama_model=self._config.vc_end_judge_llama_model,
            ),
            llama_model_path=self._config.vc_end_judge_llama_model_path,
            llama_ctx_size=self._config.vc_end_judge_llama_ctx_size,
            temperature=self._config.vc_end_judge_temperature,
            max_output_tokens=self._config.vc_end_judge_max_output_tokens,
            thinking_level=self._config.vc_end_judge_thinking_level,
        )
        self._final_llm_cfg = _LLMConfig(
            provider=(self._config.vc_final_summary_llm_provider or "").lower(),
            model=self._select_llm_model(
                provider=(self._config.vc_final_summary_llm_provider or "").lower(),
                gemini_model=self._config.vc_final_summary_gemini_model,
                llama_model=self._config.vc_final_summary_llama_model,
            ),
            llama_model_path=self._config.vc_final_summary_llama_model_path,
            llama_ctx_size=self._config.vc_final_summary_llama_ctx_size,
            temperature=self._config.vc_final_summary_temperature,
            max_output_tokens=self._config.vc_final_summary_max_output_tokens,
            thinking_level=self._config.vc_final_summary_thinking_level,
        )

    async def start(self) -> None:
        if not self._config.vc_feature_enabled:
            logger.info("VC feature disabled.")
            return

        if voice_recv is None:
            logger.warning(
                "VC feature enabled, but discord-ext-voice-recv is unavailable."
            )
            return

        if hf_pipeline is None or AutoProcessor is None or AutoModelForSpeechSeq2Seq is None:
            logger.warning(
                "VC feature enabled, but transformers ASR dependencies are unavailable."
            )
            return

        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._auto_join_loop())
        if self._post_worker_task is None or self._post_worker_task.done():
            self._post_worker_task = asyncio.create_task(self._post_process_worker())

    async def stop(self) -> None:
        monitor = self._monitor_task
        if monitor and not monitor.done():
            monitor.cancel()

        worker = self._post_worker_task
        if worker and not worker.done():
            worker.cancel()

        sessions = list(self._sessions_by_guild.values())
        for session in sessions:
            await self._leave_session(
                session,
                reason="shutdown",
                process_final_transcript=True,
            )

    def has_active_session(self) -> bool:
        return any(session.active for session in self._sessions_by_guild.values())

    def is_voice_chat_channel(self, channel: object) -> bool:
        return isinstance(channel, discord.VoiceChannel)

    async def capture_voice_chat_message(self, message: discord.Message) -> None:
        session = self._sessions_by_channel.get(getattr(message.channel, "id", 0))
        if session is None or not session.active:
            return

        content = (message.content or "").strip()
        if not content:
            return

        author_name = (
            str(getattr(message.author, "display_name", "") or "").strip()
            or str(getattr(message.author, "name", "") or "").strip()
            or str(getattr(message.author, "id", "unknown"))
        )
        offset = max(0.0, time.monotonic() - session.start_monotonic)

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return

        with session.event_lock:
            for line in lines:
                session.chat_events.append(
                    _ChatEvent(user_name=author_name, offset_seconds=offset, text=line)
                )

    async def maybe_join_from_command(self, message: discord.Message) -> bool:
        if not self.is_voice_chat_channel(message.channel):
            return False
        if not self._config.vc_feature_enabled:
            await message.channel.send("VC機能は無効化されています。")
            return True

        if self._is_indexing_active():
            await message.channel.send(
                "インデックス更新中のため、VC参加機能は停止しています。"
            )
            return True

        voice_channel = message.channel
        guild = message.guild
        if guild is None:
            await message.channel.send("サーバー内のVCチャットから実行してください。")
            return True

        current = self._sessions_by_guild.get(guild.id)
        if current and current.active:
            if current.voice_channel.id == voice_channel.id:
                await message.channel.send("既にこのVCに参加しています。")
                return True
            await message.channel.send("既に別のVCに参加中のため、移動せず拒否します。")
            return True

        await self._join_voice_channel(voice_channel, trigger="manual")
        return True

    async def maybe_quit_from_command(self, message: discord.Message) -> bool:
        if not self.is_voice_chat_channel(message.channel):
            return False

        guild = message.guild
        if guild is None:
            return True

        session = self._sessions_by_guild.get(guild.id)
        if session is None or not session.active:
            await message.channel.send("このサーバーで参加中のVCはありません。")
            return True

        if session.voice_channel.id != message.channel.id:
            await message.channel.send("現在参加中のVCではないため、退出しません。")
            return True

        await self._leave_session(
            session,
            reason="manual_quit",
            process_final_transcript=True,
        )
        return True

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        before_channel = getattr(before, "channel", None)
        if before_channel is None:
            return

        session = self._sessions_by_channel.get(before_channel.id)
        if session is None:
            return

        # Auto-quit when no human participants remain in the active VC.
        if self._config.vc_auto_quit_enabled and not session.ending:
            remaining = [m for m in before_channel.members if not getattr(m, "bot", False)]
            if not remaining:
                await self._leave_session(
                    session,
                    reason="no_participants",
                    process_final_transcript=True,
                )
                return

        bot_user = self._discord_client.user
        if bot_user is None:
            return
        if int(member.id) != int(bot_user.id):
            return

        after_channel = getattr(after, "channel", None)
        if after_channel is not None and after_channel.id == before_channel.id:
            return

        if session.intentional_disconnect:
            return

        today_key = datetime.now(_JST).strftime("%Y-%m-%d")
        self._autojoin_blocked_date[(session.guild_id, session.voice_channel.id)] = today_key

        await self._leave_session(
            session,
            reason="forced_disconnect",
            process_final_transcript=True,
        )

    def notify_rag_started(self) -> None:
        self._rag_pause_count += 1

    def notify_rag_finished(self) -> None:
        self._rag_pause_count = max(0, self._rag_pause_count - 1)

    async def _auto_join_loop(self) -> None:
        logger.info(
            "VC auto-join monitor started. enabled=%s interval=%ss",
            self._config.vc_auto_join_enabled,
            self._config.vc_participant_check_interval_seconds,
        )

        while True:
            await asyncio.sleep(
                max(2, self._config.vc_participant_check_interval_seconds)
            )

            if not self._config.vc_auto_join_enabled:
                continue
            if self._is_indexing_active():
                continue
            if (
                voice_recv is None
                or hf_pipeline is None
                or AutoProcessor is None
                or AutoModelForSpeechSeq2Seq is None
            ):
                continue

            now = datetime.now(_JST)
            if not self._is_within_schedule(now):
                continue

            for guild in self._discord_client.guilds:
                if not self._is_guild_allowed(guild.id):
                    continue
                if guild.id in self._sessions_by_guild:
                    continue

                target = self._find_target_voice_channel(guild)
                if target is None:
                    continue

                key = (guild.id, target.id)
                blocked_date = self._autojoin_blocked_date.get(key)
                today = now.strftime("%Y-%m-%d")
                if blocked_date == today:
                    continue

                participants = [member for member in target.members if not member.bot]
                if len(participants) < self._config.vc_auto_join_min_participants:
                    continue

                await self._join_voice_channel(target, trigger="auto")
                break

    async def _join_voice_channel(
        self,
        voice_channel: discord.VoiceChannel,
        *,
        trigger: str,
    ) -> None:
        if self._is_indexing_active():
            return

        guild = voice_channel.guild
        if guild is None:
            return

        if guild.id in self._sessions_by_guild:
            return

        if voice_recv is None:
            await voice_channel.send("voice_recv依存が見つからないため参加できません。")
            return

        meeting_dir, meeting_key, meeting_date, meeting_label = self._allocate_meeting_dir()
        summary_chunk_path = self._config.summery_chunk_dir / "vc" / f"{meeting_key}.jsonl"
        summary_chunk_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        except Exception:
            logger.exception("Failed to connect to VC: guild=%s channel=%s", guild.id, voice_channel.id)
            await voice_channel.send("VCへの参加に失敗しました。")
            return

        session = _MeetingSession(
            meeting_key=meeting_key,
            guild_id=guild.id,
            guild_name=guild.name,
            meeting_date=meeting_date,
            meeting_label=meeting_label,
            voice_channel=voice_channel,
            voice_client=voice_client,
            sink=_FallbackAudioSink(),
            started_at=datetime.now(_JST),
            start_monotonic=time.monotonic(),
            meeting_dir=meeting_dir,
            summary_chunk_path=summary_chunk_path,
            transcript_interval_seconds=min(
                max(30, self._config.vc_summary_transcribe_interval_seconds),
                max(30, self._config.vc_end_judge_transcribe_interval_seconds),
            ),
        )

        sink = _VoiceReceiveSink(
            on_audio_frame=lambda uid, uname, pcm: self._on_audio_frame(
                session,
                uid,
                uname,
                pcm,
            )
        )
        session.sink = sink

        try:
            listen = getattr(voice_client, "listen", None)
            if callable(listen):
                listen(sink)
            else:
                raise RuntimeError("VoiceRecvClient.listen is unavailable")
        except Exception:
            logger.exception("Failed to start voice receive: guild=%s channel=%s", guild.id, voice_channel.id)
            try:
                await voice_client.disconnect(force=True)
            except Exception:
                logger.exception("Failed to disconnect VC after listen failure")
            await voice_channel.send("音声受信の初期化に失敗しました。")
            return

        self._sessions_by_guild[guild.id] = session
        self._sessions_by_channel[voice_channel.id] = session
        self._archives[meeting_key] = _MeetingArchive(
            meeting_key=meeting_key,
            guild_id=guild.id,
            meeting_date=meeting_date,
            meeting_label=meeting_label,
            summary_chunk_path=summary_chunk_path,
        )

        session.transcribe_task = asyncio.create_task(self._transcribe_loop(session))

        logger.info(
            "Joined VC: guild=%s channel=%s trigger=%s meeting=%s",
            guild.id,
            voice_channel.id,
            trigger,
            meeting_key,
        )
        await voice_channel.send("例会VCに参加しました。音声認識を開始します。")

    async def _leave_session(
        self,
        session: _MeetingSession,
        *,
        reason: str,
        process_final_transcript: bool,
    ) -> None:
        if session.ending:
            return
        session.ending = True

        if process_final_transcript:
            before_index = session.next_transcript_index
            try:
                await self._flush_transcript(session, is_final=True)
            except Exception:
                logger.exception("Failed to flush final transcript: %s", session.meeting_key)
            else:
                if session.next_transcript_index == before_index:
                    marker_path = session.meeting_dir / "00.txt"
                    await self._enqueue_post_job(
                        _PostProcessJob(
                            meeting_key=session.meeting_key,
                            transcript_index=max(0, before_index - 1),
                            transcript_path=marker_path,
                            is_final_chunk=True,
                            marker_only=True,
                        )
                    )

        session.active = False

        transcribe_task = session.transcribe_task
        if transcribe_task and not transcribe_task.done():
            transcribe_task.cancel()

        voice_client = session.voice_client
        if voice_client is not None and getattr(voice_client, "is_connected", lambda: False)():
            session.intentional_disconnect = True
            try:
                await voice_client.disconnect(force=True)
            except Exception:
                logger.exception("Failed to disconnect VC: %s", session.meeting_key)

        try:
            cleanup = getattr(session.sink, "cleanup", None)
            if callable(cleanup):
                cleanup()
        except Exception:
            logger.exception("Failed to cleanup sink: %s", session.meeting_key)

        self._sessions_by_guild.pop(session.guild_id, None)
        self._sessions_by_channel.pop(session.voice_channel.id, None)

        logger.info(
            "Left VC: guild=%s channel=%s reason=%s meeting=%s",
            session.guild_id,
            session.voice_channel.id,
            reason,
            session.meeting_key,
        )

        try:
            await session.voice_channel.send("例会VCから退出しました。")
        except Exception:
            logger.exception("Failed to send leave message: %s", session.meeting_key)

    async def _transcribe_loop(self, session: _MeetingSession) -> None:
        try:
            while session.active:
                await asyncio.sleep(session.transcript_interval_seconds)
                await self._flush_transcript(session, is_final=False)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Transcribe loop failed: %s", session.meeting_key)

    def _on_audio_frame(
        self,
        session: _MeetingSession,
        user_id: int,
        user_name: str,
        pcm: bytes,
    ) -> None:
        if not session.active:
            return
        if not pcm:
            return

        offset = max(0.0, time.monotonic() - session.start_monotonic)
        duration = float(len(pcm)) / float(_PCM_BYTES_PER_SECOND)
        event = _AudioEvent(
            user_id=user_id,
            user_name=user_name,
            offset_seconds=offset,
            pcm=pcm,
            duration_seconds=duration,
        )
        with session.event_lock:
            session.audio_events[user_id].append(event)

    async def _flush_transcript(self, session: _MeetingSession, *, is_final: bool) -> None:
        if not session.active and not is_final:
            return

        window_start = session.last_flush_offset
        cutoff = max(0.0, time.monotonic() - session.start_monotonic)
        audio_events, chat_events = self._pop_events_until(session, cutoff=cutoff)
        if not audio_events and not chat_events:
            session.last_flush_offset = cutoff
            return

        transcript_lines = await asyncio.to_thread(
            self._build_transcript_lines,
            audio_events,
            chat_events,
            window_start,
            cutoff,
        )
        session.last_flush_offset = cutoff

        if not transcript_lines:
            return

        transcript_index = session.next_transcript_index
        session.next_transcript_index += 1
        transcript_path = session.meeting_dir / f"{transcript_index:02d}.txt"

        lines = [
            f"{_format_elapsed(line.offset_seconds)} {line.user_name}: {line.text}"
            for line in transcript_lines
            if (line.text or "").strip()
        ]
        if not lines:
            return

        transcript_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        await self._enqueue_post_job(
            _PostProcessJob(
                meeting_key=session.meeting_key,
                transcript_index=transcript_index,
                transcript_path=transcript_path,
                chunk_start_offset_seconds=window_start,
                chunk_end_offset_seconds=cutoff,
                is_final_chunk=is_final,
            )
        )

    def _pop_events_until(
        self,
        session: _MeetingSession,
        *,
        cutoff: float,
    ) -> tuple[dict[int, list[_AudioEvent]], list[_ChatEvent]]:
        audio_events: dict[int, list[_AudioEvent]] = {}
        chat_events: list[_ChatEvent] = []

        with session.event_lock:
            for user_id, queue in list(session.audio_events.items()):
                bucket: list[_AudioEvent] = []
                while queue and queue[0].offset_seconds <= cutoff:
                    bucket.append(queue.popleft())
                if bucket:
                    audio_events[user_id] = bucket
                if not queue:
                    session.audio_events.pop(user_id, None)

            while session.chat_events and session.chat_events[0].offset_seconds <= cutoff:
                chat_events.append(session.chat_events.popleft())

        return audio_events, chat_events

    def _build_transcript_lines(
        self,
        audio_events_by_user: dict[int, list[_AudioEvent]],
        chat_events: list[_ChatEvent],
        start_offset: float,
        end_offset: float,
    ) -> list[_TranscriptLine]:
        transcript_lines: list[_TranscriptLine] = []

        for events in audio_events_by_user.values():
            transcript_lines.extend(
                self._transcribe_user_events(
                    events,
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
            )

        for chat in chat_events:
            text = (chat.text or "").strip()
            if not text:
                continue
            transcript_lines.append(
                _TranscriptLine(
                    offset_seconds=max(start_offset, chat.offset_seconds),
                    user_name=chat.user_name,
                    text=text,
                )
            )

        transcript_lines.sort(
            key=lambda item: (
                item.offset_seconds,
                item.user_name,
                item.text,
            )
        )
        return transcript_lines

    def _transcribe_user_events(
        self,
        events: list[_AudioEvent],
        *,
        start_offset: float,
        end_offset: float,
    ) -> list[_TranscriptLine]:
        if not events:
            return []

        blocks = self._merge_audio_blocks(events, start_offset=start_offset, end_offset=end_offset)
        if not blocks:
            return []

        lines: list[_TranscriptLine] = []
        user_name = events[-1].user_name if events else "unknown"
        asr_pipeline, target_sample_rate = self._get_asr_pipeline()

        for block_start, pcm_bytes in blocks:
            duration = float(len(pcm_bytes)) / float(_PCM_BYTES_PER_SECOND)
            if duration < _MIN_AUDIO_BLOCK_SECONDS:
                continue

            audio = _pcm_bytes_to_mono_float32(
                pcm_bytes,
                channels=_PCM_CHANNELS,
            )
            if audio.size == 0:
                continue
            if target_sample_rate != _PCM_SAMPLE_RATE:
                audio = _resample_linear(
                    audio,
                    src_rate=_PCM_SAMPLE_RATE,
                    dst_rate=target_sample_rate,
                )
            if audio.size == 0:
                continue

            try:
                with self._asr_inference_lock:
                    result = asr_pipeline(
                        {"array": audio, "sampling_rate": target_sample_rate},
                        return_timestamps=True,
                        generate_kwargs=_build_asr_generate_kwargs(
                            language=self._config.vc_transcribe_language
                        ),
                    )
            except Exception:
                logger.exception("Transformers ASR transcription failed.")
                continue

            lines.extend(
                _transcript_lines_from_asr_result(
                    result=result,
                    block_start=block_start,
                    start_offset=start_offset,
                    user_name=user_name,
                )
            )

        return lines

    def _merge_audio_blocks(
        self,
        events: list[_AudioEvent],
        *,
        start_offset: float,
        end_offset: float,
    ) -> list[tuple[float, bytes]]:
        filtered = [
            event
            for event in sorted(events, key=lambda item: item.offset_seconds)
            if start_offset <= event.offset_seconds <= end_offset
        ]
        if not filtered:
            return []

        blocks: list[tuple[float, bytearray, float]] = []
        first = filtered[0]
        blocks.append((first.offset_seconds, bytearray(first.pcm), first.offset_seconds + first.duration_seconds))

        for event in filtered[1:]:
            block_start, block_pcm, block_end = blocks[-1]
            gap = event.offset_seconds - block_end
            if gap <= _MAX_TRANSCRIBE_GAP_SECONDS:
                if gap > 0:
                    silence_len = int(round(gap * _PCM_BYTES_PER_SECOND))
                    silence_len -= silence_len % _PCM_FRAME_BYTES
                    if silence_len > 0:
                        block_pcm.extend(b"\x00" * silence_len)
                block_pcm.extend(event.pcm)
                blocks[-1] = (
                    block_start,
                    block_pcm,
                    event.offset_seconds + event.duration_seconds,
                )
                continue

            blocks.append(
                (
                    event.offset_seconds,
                    bytearray(event.pcm),
                    event.offset_seconds + event.duration_seconds,
                )
            )

        return [(start, bytes(pcm)) for start, pcm, _ in blocks]

    def _get_asr_pipeline(self):
        with self._asr_pipeline_lock:
            if self._asr_pipeline is not None:
                return self._asr_pipeline
            if (
                hf_pipeline is None
                or AutoProcessor is None
                or AutoModelForSpeechSeq2Seq is None
                or torch is None
            ):
                raise RuntimeError("transformers ASR dependencies are unavailable")

            model_id = (self._config.vc_transcribe_model or "").strip()
            if not model_id:
                raise RuntimeError("VC_TRANSCRIBE_MODEL is empty")
            model_path = Path(model_id).expanduser()
            if not model_path.exists():
                raise RuntimeError(
                    "VC_TRANSCRIBE_MODEL local path does not exist: "
                    f"{model_path}"
                )

            device_str = _resolve_torch_device(self._config.vc_transcribe_device)
            torch_dtype = _resolve_torch_dtype(
                dtype_name=self._config.vc_transcribe_torch_dtype,
                device_str=device_str,
            )
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            processor = AutoProcessor.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            model.to(device_str)

            feature_extractor = getattr(processor, "feature_extractor", None)
            sample_rate = int(
                getattr(feature_extractor, "sampling_rate", 16_000) or 16_000
            )

            self._asr_pipeline = (
                hf_pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=getattr(processor, "tokenizer", None),
                    feature_extractor=feature_extractor,
                    torch_dtype=torch_dtype,
                    device=torch.device(device_str),
                ),
                sample_rate,
            )
            return self._asr_pipeline

    async def _enqueue_post_job(self, job: _PostProcessJob) -> None:
        async with self._post_jobs_cond:
            self._post_jobs.append(job)
            self._post_jobs_cond.notify_all()

    async def _requeue_post_job_front(self, job: _PostProcessJob) -> None:
        async with self._post_jobs_cond:
            self._post_jobs.appendleft(job)
            self._post_jobs_cond.notify_all()

    async def _post_process_worker(self) -> None:
        logger.info("VC post-process worker started.")
        try:
            while True:
                job = await self._wait_for_job()
                self._post_worker_current = job
                requeued = False
                try:
                    await self._process_post_job(job)
                except _PauseRequested:
                    requeued = True
                    await self._requeue_post_job_front(job)
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Post-process failed: meeting=%s transcript=%s",
                        job.meeting_key,
                        job.transcript_path,
                    )
                finally:
                    if not requeued and job.is_final_chunk:
                        try:
                            await self._send_final_summary(job.meeting_key)
                        except Exception:
                            logger.exception(
                                "Failed to send final summary: %s", job.meeting_key
                            )
                    self._post_worker_current = None
        except asyncio.CancelledError:
            return

    async def _wait_for_job(self) -> _PostProcessJob:
        async with self._post_jobs_cond:
            while not self._post_jobs:
                await self._post_jobs_cond.wait()
            return self._post_jobs.popleft()

    async def _process_post_job(self, job: _PostProcessJob) -> None:
        self._ensure_post_worker_not_paused()
        archive = self._archives.get(job.meeting_key)
        if archive is None:
            return

        if job.marker_only:
            if job.is_final_chunk:
                await self._generate_summary_from_pending(
                    meeting_key=job.meeting_key,
                    archive=archive,
                )
                self._reset_pending_end_judge(archive)
            return

        transcript_text = (job.transcript_path.read_text(encoding="utf-8") or "").strip()
        if not transcript_text:
            if job.is_final_chunk:
                await self._generate_summary_from_pending(
                    meeting_key=job.meeting_key,
                    archive=archive,
                )
                self._reset_pending_end_judge(archive)
            return

        chunk_seconds = max(
            0.0,
            job.chunk_end_offset_seconds - job.chunk_start_offset_seconds,
        )
        self._queue_pending_summary(
            archive=archive,
            transcript_text=transcript_text,
            transcript_index=job.transcript_index,
            chunk_seconds=chunk_seconds,
        )
        self._queue_pending_end_judge(
            archive=archive,
            transcript_text=transcript_text,
            chunk_seconds=chunk_seconds,
        )

        should_run_summary = job.is_final_chunk or (
            archive.pending_summary_seconds
            >= self._config.vc_summary_transcribe_interval_seconds
        )
        if should_run_summary:
            await self._generate_summary_from_pending(
                meeting_key=job.meeting_key,
                archive=archive,
            )

        self._ensure_post_worker_not_paused()

        if job.is_final_chunk:
            self._reset_pending_end_judge(archive)
            return

        should_run_end_judge = (
            archive.pending_end_judge_seconds
            >= self._config.vc_end_judge_transcribe_interval_seconds
        )
        if not should_run_end_judge:
            return

        pending_end_text = self._build_pending_transcript_text(
            archive.pending_end_judge_texts
        )
        if not pending_end_text:
            self._reset_pending_end_judge(archive)
            return

        end_prompt = build_end_judgement_prompt(transcript_text=pending_end_text)
        end_result = await asyncio.to_thread(
            self._generate_text_with_cfg,
            cfg=self._end_llm_cfg,
            prompt=end_prompt,
            system_prompt=_END_JUDGEMENT_SYSTEM_PROMPT,
            response_mime_type="text/plain",
        )
        self._reset_pending_end_judge(archive)

        is_end = _parse_boolean_output(end_result)
        if not is_end:
            return

        if not self._config.vc_auto_quit_enabled:
            return

        session = self._find_active_session_by_meeting_key(job.meeting_key)
        if session is None:
            return

        await self._leave_session(
            session,
            reason="auto_end_detected",
            process_final_transcript=True,
        )

    async def _generate_summary_from_pending(
        self,
        *,
        meeting_key: str,
        archive: _MeetingArchive,
    ) -> None:
        if archive.pending_summary_last_index <= 0:
            self._reset_pending_summary(archive)
            return

        pending_summary_text = self._build_pending_transcript_text(
            archive.pending_summary_texts
        )
        if not pending_summary_text:
            self._reset_pending_summary(archive)
            return

        transcript_index = archive.pending_summary_last_index
        summary_text = self._load_existing_summary_for_transcript(
            summary_chunk_path=archive.summary_chunk_path,
            transcript_index=transcript_index,
        )
        if not summary_text:
            previous_summaries = self._load_recent_summaries(
                summary_chunk_path=archive.summary_chunk_path,
                limit=self._config.vc_summary_previous_max,
            )
            summary_prompt = build_summary_prompt(
                transcript_text=pending_summary_text,
                previous_summaries=previous_summaries,
                target_characters=self._config.vc_summary_target_characters,
            )
            summary_text = await asyncio.to_thread(
                self._generate_text_with_cfg,
                cfg=self._summary_llm_cfg,
                prompt=summary_prompt,
                system_prompt=_SUMMARY_SYSTEM_PROMPT,
                response_mime_type="text/plain",
            )
            summary_text = (summary_text or "").strip()
            if summary_text:
                self._append_summary_chunk(
                    meeting_key=meeting_key,
                    transcript_index=transcript_index,
                    summary_text=summary_text,
                )

        self._reset_pending_summary(archive)

    def _queue_pending_summary(
        self,
        *,
        archive: _MeetingArchive,
        transcript_text: str,
        transcript_index: int,
        chunk_seconds: float,
    ) -> None:
        archive.pending_summary_texts.append(transcript_text)
        archive.pending_summary_seconds += max(0.0, chunk_seconds)
        archive.pending_summary_last_index = transcript_index

    def _queue_pending_end_judge(
        self,
        *,
        archive: _MeetingArchive,
        transcript_text: str,
        chunk_seconds: float,
    ) -> None:
        archive.pending_end_judge_texts.append(transcript_text)
        archive.pending_end_judge_seconds += max(0.0, chunk_seconds)

    def _reset_pending_summary(self, archive: _MeetingArchive) -> None:
        archive.pending_summary_texts.clear()
        archive.pending_summary_seconds = 0.0
        archive.pending_summary_last_index = 0

    def _reset_pending_end_judge(self, archive: _MeetingArchive) -> None:
        archive.pending_end_judge_texts.clear()
        archive.pending_end_judge_seconds = 0.0

    def _build_pending_transcript_text(self, chunks: list[str]) -> str:
        return "\n".join(item.strip() for item in chunks if (item or "").strip()).strip()

    def _append_summary_chunk(
        self,
        *,
        meeting_key: str,
        transcript_index: int,
        summary_text: str,
    ) -> None:
        archive = self._archives.get(meeting_key)
        if archive is None:
            return

        chunk_id = self._count_jsonl_lines(archive.summary_chunk_path)
        record = {
            "text": summary_text,
            "metadata": {
                "source_type": "vc_summary_chunk",
                "source_file_name": f"vc/{meeting_key}",
                "meeting_key": meeting_key,
                "meeting_date": archive.meeting_date,
                "meeting_label": archive.meeting_label,
                "transcript_index": transcript_index,
                "chunk_id": chunk_id,
                "chunk_stage": "summery",
            },
        }
        with archive.summary_chunk_path.open("a", encoding="utf-8") as fw:
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _send_final_summary(self, meeting_key: str) -> None:
        if not self._config.vc_final_summary_enabled:
            return

        archive = self._archives.get(meeting_key)
        if archive is None:
            return

        summaries = self._load_all_summaries(archive.summary_chunk_path)
        if not summaries:
            return

        prompt = build_final_summary_prompt(summary_chunks=summaries)
        final_summary = await asyncio.to_thread(
            self._generate_text_with_cfg,
            cfg=self._final_llm_cfg,
            prompt=prompt,
            system_prompt=_FINAL_SUMMARY_SYSTEM_PROMPT,
            response_mime_type="text/plain",
        )
        text = (final_summary or "").strip()
        if not text:
            return

        guild = self._discord_client.get_guild(archive.guild_id)
        if guild is None:
            return

        channel = discord.utils.get(guild.text_channels, name=_FINAL_SUMMARY_CHANNEL)
        if channel is None:
            logger.warning(
                "Final summary channel not found: guild=%s name=%s",
                guild.id,
                _FINAL_SUMMARY_CHANNEL,
            )
            return

        await channel.send(f"{text}\n\n{_ANSWER_DISCLAIMER}")

    def _load_existing_summary_for_transcript(
        self,
        *,
        summary_chunk_path: Path,
        transcript_index: int,
    ) -> str:
        if not summary_chunk_path.exists():
            return ""
        try:
            with summary_chunk_path.open("r", encoding="utf-8") as fr:
                for line in fr:
                    raw = line.strip()
                    if not raw:
                        continue
                    obj = json.loads(raw)
                    metadata = obj.get("metadata") or {}
                    if int(metadata.get("transcript_index") or -1) != transcript_index:
                        continue
                    return str(obj.get("text") or "").strip()
        except Exception:
            logger.exception(
                "Failed to read existing summary: %s", summary_chunk_path
            )
        return ""

    def _load_recent_summaries(self, *, summary_chunk_path: Path, limit: int) -> list[str]:
        if limit <= 0 or not summary_chunk_path.exists():
            return []
        rows: list[str] = []
        try:
            with summary_chunk_path.open("r", encoding="utf-8") as fr:
                for line in fr:
                    raw = line.strip()
                    if not raw:
                        continue
                    obj = json.loads(raw)
                    text = str(obj.get("text") or "").strip()
                    if text:
                        rows.append(text)
        except Exception:
            logger.exception("Failed to read summary chunks: %s", summary_chunk_path)
            return []

        if len(rows) <= limit:
            return rows
        return rows[-limit:]

    def _load_all_summaries(self, summary_chunk_path: Path) -> list[str]:
        if not summary_chunk_path.exists():
            return []
        rows: list[str] = []
        try:
            with summary_chunk_path.open("r", encoding="utf-8") as fr:
                for line in fr:
                    raw = line.strip()
                    if not raw:
                        continue
                    obj = json.loads(raw)
                    text = str(obj.get("text") or "").strip()
                    if text:
                        rows.append(text)
        except Exception:
            logger.exception("Failed to read summary chunks: %s", summary_chunk_path)
            return []
        return rows

    @staticmethod
    def _count_jsonl_lines(path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                if line.strip():
                    count += 1
        return count

    def _ensure_post_worker_not_paused(self) -> None:
        if self._rag_pause_count > 0:
            raise _PauseRequested("paused by RAG answer generation")

    def _find_active_session_by_meeting_key(self, meeting_key: str) -> _MeetingSession | None:
        for session in self._sessions_by_guild.values():
            if session.meeting_key == meeting_key and session.active:
                return session
        return None

    def _generate_text_with_cfg(
        self,
        *,
        cfg: _LLMConfig,
        prompt: str,
        system_prompt: str,
        response_mime_type: str,
    ) -> str:
        return generate_text(
            provider=cfg.provider,
            api_key=self._config.gemini_api_key,
            prompt=prompt,
            model=cfg.model,
            system_prompt=system_prompt,
            llama_model_path=cfg.llama_model_path,
            llama_ctx_size=cfg.llama_ctx_size,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            thinking_level=cfg.thinking_level,
            llama_threads=self._config.llama_threads,
            llama_gpu_layers=self._config.llama_gpu_layers,
            response_mime_type=response_mime_type,
        )

    def _allocate_meeting_dir(self) -> tuple[Path, str, str, str]:
        now = datetime.now(_JST)
        date_token = now.strftime("%Y-%m-%d")
        date_metadata = now.strftime("%Y/%m/%d")
        base_dir = self._config.raw_data_dir / "vc"
        base_dir.mkdir(parents=True, exist_ok=True)

        for number in range(1, 1000):
            key = f"{date_token}_{number:02d}"
            candidate = base_dir / key
            if candidate.exists():
                continue
            candidate.mkdir(parents=True, exist_ok=False)
            label = f"{date_metadata} 例会"
            return candidate, key, date_metadata, label

        raise RuntimeError("Failed to allocate VC meeting directory")

    def _is_within_schedule(self, now: datetime) -> bool:
        if now.tzinfo is None:
            now = now.replace(tzinfo=_JST)
        weekdays = self._config.vc_auto_join_weekdays
        if weekdays and now.weekday() not in weekdays:
            return False

        start = now.replace(
            hour=self._config.vc_auto_join_start_hour,
            minute=self._config.vc_auto_join_start_minute,
            second=0,
            microsecond=0,
        )
        end = start + timedelta(minutes=self._config.vc_auto_join_duration_minutes)
        return start <= now <= end

    def _find_target_voice_channel(
        self, guild: discord.Guild
    ) -> discord.VoiceChannel | None:
        for channel in guild.voice_channels:
            if str(channel.name).strip() == self._config.vc_target_voice_channel_name:
                return channel
        return None

    def _is_guild_allowed(self, guild_id: int) -> bool:
        allow_list = self._config.discord_guild_allow_list
        if not allow_list:
            return True
        return guild_id in set(allow_list)

    @staticmethod
    def _select_llm_model(*, provider: str, gemini_model: str, llama_model: str) -> str:
        if provider == "llama":
            return llama_model
        return gemini_model



def _resolve_torch_device(device_name: str) -> str:
    if torch is None:
        return "cpu"
    requested = (device_name or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    return requested


def _resolve_torch_dtype(*, dtype_name: str, device_str: str):
    if torch is None:
        return None
    name = (dtype_name or "auto").strip().lower()
    if name == "auto":
        if device_str.startswith("cuda"):
            return torch.float16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(name, torch.float32)


def _pcm_bytes_to_mono_float32(pcm_bytes: bytes, *, channels: int) -> np.ndarray:
    if not pcm_bytes:
        return np.array([], dtype=np.float32)

    valid_bytes = len(pcm_bytes) - (len(pcm_bytes) % _PCM_SAMPLE_WIDTH)
    if valid_bytes <= 0:
        return np.array([], dtype=np.float32)
    if valid_bytes != len(pcm_bytes):
        logger.warning(
            "Trimming odd-sized PCM bytes: input=%s trimmed=%s",
            len(pcm_bytes),
            valid_bytes,
        )

    raw = np.frombuffer(memoryview(pcm_bytes)[:valid_bytes], dtype=np.int16)
    if raw.size == 0:
        return np.array([], dtype=np.float32)
    if channels <= 1:
        return (raw.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

    valid_size = (raw.size // channels) * channels
    if valid_size <= 0:
        return np.array([], dtype=np.float32)
    reshaped = raw[:valid_size].reshape(-1, channels).astype(np.float32)
    mono = reshaped.mean(axis=1) / 32768.0
    return mono.clip(-1.0, 1.0)


def _resample_linear(
    audio: np.ndarray,
    *,
    src_rate: int,
    dst_rate: int,
) -> np.ndarray:
    if audio.size == 0:
        return np.array([], dtype=np.float32)
    if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)

    src_len = audio.shape[0]
    dst_len = int(round(src_len * (dst_rate / src_rate)))
    if dst_len <= 0:
        return np.array([], dtype=np.float32)
    if src_len == 1:
        return np.repeat(audio.astype(np.float32), dst_len)

    src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=True)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=True)
    resampled = np.interp(dst_x, src_x, audio).astype(np.float32)
    return resampled


def _build_asr_generate_kwargs(*, language: str) -> dict[str, str]:
    kwargs: dict[str, str] = {"task": "transcribe"}
    lang = (language or "").strip()
    if lang:
        kwargs["language"] = lang
    return kwargs


def _transcript_lines_from_asr_result(
    *,
    result: object,
    block_start: float,
    start_offset: float,
    user_name: str,
) -> list[_TranscriptLine]:
    lines: list[_TranscriptLine] = []
    if not isinstance(result, dict):
        text = str(result or "").strip()
        if text:
            lines.append(
                _TranscriptLine(
                    offset_seconds=max(start_offset, block_start),
                    user_name=user_name,
                    text=text,
                )
            )
        return lines

    chunks = result.get("chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            timestamp = chunk.get("timestamp")
            seg_start = _coerce_timestamp_start(timestamp)
            absolute_start = max(start_offset, block_start + seg_start)
            lines.append(
                _TranscriptLine(
                    offset_seconds=absolute_start,
                    user_name=user_name,
                    text=text,
                )
            )
        if lines:
            return lines

    text = str(result.get("text") or "").strip()
    if not text:
        return []
    lines.append(
        _TranscriptLine(
            offset_seconds=max(start_offset, block_start),
            user_name=user_name,
            text=text,
        )
    )
    return lines


def _coerce_timestamp_start(timestamp: object) -> float:
    if isinstance(timestamp, (list, tuple)) and timestamp:
        candidate = timestamp[0]
        if isinstance(candidate, (int, float)):
            return max(0.0, float(candidate))
    return 0.0


def _format_elapsed(seconds: float) -> str:
    value = max(0, int(math.floor(seconds)))
    hh = value // 3600
    mm = (value % 3600) // 60
    ss = value % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"



def _parse_boolean_output(text: str) -> bool:
    raw = (text or "").strip().lower()
    if not raw:
        return False
    if raw.startswith("true"):
        return True
    if raw.startswith("false"):
        return False

    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except Exception:
            return False
        if isinstance(payload, dict):
            value = payload.get("ended")
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes"}
    return False
