from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
import json
import logging
import threading
from zoneinfo import ZoneInfo

from kumc_agent.frontends.discord.commands import parse_command
from kumc_agent.runtime.container import build_runtime_context
from kumc_agent.usecases.chat.answer import ChatRequest
from kumc_agent.usecases.eval.ragas import EvaluateRagasRequest
from kumc_agent.usecases.indexing.build import BuildIndexRequest
from kumc_agent.usecases.chat.route import RouteRequest
from kumc_agent.utils.logging import configure_logging, default_execution_log_path

logger = logging.getLogger(__name__)
_JST = ZoneInfo("Asia/Tokyo")


def main() -> None:
    import discord

    context = build_runtime_context()
    configure_logging(
        context.config.app.log_level,
        file_path=default_execution_log_path(base_dir=context.config.base_dir),
    )

    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    client = discord.Client(intents=intents)

    command_prefix = context.config.app.command_prefix.strip()
    index_command_prefix = context.config.app.index_command_prefix.strip()
    special_channel_names = {
        name.strip()
        for name in context.config.rag.history.special_channel_names
        if name.strip()
    }

    indexing_in_progress = False
    indexing_task: asyncio.Task[None] | None = None
    indexing_cancel_event = threading.Event()
    indexing_started_at: datetime | None = None
    evaluating_task: asyncio.Task[None] | None = None
    channel_generation_tasks: dict[int, asyncio.Task[None]] = {}
    channel_cancel_events: dict[int, threading.Event] = {}
    auto_index_task: asyncio.Task[None] | None = None
    periodic_warmup_task: asyncio.Task[None] | None = None
    auto_index_last_run: date | None = None
    answer_record_lock = threading.Lock()

    context.vc.bind_discord_client(
        discord_client=client,
        is_indexing_active=lambda: indexing_in_progress,
    )

    def _bot_mention_prefixes() -> tuple[str, ...]:
        ids: set[int] = set()
        current_user = client.user
        if current_user is not None and getattr(current_user, "id", None):
            ids.add(int(current_user.id))
        prefixes: list[str] = []
        for user_id in ids:
            prefixes.append(f"<@{user_id}>")
            prefixes.append(f"<@!{user_id}>")
        return tuple(prefixes)

    def _extract_query_from_message(message: discord.Message) -> str:
        content = (message.content or "").strip()
        if not content:
            return ""

        for prefix in _bot_mention_prefixes():
            if content.startswith(prefix):
                query = content[len(prefix) :].strip()
                if query.startswith(command_prefix):
                    query = query[len(command_prefix) :].strip()
                return query

        if content.startswith(command_prefix):
            return content[len(command_prefix) :].strip()

        if getattr(message, "guild", None) is None:
            return content

        channel_name = str(getattr(message.channel, "name", "") or "").strip()
        if channel_name in special_channel_names:
            return content
        return ""

    def _strip_fast_prefix(query: str) -> tuple[str, bool]:
        normalized = (query or "").strip()
        if not normalized:
            return "", False
        parts = normalized.split(maxsplit=1)
        if parts[0].strip().lower() != "fast":
            return normalized, False
        if len(parts) == 1:
            return "", True
        return parts[1].strip(), True

    def _is_special_channel_invocation(message: discord.Message) -> bool:
        if getattr(message, "guild", None) is None:
            return False
        channel_name = str(getattr(message.channel, "name", "") or "").strip()
        return channel_name in special_channel_names

    def _history_scope_for_message(message: discord.Message) -> str:
        guild = getattr(message, "guild", None)
        guild_id = getattr(guild, "id", None)
        if guild_id is not None:
            return f"guild:{guild_id}"
        return f"channel:{message.channel.id}"

    def _question_author_from_message(message: discord.Message) -> str:
        author = getattr(message, "author", None)
        display_name = str(getattr(author, "display_name", "") or "").strip()
        user_name = str(getattr(author, "name", "") or "").strip()
        if display_name and user_name:
            return f"{display_name} (@{user_name})"
        if display_name:
            return display_name
        if user_name:
            return f"@{user_name}"
        return str(getattr(author, "id", "unknown"))

    def _question_user_id(message: discord.Message) -> str:
        value = getattr(getattr(message, "author", None), "id", None)
        return "unknown" if value is None else str(value)

    def _question_username(message: discord.Message) -> str:
        author = getattr(message, "author", None)
        username = str(getattr(author, "name", "") or "").strip()
        if username:
            return username
        display_name = str(getattr(author, "display_name", "") or "").strip()
        if display_name:
            return display_name
        return "unknown"

    def _is_maintenance_authorized(message: discord.Message) -> bool:
        allow = set(context.config.security.maintenance_command_author_ids)
        if not allow:
            return False
        author_id = getattr(getattr(message, "author", None), "id", None)
        if author_id is None:
            return False
        return int(author_id) in allow

    def _append_answer_record(
        *,
        message: discord.Message,
        query: str,
        routing_result: dict[str, object] | None,
        answer: str,
    ) -> None:
        if not context.config.ops.answer_record_log_enabled:
            return
        query_text = (query or "").strip()
        answer_text = (answer or "").strip()
        if not query_text or not answer_text:
            return
        record = {
            "timestamp": datetime.now(_JST).isoformat(timespec="seconds"),
            "questioner_user_id": _question_user_id(message),
            "questioner_username": _question_username(message),
            "question": query_text,
            "routing_result": routing_result,
            "answer": answer_text,
        }
        try:
            with answer_record_lock:
                path = context.config.ops.answer_record_log_path
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fw:
                    fw.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to append answer record.")

    def _format_jst(value: datetime) -> str:
        return value.astimezone(_JST).strftime("%Y/%m/%d %H:%M")

    def _indexing_blocked_message() -> str:
        started_at = indexing_started_at or datetime.now(_JST)
        min_end = started_at + timedelta(
            minutes=context.config.ops.index_update_estimate_min_minutes
        )
        max_end = started_at + timedelta(
            minutes=context.config.ops.index_update_estimate_max_minutes
        )
        if min_end == max_end:
            estimate = _format_jst(min_end)
        else:
            estimate = f"{_format_jst(min_end)}〜{_format_jst(max_end)}"
        return (
            "インデックス更新中のため、クエリ受付を停止しています。\n"
            f"更新開始時刻: {_format_jst(started_at)} (JST)\n"
            f"終了目安: {estimate} (JST)"
        )

    async def _resolve_referenced_message(
        message: discord.Message,
        *,
        reference_cache: dict[int, discord.Message | None],
    ) -> discord.Message | None:
        reference = getattr(message, "reference", None)
        if reference is None:
            return None
        message_id = getattr(reference, "message_id", None)
        if message_id is None:
            return None
        try:
            message_id_int = int(message_id)
        except (TypeError, ValueError):
            return None
        if message_id_int in reference_cache:
            return reference_cache[message_id_int]

        resolved = getattr(reference, "resolved", None)
        if isinstance(resolved, discord.Message):
            reference_cache[message_id_int] = resolved
            return resolved

        guild = getattr(message, "guild", None)
        channel_id = getattr(reference, "channel_id", None)
        fetch_channel = None
        if guild is not None and channel_id is not None:
            try:
                channel_id_int = int(channel_id)
            except (TypeError, ValueError):
                channel_id_int = None
            if channel_id_int is not None:
                fetch_channel = guild.get_channel_or_thread(channel_id_int)
        if fetch_channel is None:
            fetch_channel = message.channel

        fetch_message = getattr(fetch_channel, "fetch_message", None)
        if not callable(fetch_message):
            reference_cache[message_id_int] = None
            return None
        try:
            referenced = await fetch_message(message_id_int)
        except Exception:
            reference_cache[message_id_int] = None
            return None
        reference_cache[message_id_int] = referenced
        return referenced

    async def _collect_special_channel_history(
        message: discord.Message,
    ) -> list[tuple[str, str, list[str]]]:
        limit = max(0, context.config.rag.history.special_channel_history_limit)
        if limit <= 0:
            return []
        primary_messages: list[discord.Message] = []
        seen_primary_ids: set[int] = set()

        def _add_primary(candidate: discord.Message) -> None:
            text = (candidate.content or "").strip()
            if not text:
                return
            candidate_id = int(getattr(candidate, "id", 0))
            if candidate_id <= 0 or candidate_id in seen_primary_ids:
                return
            seen_primary_ids.add(candidate_id)
            primary_messages.append(candidate)

        _add_primary(message)
        try:
            async for item in message.channel.history(limit=None, oldest_first=False):
                _add_primary(item)
                if len(primary_messages) >= limit:
                    break
        except Exception:
            logger.exception("Failed to collect special-channel history.")

        reference_cache: dict[int, discord.Message | None] = {}
        all_messages: dict[int, discord.Message] = {
            int(item.id): item for item in primary_messages
        }
        for root in primary_messages:
            current = root
            visited_chain_ids: set[int] = set()
            while True:
                referenced = await _resolve_referenced_message(
                    current,
                    reference_cache=reference_cache,
                )
                if referenced is None:
                    break
                referenced_id = int(getattr(referenced, "id", 0))
                if referenced_id <= 0 or referenced_id in visited_chain_ids:
                    break
                visited_chain_ids.add(referenced_id)
                if referenced_id not in all_messages and (referenced.content or "").strip():
                    all_messages[referenced_id] = referenced
                current = referenced

        ordered = sorted(
            all_messages.values(),
            key=lambda item: (item.created_at, int(item.id)),
        )
        history: list[tuple[str, str, list[str]]] = []
        for item in ordered:
            text = (item.content or "").strip()
            if not text:
                continue
            author = _question_author_from_message(item)
            history.append((f"author: {author}\n{text}", "", []))
        return history

    async def _send_status(
        channel: discord.abc.Messageable | None,
        text: str,
    ) -> None:
        if channel is None:
            logger.info(text)
            return
        await channel.send(text)

    async def _run_build_index_job(
        *,
        channel: discord.abc.Messageable | None,
        history_query: str | None = None,
    ) -> None:
        nonlocal indexing_in_progress, indexing_task, indexing_started_at
        indexing_in_progress = True
        indexing_started_at = datetime.now(_JST)
        indexing_cancel_event.clear()
        await _send_status(channel, "インデックス更新を開始します。")
        try:
            result = await asyncio.to_thread(
                lambda: context.build_index.execute(
                    BuildIndexRequest(
                        refresh_sources=True,
                        full_rebuild=True,
                        allow_cancel=True,
                        cancel_event=indexing_cancel_event,
                    )
                )
            )
            await _send_status(
                channel,
                (
                    "インデックス更新が完了しました。"
                    f" loaded_sources={result.loaded_sources},"
                    f" documents={result.documents}, chunks={result.chunks}"
                ),
            )
        except asyncio.CancelledError:
            await _send_status(channel, "インデックス更新を中止しました。")
        except Exception as exc:
            logger.exception("Index build failed")
            await _send_status(channel, f"インデックス更新に失敗しました: {exc}")
        finally:
            indexing_in_progress = False
            indexing_task = None
            indexing_started_at = None

    async def _run_eval_job(
        *,
        channel: discord.abc.Messageable | None,
    ) -> None:
        nonlocal evaluating_task
        await _send_status(channel, "評価を開始します。")
        try:
            eval_file = context.config.app.eval_dir / "ragas.jsonl"
            result = await asyncio.to_thread(
                lambda: context.eval_ragas.execute(
                    EvaluateRagasRequest(
                        eval_file=eval_file,
                        result_path=context.config.app.eval_dir / "result" / "result.json",
                    )
                )
            )
            metric_text = ""
            if result.ragas_metrics:
                ordered_keys = (
                    "answer_relevancy",
                    "faithfulness",
                    "context_precision",
                    "context_recall",
                )
                parts: list[str] = []
                for key in ordered_keys:
                    if key in result.ragas_metrics:
                        parts.append(f"{key}={result.ragas_metrics[key]:.3f}")
                for key in sorted(result.ragas_metrics.keys()):
                    if key in ordered_keys:
                        continue
                    parts.append(f"{key}={result.ragas_metrics[key]:.3f}")
                if parts:
                    metric_text = " metrics=" + ", ".join(parts)
            await _send_status(
                channel,
                (
                    "評価が完了しました。"
                    f" total={result.total}, exact_match={result.exact_match:.3f},"
                    f" token_overlap={result.token_overlap:.3f}{metric_text}"
                ),
            )
        except asyncio.CancelledError:
            await _send_status(channel, "評価を中止しました。")
        except Exception as exc:
            logger.exception("Eval failed")
            await _send_status(channel, f"評価に失敗しました: {exc}")
        finally:
            evaluating_task = None

    async def _run_chat(
        *,
        message: discord.Message,
        query: str,
        force_fast_mode: bool,
        routing_history_override: list[tuple[str, str, list[str]]] | None,
        generation_history_override: list[tuple[str, str, list[str]]] | None,
        append_sources_to_response: bool,
        force_disable_additional_memory: bool,
        extra_mode_instruction: str | None,
    ) -> None:
        channel_id = int(message.channel.id)
        cancel_event = threading.Event()
        channel_cancel_events[channel_id] = cancel_event
        context.vc.notify_rag_started()
        try:
            answer = await asyncio.to_thread(
                lambda: context.chat_answer.execute(
                    ChatRequest(
                        query=query,
                        question_author=_question_author_from_message(message),
                        history_scope=_history_scope_for_message(message),
                        force_fast_mode=force_fast_mode,
                        force_disable_additional_memory=force_disable_additional_memory,
                        routing_history_override=routing_history_override,
                        generation_history_override=generation_history_override,
                        append_sources_to_response=append_sources_to_response,
                        extra_mode_instruction=extra_mode_instruction,
                    )
                )
            )
            if cancel_event.is_set():
                return
            await message.channel.send(answer.text)
            routing_result = answer.metadata.get("routing_decision")
            if isinstance(routing_result, dict):
                routing_payload = dict(routing_result)
            else:
                routing_payload = None
            _append_answer_record(
                message=message,
                query=query,
                routing_result=routing_payload,
                answer=answer.text,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.exception("Chat failed")
            await message.channel.send(f"エラーが発生しました: {type(exc).__name__}: {exc}")
        finally:
            context.vc.notify_rag_finished()
            channel_cancel_events.pop(channel_id, None)
            channel_generation_tasks.pop(channel_id, None)

    def _warmup_skip_reason() -> str | None:
        if indexing_in_progress or indexing_task is not None:
            return "indexing is running"
        if evaluating_task is not None:
            return "evaluation is running"
        if any(not task.done() for task in channel_generation_tasks.values()):
            return "answer generation is running"
        if context.vc.has_model_activity():
            return "voice model processing is running"
        return None

    async def _periodic_warmup_loop() -> None:
        interval_minutes = max(0, int(context.config.ops.warmup_interval_minutes))
        if interval_minutes <= 0:
            logger.info("Periodic warmup disabled.")
            return
        interval_seconds = interval_minutes * 60
        logger.info("Periodic warmup scheduler started. interval_minutes=%s", interval_minutes)
        while True:
            await asyncio.sleep(interval_seconds)
            reason = _warmup_skip_reason()
            if reason:
                logger.info("Periodic warmup skipped: %s", reason)
                continue
            try:
                await asyncio.to_thread(
                    lambda: context.chat_route.execute(RouteRequest(query="warmup"))
                )
            except Exception:
                logger.exception("Warmup failed.")

    async def _auto_index_loop() -> None:
        nonlocal auto_index_last_run, indexing_task
        logger.info(
            "Auto index scheduler started. enabled=%s time=%s weekdays=%s",
            context.config.scheduler.auto_index_enabled,
            context.config.scheduler.auto_index_time,
            context.config.scheduler.auto_index_weekdays,
        )
        while True:
            await asyncio.sleep(20)
            if not context.config.scheduler.auto_index_enabled:
                continue
            now = datetime.now(_JST)
            hhmm = now.strftime("%H:%M")
            if hhmm != context.config.scheduler.auto_index_time:
                continue
            weekdays = context.config.scheduler.auto_index_weekdays
            if weekdays and now.weekday() not in weekdays:
                continue
            if auto_index_last_run == now.date():
                continue
            auto_index_last_run = now.date()
            if indexing_in_progress or indexing_task is not None:
                logger.info("Auto index skipped: indexing already running.")
                continue
            if context.vc.has_active_session():
                logger.info("Auto index skipped: VC participation is active.")
                continue
            indexing_task = asyncio.create_task(
                _run_build_index_job(channel=None, history_query="auto_index")
            )

    @client.event
    async def on_ready() -> None:
        nonlocal auto_index_task, periodic_warmup_task
        logger.info("Logged in as %s", client.user)
        await context.vc.start()
        if auto_index_task is None or auto_index_task.done():
            auto_index_task = asyncio.create_task(_auto_index_loop())
        if periodic_warmup_task is None or periodic_warmup_task.done():
            periodic_warmup_task = asyncio.create_task(_periodic_warmup_loop())

    @client.event
    async def on_voice_state_update(
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        await context.vc.on_voice_state_update(member, before, after)

    @client.event
    async def on_message(message: discord.Message) -> None:
        nonlocal indexing_task, evaluating_task
        if message.author == client.user:
            return
        if getattr(message.author, "bot", False):
            return

        if context.vc.is_voice_chat_channel(message.channel):
            await context.vc.capture_voice_chat_message(message)

        content = (message.content or "").strip()
        parsed = parse_command(
            content=content,
            prefix=command_prefix,
            index_command_prefix=index_command_prefix,
        )

        if parsed.kind == "join_vc":
            handled = await context.vc.maybe_join_from_command(message)
            if not handled:
                await message.channel.send("`/ai join` はVCのチャット欄でのみ有効です。")
            return
        if parsed.kind == "quit_vc":
            handled = await context.vc.maybe_quit_from_command(message)
            if not handled:
                await message.channel.send("`/ai quit` はVCのチャット欄でのみ有効です。")
            return

        if parsed.kind == "build_index":
            if not _is_maintenance_authorized(message):
                await message.channel.send("このコマンドを実行する権限がありません。")
                return
            if context.vc.has_active_session():
                await message.channel.send("VC参加中のため、新規のインデックス更新は開始できません。")
                return
            if indexing_in_progress or (indexing_task is not None and not indexing_task.done()):
                await message.channel.send("インデックス更新は既に実行中です。")
                return
            indexing_task = asyncio.create_task(
                _run_build_index_job(channel=message.channel, history_query=content)
            )
            return

        if parsed.kind == "eval":
            if not _is_maintenance_authorized(message):
                await message.channel.send("このコマンドを実行する権限がありません。")
                return
            if evaluating_task is not None and not evaluating_task.done():
                await message.channel.send("評価は既に実行中です。")
                return
            evaluating_task = asyncio.create_task(_run_eval_job(channel=message.channel))
            return

        if parsed.kind == "stop":
            actions: list[str] = []
            channel_id = int(message.channel.id)
            cancel_event = channel_cancel_events.get(channel_id)
            answer_task = channel_generation_tasks.get(channel_id)
            if answer_task and not answer_task.done() and cancel_event is not None:
                cancel_event.set()
                answer_task.cancel()
                actions.append("回答生成を中止します。")
            if indexing_task is not None and not indexing_task.done():
                indexing_cancel_event.set()
                indexing_task.cancel()
                actions.append("インデックス更新を中止します。")
            if evaluating_task is not None and not evaluating_task.done():
                evaluating_task.cancel()
                actions.append("評価を中止します。")
            if not actions:
                actions.append("停止対象の処理は実行中ではありません。")
            await message.channel.send("\n".join(actions))
            return

        query = parsed.payload if parsed.kind == "chat" else _extract_query_from_message(message)
        explicit_fast = bool(parsed.force_fast_mode)
        if not query:
            return
        if not explicit_fast:
            query, explicit_fast = _strip_fast_prefix(query)
        if not query:
            return
        if indexing_in_progress:
            await message.channel.send(_indexing_blocked_message())
            return
        if (
            context.config.app.max_input_characters > 0
            and len(query) > context.config.app.max_input_characters
        ):
            await message.channel.send(
                f"入力できる最大文字数を超えています。（{context.config.app.max_input_characters}）以下で入力してください。"
            )
            return
        channel_id = int(message.channel.id)
        existing = channel_generation_tasks.get(channel_id)
        if existing is not None and not existing.done():
            await message.channel.send(
                "回答生成は既に実行中です。中止する場合は /ai stop を実行してください。"
            )
            return

        special_channel_mode = _is_special_channel_invocation(message)
        routing_history_override: list[tuple[str, str, list[str]]] | None = None
        generation_history_override: list[tuple[str, str, list[str]]] | None = None
        force_disable_additional_memory = False
        append_sources_to_response = True
        extra_mode_instruction: str | None = None
        if not special_channel_mode:
            history = await _collect_special_channel_history(message)
            routing_history_override = history
            generation_history_override = history
            force_disable_additional_memory = True
            append_sources_to_response = False
            extra_mode_instruction = context.config.rag.history.special_channel_custom_instruction
            logger.info(
                "Non-special channel mode override applied. channel=%s turns=%s",
                getattr(message.channel, "name", ""),
                len(history),
            )

        force_fast_mode = explicit_fast or context.vc.should_use_fast_model_for_query()
        task = asyncio.create_task(
            _run_chat(
                message=message,
                query=query,
                force_fast_mode=force_fast_mode,
                routing_history_override=routing_history_override,
                generation_history_override=generation_history_override,
                append_sources_to_response=append_sources_to_response,
                force_disable_additional_memory=force_disable_additional_memory,
                extra_mode_instruction=extra_mode_instruction,
            )
        )
        channel_generation_tasks[channel_id] = task

    token = context.config.integrations.discord.bot_token
    if not token:
        raise RuntimeError("KUMC_DISCORD_BOT_TOKEN is not set.")
    client.run(token, log_handler=None)


if __name__ == "__main__":
    main()
