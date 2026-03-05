from __future__ import annotations

import asyncio
import logging

from kumc_agent.runtime.container import build_runtime_context
from kumc_agent.usecases.chat.answer import ChatRequest
from kumc_agent.usecases.eval.ragas import EvaluateRagasRequest
from kumc_agent.usecases.indexing.build import BuildIndexRequest
from kumc_agent.frontends.discord.commands import parse_command
from kumc_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    import discord

    context = build_runtime_context()
    configure_logging(context.config.app.log_level)

    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    client = discord.Client(intents=intents)
    indexing_in_progress = False

    command_prefix = context.config.app.command_prefix.strip()
    index_command_prefix = context.config.app.index_command_prefix.strip()
    context.vc.bind_discord_client(
        discord_client=client,
        is_indexing_active=lambda: indexing_in_progress,
    )

    @client.event
    async def on_ready() -> None:
        logger.info("Logged in as %s", client.user)
        await context.vc.start()

    @client.event
    async def on_voice_state_update(
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        await context.vc.on_voice_state_update(member, before, after)

    @client.event
    async def on_message(message: discord.Message) -> None:
        nonlocal indexing_in_progress
        if message.author == client.user:
            return
        if getattr(message.author, "bot", False):
            return

        if context.vc.is_voice_chat_channel(message.channel):
            await context.vc.capture_voice_chat_message(message)

        parsed = parse_command(
            content=message.content,
            prefix=command_prefix,
            index_command_prefix=index_command_prefix,
        )

        if parsed.kind == "none":
            return

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
            if context.vc.has_active_session():
                await message.channel.send("VC参加中のため、新規のインデックス更新は開始できません。")
                return
            if indexing_in_progress:
                await message.channel.send("インデックス更新は既に実行中です。")
                return
            await message.channel.send("Index build started.")
            indexing_in_progress = True

            def _run_build() -> str:
                result = context.build_index.execute(BuildIndexRequest(refresh_sources=True))
                return (
                    f"Index build completed: loaded_sources={result.loaded_sources}, "
                    f"documents={result.documents}, chunks={result.chunks}"
                )

            try:
                text = await asyncio.to_thread(_run_build)
            except Exception as exc:  # pragma: no cover - runtime dependent
                logger.exception("Index build failed")
                text = f"Index build failed: {exc}"
            finally:
                indexing_in_progress = False
            await message.channel.send(text)
            return

        if parsed.kind == "eval":
            eval_file = context.config.app.eval_dir / "ragas.jsonl"

            def _run_eval() -> str:
                result = context.eval_ragas.execute(
                    EvaluateRagasRequest(
                        eval_file=eval_file,
                        result_path=context.config.app.eval_dir / "result" / "result.json",
                    )
                )
                return (
                    f"Eval done: total={result.total}, "
                    f"exact_match={result.exact_match:.3f}, token_overlap={result.token_overlap:.3f}"
                )

            text = await asyncio.to_thread(_run_eval)
            await message.channel.send(text)
            return

        if parsed.kind == "stop":
            await message.channel.send("Stop command accepted. (No running cancellable task)")
            return

        if parsed.kind == "chat":
            query = parsed.payload
            if not query:
                await message.channel.send("Please provide a query.")
                return
            if (
                context.config.app.max_input_characters > 0
                and len(query) > context.config.app.max_input_characters
            ):
                await message.channel.send("Input is too long.")
                return

            context.vc.notify_rag_started()
            try:
                answer = await asyncio.to_thread(
                    lambda: context.chat_answer.execute(ChatRequest(query=query))
                )
            finally:
                context.vc.notify_rag_finished()
            await message.channel.send(answer.text)

    token = context.config.integrations.discord.bot_token
    if not token:
        raise RuntimeError("KUMC_DISCORD_BOT_TOKEN is not set.")
    client.run(token, log_handler=None)


if __name__ == "__main__":
    main()
