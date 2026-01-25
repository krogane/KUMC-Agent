import asyncio
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
import discord

from pipeline.rag_pipeline import RagPipeline
from config import AppConfig, EmbeddingFactory


# Config
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env", override=True)
APP_CONFIG = AppConfig.from_here(base_dir=BASE_DIR)

INDEX_DIR = APP_CONFIG.index_dir

COMMAND_PREFIX = APP_CONFIG.command_prefix
BUILD_INDEX_COMMAND = APP_CONFIG.index_command_prefix
BUILD_INDEX_PATH = APP_CONFIG.base_dir / "app" / "src" / "indexing" / "build_index.py"

DISCORD_BOT_TOKEN = APP_CONFIG.discord_bot_token
GEMINI_API_KEY = APP_CONFIG.gemini_api_key


# Bootstrap
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Discord Client
intents = discord.Intents.default()
intents.message_content = True
discord_client = discord.Client(intents=intents)


# RAG Pipeline
_embedding_factory = EmbeddingFactory(APP_CONFIG.embedding_model)
rag_pipeline = RagPipeline(
    index_dir=INDEX_DIR,
    embedding_factory=_embedding_factory,
    llm_api_key=GEMINI_API_KEY or "",
    config=APP_CONFIG,
)

is_indexing = False
indexing_task: asyncio.Task[None] | None = None


async def _run_build_index(channel: discord.abc.Messageable) -> None:
    global is_indexing, indexing_task
    is_indexing = True
    try:
        await channel.send("インデックス更新を開始します。")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(BUILD_INDEX_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.error(
                "build_index failed with code %s: %s",
                process.returncode,
                (stderr or b"").decode("utf-8", errors="replace"),
            )
            await channel.send("インデックス更新に失敗しました。ログを確認してください。")
            return

        if stdout:
            logger.info(
                "build_index completed: %s",
                stdout.decode("utf-8", errors="replace"),
            )
        rag_pipeline.refresh_index()
        await channel.send("インデックス更新が完了しました。クエリ受付を再開します。")
    except Exception:
        logger.exception("Failed to run build_index")
        await channel.send("インデックス更新に失敗しました。ログを確認してください。")
    finally:
        is_indexing = False
        indexing_task = None


# Discord events
@discord_client.event
async def on_ready():
    logger.info("Logged in as %s", discord_client.user)


@discord_client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    content = (message.content or "").strip()
    if content == BUILD_INDEX_COMMAND:
        global indexing_task, is_indexing
        if indexing_task and not indexing_task.done():
            await message.channel.send("インデックス更新は既に実行中です。")
            return
        is_indexing = True
        indexing_task = asyncio.create_task(_run_build_index(message.channel))
        return

    if is_indexing:
        if content.startswith(COMMAND_PREFIX):
            await message.channel.send("インデックス更新中のため、クエリ受付を停止しています。")
        return

    if not content.startswith(COMMAND_PREFIX):
        return

    query = content[len(COMMAND_PREFIX) :].strip()
    if not query:
        return

    try:
        answer = await asyncio.to_thread(rag_pipeline.answer, query)
        await message.channel.send(answer)
    except Exception as e:
        logger.exception("Failed to handle /llm request")
        await message.channel.send(f"エラーが発生しました: {type(e).__name__}: {e}")


def main() -> None:
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set. Please set it in .env")

    discord_client.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
