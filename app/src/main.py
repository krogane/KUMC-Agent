import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import discord

from rag_pipeline import RagPipeline
from config import AppConfig, EmbeddingFactory, DEFAULT_SYSTEM_RULES
# コンフィグ
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

APP_CONFIG = AppConfig.from_here(system_rules=DEFAULT_SYSTEM_RULES, base_dir=BASE_DIR)
INDEX_DIR = APP_CONFIG.index_dir
COMMAND_PREFIX: str = "/ai "

LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR


# Bootstrap
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Discord client
intents = discord.Intents.default()
intents.message_content = True

discord_client = discord.Client(intents=intents)


_embedding_factory = EmbeddingFactory(APP_CONFIG.embedding_model_name)
rag_pipeline = RagPipeline(
    index_dir=INDEX_DIR,
    embedding_factory=_embedding_factory,
    llm_api_key=GEMINI_API_KEY or "",
    config=APP_CONFIG,
)


# Discord events
@discord_client.event
async def on_ready():
    logger.info("Logged in as %s", discord_client.user)


@discord_client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    content = (message.content or "").strip()
    if not content.startswith(COMMAND_PREFIX):
        return

    query = content[len(COMMAND_PREFIX) :].strip()
    if not query:
        await message.channel.send("質問が空です。例: /llm 〇〇について教えて")
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
