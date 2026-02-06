import asyncio
import json
import logging
import sys
import threading
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv
import discord
from langchain_core.documents import Document

from pipeline.rag_pipeline import GenerationCancelled, RagPipeline
from config import AppConfig, EmbeddingFactory
from indexing.llm_client import generate_text
from pipeline.function_calling import decide_tools
from pipeline.llm_clients import (
    generate_with_gemini_config,
    generate_with_llama_config,
)
from pipeline.prompts import (
    QUERY_TRANSFORM_SYSTEM_PROMPT,
    build_query_transform_prompt,
)


# Config
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env", override=True)
APP_CONFIG = AppConfig.from_here(base_dir=BASE_DIR)

INDEX_DIR = APP_CONFIG.index_dir

COMMAND_PREFIX = APP_CONFIG.command_prefix
BUILD_INDEX_COMMAND = APP_CONFIG.index_command_prefix
BUILD_INDEX_PATH = APP_CONFIG.base_dir / "app" / "src" / "indexing" / "build_index.py"
EVAL_COMMAND = "/ai eval"
STOP_COMMAND = "/ai stop"
EVAL_SCRIPT_PATH = APP_CONFIG.base_dir / "app" / "src" / "eval" / "evaluate_ragas.py"
EVAL_METRICS_PREFIX = "EVAL_METRICS_JSON:"
AUTO_INDEX_ENABLED = APP_CONFIG.auto_index_enabled
AUTO_INDEX_WEEKDAYS = APP_CONFIG.auto_index_weekdays
AUTO_INDEX_HOUR = APP_CONFIG.auto_index_hour
AUTO_INDEX_MINUTE = APP_CONFIG.auto_index_minute
MAX_INPUT_CHARACTERS = APP_CONFIG.max_input_characters
KUMC_AGENT_CHANNEL_NAME = "kumc-agent"
BOT_MENTION_USER_ID = 1457352598209171520

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
indexing_process: asyncio.subprocess.Process | None = None
indexing_stop_requested = False
auto_index_task: asyncio.Task[None] | None = None
auto_index_last_run: date | None = None
is_evaluating = False
evaluating_task: asyncio.Task[None] | None = None
channel_generation_tasks: dict[int, asyncio.Task[None]] = {}
channel_cancel_events: dict[int, threading.Event] = {}


def _bot_mention_prefixes() -> tuple[str, ...]:
    ids: set[int] = {BOT_MENTION_USER_ID}
    current_user = discord_client.user
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
            if query.startswith(COMMAND_PREFIX):
                query = query[len(COMMAND_PREFIX) :].strip()
            return query

    if content.startswith(COMMAND_PREFIX):
        return content[len(COMMAND_PREFIX) :].strip()

    channel_name = str(getattr(message.channel, "name", "") or "").strip()
    if channel_name == KUMC_AGENT_CHANNEL_NAME:
        return content
    return ""


def _warmup_embedding() -> None:
    embeddings = _embedding_factory.get_embeddings()
    embeddings.embed_query("warmup")
    embeddings.embed_documents(["warmup document"])


def _warmup_max_tokens(value: int) -> int:
    try:
        raw = int(value)
    except (TypeError, ValueError):
        raw = 1
    return max(1, min(8, raw))


def _warmup_faiss_index() -> None:
    rag_pipeline._vectorstore()


def _warmup_reranker() -> None:
    model_path = (APP_CONFIG.cross_encoder_model_path or "").strip()
    if not model_path:
        logger.info("Warmup: cross-encoder reranker skipped (model path not set).")
        return
    doc = Document(page_content="warmup", metadata={})
    rag_pipeline._reranker.score_documents(query="warmup", docs=[doc])


def _warmup_function_calling() -> None:
    decide_tools(query="warmup", config=APP_CONFIG)


def _warmup_answer_llm() -> None:
    provider = (APP_CONFIG.llm_provider or "").lower()
    if provider == "gemini":
        generate_with_gemini_config(
            api_key=GEMINI_API_KEY or "",
            prompt="こんにちは",
            system_rules=APP_CONFIG.system_rules,
            model=APP_CONFIG.genai_model,
            temperature=APP_CONFIG.temperature,
            max_output_tokens=_warmup_max_tokens(APP_CONFIG.max_output_tokens),
            thinking_level=APP_CONFIG.thinking_level,
        )
        return
    if provider == "llama":
        generate_with_llama_config(
            messages=[
                {"role": "system", "content": "You are a warmup assistant."},
                {"role": "user", "content": "hello"},
            ],
            model_path=APP_CONFIG.llama_model_path,
            ctx_size=APP_CONFIG.llama_ctx_size,
            threads=APP_CONFIG.llama_threads,
            gpu_layers=APP_CONFIG.llama_gpu_layers,
            temperature=APP_CONFIG.temperature,
            max_output_tokens=_warmup_max_tokens(APP_CONFIG.max_output_tokens),
            stop=["\n---"],
        )
        return
    raise ValueError(f"Unsupported llm_provider: {APP_CONFIG.llm_provider}")


def _warmup_no_rag_llm() -> None:
    provider = (APP_CONFIG.no_rag_llm_provider or "").lower()
    if provider == "gemini":
        generate_with_gemini_config(
            api_key=GEMINI_API_KEY or "",
            prompt="こんにちは",
            system_rules=APP_CONFIG.system_rules,
            model=APP_CONFIG.no_rag_genai_model,
            temperature=APP_CONFIG.no_rag_temperature,
            max_output_tokens=_warmup_max_tokens(
                APP_CONFIG.no_rag_max_output_tokens
            ),
            thinking_level=APP_CONFIG.no_rag_thinking_level,
        )
        return
    if provider == "llama":
        generate_with_llama_config(
            messages=[
                {"role": "system", "content": "You are a warmup assistant."},
                {"role": "user", "content": "hello"},
            ],
            model_path=APP_CONFIG.no_rag_llama_model_path,
            ctx_size=APP_CONFIG.no_rag_llama_ctx_size,
            threads=APP_CONFIG.llama_threads,
            gpu_layers=APP_CONFIG.llama_gpu_layers,
            temperature=APP_CONFIG.no_rag_temperature,
            max_output_tokens=_warmup_max_tokens(
                APP_CONFIG.no_rag_max_output_tokens
            ),
        )
        return
    raise ValueError(
        f"Unsupported no_rag_llm_provider: {APP_CONFIG.no_rag_llm_provider}"
    )


def _warmup_query_transform_llm() -> None:
    provider = (APP_CONFIG.query_transform_provider or "").lower()
    if provider not in {"gemini", "llama"}:
        logger.info(
            "Warmup: query transform skipped (unsupported provider=%s).",
            APP_CONFIG.query_transform_provider,
        )
        return
    prompt = build_query_transform_prompt(query="warmup")
    model = (
        APP_CONFIG.query_transform_llama_model
        if provider == "llama"
        else APP_CONFIG.query_transform_gemini_model
    )
    generate_text(
        provider=provider,
        api_key=GEMINI_API_KEY or "",
        prompt=prompt,
        model=model,
        system_prompt=QUERY_TRANSFORM_SYSTEM_PROMPT,
        llama_model_path=APP_CONFIG.query_transform_llama_model_path,
        llama_ctx_size=APP_CONFIG.query_transform_llama_ctx_size,
        temperature=APP_CONFIG.query_transform_temperature,
        max_output_tokens=_warmup_max_tokens(
            APP_CONFIG.query_transform_max_output_tokens
        ),
        thinking_level=APP_CONFIG.thinking_level,
        llama_threads=APP_CONFIG.llama_threads,
        llama_gpu_layers=APP_CONFIG.llama_gpu_layers,
        response_mime_type="text/plain",
    )


def _warmup_models() -> None:
    logger.info("Warmup started.")
    steps = [
        ("embedding", _warmup_embedding),
        ("faiss_index", _warmup_faiss_index),
        ("cross_encoder_reranker", _warmup_reranker),
        ("function_calling", _warmup_function_calling),
        ("answer_llm", _warmup_answer_llm),
        ("no_rag_llm", _warmup_no_rag_llm),
        ("query_transform_llm", _warmup_query_transform_llm),
    ]
    for name, action in steps:
        try:
            action()
            logger.info("Warmup complete: %s", name)
        except Exception:
            logger.exception("Warmup failed: %s", name)
    logger.info("Warmup finished.")


async def _send_status(
    channel: discord.abc.Messageable | None, message: str
) -> None:
    if channel is None:
        logger.info(message)
        return
    await channel.send(message)


async def _run_build_index(channel: discord.abc.Messageable | None) -> None:
    global is_indexing, indexing_task, indexing_process, indexing_stop_requested
    is_indexing = True
    try:
        await _send_status(channel, "インデックス更新を開始します。")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(BUILD_INDEX_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        indexing_process = process
        if indexing_stop_requested and process.returncode is None:
            process.terminate()
        stdout, stderr = await process.communicate()
        if indexing_stop_requested:
            await _send_status(channel, "インデックス更新を中止しました。")
            return
        if process.returncode != 0:
            logger.error(
                "build_index failed with code %s: %s",
                process.returncode,
                (stderr or b"").decode("utf-8", errors="replace"),
            )
            await _send_status(
                channel, "インデックス更新に失敗しました。ログを確認してください。"
            )
            return

        if stdout:
            logger.info(
                "build_index completed: %s",
                stdout.decode("utf-8", errors="replace"),
            )
        rag_pipeline.refresh_index()
        await _send_status(
            channel, "インデックス更新が完了しました。クエリ受付を再開します。"
        )
    except Exception:
        logger.exception("Failed to run build_index")
        await _send_status(
            channel, "インデックス更新に失敗しました。ログを確認してください。"
        )
    finally:
        is_indexing = False
        indexing_task = None
        indexing_process = None
        indexing_stop_requested = False


def _parse_eval_metrics(output: bytes) -> dict[str, float] | None:
    text = output.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if line.startswith(EVAL_METRICS_PREFIX):
            payload = line[len(EVAL_METRICS_PREFIX) :].strip()
            if not payload:
                return None
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("Failed to parse eval metrics JSON: %s", payload)
                return None
            if isinstance(data, dict):
                return {
                    key: float(value)
                    for key, value in data.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            return None
    return None


def _format_eval_metrics(metrics: dict[str, float]) -> str:
    preferred_order = [
        "answer_relevancy",
        "context_recall",
    ]
    parts: list[str] = []
    for key in preferred_order:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    for key in sorted(metrics.keys()):
        if key in preferred_order:
            continue
        parts.append(f"{key}={metrics[key]:.4f}")
    return ", ".join(parts)


async def _run_eval(channel: discord.abc.Messageable | None) -> None:
    global is_evaluating, evaluating_task
    is_evaluating = True
    try:
        await _send_status(channel, "評価を開始します。")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(EVAL_SCRIPT_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.error(
                "evaluate_ragas failed with code %s: %s",
                process.returncode,
                (stderr or b"").decode("utf-8", errors="replace"),
            )
            await _send_status(channel, "評価に失敗しました。ログを確認してください。")
            return

        if stdout:
            logger.info(
                "evaluate_ragas completed: %s",
                stdout.decode("utf-8", errors="replace"),
            )

        metrics = _parse_eval_metrics(stdout or b"")
        if metrics:
            await _send_status(
                channel,
                f"評価が完了しました。最終指標: {_format_eval_metrics(metrics)}",
            )
        else:
            await _send_status(channel, "評価が完了しました。")
    except Exception:
        logger.exception("Failed to run evaluate_ragas")
        await _send_status(channel, "評価に失敗しました。ログを確認してください。")
    finally:
        is_evaluating = False
        evaluating_task = None


async def _run_answer(message: discord.Message, query: str) -> None:
    channel = message.channel
    channel_id = channel.id
    cancel_event = threading.Event()
    channel_cancel_events[channel_id] = cancel_event
    try:
        loop = asyncio.get_running_loop()

        def _notify_research_start() -> None:
            if channel is None:
                return
            future = asyncio.run_coroutine_threadsafe(
                channel.send("詳細な検索を行っています…"),
                loop,
            )
            future.add_done_callback(_handle_research_status)

        def _handle_research_status(fut: asyncio.Future) -> None:
            try:
                fut.result()
            except Exception:
                logger.exception("Failed to send research status")

        def _notify_memory_start() -> None:
            if channel is None:
                return
            future = asyncio.run_coroutine_threadsafe(
                channel.send("過去のチャットを思い出しています…"),
                loop,
            )
            future.add_done_callback(_handle_memory_status)

        def _handle_memory_status(fut: asyncio.Future) -> None:
            try:
                fut.result()
            except Exception:
                logger.exception("Failed to send memory status")

        def _notify_research_and_memory_start() -> None:
            if channel is None:
                return
            future = asyncio.run_coroutine_threadsafe(
                channel.send(
                    "詳細な検索を行っています…\n過去のチャットを思い出しています…"
                ),
                loop,
            )
            future.add_done_callback(_handle_research_and_memory_status)

        def _handle_research_and_memory_status(fut: asyncio.Future) -> None:
            try:
                fut.result()
            except Exception:
                logger.exception("Failed to send research+memory status")

        answer = await asyncio.to_thread(
            rag_pipeline.answer_with_routing,
            query,
            on_research_start=_notify_research_start,
            on_memory_start=_notify_memory_start,
            on_research_and_memory_start=_notify_research_and_memory_start,
            cancel_event=cancel_event,
        )
        if cancel_event.is_set():
            return
        await channel.send(answer)
    except GenerationCancelled:
        return
    except Exception as e:
        logger.exception("Failed to handle /llm request")
        await channel.send(f"エラーが発生しました: {type(e).__name__}: {e}")
    finally:
        channel_cancel_events.pop(channel_id, None)
        channel_generation_tasks.pop(channel_id, None)


def _should_run_auto_index(now: datetime) -> bool:
    if not AUTO_INDEX_ENABLED:
        return False
    if AUTO_INDEX_WEEKDAYS and now.weekday() not in AUTO_INDEX_WEEKDAYS:
        return False
    if now.hour != AUTO_INDEX_HOUR or now.minute != AUTO_INDEX_MINUTE:
        return False
    if auto_index_last_run == now.date():
        return False
    return True


async def _auto_index_loop() -> None:
    global auto_index_last_run, indexing_task, is_indexing, indexing_stop_requested
    logger.info(
        "Auto index scheduler started. enabled=%s time=%02d:%02d weekdays=%s",
        AUTO_INDEX_ENABLED,
        AUTO_INDEX_HOUR,
        AUTO_INDEX_MINUTE,
        ",".join(str(day) for day in AUTO_INDEX_WEEKDAYS) or "all",
    )
    while True:
        await asyncio.sleep(20)
        now = datetime.now()
        if not _should_run_auto_index(now):
            continue
        if indexing_task and not indexing_task.done():
            logger.info("Auto index skipped: indexing already running.")
            auto_index_last_run = now.date()
            continue
        is_indexing = True
        indexing_stop_requested = False
        auto_index_last_run = now.date()
        indexing_task = asyncio.create_task(_run_build_index(None))


# Discord events
@discord_client.event
async def on_ready():
    logger.info("Logged in as %s", discord_client.user)
    global auto_index_task
    if AUTO_INDEX_ENABLED and (auto_index_task is None or auto_index_task.done()):
        auto_index_task = asyncio.create_task(_auto_index_loop())


@discord_client.event
async def on_message(message: discord.Message):
    global evaluating_task, indexing_process, indexing_stop_requested, indexing_task
    global is_evaluating, is_indexing
    if message.author.bot:
        return

    content = (message.content or "").strip()
    if content == BUILD_INDEX_COMMAND:
        if indexing_task and not indexing_task.done():
            await message.channel.send("インデックス更新は既に実行中です。")
            return
        indexing_stop_requested = False
        is_indexing = True
        indexing_task = asyncio.create_task(_run_build_index(message.channel))
        return
    if content == EVAL_COMMAND:
        if evaluating_task and not evaluating_task.done():
            await message.channel.send("評価は既に実行中です。")
            return
        is_evaluating = True
        evaluating_task = asyncio.create_task(_run_eval(message.channel))
        return
    if content == STOP_COMMAND:
        channel_id = message.channel.id
        actions: list[str] = []
        cancel_event = channel_cancel_events.get(channel_id)
        task = channel_generation_tasks.get(channel_id)
        if task and not task.done() and cancel_event:
            if not cancel_event.is_set():
                cancel_event.set()
            actions.append("回答生成を中止します。")
        if indexing_task and not indexing_task.done():
            indexing_stop_requested = True
            if indexing_process and indexing_process.returncode is None:
                indexing_process.terminate()
            actions.append("インデックス更新を中止します。")
        if not actions:
            actions.append("停止対象の処理は実行中ではありません。")
        await message.channel.send("\n".join(actions))
        return

    if is_indexing:
        if _extract_query_from_message(message):
            await message.channel.send("インデックス更新中のため、クエリ受付を停止しています。")
        return

    query = _extract_query_from_message(message)
    if not query:
        return
    if MAX_INPUT_CHARACTERS > 0 and len(query) > MAX_INPUT_CHARACTERS:
        await message.channel.send(
            f"入力できる最大文字数を超えています。（{MAX_INPUT_CHARACTERS}）以下で入力してください。"
        )
        return

    channel_id = message.channel.id
    existing = channel_generation_tasks.get(channel_id)
    if existing and not existing.done():
        await message.channel.send(
            "回答生成は既に実行中です。中止する場合は /ai stop を実行してください。"
        )
        return
    task = asyncio.create_task(_run_answer(message, query))
    channel_generation_tasks[channel_id] = task


def main() -> None:
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set. Please set it in .env")

    _warmup_models()
    discord_client.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
