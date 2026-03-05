import asyncio
import json
import logging
import sys
import threading
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import discord
from langchain_core.documents import Document

from kumc_agent.infra.legacy.pipeline.rag_pipeline import GenerationCancelled, RagPipeline
from kumc_agent.infra.legacy.pipeline.prompts import ChatHistoryEntry
from kumc_agent.infra.legacy.config import AppConfig, EmbeddingFactory
from kumc_agent.infra.legacy.pipeline.function_calling import FunctionRoutingDecision, decide_tools
from kumc_agent.infra.legacy.pipeline.llm_clients import (
    generate_with_llama_config,
)
from kumc_agent.infra.legacy.vc import VoiceMeetingManager


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
FAST_QUERY_PREFIX = "fast"
FAST_MODEL_NOTICE = "※負荷軽減のために軽量モデルを使用しました。"
EVAL_SCRIPT_PATH = APP_CONFIG.base_dir / "app" / "src" / "eval" / "evaluate_ragas.py"
EVAL_METRICS_PREFIX = "EVAL_METRICS_JSON:"
AUTO_INDEX_TIMEZONE = ZoneInfo("Asia/Tokyo")
AUTO_INDEX_ENABLED = APP_CONFIG.auto_index_enabled
AUTO_INDEX_WEEKDAYS = APP_CONFIG.auto_index_weekdays
AUTO_INDEX_HOUR = APP_CONFIG.auto_index_hour
AUTO_INDEX_MINUTE = APP_CONFIG.auto_index_minute
WARMUP_INTERVAL_MINUTES = max(0, APP_CONFIG.warmup_interval_minutes)
MAX_INPUT_CHARACTERS = APP_CONFIG.max_input_characters
SPECIAL_CHANNEL_HISTORY_LIMIT = max(0, APP_CONFIG.special_channel_history_limit)
KUMC_AGENT_CHANNEL_NAME = "kumc-agent"
BOT_MENTION_USER_ID = 1457352598209171520
MAINTENANCE_COMMAND_AUTHOR_ID_SET = set(APP_CONFIG.maintenance_command_author_ids)

DISCORD_BOT_TOKEN = APP_CONFIG.discord_bot_token
GEMINI_API_KEY = APP_CONFIG.gemini_api_key
_answer_record_log_path_candidate = Path(APP_CONFIG.answer_record_log_path).expanduser()
ANSWER_RECORD_LOG_PATH = (
    _answer_record_log_path_candidate
    if _answer_record_log_path_candidate.is_absolute()
    else APP_CONFIG.base_dir / _answer_record_log_path_candidate
)


# Bootstrap
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Discord Client
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
discord_client = discord.Client(intents=intents)


# RAG Pipeline
_embedding_factory = EmbeddingFactory(
    APP_CONFIG.embedding_model,
    api_key=APP_CONFIG.gemini_api_key,
)
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
indexing_started_at: datetime | None = None
auto_index_task: asyncio.Task[None] | None = None
auto_index_last_run: date | None = None
periodic_warmup_task: asyncio.Task[None] | None = None
is_evaluating = False
evaluating_task: asyncio.Task[None] | None = None
channel_generation_tasks: dict[int, asyncio.Task[None]] = {}
channel_cancel_events: dict[int, threading.Event] = {}
_warmup_lock = threading.Lock()
_answer_record_log_lock = threading.Lock()
voice_meeting_manager = VoiceMeetingManager(
    discord_client=discord_client,
    config=APP_CONFIG,
    is_indexing_active=lambda: is_indexing,
)
JOIN_COMMAND = f"{COMMAND_PREFIX.strip()} join".strip().lower()
QUIT_COMMAND = f"{COMMAND_PREFIX.strip()} quit".strip().lower()


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

    if getattr(message, "guild", None) is None:
        return content

    channel_name = str(getattr(message.channel, "name", "") or "").strip()
    if channel_name == KUMC_AGENT_CHANNEL_NAME:
        return content
    return ""


def _strip_fast_prefix(query: str) -> tuple[str, bool]:
    normalized = (query or "").strip()
    if not normalized:
        return "", False
    parts = normalized.split(maxsplit=1)
    head = parts[0].strip().lower()
    if head != FAST_QUERY_PREFIX:
        return normalized, False
    if len(parts) == 1:
        return "", True
    return parts[1].strip(), True


def _has_explicit_fast_prefix(message: discord.Message) -> bool:
    content = (message.content or "").strip()
    if not content:
        return False
    for prefix in _bot_mention_prefixes():
        if content.startswith(prefix):
            content = content[len(prefix) :].strip()
            break
    command_prefix = COMMAND_PREFIX.strip().lower()
    if not command_prefix:
        return False
    lowered = content.lower()
    if not lowered.startswith(command_prefix):
        return False
    rest = content[len(command_prefix) :].strip()
    if not rest:
        return False
    return rest.split(maxsplit=1)[0].strip().lower() == FAST_QUERY_PREFIX


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
    author_id = str(getattr(author, "id", "") or "").strip()
    return author_id or "unknown"


def _question_user_id_from_message(message: discord.Message) -> str:
    author = getattr(message, "author", None)
    author_id = getattr(author, "id", None)
    if author_id is None:
        return "unknown"
    return str(author_id)


def _question_username_from_message(message: discord.Message) -> str:
    author = getattr(message, "author", None)
    username = str(getattr(author, "name", "") or "").strip()
    if username:
        return username
    display_name = str(getattr(author, "display_name", "") or "").strip()
    if display_name:
        return display_name
    return "unknown"


def _is_special_channel_invocation(message: discord.Message) -> bool:
    if getattr(message, "guild", None) is None:
        return False
    channel_name = str(getattr(message.channel, "name", "") or "").strip()
    if channel_name == KUMC_AGENT_CHANNEL_NAME:
        return False
    content = (message.content or "").strip()
    if not content:
        return False
    if any(content.startswith(prefix) for prefix in _bot_mention_prefixes()):
        return True
    return content.startswith(COMMAND_PREFIX)


def _history_entry_from_channel_message(
    message: discord.Message,
) -> ChatHistoryEntry | None:
    text = (message.content or "").strip()
    if not text:
        return None
    author = _question_author_from_message(message)
    return (f"author: {author}\n{text}", "", [])


async def _resolve_referenced_message(
    message: discord.Message,
    *,
    reference_cache: dict[int, discord.Message | None],
) -> discord.Message | None:
    reference = getattr(message, "reference", None)
    if reference is None:
        return None
    referenced_message_id = getattr(reference, "message_id", None)
    if referenced_message_id is None:
        return None
    try:
        message_id = int(referenced_message_id)
    except (TypeError, ValueError):
        return None
    if message_id in reference_cache:
        return reference_cache[message_id]

    resolved = getattr(reference, "resolved", None)
    if isinstance(resolved, discord.Message):
        reference_cache[message_id] = resolved
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
            try:
                if channel_id_int is not None:
                    fetch_channel = await guild.fetch_channel(channel_id_int)
            except Exception:
                fetch_channel = None
    if fetch_channel is None:
        fetch_channel = message.channel

    fetch_message = getattr(fetch_channel, "fetch_message", None)
    if not callable(fetch_message):
        reference_cache[message_id] = None
        return None

    try:
        referenced = await fetch_message(message_id)
    except Exception as exc:
        logger.info(
            "Failed to fetch referenced message. channel_id=%s message_id=%s error=%s",
            getattr(fetch_channel, "id", "unknown"),
            message_id,
            exc,
        )
        reference_cache[message_id] = None
        return None

    reference_cache[message_id] = referenced
    return referenced


async def _collect_special_channel_history(
    message: discord.Message,
    *,
    history_limit: int,
) -> tuple[list[ChatHistoryEntry], int, int]:
    if history_limit <= 0:
        return [], 0, 0

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
            if len(primary_messages) >= history_limit:
                break
    except Exception:
        logger.exception(
            "Failed to collect special-channel history. channel_id=%s",
            message.channel.id,
        )

    if len(primary_messages) > history_limit:
        primary_messages = primary_messages[:history_limit]

    reference_cache: dict[int, discord.Message | None] = {}
    all_messages: dict[int, discord.Message] = {
        int(item.id): item for item in primary_messages
    }
    expanded_reference_count = 0

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
                expanded_reference_count += 1
            current = referenced

    ordered_messages = sorted(
        all_messages.values(),
        key=lambda item: (item.created_at, int(item.id)),
    )
    history_entries: list[ChatHistoryEntry] = []
    for item in ordered_messages:
        entry = _history_entry_from_channel_message(item)
        if entry is not None:
            history_entries.append(entry)

    return history_entries, len(primary_messages), expanded_reference_count


def _history_scope_for_message(message: discord.Message) -> str:
    guild = getattr(message, "guild", None)
    guild_id = getattr(guild, "id", None)
    if guild_id is not None:
        return f"guild:{guild_id}"
    return f"channel:{message.channel.id}"


def _is_maintenance_command_authorized(message: discord.Message) -> bool:
    if not MAINTENANCE_COMMAND_AUTHOR_ID_SET:
        return False
    author_id = getattr(getattr(message, "author", None), "id", None)
    if author_id is None:
        return False
    return int(author_id) in MAINTENANCE_COMMAND_AUTHOR_ID_SET


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
    if not APP_CONFIG.rerank_enabled:
        logger.info("Warmup: cross-encoder reranker skipped (rerank disabled).")
        return
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
        logger.info(
            "Warmup: answer LLM skipped (Gemini API warmup is disabled)."
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
        logger.info(
            "Warmup: no-rag LLM skipped (Gemini API warmup is disabled)."
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


def _warmup_refusal_llm() -> None:
    provider = (APP_CONFIG.refusal_llm_provider or "").lower()
    if provider == "gemini":
        logger.info(
            "Warmup: refusal LLM skipped (Gemini API warmup is disabled)."
        )
        return
    if provider == "llama":
        generate_with_llama_config(
            messages=[
                {"role": "system", "content": "You are a refusal warmup assistant."},
                {"role": "user", "content": "hello"},
            ],
            model_path=APP_CONFIG.refusal_llama_model_path,
            ctx_size=APP_CONFIG.refusal_llama_ctx_size,
            threads=APP_CONFIG.llama_threads,
            gpu_layers=APP_CONFIG.llama_gpu_layers,
            temperature=APP_CONFIG.refusal_temperature,
            max_output_tokens=_warmup_max_tokens(
                APP_CONFIG.refusal_max_output_tokens
            ),
            stop=["\n---"],
        )
        return
    raise ValueError(
        f"Unsupported refusal_llm_provider: {APP_CONFIG.refusal_llm_provider}"
    )


def _warmup_rag_idea_llm() -> None:
    provider = (APP_CONFIG.llm_provider or "").lower()
    if provider != "llama":
        logger.info("Warmup: rag-idea LLM skipped (provider=%s).", provider)
        return
    generate_with_llama_config(
        messages=[
            {"role": "system", "content": "You are a creative warmup assistant."},
            {"role": "user", "content": "hello"},
        ],
        model_path=APP_CONFIG.llama_model_path,
        ctx_size=APP_CONFIG.llama_ctx_size,
        threads=APP_CONFIG.llama_threads,
        gpu_layers=APP_CONFIG.llama_gpu_layers,
        temperature=APP_CONFIG.rag_idea_temperature,
        max_output_tokens=_warmup_max_tokens(
            APP_CONFIG.max_output_tokens
        ),
        stop=["\n---"],
    )


def _warmup_models(*, trigger: str) -> None:
    if not _warmup_lock.acquire(blocking=False):
        logger.info(
            "Warmup skipped (already running). trigger=%s",
            trigger,
        )
        return
    try:
        logger.info("Warmup started. trigger=%s", trigger)
        steps = [
            ("embedding", _warmup_embedding),
            ("faiss_index", _warmup_faiss_index),
            ("cross_encoder_reranker", _warmup_reranker),
            ("function_calling", _warmup_function_calling),
            ("answer_llm", _warmup_answer_llm),
            ("no_rag_llm", _warmup_no_rag_llm),
            ("refusal_llm", _warmup_refusal_llm),
            ("rag_idea_llm", _warmup_rag_idea_llm),
        ]
        for name, action in steps:
            try:
                action()
                logger.info("Warmup complete: %s", name)
            except Exception:
                logger.exception("Warmup failed: %s", name)
        logger.info("Warmup finished. trigger=%s", trigger)
    finally:
        _warmup_lock.release()


async def _send_status(
    channel: discord.abc.Messageable | None,
    message: str,
    *,
    history_scope: str | int | None = None,
    history_query: str | None = None,
) -> None:
    if channel is None:
        logger.info(message)
        return
    await channel.send(message)
    query = (history_query or "").strip()
    response = (message or "").strip()
    if not query or not response:
        return
    rag_pipeline._record_history(
        query=query,
        answer=response,
        sources=[],
        history_scope=history_scope,
    )


def _history_query_from_message(
    message: discord.Message,
    *,
    fallback: str | None = None,
) -> str:
    if fallback:
        query = fallback.strip()
        if query:
            return query
    extracted = _extract_query_from_message(message)
    if extracted:
        return extracted
    return (message.content or "").strip()


def _append_answer_record(
    *,
    message: discord.Message,
    query: str,
    routing_decision: FunctionRoutingDecision | None,
    answer: str,
) -> None:
    if not APP_CONFIG.answer_record_log_enabled:
        return

    query_text = (query or "").strip()
    answer_text = (answer or "").strip()
    if not query_text or not answer_text:
        return

    record = {
        "timestamp": datetime.now(AUTO_INDEX_TIMEZONE).isoformat(timespec="seconds"),
        "questioner_user_id": _question_user_id_from_message(message),
        "questioner_username": _question_username_from_message(message),
        "question": query_text,
        "routing_result": asdict(routing_decision) if routing_decision is not None else None,
        "answer": answer_text,
    }

    try:
        with _answer_record_log_lock:
            ANSWER_RECORD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with ANSWER_RECORD_LOG_PATH.open("a", encoding="utf-8") as fw:
                fw.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception(
            "Failed to write answer record log. path=%s",
            ANSWER_RECORD_LOG_PATH,
        )


async def _send_message_with_history(
    message: discord.Message,
    response: str,
    *,
    query: str | None = None,
) -> None:
    await _send_status(
        message.channel,
        response,
        history_scope=_history_scope_for_message(message),
        history_query=_history_query_from_message(message, fallback=query),
    )


def _format_jst_timestamp(value: datetime) -> str:
    return value.astimezone(AUTO_INDEX_TIMEZONE).strftime("%Y/%m/%d %H:%M")


def _build_indexing_blocked_message() -> str:
    started_at = indexing_started_at or datetime.now(AUTO_INDEX_TIMEZONE)
    min_minutes = APP_CONFIG.index_update_estimate_min_minutes
    max_minutes = APP_CONFIG.index_update_estimate_max_minutes
    min_end = started_at + timedelta(minutes=min_minutes)
    max_end = started_at + timedelta(minutes=max_minutes)

    if min_minutes == max_minutes:
        estimate_text = _format_jst_timestamp(min_end)
    else:
        estimate_text = (
            f"{_format_jst_timestamp(min_end)}〜"
            f"{_format_jst_timestamp(max_end)}"
        )
    return (
        "インデックス更新中のため、クエリ受付を停止しています。\n"
        f"更新開始時刻: {_format_jst_timestamp(started_at)} (JST)\n"
        f"終了目安: {estimate_text} (JST)"
    )


async def _run_build_index(
    channel: discord.abc.Messageable | None,
    *,
    history_scope: str | int | None = None,
    history_query: str | None = None,
) -> None:
    global is_indexing, indexing_task, indexing_process, indexing_stop_requested
    global indexing_started_at
    is_indexing = True
    if indexing_started_at is None:
        indexing_started_at = datetime.now(AUTO_INDEX_TIMEZONE)
    try:
        await _send_status(
            channel,
            "インデックス更新を開始します。",
            history_scope=history_scope,
            history_query=history_query,
        )
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
            await _send_status(
                channel,
                "インデックス更新を中止しました。",
                history_scope=history_scope,
                history_query=history_query,
            )
            return
        if process.returncode != 0:
            logger.error(
                "build_index failed with code %s: %s",
                process.returncode,
                (stderr or b"").decode("utf-8", errors="replace"),
            )
            await _send_status(
                channel,
                "インデックス更新に失敗しました。ログを確認してください。",
                history_scope=history_scope,
                history_query=history_query,
            )
            return

        if stdout:
            logger.info(
                "build_index completed: %s",
                stdout.decode("utf-8", errors="replace"),
            )
        rag_pipeline.refresh_index()
        await asyncio.to_thread(_warmup_models, trigger="index_update")
        await _send_status(
            channel,
            "インデックス更新が完了しました。クエリ受付を再開します。",
            history_scope=history_scope,
            history_query=history_query,
        )
    except Exception:
        logger.exception("Failed to run build_index")
        await _send_status(
            channel,
            "インデックス更新に失敗しました。ログを確認してください。",
            history_scope=history_scope,
            history_query=history_query,
        )
    finally:
        is_indexing = False
        indexing_task = None
        indexing_process = None
        indexing_stop_requested = False
        indexing_started_at = None


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
        "faithfulness",
        "context_precision",
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


async def _run_eval(
    channel: discord.abc.Messageable | None,
    *,
    history_scope: str | int | None = None,
    history_query: str | None = None,
) -> None:
    global is_evaluating, evaluating_task
    is_evaluating = True
    try:
        await _send_status(
            channel,
            "評価を開始します。",
            history_scope=history_scope,
            history_query=history_query,
        )
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
            await _send_status(
                channel,
                "評価に失敗しました。ログを確認してください。",
                history_scope=history_scope,
                history_query=history_query,
            )
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
                history_scope=history_scope,
                history_query=history_query,
            )
        else:
            await _send_status(
                channel,
                "評価が完了しました。",
                history_scope=history_scope,
                history_query=history_query,
            )
    except Exception:
        logger.exception("Failed to run evaluate_ragas")
        await _send_status(
            channel,
            "評価に失敗しました。ログを確認してください。",
            history_scope=history_scope,
            history_query=history_query,
        )
    finally:
        is_evaluating = False
        evaluating_task = None


async def _run_answer(
    message: discord.Message,
    query: str,
    *,
    force_fast_mode: bool = False,
) -> None:
    channel = message.channel
    channel_id = channel.id
    question_author = _question_author_from_message(message)
    history_scope = _history_scope_for_message(message)
    cancel_event = threading.Event()
    channel_cancel_events[channel_id] = cancel_event
    voice_meeting_manager.notify_rag_started()
    try:
        routing_decision: FunctionRoutingDecision | None = None

        def _capture_routing_decision(decision: FunctionRoutingDecision) -> None:
            nonlocal routing_decision
            routing_decision = decision

        special_channel_mode = _is_special_channel_invocation(message)
        routing_history_override: list[ChatHistoryEntry] | None = None
        generation_history_override: list[ChatHistoryEntry] | None = None
        force_disable_additional_memory = False
        append_sources_to_response = True
        extra_mode_instruction: str | None = None
        if special_channel_mode:
            (
                special_history,
                primary_message_count,
                expanded_reference_count,
            ) = await _collect_special_channel_history(
                message,
                history_limit=SPECIAL_CHANNEL_HISTORY_LIMIT,
            )
            routing_history_override = special_history
            generation_history_override = special_history
            force_disable_additional_memory = True
            append_sources_to_response = False
            extra_mode_instruction = APP_CONFIG.special_channel_custom_instruction
            logger.info(
                "Special channel mode applied. channel_id=%s primary_messages=%s history_turns=%s expanded_references=%s custom_instruction=%s",
                channel_id,
                primary_message_count,
                len(special_history),
                expanded_reference_count,
                bool((extra_mode_instruction or "").strip()),
            )

        answer = await asyncio.to_thread(
            rag_pipeline.answer_with_routing,
            query,
            question_author=question_author,
            on_routing_decided=_capture_routing_decision,
            history_scope=history_scope,
            routing_history_override=routing_history_override,
            generation_history_override=generation_history_override,
            force_disable_additional_memory=force_disable_additional_memory,
            append_sources_to_response=append_sources_to_response,
            extra_mode_instruction=extra_mode_instruction,
            force_fast_mode=force_fast_mode,
            cancel_event=cancel_event,
        )
        if cancel_event.is_set():
            return
        if force_fast_mode:
            answer = (
                FAST_MODEL_NOTICE
                if not answer
                else f"{FAST_MODEL_NOTICE}\n\n{answer}"
            )
        _append_answer_record(
            message=message,
            query=query,
            routing_decision=routing_decision,
            answer=answer,
        )
        await channel.send(answer)
    except GenerationCancelled:
        return
    except Exception as e:
        logger.exception("Failed to handle /llm request")
        await _send_status(
            channel,
            f"エラーが発生しました: {type(e).__name__}: {e}",
            history_scope=history_scope,
            history_query=query,
        )
    finally:
        voice_meeting_manager.notify_rag_finished()
        channel_cancel_events.pop(channel_id, None)
        channel_generation_tasks.pop(channel_id, None)


def _has_active_answer_generation() -> bool:
    return any(not task.done() for task in channel_generation_tasks.values())


def _warmup_skip_reason() -> str | None:
    if _warmup_lock.locked():
        return "warmup already running"
    if is_indexing or (indexing_task is not None and not indexing_task.done()):
        return "indexing is running"
    if is_evaluating or (evaluating_task is not None and not evaluating_task.done()):
        return "evaluation is running"
    if _has_active_answer_generation():
        return "answer generation is running"
    if voice_meeting_manager.has_model_activity():
        return "voice model processing is running"
    return None


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
    global indexing_started_at
    logger.info(
        "Auto index scheduler started. enabled=%s time=%02d:%02d weekdays=%s timezone=%s",
        AUTO_INDEX_ENABLED,
        AUTO_INDEX_HOUR,
        AUTO_INDEX_MINUTE,
        ",".join(str(day) for day in AUTO_INDEX_WEEKDAYS) or "all",
        AUTO_INDEX_TIMEZONE.key,
    )
    while True:
        await asyncio.sleep(20)
        now = datetime.now(AUTO_INDEX_TIMEZONE)
        if not _should_run_auto_index(now):
            continue
        if indexing_task and not indexing_task.done():
            logger.info("Auto index skipped: indexing already running.")
            auto_index_last_run = now.date()
            continue
        if voice_meeting_manager.has_active_session():
            logger.info("Auto index skipped: VC participation is active.")
            auto_index_last_run = now.date()
            continue
        is_indexing = True
        indexing_stop_requested = False
        indexing_started_at = now
        auto_index_last_run = now.date()
        indexing_task = asyncio.create_task(_run_build_index(None))


async def _periodic_warmup_loop() -> None:
    if WARMUP_INTERVAL_MINUTES <= 0:
        logger.info("Periodic warmup disabled. interval_minutes=%s", WARMUP_INTERVAL_MINUTES)
        return
    interval_seconds = WARMUP_INTERVAL_MINUTES * 60
    logger.info(
        "Periodic warmup scheduler started. interval_minutes=%s",
        WARMUP_INTERVAL_MINUTES,
    )
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            reason = _warmup_skip_reason()
            if reason:
                logger.info("Periodic warmup skipped: %s", reason)
                continue
            await asyncio.to_thread(_warmup_models, trigger="periodic")
    except asyncio.CancelledError:
        return


# Discord events
@discord_client.event
async def on_ready():
    logger.info("Logged in as %s", discord_client.user)
    await voice_meeting_manager.start()
    global auto_index_task, periodic_warmup_task
    if AUTO_INDEX_ENABLED and (auto_index_task is None or auto_index_task.done()):
        auto_index_task = asyncio.create_task(_auto_index_loop())
    if WARMUP_INTERVAL_MINUTES > 0 and (
        periodic_warmup_task is None or periodic_warmup_task.done()
    ):
        periodic_warmup_task = asyncio.create_task(_periodic_warmup_loop())


@discord_client.event
async def on_voice_state_update(
    member: discord.Member,
    before: discord.VoiceState,
    after: discord.VoiceState,
):
    await voice_meeting_manager.on_voice_state_update(member, before, after)


@discord_client.event
async def on_message(message: discord.Message):
    global evaluating_task, indexing_process, indexing_stop_requested, indexing_task
    global indexing_started_at
    global is_evaluating, is_indexing
    if message.author.bot:
        return

    content = (message.content or "").strip()
    if voice_meeting_manager.is_voice_chat_channel(message.channel):
        await voice_meeting_manager.capture_voice_chat_message(message)
    lower_content = content.lower()

    if lower_content == JOIN_COMMAND:
        handled = await voice_meeting_manager.maybe_join_from_command(message)
        if handled:
            return
        await _send_message_with_history(
            message,
            "`/ai join` はVCのチャット欄でのみ有効です。",
            query=content,
        )
        return
    if lower_content == QUIT_COMMAND:
        handled = await voice_meeting_manager.maybe_quit_from_command(message)
        if handled:
            return
        await _send_message_with_history(
            message,
            "`/ai quit` はVCのチャット欄でのみ有効です。",
            query=content,
        )
        return

    if content == BUILD_INDEX_COMMAND:
        if not _is_maintenance_command_authorized(message):
            await _send_message_with_history(
                message,
                "このコマンドを実行する権限がありません。",
                query=content,
            )
            return
        if voice_meeting_manager.has_active_session():
            await _send_message_with_history(
                message,
                "VC参加中のため、新規のインデックス更新は開始できません。",
                query=content,
            )
            return
        if indexing_task and not indexing_task.done():
            await _send_message_with_history(
                message,
                "インデックス更新は既に実行中です。",
                query=content,
            )
            return
        indexing_stop_requested = False
        is_indexing = True
        indexing_started_at = datetime.now(AUTO_INDEX_TIMEZONE)
        indexing_task = asyncio.create_task(
            _run_build_index(
                message.channel,
                history_scope=_history_scope_for_message(message),
                history_query=content,
            )
        )
        return
    if content == EVAL_COMMAND:
        if not _is_maintenance_command_authorized(message):
            await _send_message_with_history(
                message,
                "このコマンドを実行する権限がありません。",
                query=content,
            )
            return
        if evaluating_task and not evaluating_task.done():
            await _send_message_with_history(
                message,
                "評価は既に実行中です。",
                query=content,
            )
            return
        is_evaluating = True
        evaluating_task = asyncio.create_task(
            _run_eval(
                message.channel,
                history_scope=_history_scope_for_message(message),
                history_query=content,
            )
        )
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
        await _send_message_with_history(
            message,
            "\n".join(actions),
            query=content,
        )
        return

    if is_indexing:
        query = _extract_query_from_message(message)
        if query:
            await _send_message_with_history(
                message,
                _build_indexing_blocked_message(),
                query=query,
            )
        return

    query = _extract_query_from_message(message)
    if not query:
        return
    has_fast_prefix = _has_explicit_fast_prefix(message)
    if has_fast_prefix:
        query, _ = _strip_fast_prefix(query)
    if not query:
        return
    vc_summary_fast_mode = voice_meeting_manager.should_use_fast_model_for_query()
    force_fast_mode = has_fast_prefix or vc_summary_fast_mode
    if MAX_INPUT_CHARACTERS > 0 and len(query) > MAX_INPUT_CHARACTERS:
        await _send_message_with_history(
            message,
            f"入力できる最大文字数を超えています。（{MAX_INPUT_CHARACTERS}）以下で入力してください。",
            query=query,
        )
        return

    channel_id = message.channel.id
    existing = channel_generation_tasks.get(channel_id)
    if existing and not existing.done():
        await _send_message_with_history(
            message,
            "回答生成は既に実行中です。中止する場合は /ai stop を実行してください。",
            query=query,
        )
        return
    task = asyncio.create_task(
        _run_answer(
            message,
            query,
            force_fast_mode=force_fast_mode,
        )
    )
    channel_generation_tasks[channel_id] = task


def main() -> None:
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set. Please set it in .env")

    _warmup_models(trigger="startup")
    discord_client.run(DISCORD_BOT_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
