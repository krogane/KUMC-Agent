from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency in test runtime
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

from kumc_agent.config.env_map import ENV_BINDINGS
from kumc_agent.config.merge import MergeError, deep_merge
from kumc_agent.config.schema import (
    AppSection,
    AutonomousAgentBudgetSection,
    AutonomousAgentLookaheadSection,
    AutonomousAgentSection,
    DatabaseSection,
    EmbeddingSection,
    EventManagementSection,
    FeatureSection,
    FunctionCallSection,
    ImageSearchFeatureSection,
    InfrastructureSection,
    MemberSearchFeatureSection,
    RagGenerationProfileSection,
    RagGenerationSection,
    IndexingChunkingSection,
    IndexingRefreshSection,
    IndexingSection,
    IndexingStagesSection,
    IntegrationCraftersColonySection,
    IntegrationDiscordSection,
    IntegrationDriveSection,
    IntegrationHatenablogSection,
    IntegrationMinecraftWikiSection,
    IntegrationNotionSection,
    IntegrationOpenClawSection,
    IntegrationSection,
    LLMSection,
    MinecraftWikiRagChunkingSection,
    MinecraftWikiRagRetrievalSection,
    MinecraftWikiRagSection,
    MigrationSection,
    ModelSection,
    ObjectStorageSection,
    OpsRagasMetricsSection,
    OpsSection,
    ProviderSection,
    RagHistorySection,
    RagPromptTextSection,
    RagRoutingSection,
    RagRoutingTaskSection,
    RagRoutingTasksSection,
    RagSection,
    RerankerSection,
    RetrievalSection,
    RuntimeConfig,
    SchedulerSection,
    SecuritySection,
    ServerManagementBackupSection,
    ServerManagementDockerPsSection,
    ServerManagementExecutionSection,
    ServerManagementSection,
    ServerManagementServerSection,
    SourcesSection,
    RedisSection,
    RiskFeatureFlagsSection,
    SummarizationSection,
    TaskManagementSection,
    VCSection,
)

MAIN_FILES = (
    "app.yaml",
    "infrastructure.yaml",
    "providers.yaml",
    "security.yaml",
    "scheduler.yaml",
    "autonomous_agent.yaml",
    "features.yaml",
    "model.yaml",
    "rag.yaml",
    "indexing.yaml",
    "evaluation.yaml",
    "integrations.yaml",
    "summarization.yaml",
    "server_management.yaml",
    "event_management.yaml",
    "task_management.yaml",
    "vc.yaml",
)


class ConfigLoadError(RuntimeError):
    pass


def _yaml_load(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        yaml = None

    if not path.exists():
        raise ConfigLoadError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(text)
    else:
        payload = _fallback_yaml_load(text)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigLoadError(f"Config root must be object: {path}")
    return payload


def _fallback_yaml_load(text: str) -> dict[str, Any]:
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if raw.lstrip().startswith("#"):
            continue
        lines.append(raw.rstrip())
    if not lines:
        return {}
    node, index = _parse_yaml_node(lines, 0, _indent_of(lines[0]))
    if index < len(lines):
        raise ConfigLoadError("Failed to parse YAML payload completely.")
    if not isinstance(node, dict):
        raise ConfigLoadError("Fallback YAML parser expects mapping root.")
    return node


def _parse_yaml_node(
    lines: list[str],
    index: int,
    indent: int,
) -> tuple[object, int]:
    if index >= len(lines):
        return {}, index
    current = lines[index]
    if _indent_of(current) < indent:
        return {}, index
    stripped = current[_indent_of(current) :]
    if stripped.startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_dict(lines, index, indent)


def _parse_yaml_dict(
    lines: list[str],
    index: int,
    indent: int,
) -> tuple[dict[str, object], int]:
    result: dict[str, object] = {}
    i = index
    while i < len(lines):
        line = lines[i]
        current_indent = _indent_of(line)
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ConfigLoadError(f"Invalid YAML indentation near: {line}")
        stripped = line[current_indent:]
        if stripped.startswith("- "):
            break
        if ":" not in stripped:
            raise ConfigLoadError(f"Invalid YAML mapping row: {line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ConfigLoadError(f"Invalid YAML key: {line}")
        if value:
            result[key] = _parse_yaml_scalar(value)
            i += 1
            continue

        i += 1
        if i >= len(lines) or _indent_of(lines[i]) <= current_indent:
            result[key] = {}
            continue
        child_indent = _indent_of(lines[i])
        child, i = _parse_yaml_node(lines, i, child_indent)
        result[key] = child
    return result, i


def _parse_yaml_list(
    lines: list[str],
    index: int,
    indent: int,
) -> tuple[list[object], int]:
    result: list[object] = []
    i = index
    while i < len(lines):
        line = lines[i]
        current_indent = _indent_of(line)
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ConfigLoadError(f"Invalid YAML list indentation near: {line}")
        stripped = line[current_indent:]
        if not stripped.startswith("- "):
            break
        value = stripped[2:].strip()
        if value:
            result.append(_parse_yaml_scalar(value))
            i += 1
            continue
        i += 1
        if i >= len(lines) or _indent_of(lines[i]) <= current_indent:
            result.append({})
            continue
        child_indent = _indent_of(lines[i])
        child, i = _parse_yaml_node(lines, i, child_indent)
        result.append(child)
    return result, i


def _parse_yaml_scalar(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?\d+\.\d+", value):
        return float(value)
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(part.strip()) for part in inner.split(",")]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        payload: dict[str, object] = {}
        for part in inner.split(","):
            token = part.strip()
            if not token:
                continue
            if ":" not in token:
                raise ConfigLoadError(f"Invalid inline mapping token: {token}")
            key, raw = token.split(":", 1)
            payload[key.strip().strip("'\"")] = _parse_yaml_scalar(raw.strip())
        return payload
    return value


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _set_deep(target: dict[str, Any], path: str, value: object) -> None:
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        if part not in current:
            raise ConfigLoadError(f"Unknown config key from env: {path}")
        next_value = current[part]
        if not isinstance(next_value, dict):
            raise ConfigLoadError(f"Invalid env target path: {path}")
        current = next_value
    leaf = parts[-1]
    if leaf not in current:
        raise ConfigLoadError(f"Unknown config key from env: {path}")
    current[leaf] = value


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    for binding in ENV_BINDINGS:
        raw = os.getenv(binding.env_name)
        if raw is None:
            continue
        parsed = binding.parser(raw)
        _set_deep(updated, binding.path, parsed)
    return updated


def _resolve_path(base_dir: Path, value: str) -> Path:
    expanded = os.path.expandvars(value)
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_optional_path_str(base_dir: Path, value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    return str(_resolve_path(base_dir, raw))


def _backfill_default_config_values(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    infrastructure = updated.get("infrastructure")
    if not isinstance(infrastructure, dict):
        infrastructure = {}
        updated["infrastructure"] = infrastructure
    database = infrastructure.get("database")
    if not isinstance(database, dict):
        database = {}
        infrastructure["database"] = database
    database.setdefault("url", "")
    database.setdefault("connect_timeout_seconds", 3.0)
    database.setdefault("application_name", "kumc-agent")
    redis = infrastructure.get("redis")
    if not isinstance(redis, dict):
        redis = {}
        infrastructure["redis"] = redis
    redis.setdefault("url", "")
    redis.setdefault("socket_timeout_seconds", 3.0)
    object_storage = infrastructure.get("object_storage")
    if not isinstance(object_storage, dict):
        object_storage = {}
        infrastructure["object_storage"] = object_storage
    object_storage.setdefault("endpoint_url", "")
    object_storage.setdefault("bucket", "")
    object_storage.setdefault("region", "ap-northeast-1")
    object_storage.setdefault("access_key_id", "")
    object_storage.setdefault("secret_access_key", "")
    object_storage.setdefault("prefix", "kumc-agent")
    object_storage.setdefault("use_ssl", True)
    migrations = infrastructure.get("migrations")
    if not isinstance(migrations, dict):
        migrations = {}
        infrastructure["migrations"] = migrations
    migrations.setdefault("directory", "infrastructure/migrations")
    migrations.setdefault("table_name", "schema_migrations")

    scheduler = updated.get("scheduler")
    if not isinstance(scheduler, dict):
        scheduler = {}
        updated["scheduler"] = scheduler
    scheduler.setdefault("auto_index_max_runtime_minutes", 120)
    scheduler.setdefault("auto_index_lock_ttl_minutes", 180)
    scheduler.setdefault("auto_index_timezone", "Asia/Tokyo")
    scheduler.setdefault("quality_min_chunk_ratio", 0.5)
    scheduler.setdefault("quality_smoke_queries", [])
    scheduler.setdefault("rollback_keep_snapshots", 3)

    security = updated.get("security")
    if not isinstance(security, dict):
        security = {}
        updated["security"] = security
    security.setdefault("maintenance_command_author_ids", [])
    security.setdefault("discord_guild_allow_list", [])
    security.setdefault("discord_member_profile_guild_ids", [])

    autonomous_agent = updated.get("autonomous_agent")
    if not isinstance(autonomous_agent, dict):
        autonomous_agent = {}
        updated["autonomous_agent"] = autonomous_agent
    autonomous_agent.setdefault("enabled", False)
    autonomous_agent.setdefault("schedule_times", ["08:00", "13:00", "20:00"])
    autonomous_agent.setdefault("timezone", "Asia/Tokyo")
    autonomous_agent.setdefault("scopes", ["tasks", "events", "rag_delta", "server_ops", "automation"])
    autonomous_agent.setdefault("notification_channel_id", "")
    autonomous_agent.setdefault("dry_run", True)
    autonomous_agent.setdefault("duplicate_suppression_hours", 24)
    lookahead = autonomous_agent.get("lookahead_days")
    if not isinstance(lookahead, dict):
        lookahead = {}
        autonomous_agent["lookahead_days"] = lookahead
    lookahead.setdefault("tasks", 2)
    lookahead.setdefault("events", 7)
    budget = autonomous_agent.get("budget")
    if not isinstance(budget, dict):
        budget = {}
        autonomous_agent["budget"] = budget
    budget.setdefault("max_steps", 10)
    budget.setdefault("max_search_calls", 6)
    budget.setdefault("max_replans", 1)
    budget.setdefault("max_cost_usd", 0.50)
    budget.setdefault("max_latency_seconds", 120.0)

    task_management = updated.get("task_management")
    if not isinstance(task_management, dict):
        task_management = {}
        updated["task_management"] = task_management
    task_management.setdefault("approval_batch_interval_days", 7)
    task_management.setdefault("due_soon_notice_days", 1)
    task_management.setdefault("notification_channel_id", "")
    task_management.setdefault("admin_user_ids", [])
    task_management.setdefault("admin_role_ids", [])
    task_management.setdefault("prompt_name", "task_extraction.md")
    task_management.setdefault("auto_extract_after_index_update", True)

    event_management = updated.get("event_management")
    if not isinstance(event_management, dict):
        event_management = {}
        updated["event_management"] = event_management
    event_management.setdefault("approval_batch_interval_days", 7)
    event_management.setdefault("notification_before_days", 1)
    event_management.setdefault("notification_channel_id", "")
    event_management.setdefault("admin_user_ids", [])
    event_management.setdefault("admin_role_ids", [])
    event_management.setdefault("prompt_name", "event_extraction.md")
    event_management.setdefault("auto_extract_after_index_update", True)
    event_management.setdefault("timezone", "Asia/Tokyo")

    features = updated.get("features")
    if not isinstance(features, dict):
        features = {}
        updated["features"] = features
    retrieval = features.get("retrieval")
    if not isinstance(retrieval, dict):
        retrieval = {}
        features["retrieval"] = retrieval
    retrieval.setdefault("rrf_k", 60)
    risk_flags = features.get("risk_flags")
    if not isinstance(risk_flags, dict):
        risk_flags = {}
        features["risk_flags"] = risk_flags
    risk_flags.setdefault("action_execution", "approval_required")
    risk_flags.setdefault("external_posting", "approval_required")
    risk_flags.setdefault("minecraft_server_ops", "approval_required")
    risk_flags.setdefault("accounting_finalize", "approval_required")
    risk_flags.setdefault("auto_reply", "approval_required")
    risk_flags.setdefault("automation_auto_run", "disabled")
    risk_flags.setdefault("vc_recording", "disabled")
    risk_flags.setdefault("image_generation", "approval_required")
    image_search = features.get("image_search")
    if not isinstance(image_search, dict):
        image_search = {}
        features["image_search"] = image_search
    image_search.setdefault("enabled", True)
    image_search.setdefault("limit", retrieval.get("top_k", 8))
    image_search.setdefault("dense_top_k", retrieval.get("dense_top_k", 24))
    image_search.setdefault("feature_top_k", retrieval.get("sparse_top_k", 16))
    image_search.setdefault("rrf_k", retrieval.get("rrf_k", 60))
    image_search.setdefault("ocr_text_char_limit", 800)
    image_search.setdefault("surrounding_text_char_limit", 1200)
    image_search.setdefault("caption_model", "")
    image_search.setdefault("ocr_model", "")
    image_search.setdefault("feature_model", "openai/clip-vit-base-patch32")
    image_search.setdefault("feature_dimensions", 512)
    image_search.setdefault("duplicate_group_limit", 1)
    member_search = features.get("member_search")
    if not isinstance(member_search, dict):
        member_search = {}
        features["member_search"] = member_search
    member_search.setdefault("exclude_role_names", [])

    rag = updated.get("rag")
    if not isinstance(rag, dict):
        rag = {}
        updated["rag"] = rag
    generation = rag.get("generation")
    if not isinstance(generation, dict):
        generation = {}
        rag["generation"] = generation
    idea_generation = generation.get("idea_generation")
    if not isinstance(idea_generation, dict):
        idea_generation = {}
        generation["idea_generation"] = idea_generation
    idea_generation.setdefault("prompt_name", "answer_idea")
    idea_generation.setdefault("temperature", 0.0)

    summarization = updated.get("summarization")
    if not isinstance(summarization, dict):
        summarization = {}
        updated["summarization"] = summarization
    summarization.setdefault("target_characters", 200)

    integrations = updated.get("integrations")
    if not isinstance(integrations, dict):
        integrations = {}
        updated["integrations"] = integrations
    integrations.setdefault("openai_api_key", "")
    openclaw = integrations.get("openclaw")
    if not isinstance(openclaw, dict):
        openclaw = {}
        integrations["openclaw"] = openclaw
    openclaw.setdefault("enabled", True)
    openclaw.setdefault("agent", "main")
    openclaw.setdefault("model", "")
    openclaw.setdefault("lite_agent", "")
    openclaw.setdefault("lite_model", "")
    openclaw.setdefault("config_dir", "configs/openclaw")
    minecraft_wiki = integrations.get("minecraft_wiki")
    if not isinstance(minecraft_wiki, dict):
        minecraft_wiki = {}
        integrations["minecraft_wiki"] = minecraft_wiki
    minecraft_wiki.setdefault("page_titles", [])
    minecraft_wiki.setdefault("api_url", "https://ja.minecraft.wiki/api.php")
    minecraft_wiki.setdefault("page_url_base", "https://ja.minecraft.wiki/w/")
    minecraft_wiki.setdefault("max_pages", 20)
    minecraft_wiki.setdefault("rate_limit_per_minute", 30)
    minecraft_wiki.setdefault("request_interval_seconds", 1.0)
    minecraft_wiki.setdefault("namespaces", [0])
    minecraft_wiki.setdefault("full_backfill_enabled", False)

    minecraft_wiki_rag = updated.get("minecraft_wiki_rag")
    if not isinstance(minecraft_wiki_rag, dict):
        minecraft_wiki_rag = {}
        updated["minecraft_wiki_rag"] = minecraft_wiki_rag
    minecraft_wiki_chunking = minecraft_wiki_rag.get("chunking")
    if not isinstance(minecraft_wiki_chunking, dict):
        minecraft_wiki_chunking = {}
        minecraft_wiki_rag["chunking"] = minecraft_wiki_chunking
    minecraft_wiki_retrieval = minecraft_wiki_rag.get("retrieval")
    if not isinstance(minecraft_wiki_retrieval, dict):
        minecraft_wiki_retrieval = {}
        minecraft_wiki_rag["retrieval"] = minecraft_wiki_retrieval
    indexing = updated.get("indexing")
    indexing_chunking = {}
    if isinstance(indexing, dict) and isinstance(indexing.get("chunking"), dict):
        indexing_chunking = indexing["chunking"]
    for key, fallback in (
        ("first_recursive_chunk_size", indexing_chunking.get("first_recursive_chunk_size", 1024)),
        ("first_recursive_chunk_overlap", indexing_chunking.get("first_recursive_chunk_overlap", 128)),
        ("second_recursive_chunk_size", indexing_chunking.get("second_recursive_chunk_size", 384)),
        ("second_recursive_chunk_overlap", indexing_chunking.get("second_recursive_chunk_overlap", 64)),
        ("summary_characters", indexing_chunking.get("summary_characters", 200)),
        ("summary_batch_size", indexing_chunking.get("summary_batch_size", 1)),
        ("summary_llm_provider", indexing_chunking.get("summary_llm_provider", "none")),
        ("summary_gemini_model", indexing_chunking.get("summary_gemini_model", "")),
        ("summary_temperature", indexing_chunking.get("summary_temperature", 0.0)),
        ("summary_max_output_tokens", indexing_chunking.get("summary_max_output_tokens", 1024)),
        ("summary_thinking_level", indexing_chunking.get("summary_thinking_level", "minimal")),
    ):
        minecraft_wiki_chunking.setdefault(key, fallback)
    for key, fallback in (
        ("top_k", retrieval.get("top_k", 8)),
        ("dense_top_k", retrieval.get("dense_top_k", 15)),
        ("sparse_top_k", retrieval.get("sparse_top_k", 15)),
        (
            "sparse_initial_sparse_top_k",
            retrieval.get("sparse_initial_sparse_top_k", retrieval.get("sparse_top_k", 15)),
        ),
        ("sparse_normalized_ratio", retrieval.get("sparse_normalized_ratio")),
        ("rerank_pool_size", retrieval.get("rerank_pool_size", 20)),
        ("rrf_k", retrieval.get("rrf_k", 60)),
        ("mmr_lambda", retrieval.get("mmr_lambda", 0.75)),
        ("parent_doc_enabled", retrieval.get("parent_doc_enabled", True)),
        ("parent_chunk_cap", retrieval.get("parent_chunk_cap", 2)),
        ("sudachi_mode", retrieval.get("sudachi_mode", "B")),
        ("sparse_bm25_k1", retrieval.get("sparse_bm25_k1", 1.5)),
        ("sparse_bm25_b", retrieval.get("sparse_bm25_b", 0.75)),
        ("sparse_use_normalized_form", retrieval.get("sparse_use_normalized_form", True)),
        ("sparse_remove_symbols", retrieval.get("sparse_remove_symbols", True)),
    ):
        minecraft_wiki_retrieval.setdefault(key, fallback)
    server_management = updated.get("server_management")
    if not isinstance(server_management, dict):
        server_management = {}
        updated["server_management"] = server_management
    server_management.setdefault("default_server_name", "default")
    docker_ps = server_management.get("docker_ps")
    if not isinstance(docker_ps, dict):
        docker_ps = {}
        server_management["docker_ps"] = docker_ps
    docker_ps.setdefault("container_name_prefixes", [])
    server_management.setdefault("servers", [])
    execution = server_management.get("execution")
    if not isinstance(execution, dict):
        execution = {}
        server_management["execution"] = execution
    execution.setdefault("timeout_seconds", 120)
    execution.setdefault("stdout_char_limit", 4000)
    execution.setdefault("stderr_char_limit", 4000)
    backup = server_management.get("backup")
    if not isinstance(backup, dict):
        backup = {}
        server_management["backup"] = backup
    backup.setdefault("backup_dir", "data/minecraft/backups")
    backup.setdefault("max_backups", 10)
    return updated


def load_runtime_config(*, base_dir: Path | None = None) -> RuntimeConfig:
    resolved_base_dir = base_dir or Path(__file__).resolve().parents[3]
    load_dotenv(resolved_base_dir / ".env", override=False)

    merged: dict[str, Any] = {}
    for file_name in MAIN_FILES:
        path = resolved_base_dir / "configs" / "main" / file_name
        if (
            file_name
            in {
                "server_management.yaml",
                "autonomous_agent.yaml",
                "event_management.yaml",
                "task_management.yaml",
            }
            and not path.exists()
        ):
            payload = {}
        else:
            payload = _yaml_load(path)
        try:
            merged = deep_merge(merged, payload, allow_new_keys=True)
        except MergeError as exc:
            raise ConfigLoadError(str(exc)) from exc
    merged = _backfill_default_config_values(merged)

    try:
        merged = _apply_env_overrides(merged)
    except (ValueError, ConfigLoadError) as exc:
        raise ConfigLoadError(str(exc)) from exc

    return _to_runtime_config(
        merged=merged,
        base_dir=resolved_base_dir,
    )


def _to_runtime_config(
    *,
    merged: dict[str, Any],
    base_dir: Path,
) -> RuntimeConfig:
    app = merged["app"]
    providers = merged["providers"]
    security = merged["security"]
    scheduler = merged["scheduler"]
    autonomous_agent = merged.get("autonomous_agent", {})
    task_management = merged.get("task_management", {})
    event_management = merged.get("event_management", {})
    infrastructure = merged.get("infrastructure", {})
    features = merged["features"]
    image_search_raw = features.get("image_search", {})
    if not isinstance(image_search_raw, dict):
        image_search_raw = {}
    minecraft_wiki_rag = merged.get("minecraft_wiki_rag", {})
    rag = merged.get("rag", {})
    indexing = merged.get("indexing", {})
    ops = merged.get("ops", {})
    summarization = merged.get("summarization", {})
    integrations = merged.get("integrations", {})
    model = merged.get("model", {})
    vc = merged.get("vc", {})
    server_management = merged.get("server_management", {})

    rag_routing = rag.get("routing", {})
    rag_history = rag.get("history", {})
    rag_generation = rag.get("generation", {})
    rag_prompt_texts = rag.get("prompt_texts", {})
    rag_generation_rag = rag_generation.get("rag", {})
    rag_generation_no_rag = rag_generation.get("no_rag", {})
    rag_generation_idea = rag_generation.get("idea_generation", {})
    indexing_chunking = indexing.get("chunking", {})
    indexing_stages = indexing.get("stages", {})
    indexing_refresh = indexing.get("refresh", {})
    ops_ragas_metrics = ops.get("ragas_metrics", {})
    database = infrastructure.get("database", {})
    redis = infrastructure.get("redis", {})
    object_storage = infrastructure.get("object_storage", {})
    migrations = infrastructure.get("migrations", {})
    notion = integrations.get("notion", {})
    if not isinstance(notion, dict):
        notion = {}
    notion_database_ids_raw = notion.get("database_ids", [])
    if not isinstance(notion_database_ids_raw, list):
        notion_database_ids_raw = []
    minecraft_wiki = integrations.get("minecraft_wiki", {})
    if not isinstance(minecraft_wiki, dict):
        minecraft_wiki = {}
    minecraft_wiki_namespaces_raw = minecraft_wiki.get("namespaces", [0])
    if isinstance(minecraft_wiki_namespaces_raw, str):
        minecraft_wiki_namespaces = [
            int(part.strip())
            for part in minecraft_wiki_namespaces_raw.split(",")
            if part.strip()
        ]
    elif isinstance(minecraft_wiki_namespaces_raw, list):
        minecraft_wiki_namespaces = [
            int(part)
            for part in minecraft_wiki_namespaces_raw
            if str(part).strip()
        ]
    else:
        minecraft_wiki_namespaces = [0]
    if not minecraft_wiki_namespaces:
        minecraft_wiki_namespaces = [0]
    minecraft_page_titles_raw = minecraft_wiki.get("page_titles", [])
    if isinstance(minecraft_page_titles_raw, str):
        minecraft_page_titles = [
            part.strip()
            for part in minecraft_page_titles_raw.split(",")
            if part.strip()
        ]
    elif isinstance(minecraft_page_titles_raw, list):
        minecraft_page_titles = [
            str(part).strip()
            for part in minecraft_page_titles_raw
            if str(part).strip()
        ]
    else:
        minecraft_page_titles = []
    if not isinstance(minecraft_wiki_rag, dict):
        minecraft_wiki_rag = {}
    minecraft_wiki_rag_chunking = minecraft_wiki_rag.get("chunking", {})
    if not isinstance(minecraft_wiki_rag_chunking, dict):
        minecraft_wiki_rag_chunking = {}
    minecraft_wiki_rag_retrieval = minecraft_wiki_rag.get("retrieval", {})
    if not isinstance(minecraft_wiki_rag_retrieval, dict):
        minecraft_wiki_rag_retrieval = {}

    special_channel_names_raw = rag_history.get("special_channel_names", ["kumc-agent"])
    if isinstance(special_channel_names_raw, str):
        special_channel_names = [
            part.strip()
            for part in special_channel_names_raw.split(",")
            if part.strip()
        ]
    elif isinstance(special_channel_names_raw, list):
        special_channel_names = [str(part).strip() for part in special_channel_names_raw if str(part).strip()]
    else:
        special_channel_names = ["kumc-agent"]
    if not special_channel_names:
        special_channel_names = ["kumc-agent"]

    def _build_generation_profile(
        section: dict[str, Any],
        *,
        default_prompt_name: str,
        fallback: RagGenerationProfileSection | None = None,
    ) -> RagGenerationProfileSection:
        provider_default = (
            fallback.provider
            if fallback is not None
            else str(providers["llm"]["provider"])
        )
        gemini_model_default = (
            fallback.gemini_model
            if fallback is not None
            else str(providers["llm"]["gemini_model"])
        )
        temperature_default = (
            fallback.temperature
            if fallback is not None
            else float(providers["llm"]["temperature"])
        )
        max_output_tokens_default = (
            fallback.max_output_tokens
            if fallback is not None
            else int(providers["llm"]["max_output_tokens"])
        )
        thinking_level_default = (
            fallback.thinking_level
            if fallback is not None
            else str(providers["llm"]["thinking_level"])
        )
        prompt_name_default = (
            fallback.prompt_name if fallback is not None else default_prompt_name
        )
        return RagGenerationProfileSection(
            provider=str(section.get("provider", provider_default)),
            gemini_model=str(section.get("gemini_model", gemini_model_default)),
            temperature=float(section.get("temperature", temperature_default)),
            max_output_tokens=int(
                section.get("max_output_tokens", max_output_tokens_default)
            ),
            thinking_level=str(section.get("thinking_level", thinking_level_default)),
            prompt_name=str(section.get("prompt_name", prompt_name_default)),
        )

    rag_generation_profile = _build_generation_profile(
        rag_generation_rag,
        default_prompt_name="answer_rag",
    )
    no_rag_generation_profile = _build_generation_profile(
        rag_generation_no_rag,
        default_prompt_name="answer_no_rag",
        fallback=rag_generation_profile,
    )
    idea_generation_profile = _build_generation_profile(
        rag_generation_idea,
        default_prompt_name="answer_idea",
        fallback=no_rag_generation_profile,
    )
    routing_provider = str(
        rag_routing.get(
            "provider",
            providers["function_call"].get("provider", "gemini"),
        )
    )
    routing_gemini_model = str(
        rag_routing.get(
            "gemini_model",
            providers["function_call"].get("gemini_model", ""),
        )
    )
    routing_prompt_name = str(rag_routing.get("prompt_name", "routing")).strip() or "routing"
    routing_tasks_raw = rag_routing.get("tasks", {})
    if not isinstance(routing_tasks_raw, dict):
        routing_tasks_raw = {}

    def _build_routing_task(task_name: str) -> RagRoutingTaskSection:
        task_raw = routing_tasks_raw.get(task_name, {})
        if not isinstance(task_raw, dict):
            task_raw = {}
        provider_value = str(task_raw.get("provider", routing_provider)).strip()
        gemini_model_value = str(
            task_raw.get("gemini_model", routing_gemini_model)
        ).strip()
        prompt_name_value = str(
            task_raw.get("prompt_name", routing_prompt_name)
        ).strip()
        return RagRoutingTaskSection(
            provider=provider_value or routing_provider,
            gemini_model=gemini_model_value or routing_gemini_model,
            prompt_name=prompt_name_value or routing_prompt_name,
        )

    routing_tasks = RagRoutingTasksSection(
        target_model=_build_routing_task("target_model"),
        use_additional_memory=_build_routing_task("use_additional_memory"),
        include_capabilities_info=_build_routing_task("include_capabilities_info"),
        needs_additional_query=_build_routing_task("needs_additional_query"),
        additional_queries=_build_routing_task("additional_queries"),
        material_names=_build_routing_task("material_names"),
        recency_mode=_build_routing_task("recency_mode"),
    )

    runtime = RuntimeConfig(
        base_dir=base_dir,
        app=AppSection(
            command_prefix=str(app["command_prefix"]),
            index_command_prefix=str(app["index_command_prefix"]),
            max_input_characters=int(app["max_input_characters"]),
            log_level=str(app["log_level"]),
            data_dir=_resolve_path(base_dir, str(app["data_dir"])),
            raw_dir=_resolve_path(base_dir, str(app["raw_dir"])),
            chunks_path=_resolve_path(base_dir, str(app["chunks_path"])),
            index_dir=_resolve_path(base_dir, str(app["index_dir"])),
            eval_dir=_resolve_path(base_dir, str(app["eval_dir"])),
            cache_dir=_resolve_path(base_dir, str(app["cache_dir"])),
            answer_record_log_path=_resolve_path(
                base_dir,
                str(app["answer_record_log_path"]),
            ),
            source_max_count=int(app["source_max_count"]),
        ),
        providers=ProviderSection(
            llm=LLMSection(
                provider=str(providers["llm"]["provider"]),
                gemini_model=str(providers["llm"]["gemini_model"]),
                temperature=float(providers["llm"]["temperature"]),
                max_output_tokens=int(providers["llm"]["max_output_tokens"]),
                thinking_level=str(providers["llm"]["thinking_level"]),
            ),
            embeddings=EmbeddingSection(
                provider=str(providers["embeddings"]["provider"]),
                model=str(providers["embeddings"]["model"]),
                dimensions=int(providers["embeddings"]["dimensions"]),
            ),
            reranker=RerankerSection(
                model=str(providers["reranker"]["model"]),
                enabled=bool(providers["reranker"]["enabled"]),
            ),
            function_call=FunctionCallSection(
                enabled=bool(providers["function_call"]["enabled"]),
                provider=str(providers["function_call"]["provider"]),
                gemini_model=str(providers["function_call"]["gemini_model"]),
            ),
        ),
        security=SecuritySection(
            maintenance_command_author_ids=[
                int(v) for v in security["maintenance_command_author_ids"]
            ],
            discord_guild_allow_list=[int(v) for v in security["discord_guild_allow_list"]],
            discord_member_profile_guild_ids=[
                int(v) for v in security.get("discord_member_profile_guild_ids", [])
            ],
        ),
        scheduler=SchedulerSection(
            auto_index_enabled=bool(scheduler["auto_index_enabled"]),
            auto_index_time=str(scheduler["auto_index_time"]),
            auto_index_weekdays=[int(v) for v in scheduler["auto_index_weekdays"]],
            auto_index_timezone=str(scheduler.get("auto_index_timezone", "Asia/Tokyo")),
            auto_index_max_runtime_minutes=int(
                scheduler.get("auto_index_max_runtime_minutes", 120)
            ),
            auto_index_lock_ttl_minutes=int(
                scheduler.get("auto_index_lock_ttl_minutes", 180)
            ),
            quality_min_chunk_ratio=float(
                scheduler.get("quality_min_chunk_ratio", 0.5)
            ),
            quality_smoke_queries=[
                str(v) for v in scheduler.get("quality_smoke_queries", [])
            ],
            rollback_keep_snapshots=int(scheduler.get("rollback_keep_snapshots", 3)),
        ),
        autonomous_agent=AutonomousAgentSection(
            enabled=bool(autonomous_agent.get("enabled", False)),
            schedule_times=[str(value) for value in autonomous_agent.get("schedule_times", [])],
            timezone=str(autonomous_agent.get("timezone", "Asia/Tokyo")),
            scopes=[str(value) for value in autonomous_agent.get("scopes", [])],
            notification_channel_id=str(autonomous_agent.get("notification_channel_id", "")),
            dry_run=bool(autonomous_agent.get("dry_run", True)),
            lookahead_days=AutonomousAgentLookaheadSection(
                tasks=int((autonomous_agent.get("lookahead_days") or {}).get("tasks", 2)),
                events=int((autonomous_agent.get("lookahead_days") or {}).get("events", 7)),
            ),
            duplicate_suppression_hours=int(
                autonomous_agent.get("duplicate_suppression_hours", 24)
            ),
            budget=AutonomousAgentBudgetSection(
                max_steps=int((autonomous_agent.get("budget") or {}).get("max_steps", 10)),
                max_search_calls=int(
                    (autonomous_agent.get("budget") or {}).get("max_search_calls", 6)
                ),
                max_replans=int((autonomous_agent.get("budget") or {}).get("max_replans", 1)),
                max_cost_usd=float(
                    (autonomous_agent.get("budget") or {}).get("max_cost_usd", 0.50)
                ),
                max_latency_seconds=float(
                    (autonomous_agent.get("budget") or {}).get("max_latency_seconds", 120.0)
                ),
            ),
        ),
        task_management=TaskManagementSection(
            approval_batch_interval_days=int(
                task_management.get("approval_batch_interval_days", 7)
            ),
            due_soon_notice_days=int(task_management.get("due_soon_notice_days", 1)),
            notification_channel_id=str(
                task_management.get("notification_channel_id", "")
            ),
            admin_user_ids=[
                str(value) for value in task_management.get("admin_user_ids", [])
            ],
            admin_role_ids=[
                str(value) for value in task_management.get("admin_role_ids", [])
            ],
            prompt_name=str(task_management.get("prompt_name", "task_extraction.md")),
            auto_extract_after_index_update=bool(
                task_management.get("auto_extract_after_index_update", True)
            ),
        ),
        event_management=EventManagementSection(
            approval_batch_interval_days=int(
                event_management.get("approval_batch_interval_days", 7)
            ),
            notification_before_days=int(
                event_management.get("notification_before_days", 1)
            ),
            notification_channel_id=str(
                event_management.get("notification_channel_id", "")
            ),
            admin_user_ids=[
                str(value) for value in event_management.get("admin_user_ids", [])
            ],
            admin_role_ids=[
                str(value) for value in event_management.get("admin_role_ids", [])
            ],
            prompt_name=str(event_management.get("prompt_name", "event_extraction.md")),
            auto_extract_after_index_update=bool(
                event_management.get("auto_extract_after_index_update", True)
            ),
            timezone=str(event_management.get("timezone", "Asia/Tokyo")),
        ),
        infrastructure=InfrastructureSection(
            database=DatabaseSection(
                url=str(database.get("url", "")),
                connect_timeout_seconds=float(
                    database.get("connect_timeout_seconds", 3.0)
                ),
                application_name=str(database.get("application_name", "kumc-agent")),
            ),
            redis=RedisSection(
                url=str(redis.get("url", "")),
                socket_timeout_seconds=float(
                    redis.get("socket_timeout_seconds", 3.0)
                ),
            ),
            object_storage=ObjectStorageSection(
                endpoint_url=str(object_storage.get("endpoint_url", "")),
                bucket=str(object_storage.get("bucket", "")),
                region=str(object_storage.get("region", "ap-northeast-1")),
                access_key_id=str(object_storage.get("access_key_id", "")),
                secret_access_key=str(object_storage.get("secret_access_key", "")),
                prefix=str(object_storage.get("prefix", "kumc-agent")),
                use_ssl=bool(object_storage.get("use_ssl", True)),
            ),
            migrations=MigrationSection(
                directory=_resolve_path(
                    base_dir,
                    str(migrations.get("directory", "infrastructure/migrations")),
                ),
                table_name=str(migrations.get("table_name", "schema_migrations")),
            ),
        ),
        features=FeatureSection(
            rag=bool(features["rag"]),
            indexing=bool(features["indexing"]),
            eval=bool(features["eval"]),
            summarization=bool(features["summarization"]),
            vc=bool(features["vc"]),
            docgen=bool(features["docgen"]),
            http=bool(features["http"]),
            recency_mode=str(features.get("recency_mode", "off")),
            sources=SourcesSection(
                drive=bool(features["sources"]["drive"]),
                discord=bool(features["sources"]["discord"]),
                hatenablog=bool(features["sources"]["hatenablog"]),
                crafters_colony=bool(features["sources"]["crafters_colony"]),
                x=bool(features["sources"].get("x", True)),
                notion=bool(features["sources"].get("notion", False)),
                minecraft_wiki=bool(
                    features["sources"].get("minecraft_wiki", False)
                ),
            ),
            retrieval=RetrievalSection(
                top_k=int(features["retrieval"]["top_k"]),
                dense_top_k=int(features["retrieval"]["dense_top_k"]),
                sparse_top_k=int(features["retrieval"]["sparse_top_k"]),
                sparse_initial_sparse_top_k=int(
                    features["retrieval"].get(
                        "sparse_initial_sparse_top_k",
                        features["retrieval"]["sparse_top_k"],
                    )
                ),
                sparse_normalized_ratio=(
                    None
                    if features["retrieval"].get("sparse_normalized_ratio") is None
                    else float(features["retrieval"].get("sparse_normalized_ratio"))
                ),
                rerank_pool_size=int(features["retrieval"]["rerank_pool_size"]),
                rrf_k=int(features["retrieval"].get("rrf_k", 60)),
                mmr_lambda=float(features["retrieval"]["mmr_lambda"]),
                recency_weight_soft=float(
                    features["retrieval"].get(
                        "recency_weight_soft",
                        features["retrieval"].get("recency_weight", 0.20),
                    )
                ),
                recency_weight_hard=float(
                    features["retrieval"].get("recency_weight_hard", 0.45)
                ),
                recency_half_life_days=float(
                    features["retrieval"].get("recency_half_life_days", 45.0)
                ),
                parent_doc_enabled=bool(
                    features["retrieval"].get("parent_doc_enabled", True)
                ),
                parent_chunk_cap=int(
                    features["retrieval"].get("parent_chunk_cap", 2)
                ),
                material_full_text_char_limit=int(
                    features["retrieval"].get("material_full_text_char_limit", 3000)
                ),
                sudachi_mode=str(features["retrieval"].get("sudachi_mode", "B")),
                sparse_bm25_k1=float(features["retrieval"].get("sparse_bm25_k1", 1.5)),
                sparse_bm25_b=float(features["retrieval"].get("sparse_bm25_b", 0.75)),
                sparse_use_normalized_form=bool(
                    features["retrieval"].get("sparse_use_normalized_form", True)
                ),
                sparse_remove_symbols=bool(
                    features["retrieval"].get("sparse_remove_symbols", True)
                ),
            ),
            risk_flags=RiskFeatureFlagsSection(
                action_execution=str(
                    features["risk_flags"].get(
                        "action_execution",
                        "approval_required",
                    )
                ),
                external_posting=str(
                    features["risk_flags"].get(
                        "external_posting",
                        "approval_required",
                    )
                ),
                minecraft_server_ops=str(
                    features["risk_flags"].get(
                        "minecraft_server_ops",
                        "approval_required",
                    )
                ),
                accounting_finalize=str(
                    features["risk_flags"].get(
                        "accounting_finalize",
                        "approval_required",
                    )
                ),
                auto_reply=str(
                    features["risk_flags"].get("auto_reply", "approval_required")
                ),
                automation_auto_run=str(
                    features["risk_flags"].get("automation_auto_run", "disabled")
                ),
                vc_recording=str(
                    features["risk_flags"].get("vc_recording", "disabled")
                ),
                image_generation=str(
                    features["risk_flags"].get(
                        "image_generation",
                        "approval_required",
                    )
                ),
            ),
            image_search=ImageSearchFeatureSection(
                enabled=bool(image_search_raw.get("enabled", True)),
                limit=int(image_search_raw.get("limit", features["retrieval"]["top_k"])),
                dense_top_k=int(
                    image_search_raw.get(
                        "dense_top_k",
                        features["retrieval"]["dense_top_k"],
                    )
                ),
                feature_top_k=int(
                    image_search_raw.get(
                        "feature_top_k",
                        features["retrieval"]["sparse_top_k"],
                    )
                ),
                rrf_k=int(image_search_raw.get("rrf_k", features["retrieval"].get("rrf_k", 60))),
                ocr_text_char_limit=int(image_search_raw.get("ocr_text_char_limit", 800)),
                surrounding_text_char_limit=int(
                    image_search_raw.get("surrounding_text_char_limit", 1200)
                ),
                caption_model=str(image_search_raw.get("caption_model", "")),
                ocr_model=str(image_search_raw.get("ocr_model", "")),
                feature_model=str(image_search_raw.get("feature_model", "openai/clip-vit-base-patch32")),
                feature_dimensions=int(image_search_raw.get("feature_dimensions", 512)),
                duplicate_group_limit=int(image_search_raw.get("duplicate_group_limit", 1)),
            ),
            member_search=MemberSearchFeatureSection(
                exclude_role_names=[
                    str(value)
                    for value in features.get("member_search", {}).get("exclude_role_names", [])
                ],
            ),
        ),
        minecraft_wiki_rag=MinecraftWikiRagSection(
            chunking=MinecraftWikiRagChunkingSection(
                first_recursive_chunk_size=int(
                    minecraft_wiki_rag_chunking.get(
                        "first_recursive_chunk_size",
                        indexing_chunking.get("first_recursive_chunk_size", 1024),
                    )
                ),
                first_recursive_chunk_overlap=int(
                    minecraft_wiki_rag_chunking.get(
                        "first_recursive_chunk_overlap",
                        indexing_chunking.get("first_recursive_chunk_overlap", 128),
                    )
                ),
                second_recursive_chunk_size=int(
                    minecraft_wiki_rag_chunking.get(
                        "second_recursive_chunk_size",
                        indexing_chunking.get("second_recursive_chunk_size", 384),
                    )
                ),
                second_recursive_chunk_overlap=int(
                    minecraft_wiki_rag_chunking.get(
                        "second_recursive_chunk_overlap",
                        indexing_chunking.get("second_recursive_chunk_overlap", 64),
                    )
                ),
                summary_characters=int(
                    minecraft_wiki_rag_chunking.get(
                        "summary_characters",
                        indexing_chunking.get("summary_characters", 200),
                    )
                ),
                summary_batch_size=max(
                    1,
                    int(
                        minecraft_wiki_rag_chunking.get(
                            "summary_batch_size",
                            indexing_chunking.get("summary_batch_size", 1),
                        )
                    ),
                ),
                summary_llm_provider=str(
                    minecraft_wiki_rag_chunking.get(
                        "summary_llm_provider",
                        indexing_chunking.get("summary_llm_provider", "none"),
                    )
                ),
                summary_gemini_model=str(
                    minecraft_wiki_rag_chunking.get(
                        "summary_gemini_model",
                        indexing_chunking.get(
                            "summary_gemini_model",
                            providers["llm"]["gemini_model"],
                        ),
                    )
                ),
                summary_temperature=float(
                    minecraft_wiki_rag_chunking.get(
                        "summary_temperature",
                        indexing_chunking.get(
                            "summary_temperature",
                            providers["llm"]["temperature"],
                        ),
                    )
                ),
                summary_max_output_tokens=int(
                    minecraft_wiki_rag_chunking.get(
                        "summary_max_output_tokens",
                        indexing_chunking.get(
                            "summary_max_output_tokens",
                            providers["llm"]["max_output_tokens"],
                        ),
                    )
                ),
                summary_thinking_level=str(
                    minecraft_wiki_rag_chunking.get(
                        "summary_thinking_level",
                        indexing_chunking.get(
                            "summary_thinking_level",
                            providers["llm"]["thinking_level"],
                        ),
                    )
                ),
            ),
            retrieval=MinecraftWikiRagRetrievalSection(
                top_k=int(
                    minecraft_wiki_rag_retrieval.get(
                        "top_k",
                        features["retrieval"]["top_k"],
                    )
                ),
                dense_top_k=int(
                    minecraft_wiki_rag_retrieval.get(
                        "dense_top_k",
                        features["retrieval"]["dense_top_k"],
                    )
                ),
                sparse_top_k=int(
                    minecraft_wiki_rag_retrieval.get(
                        "sparse_top_k",
                        features["retrieval"]["sparse_top_k"],
                    )
                ),
                sparse_initial_sparse_top_k=int(
                    minecraft_wiki_rag_retrieval.get(
                        "sparse_initial_sparse_top_k",
                        features["retrieval"].get(
                            "sparse_initial_sparse_top_k",
                            features["retrieval"]["sparse_top_k"],
                        ),
                    )
                ),
                sparse_normalized_ratio=(
                    None
                    if minecraft_wiki_rag_retrieval.get(
                        "sparse_normalized_ratio",
                        features["retrieval"].get("sparse_normalized_ratio"),
                    )
                    is None
                    else float(
                        minecraft_wiki_rag_retrieval.get(
                            "sparse_normalized_ratio",
                            features["retrieval"].get("sparse_normalized_ratio"),
                        )
                    )
                ),
                rerank_pool_size=int(
                    minecraft_wiki_rag_retrieval.get(
                        "rerank_pool_size",
                        features["retrieval"]["rerank_pool_size"],
                    )
                ),
                rrf_k=int(
                    minecraft_wiki_rag_retrieval.get(
                        "rrf_k",
                        features["retrieval"].get("rrf_k", 60),
                    )
                ),
                mmr_lambda=float(
                    minecraft_wiki_rag_retrieval.get(
                        "mmr_lambda",
                        features["retrieval"]["mmr_lambda"],
                    )
                ),
                parent_doc_enabled=bool(
                    minecraft_wiki_rag_retrieval.get(
                        "parent_doc_enabled",
                        features["retrieval"].get("parent_doc_enabled", True),
                    )
                ),
                parent_chunk_cap=int(
                    minecraft_wiki_rag_retrieval.get(
                        "parent_chunk_cap",
                        features["retrieval"].get("parent_chunk_cap", 2),
                    )
                ),
                sudachi_mode=str(
                    minecraft_wiki_rag_retrieval.get(
                        "sudachi_mode",
                        features["retrieval"].get("sudachi_mode", "B"),
                    )
                ),
                sparse_bm25_k1=float(
                    minecraft_wiki_rag_retrieval.get(
                        "sparse_bm25_k1",
                        features["retrieval"].get("sparse_bm25_k1", 1.5),
                    )
                ),
                sparse_bm25_b=float(
                    minecraft_wiki_rag_retrieval.get(
                        "sparse_bm25_b",
                        features["retrieval"].get("sparse_bm25_b", 0.75),
                    )
                ),
                sparse_use_normalized_form=bool(
                    minecraft_wiki_rag_retrieval.get(
                        "sparse_use_normalized_form",
                        features["retrieval"].get("sparse_use_normalized_form", True),
                    )
                ),
                sparse_remove_symbols=bool(
                    minecraft_wiki_rag_retrieval.get(
                        "sparse_remove_symbols",
                        features["retrieval"].get("sparse_remove_symbols", True),
                    )
                ),
            ),
        ),
        rag=RagSection(
            routing=RagRoutingSection(
                enabled=bool(
                    rag_routing.get(
                        "enabled",
                        providers["function_call"].get("enabled", True),
                    )
                ),
                provider=routing_provider,
                gemini_model=routing_gemini_model,
                prompt_name=routing_prompt_name,
                temperature=float(rag_routing.get("temperature", 0.0)),
                max_new_tokens=int(rag_routing.get("max_new_tokens", 64)),
                max_retries=int(rag_routing.get("max_retries", 2)),
                log_enabled=bool(rag_routing.get("log_enabled", False)),
                material_search_max_names=int(
                    rag_routing.get("material_search_max_names", 3)
                ),
                tasks=routing_tasks,
            ),
            history=RagHistorySection(
                enabled=bool(rag_history.get("enabled", False)),
                max_turns=int(rag_history.get("max_turns", 5)),
                prompt_default_turns=int(rag_history.get("prompt_default_turns", 3)),
                prompt_additional_turns=int(
                    rag_history.get("prompt_additional_turns", 10)
                ),
                special_channel_history_limit=int(
                    rag_history.get("special_channel_history_limit", 30)
                ),
                special_channel_names=special_channel_names,
                special_channel_custom_instruction=str(
                    rag_history.get("special_channel_custom_instruction", "")
                ),
            ),
            generation=RagGenerationSection(
                rag=rag_generation_profile,
                no_rag=no_rag_generation_profile,
                idea_generation=idea_generation_profile,
            ),
            prompt_texts=RagPromptTextSection(
                empty_context=str(
                    rag_prompt_texts.get("empty_context", "(コンテキストなし)")
                ),
                empty_history=str(
                    rag_prompt_texts.get("empty_history", "(履歴なし)")
                ),
                history_user_prefix=str(
                    rag_prompt_texts.get("history_user_prefix", "ユーザー: ")
                ),
                history_assistant_prefix=str(
                    rag_prompt_texts.get(
                        "history_assistant_prefix",
                        "アシスタント: ",
                    )
                ),
                history_sources_label=str(
                    rag_prompt_texts.get("history_sources_label", "参照ソース:")
                ),
                gemini_header_chat_history=str(
                    rag_prompt_texts.get(
                        "gemini_header_chat_history",
                        "# チャット履歴",
                    )
                ),
                gemini_header_retry_history=str(
                    rag_prompt_texts.get(
                        "gemini_header_retry_history",
                        "# 再検索前の質問と回答",
                    )
                ),
                gemini_header_circle_info=str(
                    rag_prompt_texts.get(
                        "gemini_header_circle_info",
                        "# サークルの基本情報",
                    )
                ),
                gemini_header_capabilities=str(
                    rag_prompt_texts.get(
                        "gemini_header_capabilities",
                        "# チャットボット自身の機能情報",
                    )
                ),
                gemini_header_context=str(
                    rag_prompt_texts.get("gemini_header_context", "# コンテキスト")
                ),
                gemini_header_output_format=str(
                    rag_prompt_texts.get(
                        "gemini_header_output_format",
                        "# 出力形式",
                    )
                ),
                gemini_header_instructions=str(
                    rag_prompt_texts.get(
                        "gemini_header_instructions",
                        "## 指示",
                    )
                ),
                gemini_header_question=str(
                    rag_prompt_texts.get(
                        "gemini_header_question",
                        "# ユーザーの質問",
                    )
                ),
            ),
            fast_model_notice=str(
                rag.get(
                    "fast_model_notice",
                    "※負荷軽減のために軽量モードを使用しました。",
                )
            ),
            answer_json_max_retries=max(
                0,
                int(rag.get("answer_json_max_retries", 2)),
            ),
        ),
        indexing=IndexingSection(
            chunking=IndexingChunkingSection(
                first_recursive_chunk_size=int(
                    indexing_chunking.get("first_recursive_chunk_size", 1024)
                ),
                first_recursive_chunk_overlap=int(
                    indexing_chunking.get("first_recursive_chunk_overlap", 128)
                ),
                second_recursive_chunk_size=int(
                    indexing_chunking.get("second_recursive_chunk_size", 128)
                ),
                second_recursive_chunk_overlap=int(
                    indexing_chunking.get("second_recursive_chunk_overlap", 32)
                ),
                summary_characters=int(indexing_chunking.get("summary_characters", 200)),
                summary_batch_size=max(
                    1,
                    int(indexing_chunking.get("summary_batch_size", 1)),
                ),
                summary_llm_provider=str(
                    indexing_chunking.get("summary_llm_provider", "none")
                ),
                summary_gemini_model=str(
                    indexing_chunking.get(
                        "summary_gemini_model",
                        providers["llm"]["gemini_model"],
                    )
                ),
                summary_temperature=float(
                    indexing_chunking.get(
                        "summary_temperature",
                        providers["llm"]["temperature"],
                    )
                ),
                summary_max_output_tokens=int(
                    indexing_chunking.get(
                        "summary_max_output_tokens",
                        providers["llm"]["max_output_tokens"],
                    )
                ),
                summary_thinking_level=str(
                    indexing_chunking.get(
                        "summary_thinking_level",
                        providers["llm"]["thinking_level"],
                    )
                ),
            ),
            stages=IndexingStagesSection(
                second_recursive_enabled=bool(
                    indexing_stages.get("second_recursive_enabled", True)
                ),
                sparse_second_recursive_enabled=bool(
                    indexing_stages.get("sparse_second_recursive_enabled", True)
                ),
                summary_enabled=bool(indexing_stages.get("summary_enabled", True)),
            ),
            refresh=IndexingRefreshSection(
                clear_raw_data=bool(indexing_refresh.get("clear_raw_data", False)),
                clear_first_recursive_chunk_data=bool(
                    indexing_refresh.get("clear_first_recursive_chunk_data", False)
                ),
                clear_second_recursive_chunk_data=bool(
                    indexing_refresh.get("clear_second_recursive_chunk_data", False)
                ),
                clear_summary_chunk_data=bool(
                    indexing_refresh.get("clear_summary_chunk_data", False)
                ),
                update_raw_data=bool(indexing_refresh.get("update_raw_data", True)),
                update_first_recursive_chunk_data=bool(
                    indexing_refresh.get("update_first_recursive_chunk_data", True)
                ),
                update_second_recursive_chunk_data=bool(
                    indexing_refresh.get("update_second_recursive_chunk_data", True)
                ),
                update_sparse_second_recursive_chunk_data=bool(
                    indexing_refresh.get(
                        "update_sparse_second_recursive_chunk_data",
                        True,
                    )
                ),
                update_summary_chunk_data=bool(
                    indexing_refresh.get("update_summary_chunk_data", True)
                ),
            ),
        ),
        ops=OpsSection(
            index_update_estimate_min_minutes=int(
                ops.get("index_update_estimate_min_minutes", 30)
            ),
            index_update_estimate_max_minutes=int(
                ops.get("index_update_estimate_max_minutes", 60)
            ),
            ragas_answer_generation_batch_size=max(
                0,
                int(
                    ops.get(
                        "ragas_answer_generation_batch_size",
                        ops.get("ragas_batch_size", 10),
                    )
                ),
            ),
            ragas_batch_size=max(
                0,
                int(ops.get("ragas_batch_size", 10)),
            ),
            ragas_max_workers=max(
                0,
                int(ops.get("ragas_max_workers", 16)),
            ),
            ragas_timeout_seconds=max(
                0.0,
                float(ops.get("ragas_timeout_seconds", 180.0)),
            ),
            ragas_max_retries=max(
                0,
                int(ops.get("ragas_max_retries", 2)),
            ),
            ragas_answer_cache_enabled=bool(
                ops.get("ragas_answer_cache_enabled", True)
            ),
            ragas_answer_cache_path=_resolve_path(
                base_dir,
                str(
                    ops.get(
                        "ragas_answer_cache_path",
                        "data/eval/cache/ragas_answers.jsonl",
                    )
                ),
            ),
            ragas_disable_history_for_eval=bool(
                ops.get("ragas_disable_history_for_eval", True)
            ),
            ragas_metrics=OpsRagasMetricsSection(
                answer_relevancy_enabled=bool(
                    ops_ragas_metrics.get("answer_relevancy_enabled", True)
                ),
                faithfulness_enabled=bool(
                    ops_ragas_metrics.get("faithfulness_enabled", True)
                ),
                context_precision_enabled=bool(
                    ops_ragas_metrics.get("context_precision_enabled", True)
                ),
                context_recall_enabled=bool(
                    ops_ragas_metrics.get("context_recall_enabled", True)
                ),
            ),
            answer_record_log_enabled=bool(
                ops.get("answer_record_log_enabled", True)
            ),
            answer_record_log_path=_resolve_path(
                base_dir,
                str(
                    ops.get(
                        "answer_record_log_path",
                        app.get("answer_record_log_path", "logs/answer_records.jsonl"),
                    )
                ),
            ),
        ),
        summarization=SummarizationSection(
            target_characters=max(
                1,
                int(summarization.get("target_characters", 200)),
            ),
        ),
        integrations=IntegrationSection(
            discord=IntegrationDiscordSection(
                bot_token=str(integrations.get("discord", {}).get("bot_token", "")),
            ),
            openclaw=IntegrationOpenClawSection(
                enabled=bool(integrations.get("openclaw", {}).get("enabled", True)),
                agent=str(integrations.get("openclaw", {}).get("agent", "main")),
                model=str(integrations.get("openclaw", {}).get("model", "")),
                lite_agent=str(
                    integrations.get("openclaw", {}).get("lite_agent", "")
                ),
                lite_model=str(
                    integrations.get("openclaw", {}).get("lite_model", "")
                ),
                config_dir=_resolve_path(
                    base_dir,
                    str(
                        integrations.get("openclaw", {}).get(
                            "config_dir",
                            "configs/openclaw",
                        )
                    ),
                ),
            ),
            drive=IntegrationDriveSection(
                folder_id=str(integrations.get("drive", {}).get("folder_id", "")),
                google_application_credentials=_resolve_optional_path_str(
                    base_dir,
                    str(
                        integrations.get("drive", {}).get(
                            "google_application_credentials",
                            "",
                        )
                    ),
                ),
                max_files=int(integrations.get("drive", {}).get("max_files", 0)),
                batch_size=max(
                    1,
                    int(integrations.get("drive", {}).get("batch_size", 20)),
                ),
                download_max_retries=max(
                    0,
                    int(integrations.get("drive", {}).get("download_max_retries", 3)),
                ),
                download_retry_initial_delay_seconds=max(
                    0.0,
                    float(
                        integrations.get("drive", {}).get(
                            "download_retry_initial_delay_seconds",
                            0.5,
                        )
                    ),
                ),
                download_retry_max_delay_seconds=max(
                    0.0,
                    float(
                        integrations.get("drive", {}).get(
                            "download_retry_max_delay_seconds",
                            8.0,
                        )
                    ),
                ),
                download_retry_backoff_multiplier=max(
                    1.0,
                    float(
                        integrations.get("drive", {}).get(
                            "download_retry_backoff_multiplier",
                            2.0,
                        )
                    ),
                ),
                pdf_ocr_model_path=str(
                    integrations.get("drive", {}).get(
                        "pdf_ocr_model_path",
                        "model/ocr/PaddlePaddle/PP-OCRv5_mobile",
                    )
                ),
            ),
            hatenablog=IntegrationHatenablogSection(
                blog_url=str(
                    integrations.get("hatenablog", {}).get(
                        "blog_url",
                        "",
                    )
                ),
            ),
            crafters_colony=IntegrationCraftersColonySection(
                author_url=str(
                    integrations.get("crafters_colony", {}).get("author_url", "")
                ),
                max_pages=int(
                    integrations.get("crafters_colony", {}).get("max_pages", 100)
                ),
                max_articles=int(
                    integrations.get("crafters_colony", {}).get("max_articles", 0)
                ),
            ),
            notion=IntegrationNotionSection(
                api_token=str(notion.get("api_token", "")),
                database_ids=[
                    str(value).strip()
                    for value in notion_database_ids_raw
                    if str(value).strip()
                ],
            ),
            minecraft_wiki=IntegrationMinecraftWikiSection(
                page_titles=minecraft_page_titles,
                api_url=str(
                    minecraft_wiki.get("api_url", "https://ja.minecraft.wiki/api.php")
                ),
                page_url_base=str(
                    minecraft_wiki.get("page_url_base", "https://ja.minecraft.wiki/w/")
                ),
                max_pages=max(0, int(minecraft_wiki.get("max_pages", 20))),
                rate_limit_per_minute=max(
                    0,
                    int(minecraft_wiki.get("rate_limit_per_minute", 30)),
                ),
                request_interval_seconds=max(
                    0.0,
                    float(minecraft_wiki.get("request_interval_seconds", 1.0)),
                ),
                namespaces=minecraft_wiki_namespaces,
                full_backfill_enabled=bool(
                    minecraft_wiki.get("full_backfill_enabled", False)
                ),
            ),
            openai_api_key=str(integrations.get("openai_api_key", "")),
            gemini_api_key=str(integrations.get("gemini_api_key", "")),
            gemini_requests_per_minute=max(
                0,
                int(integrations.get("gemini_requests_per_minute", 60)),
            ),
            gemini_embedding_requests_per_minute=max(
                0,
                int(
                    integrations.get(
                        "gemini_embedding_requests_per_minute",
                        integrations.get("gemini_requests_per_minute", 60),
                    )
                ),
            ),
            gemini_summary_requests_per_minute=max(
                0,
                int(
                    integrations.get(
                        "gemini_summary_requests_per_minute",
                        integrations.get("gemini_requests_per_minute", 60),
                    )
                ),
            ),
            gemini_ragas_requests_per_minute=max(
                0,
                int(
                    integrations.get(
                        "gemini_ragas_requests_per_minute",
                        integrations.get("gemini_requests_per_minute", 60),
                    )
                ),
            ),
            gemini_ragas_embedding_requests_per_minute=max(
                0,
                int(
                    integrations.get(
                        "gemini_ragas_embedding_requests_per_minute",
                        integrations.get(
                            "gemini_ragas_requests_per_minute",
                            integrations.get(
                                "gemini_embedding_requests_per_minute",
                                integrations.get("gemini_requests_per_minute", 60),
                            ),
                        ),
                    )
                ),
            ),
        ),
        model=ModelSection(
            root_dir=_resolve_path(
                base_dir,
                str(model.get("root_dir", "model")),
            ),
            llm_dir=_resolve_path(
                base_dir,
                str(model.get("llm_dir", "model/llm")),
            ),
            embedding_dir=_resolve_path(
                base_dir,
                str(model.get("embedding_dir", "model/embedding")),
            ),
            cross_encoder_dir=_resolve_path(
                base_dir,
                str(model.get("cross_encoder_dir", "model/cross-encoder")),
            ),
            whisper_dir=_resolve_path(
                base_dir,
                str(model.get("whisper_dir", "model/whisper")),
            ),
            ocr_dir=_resolve_path(
                base_dir,
                str(model.get("ocr_dir", "model/ocr")),
            ),
        ),
        vc=VCSection(
            feature_enabled=bool(vc.get("feature_enabled", False)),
            auto_join_enabled=bool(vc.get("auto_join_enabled", False)),
            auto_join_weekdays=[int(v) for v in vc.get("auto_join_weekdays", [5])],
            auto_join_time=str(vc.get("auto_join_time", "20:00")),
            auto_join_duration_minutes=int(vc.get("auto_join_duration_minutes", 30)),
            target_voice_channel_name=str(vc.get("target_voice_channel_name", "例会")),
            auto_join_min_participants=int(vc.get("auto_join_min_participants", 3)),
            participant_check_interval_seconds=int(
                vc.get("participant_check_interval_seconds", 10)
            ),
            summary_transcribe_interval_seconds=int(
                vc.get("summary_transcribe_interval_seconds", 300)
            ),
            transcribe_model=str(vc.get("transcribe_model", "model/whisper/openai/whisper-large-v3-turbo")),
            transcribe_device=str(vc.get("transcribe_device", "auto")),
            transcribe_torch_dtype=str(vc.get("transcribe_torch_dtype", "auto")),
            transcribe_language=str(vc.get("transcribe_language", "ja")),
            auto_quit_enabled=bool(vc.get("auto_quit_enabled", True)),
            final_summary_enabled=bool(vc.get("final_summary_enabled", True)),
            summary_previous_max=int(vc.get("summary_previous_max", 2)),
            summary_target_characters=int(vc.get("summary_target_characters", 100)),
            summary_llm_provider=str(vc.get("summary_llm_provider", "gemini")),
            summary_gemini_model=str(
                vc.get("summary_gemini_model", providers["llm"]["gemini_model"])
            ),
            summary_temperature=float(vc.get("summary_temperature", 0.2)),
            summary_max_output_tokens=int(vc.get("summary_max_output_tokens", 256)),
            summary_thinking_level=str(vc.get("summary_thinking_level", "minimal")),
            minutes_enabled=bool(vc.get("minutes_enabled", True)),
            minutes_drive_dir=str(vc.get("minutes_drive_dir", "議事録")),
            minutes_fetch_max_retries=int(vc.get("minutes_fetch_max_retries", 2)),
            minutes_apply_max_retries=int(vc.get("minutes_apply_max_retries", 2)),
            minutes_llm_max_retries=int(vc.get("minutes_llm_max_retries", 2)),
            minutes_history_summary_max=int(vc.get("minutes_history_summary_max", 2)),
            minutes_image_batch_size=int(vc.get("minutes_image_batch_size", 10)),
            minutes_edit_llm_provider=str(
                vc.get("minutes_edit_llm_provider", "gemini")
            ),
            minutes_edit_gemini_model=str(
                vc.get("minutes_edit_gemini_model", providers["llm"]["gemini_model"])
            ),
            minutes_edit_temperature=float(vc.get("minutes_edit_temperature", 0.2)),
            minutes_edit_max_output_tokens=int(
                vc.get("minutes_edit_max_output_tokens", 1024)
            ),
            minutes_edit_thinking_level=str(
                vc.get("minutes_edit_thinking_level", "minimal")
            ),
            final_summary_llm_provider=str(
                vc.get("final_summary_llm_provider", "gemini")
            ),
            final_summary_gemini_model=str(
                vc.get("final_summary_gemini_model", providers["llm"]["gemini_model"])
            ),
            final_summary_temperature=float(vc.get("final_summary_temperature", 0.0)),
            final_summary_max_output_tokens=int(
                vc.get("final_summary_max_output_tokens", 1024)
            ),
            final_summary_thinking_level=str(
                vc.get("final_summary_thinking_level", "minimal")
            ),
        ),
        server_management=ServerManagementSection(
            default_server_name=str(
                server_management.get("default_server_name", "default")
            ),
            docker_ps=ServerManagementDockerPsSection(
                container_name_prefixes=[
                    str(value)
                    for value in (
                        server_management.get("docker_ps", {}).get(
                            "container_name_prefixes",
                            [],
                        )
                        or []
                    )
                    if str(value)
                ],
            ),
            servers=[
                ServerManagementServerSection(
                    name=str(item.get("name", "")),
                    compose_dir=_resolve_path(base_dir, str(item.get("compose_dir", ""))),
                    services=[
                        str(value)
                        for value in (item.get("services", []) or [])
                        if str(value)
                    ],
                    allow_file_search_paths=[
                        _resolve_path(base_dir, str(value))
                        for value in (item.get("allow_file_search_paths", []) or [])
                        if str(value)
                    ],
                    critical_operations_enabled=bool(
                        item.get("critical_operations_enabled", False)
                    ),
                )
                for item in (server_management.get("servers", []) or [])
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            ],
            execution=ServerManagementExecutionSection(
                timeout_seconds=int(
                    server_management.get("execution", {}).get("timeout_seconds", 120)
                ),
                stdout_char_limit=int(
                    server_management.get("execution", {}).get("stdout_char_limit", 4000)
                ),
                stderr_char_limit=int(
                    server_management.get("execution", {}).get("stderr_char_limit", 4000)
                ),
            ),
            backup=ServerManagementBackupSection(
                backup_dir=_resolve_path(
                    base_dir,
                    str(
                        server_management.get("backup", {}).get(
                            "backup_dir",
                            "data/minecraft/backups",
                        )
                    ),
                ),
                max_backups=int(
                    server_management.get("backup", {}).get("max_backups", 10)
                ),
            ),
        ),
    )

    _validate_required(runtime)
    return runtime


def _validate_required(config: RuntimeConfig) -> None:
    missing = config.required_env_missing
    if missing:
        missing_text = ", ".join(missing)
        raise ConfigLoadError(f"Missing required environment variables: {missing_text}")
