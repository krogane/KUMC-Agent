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
    EmbeddingSection,
    FeatureSection,
    FunctionCallSection,
    RagGenerationProfileSection,
    RagGenerationSection,
    RagIdeaGenerationSection,
    IndexingChunkingSection,
    IndexingRefreshSection,
    IndexingSection,
    IndexingStagesSection,
    IntegrationCraftersColonySection,
    IntegrationDiscordSection,
    IntegrationDriveSection,
    IntegrationOpenClawSection,
    IntegrationSection,
    LLMSection,
    ModelSection,
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
    SourcesSection,
    VCSection,
)

OPS_FILES = (
    "app.yaml",
    "providers.yaml",
    "security.yaml",
    "scheduler.yaml",
    "features.yaml",
    "model.yaml",
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
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_experiment_path(base_dir: Path, profile: str) -> Path:
    profile_clean = (profile or "rag/baseline").strip().replace("\\", "/")
    if not profile_clean:
        profile_clean = "rag/baseline"
    if not profile_clean.endswith(".yaml"):
        profile_clean += ".yaml"
    return base_dir / "configs" / "experiments" / profile_clean


def _backfill_default_config_values(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    integrations = updated.get("integrations")
    if not isinstance(integrations, dict):
        integrations = {}
        updated["integrations"] = integrations
    openclaw = integrations.get("openclaw")
    if not isinstance(openclaw, dict):
        openclaw = {}
        integrations["openclaw"] = openclaw
    openclaw.setdefault("enabled", True)
    openclaw.setdefault("agent", "main")
    openclaw.setdefault("model", "")
    openclaw.setdefault("config_dir", "configs/openclaw")
    return updated


def load_runtime_config(*, base_dir: Path | None = None) -> RuntimeConfig:
    resolved_base_dir = base_dir or Path(__file__).resolve().parents[3]
    load_dotenv(resolved_base_dir / ".env", override=False)

    merged: dict[str, Any] = {}
    for file_name in OPS_FILES:
        payload = _yaml_load(resolved_base_dir / "configs" / "ops" / file_name)
        try:
            merged = deep_merge(merged, payload, allow_new_keys=True)
        except MergeError as exc:
            raise ConfigLoadError(str(exc)) from exc
    merged = _backfill_default_config_values(merged)

    try:
        merged = _apply_env_overrides(merged)
    except (ValueError, ConfigLoadError) as exc:
        raise ConfigLoadError(str(exc)) from exc

    experiment_profile = os.getenv("KUMC_EXPERIMENT_PROFILE", "rag/baseline")
    experiment_path = _resolve_experiment_path(resolved_base_dir, experiment_profile)
    experiment_payload = _yaml_load(experiment_path)

    try:
        merged = deep_merge(merged, experiment_payload, allow_new_keys=False)
    except MergeError as exc:
        raise ConfigLoadError(str(exc)) from exc

    return _to_runtime_config(
        merged=merged,
        experiment_profile=experiment_profile,
        base_dir=resolved_base_dir,
        experiments=experiment_payload,
    )


def _to_runtime_config(
    *,
    merged: dict[str, Any],
    experiment_profile: str,
    base_dir: Path,
    experiments: dict[str, Any],
) -> RuntimeConfig:
    app = merged["app"]
    providers = merged["providers"]
    security = merged["security"]
    scheduler = merged["scheduler"]
    features = merged["features"]
    rag = merged.get("rag", {})
    indexing = merged.get("indexing", {})
    ops = merged.get("ops", {})
    integrations = merged.get("integrations", {})
    model = merged.get("model", {})
    vc = merged.get("vc", {})

    rag_routing = rag.get("routing", {})
    rag_history = rag.get("history", {})
    rag_generation = rag.get("generation", {})
    rag_prompt_texts = rag.get("prompt_texts", {})
    rag_generation_rag = rag_generation.get("rag", {})
    rag_generation_no_rag = rag_generation.get("no_rag", {})
    rag_generation_refusal = rag_generation.get("refusal", {})
    rag_generation_idea = rag_generation.get("idea_generation", {})
    indexing_chunking = indexing.get("chunking", {})
    indexing_stages = indexing.get("stages", {})
    indexing_refresh = indexing.get("refresh", {})
    ops_ragas_metrics = ops.get("ragas_metrics", {})

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
        llama_model_path_default = (
            fallback.llama_model_path
            if fallback is not None
            else str(providers["llm"]["llama_model_path"])
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
            llama_model_path=str(
                section.get("llama_model_path", llama_model_path_default)
            ),
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
    refusal_generation_profile = _build_generation_profile(
        rag_generation_refusal,
        default_prompt_name="answer_refusal",
        fallback=no_rag_generation_profile,
    )
    idea_generation_profile = RagIdeaGenerationSection(
        prompt_name=str(
            rag_generation_idea.get(
                "prompt_name",
                "answer_idea",
            )
        ),
        temperature=float(
            rag_generation_idea.get(
                "temperature",
                rag_generation_profile.temperature,
            )
        ),
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
    routing_llama_model_path = str(
        rag_routing.get(
            "llama_model_path",
            providers["function_call"].get("llama_model_path", ""),
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
        llama_model_path_value = str(
            task_raw.get("llama_model_path", routing_llama_model_path)
        ).strip()
        prompt_name_value = str(
            task_raw.get("prompt_name", routing_prompt_name)
        ).strip()
        return RagRoutingTaskSection(
            provider=provider_value or routing_provider,
            gemini_model=gemini_model_value or routing_gemini_model,
            llama_model_path=llama_model_path_value or routing_llama_model_path,
            prompt_name=prompt_name_value or routing_prompt_name,
        )

    routing_tasks = RagRoutingTasksSection(
        target_model=_build_routing_task("target_model"),
        use_additional_memory=_build_routing_task("use_additional_memory"),
        include_capabilities_info=_build_routing_task("include_capabilities_info"),
        idea_generation=_build_routing_task("idea_generation"),
        needs_additional_query=_build_routing_task("needs_additional_query"),
        additional_queries=_build_routing_task("additional_queries"),
        material_names=_build_routing_task("material_names"),
        recency_mode=_build_routing_task("recency_mode"),
    )

    runtime = RuntimeConfig(
        base_dir=base_dir,
        experiment_profile=experiment_profile,
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
                llama_model_path=str(providers["llm"]["llama_model_path"]),
                temperature=float(providers["llm"]["temperature"]),
                max_output_tokens=int(providers["llm"]["max_output_tokens"]),
                thinking_level=str(providers["llm"]["thinking_level"]),
                threads=int(providers["llm"].get("threads", 4)),
                gpu_layers=int(providers["llm"].get("gpu_layers", 0)),
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
                llama_model_path=str(providers["function_call"]["llama_model_path"]),
            ),
        ),
        security=SecuritySection(
            maintenance_command_author_ids=[
                int(v) for v in security["maintenance_command_author_ids"]
            ],
            discord_guild_allow_list=[int(v) for v in security["discord_guild_allow_list"]],
            refusal_keywords=[str(v) for v in security["refusal_keywords"]],
        ),
        scheduler=SchedulerSection(
            auto_index_enabled=bool(scheduler["auto_index_enabled"]),
            auto_index_time=str(scheduler["auto_index_time"]),
            auto_index_weekdays=[int(v) for v in scheduler["auto_index_weekdays"]],
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
                rerank_pool_size=int(features["retrieval"]["rerank_pool_size"]),
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
                llama_model_path=routing_llama_model_path,
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
                refusal=refusal_generation_profile,
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
                llama_header_question=str(
                    rag_prompt_texts.get("llama_header_question", "### Question")
                ),
                llama_header_previous_attempt=str(
                    rag_prompt_texts.get(
                        "llama_header_previous_attempt",
                        "### Previous attempt (Question/Answer)",
                    )
                ),
                llama_header_circle_info=str(
                    rag_prompt_texts.get(
                        "llama_header_circle_info",
                        "### サークルの基本情報",
                    )
                ),
                llama_header_capabilities=str(
                    rag_prompt_texts.get(
                        "llama_header_capabilities",
                        "### チャットボット自身の機能情報",
                    )
                ),
                llama_header_context=str(
                    rag_prompt_texts.get("llama_header_context", "### Context")
                ),
                llama_header_output_format=str(
                    rag_prompt_texts.get(
                        "llama_header_output_format",
                        "### Output format",
                    )
                ),
                llama_header_instructions=str(
                    rag_prompt_texts.get(
                        "llama_header_instructions",
                        "## 指示",
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
                summary_llama_model_path=str(
                    indexing_chunking.get(
                        "summary_llama_model_path",
                        providers["llm"]["llama_model_path"],
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
                proposition_llm_provider=str(
                    indexing_chunking.get(
                        "proposition_llm_provider",
                        providers["llm"]["provider"],
                    )
                ),
                proposition_gemini_model=str(
                    indexing_chunking.get(
                        "proposition_gemini_model",
                        providers["llm"]["gemini_model"],
                    )
                ),
                proposition_llama_model_path=str(
                    indexing_chunking.get(
                        "proposition_llama_model_path",
                        providers["llm"]["llama_model_path"],
                    )
                ),
                proposition_temperature=float(
                    indexing_chunking.get(
                        "proposition_temperature",
                        providers["llm"]["temperature"],
                    )
                ),
                proposition_max_output_tokens=int(
                    indexing_chunking.get(
                        "proposition_max_output_tokens",
                        providers["llm"]["max_output_tokens"],
                    )
                ),
                proposition_thinking_level=str(
                    indexing_chunking.get(
                        "proposition_thinking_level",
                        providers["llm"]["thinking_level"],
                    )
                ),
                proposition_max_retries=max(
                    1,
                    int(indexing_chunking.get("proposition_max_retries", 2)),
                ),
                raptor_llm_provider=str(
                    indexing_chunking.get(
                        "raptor_llm_provider",
                        providers["llm"]["provider"],
                    )
                ),
                raptor_gemini_model=str(
                    indexing_chunking.get(
                        "raptor_gemini_model",
                        providers["llm"]["gemini_model"],
                    )
                ),
                raptor_llama_model_path=str(
                    indexing_chunking.get(
                        "raptor_llama_model_path",
                        providers["llm"]["llama_model_path"],
                    )
                ),
                raptor_temperature=float(
                    indexing_chunking.get(
                        "raptor_temperature",
                        providers["llm"]["temperature"],
                    )
                ),
                raptor_max_output_tokens=int(
                    indexing_chunking.get(
                        "raptor_max_output_tokens",
                        providers["llm"]["max_output_tokens"],
                    )
                ),
                raptor_thinking_level=str(
                    indexing_chunking.get(
                        "raptor_thinking_level",
                        providers["llm"]["thinking_level"],
                    )
                ),
                raptor_max_retries=max(
                    1,
                    int(indexing_chunking.get("raptor_max_retries", 2)),
                ),
                raptor_cluster_max_tokens=max(
                    32,
                    int(indexing_chunking.get("raptor_cluster_max_tokens", 1024)),
                ),
                raptor_stop_chunk_count=max(
                    1,
                    int(indexing_chunking.get("raptor_stop_chunk_count", 20)),
                ),
                raptor_k_max=max(
                    2,
                    int(indexing_chunking.get("raptor_k_max", 8)),
                ),
                raptor_k_selection=str(
                    indexing_chunking.get("raptor_k_selection", "elbow")
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
                proposition_enabled=bool(
                    indexing_stages.get("proposition_enabled", False)
                ),
                raptor_enabled=bool(indexing_stages.get("raptor_enabled", False)),
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
                clear_proposition_chunk_data=bool(
                    indexing_refresh.get("clear_proposition_chunk_data", False)
                ),
                clear_raptor_chunk_data=bool(
                    indexing_refresh.get("clear_raptor_chunk_data", False)
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
                update_proposition_chunk_data=bool(
                    indexing_refresh.get("update_proposition_chunk_data", True)
                ),
                update_raptor_chunk_data=bool(
                    indexing_refresh.get("update_raptor_chunk_data", True)
                ),
            ),
        ),
        ops=OpsSection(
            warmup_interval_minutes=int(ops.get("warmup_interval_minutes", 60)),
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
        integrations=IntegrationSection(
            discord=IntegrationDiscordSection(
                bot_token=str(integrations.get("discord", {}).get("bot_token", "")),
            ),
            openclaw=IntegrationOpenClawSection(
                enabled=bool(integrations.get("openclaw", {}).get("enabled", True)),
                agent=str(integrations.get("openclaw", {}).get("agent", "main")),
                model=str(integrations.get("openclaw", {}).get("model", "")),
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
                google_application_credentials=str(
                    integrations.get("drive", {}).get(
                        "google_application_credentials",
                        "",
                    )
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
            summary_llama_model_path=str(
                vc.get("summary_llama_model_path", providers["llm"]["llama_model_path"])
            ),
            summary_llama_ctx_size=int(vc.get("summary_llama_ctx_size", 4096)),
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
            minutes_edit_llama_model_path=str(
                vc.get("minutes_edit_llama_model_path", providers["llm"]["llama_model_path"])
            ),
            minutes_edit_llama_ctx_size=int(vc.get("minutes_edit_llama_ctx_size", 4096)),
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
            final_summary_llama_model_path=str(
                vc.get("final_summary_llama_model_path", providers["llm"]["llama_model_path"])
            ),
            final_summary_llama_ctx_size=int(
                vc.get("final_summary_llama_ctx_size", 4096)
            ),
            final_summary_temperature=float(vc.get("final_summary_temperature", 0.0)),
            final_summary_max_output_tokens=int(
                vc.get("final_summary_max_output_tokens", 1024)
            ),
            final_summary_thinking_level=str(
                vc.get("final_summary_thinking_level", "minimal")
            ),
        ),
        experiments=experiments,
    )

    _validate_required(runtime)
    return runtime


def _validate_required(config: RuntimeConfig) -> None:
    missing = config.required_env_missing
    if missing:
        missing_text = ", ".join(missing)
        raise ConfigLoadError(f"Missing required environment variables: {missing_text}")
