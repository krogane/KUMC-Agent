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
    IndexingChunkingSection,
    IndexingRefreshSection,
    IndexingSection,
    IndexingStagesSection,
    IntegrationCraftersColonySection,
    IntegrationDiscordSection,
    IntegrationDriveSection,
    IntegrationSection,
    LLMSection,
    ModelSection,
    OpsSection,
    ProviderSection,
    RagHistorySection,
    RagRoutingSection,
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
    indexing_chunking = indexing.get("chunking", {})
    indexing_stages = indexing.get("stages", {})
    indexing_refresh = indexing.get("refresh", {})

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
            ),
            retrieval=RetrievalSection(
                top_k=int(features["retrieval"]["top_k"]),
                dense_top_k=int(features["retrieval"]["dense_top_k"]),
                sparse_top_k=int(features["retrieval"]["sparse_top_k"]),
                rerank_pool_size=int(features["retrieval"]["rerank_pool_size"]),
                mmr_lambda=float(features["retrieval"]["mmr_lambda"]),
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
                provider=str(
                    rag_routing.get(
                        "provider",
                        providers["function_call"].get("provider", "gemini"),
                    )
                ),
                gemini_model=str(
                    rag_routing.get(
                        "gemini_model",
                        providers["function_call"].get("gemini_model", ""),
                    )
                ),
                llama_model_path=str(
                    rag_routing.get(
                        "llama_model_path",
                        providers["function_call"].get("llama_model_path", ""),
                    )
                ),
                temperature=float(rag_routing.get("temperature", 0.0)),
                max_new_tokens=int(rag_routing.get("max_new_tokens", 64)),
                max_retries=int(rag_routing.get("max_retries", 2)),
                log_enabled=bool(rag_routing.get("log_enabled", False)),
                material_search_max_names=int(
                    rag_routing.get("material_search_max_names", 3)
                ),
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
            fast_model_notice=str(
                rag.get(
                    "fast_model_notice",
                    "※負荷軽減のために軽量モードを使用しました。",
                )
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
            drive=IntegrationDriveSection(
                folder_id=str(integrations.get("drive", {}).get("folder_id", "")),
                google_application_credentials=str(
                    integrations.get("drive", {}).get(
                        "google_application_credentials",
                        "",
                    )
                ),
                max_files=int(integrations.get("drive", {}).get("max_files", 0)),
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
