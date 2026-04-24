from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.document import Document
from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever
from kumc_agent.infra.storage.filesystem import FileSystemStorage
from kumc_agent.utils.hashing import stable_hash

KEYWORD_CORPUS_SPARSE = "sparse"
KEYWORD_CORPUS_SPARSE_SECOND_REC = "sparse_second_rec"
KEYWORD_CORPUS_SECOND_REC_SPARSE = "second_rec_sparse"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexBuildResult:
    loaded_sources: int
    documents: int
    chunks: int
    index_dir: Path


class IndexingService:
    def __init__(
        self,
        *,
        storage: FileSystemStorage,
        embedder: EmbedderPort,
        faiss_index: FaissLikeIndex,
        bm25_index: SudachiBM25Retriever,
        raw_dir: Path,
        app_config: RuntimeConfig,
        summary_llm: LLMPort | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._faiss_index = faiss_index
        self._bm25_index = bm25_index
        self._raw_dir = raw_dir
        self._runtime = app_config
        self._summary_llm = summary_llm
        self._chunks_root = self._runtime.app.data_dir / "chunks"
        self._first_rec_dir = self._chunks_root / "first_rec_chunk"
        self._second_rec_dir = self._chunks_root / "second_rec_chunk"
        self._sparse_second_rec_dir = self._chunks_root / "sparse_second_rec_chunk"
        self._summary_dir = self._chunks_root / "summary_chunk"
        self._prop_dir = self._chunks_root / "prop_chunk"
        self._raptor_dir = self._chunks_root / "raptor_chunk"
        self._raw_docs_dir = self._raw_dir / "docs"
        self._raw_sheets_dir = self._raw_dir / "sheets"
        self._raw_messages_dir = self._raw_dir / "messages"
        self._raw_x_dir = self._raw_dir / "x"
        self._raw_vc_dir = self._raw_dir / "vc"
        self._raw_hatenablog_dir = self._raw_dir / "hatenablog"
        self._raw_crafters_colony_dir = self._raw_dir / "crafters_colony"
        self._raw_notion_dir = self._raw_dir / "notion"

        self._first_rec_docs_dir = self._first_rec_dir / "docs"
        self._first_rec_sheets_dir = self._first_rec_dir / "sheets"
        self._first_rec_messages_dir = self._first_rec_dir / "messages"
        self._first_rec_x_dir = self._first_rec_dir / "x"
        self._first_rec_hatenablog_dir = self._first_rec_dir / "hatenablog"
        self._first_rec_crafters_colony_dir = self._first_rec_dir / "crafters_colony"
        self._first_rec_notion_dir = self._first_rec_dir / "notion"

        self._second_rec_docs_dir = self._second_rec_dir / "docs"
        self._second_rec_sheets_dir = self._second_rec_dir / "sheets"
        self._second_rec_messages_dir = self._second_rec_dir / "messages"
        self._second_rec_x_dir = self._second_rec_dir / "x"
        self._second_rec_vc_dir = self._second_rec_dir / "vc"
        self._second_rec_hatenablog_dir = self._second_rec_dir / "hatenablog"
        self._second_rec_crafters_colony_dir = self._second_rec_dir / "crafters_colony"
        self._second_rec_notion_dir = self._second_rec_dir / "notion"

        self._sparse_second_rec_docs_dir = self._sparse_second_rec_dir / "docs"
        self._sparse_second_rec_sheets_dir = self._sparse_second_rec_dir / "sheets"
        self._sparse_second_rec_messages_dir = self._sparse_second_rec_dir / "messages"
        self._sparse_second_rec_x_dir = self._sparse_second_rec_dir / "x"
        self._sparse_second_rec_vc_dir = self._sparse_second_rec_dir / "vc"
        self._sparse_second_rec_hatenablog_dir = self._sparse_second_rec_dir / "hatenablog"
        self._sparse_second_rec_crafters_colony_dir = (
            self._sparse_second_rec_dir / "crafters_colony"
        )
        self._sparse_second_rec_notion_dir = self._sparse_second_rec_dir / "notion"

        self._summary_docs_dir = self._summary_dir / "docs"
        self._summary_sheets_dir = self._summary_dir / "sheets"
        self._summary_messages_dir = self._summary_dir / "messages"
        self._summary_x_dir = self._summary_dir / "x"
        self._summary_hatenablog_dir = self._summary_dir / "hatenablog"
        self._summary_crafters_colony_dir = self._summary_dir / "crafters_colony"
        self._summary_notion_dir = self._summary_dir / "notion"

        self._prop_docs_dir = self._prop_dir / "docs"
        self._prop_sheets_dir = self._prop_dir / "sheets"
        self._prop_hatenablog_dir = self._prop_dir / "hatenablog"
        self._prop_crafters_colony_dir = self._prop_dir / "crafters_colony"
        self._prop_notion_dir = self._prop_dir / "notion"

    def build(
        self,
        *,
        loaded_sources: int,
        full_rebuild: bool = False,
        stage_selection: tuple[str, ...] | None = None,
        allow_cancel: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> IndexBuildResult:
        selected = {
            value.strip()
            for value in (stage_selection or ())
            if value and value.strip()
        }

        self._apply_clear_flags(full_rebuild=full_rebuild)
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)
        self._ensure_raw_source_dirs()

        documents = self._parse_documents_from_raw()
        self._storage.save_documents(documents)

        legacy_cfg = self._build_legacy_app_config()
        self._ensure_legacy_prompt_env_defaults()
        self._run_legacy_chunk_pipeline(
            legacy_cfg=legacy_cfg,
            selected=selected,
            allow_cancel=allow_cancel,
            cancel_event=cancel_event,
        )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        index_chunks = self._load_index_chunks_from_legacy_dirs(legacy_cfg=legacy_cfg)
        self._storage.save_chunks(index_chunks)

        dense_texts = [self._chunk_embedding_text_for_dense(chunk) for chunk in index_chunks]
        embeddings = self._embedder.embed_documents(dense_texts)
        self._faiss_index.build(chunks=index_chunks, embeddings=embeddings)
        self._bm25_index.build(index_chunks)

        self._build_keyword_inverted_indexes(legacy_cfg=legacy_cfg)
        self._build_material_catalog_legacy(legacy_cfg=legacy_cfg)

        return IndexBuildResult(
            loaded_sources=loaded_sources,
            documents=len(documents),
            chunks=len(index_chunks),
            index_dir=self._faiss_index._index_dir,  # noqa: SLF001
        )

    def update(
        self,
        *,
        loaded_sources: int,
        full_rebuild: bool = False,
        stage_selection: tuple[str, ...] | None = None,
        allow_cancel: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> IndexBuildResult:
        return self.build(
            loaded_sources=loaded_sources,
            full_rebuild=full_rebuild,
            stage_selection=stage_selection,
            allow_cancel=allow_cancel,
            cancel_event=cancel_event,
        )

    def _apply_clear_flags(self, *, full_rebuild: bool) -> None:
        refresh = self._runtime.indexing.refresh
        clear_all = full_rebuild
        if clear_all or refresh.clear_raw_data:
            self._clear_dir_contents(self._raw_dir)
        if clear_all or refresh.clear_first_recursive_chunk_data:
            self._clear_dir_contents(self._first_rec_dir)
        if clear_all or refresh.clear_second_recursive_chunk_data:
            self._clear_dir_contents(self._second_rec_dir)
            self._clear_dir_contents(self._sparse_second_rec_dir)
        if clear_all or refresh.clear_summary_chunk_data:
            self._clear_dir_contents(self._summary_dir)
        if clear_all or refresh.clear_proposition_chunk_data:
            self._clear_dir_contents(self._prop_dir)
        if clear_all or refresh.clear_raptor_chunk_data:
            self._clear_dir_contents(self._raptor_dir)

    def _ensure_raw_source_dirs(self) -> None:
        for path in (
            self._raw_docs_dir,
            self._raw_sheets_dir,
            self._raw_messages_dir,
            self._raw_x_dir,
            self._raw_vc_dir,
            self._raw_hatenablog_dir,
            self._raw_crafters_colony_dir,
            self._raw_notion_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _build_legacy_app_config(self):
        from kumc_agent.infra.indexing.config import AppConfig as LegacyAppConfig

        chunking = self._runtime.indexing.chunking
        stages = self._runtime.indexing.stages
        refresh = self._runtime.indexing.refresh
        providers_llm = self._runtime.providers.llm
        retrieval = self._runtime.features.retrieval
        integrations = self._runtime.integrations

        def _model_label(path_value: str) -> str:
            cleaned = str(path_value or "").strip()
            if not cleaned:
                return ""
            return Path(cleaned).name

        return LegacyAppConfig(
            base_dir=self._runtime.base_dir,
            raw_data_dir=self._raw_dir,
            first_rec_chunk_dir=self._first_rec_dir,
            second_rec_chunk_dir=self._second_rec_dir,
            sparse_second_rec_chunk_dir=self._sparse_second_rec_dir,
            summery_chunk_dir=self._summary_dir,
            prop_chunk_dir=self._prop_dir,
            raptor_chunk_dir=self._raptor_dir,
            index_dir=self._runtime.app.index_dir,
            discord_bot_token=self._runtime.integrations.discord.bot_token,
            discord_guild_allow_list=tuple(self._runtime.security.discord_guild_allow_list),
            gemini_api_key=integrations.gemini_api_key,
            gemini_requests_per_minute=integrations.gemini_requests_per_minute,
            gemini_summary_requests_per_minute=(
                integrations.gemini_summary_requests_per_minute
            ),
            drive_folder_id=integrations.drive.folder_id,
            google_application_credentials=(
                integrations.drive.google_application_credentials
            ),
            drive_max_files=integrations.drive.max_files,
            crafters_colony_author_url=integrations.crafters_colony.author_url,
            crafters_colony_max_pages=integrations.crafters_colony.max_pages,
            crafters_colony_max_articles=integrations.crafters_colony.max_articles,
            pdf_ocr_model_path=integrations.drive.pdf_ocr_model_path,
            embedding_model=self._runtime.providers.embeddings.model,
            raptor_embedding_model=self._runtime.providers.embeddings.model,
            first_rec_chunk_size=chunking.first_recursive_chunk_size,
            first_rec_chunk_overlap=chunking.first_recursive_chunk_overlap,
            second_rec_enabled=stages.second_recursive_enabled,
            second_rec_chunk_size=chunking.second_recursive_chunk_size,
            second_rec_chunk_overlap=chunking.second_recursive_chunk_overlap,
            summery_enabled=stages.summary_enabled,
            summery_characters=chunking.summary_characters,
            summery_provider=chunking.summary_llm_provider,
            summery_gemini_model=chunking.summary_gemini_model,
            summery_llama_model=_model_label(chunking.summary_llama_model_path),
            summery_llama_model_path=chunking.summary_llama_model_path,
            summery_llama_ctx_size=4096,
            summery_temperature=chunking.summary_temperature,
            summery_max_output_tokens=chunking.summary_max_output_tokens,
            summery_max_retries=2,
            summery_batch_size=chunking.summary_batch_size,
            llm_provider=providers_llm.provider,
            genai_model=providers_llm.gemini_model,
            llama_model_path=providers_llm.llama_model_path,
            llama_ctx_size=4096,
            llama_gpu_layers=providers_llm.gpu_layers,
            llama_threads=providers_llm.threads,
            temperature=providers_llm.temperature,
            max_output_tokens=providers_llm.max_output_tokens,
            prop_enabled=stages.proposition_enabled,
            prop_provider=chunking.proposition_llm_provider,
            prop_gemini_model=chunking.proposition_gemini_model,
            prop_llama_model=_model_label(chunking.proposition_llama_model_path),
            prop_llama_model_path=chunking.proposition_llama_model_path,
            prop_llama_ctx_size=4096,
            prop_temperature=chunking.proposition_temperature,
            prop_max_output_tokens=chunking.proposition_max_output_tokens,
            prop_max_retries=chunking.proposition_max_retries,
            raptor_enabled=stages.raptor_enabled,
            raptor_cluster_max_tokens=chunking.raptor_cluster_max_tokens,
            raptor_summery_max_tokens=chunking.raptor_max_output_tokens,
            raptor_stop_chunk_count=chunking.raptor_stop_chunk_count,
            raptor_k_max=chunking.raptor_k_max,
            raptor_k_selection=chunking.raptor_k_selection,
            raptor_summery_provider=chunking.raptor_llm_provider,
            raptor_summery_gemini_model=chunking.raptor_gemini_model,
            raptor_summery_llama_model=_model_label(chunking.raptor_llama_model_path),
            raptor_summery_llama_model_path=chunking.raptor_llama_model_path,
            raptor_summery_llama_ctx_size=4096,
            raptor_summery_temperature=chunking.raptor_temperature,
            raptor_summery_max_retries=chunking.raptor_max_retries,
            clear_raw_data=refresh.clear_raw_data,
            clear_first_rec_chunk_data=refresh.clear_first_recursive_chunk_data,
            clear_second_rec_chunk_data=refresh.clear_second_recursive_chunk_data,
            clear_summery_chunk_data=refresh.clear_summary_chunk_data,
            clear_prop_chunk_data=refresh.clear_proposition_chunk_data,
            clear_raptor_chunk_data=refresh.clear_raptor_chunk_data,
            update_raw_data=refresh.update_raw_data,
            update_first_rec_chunk_data=refresh.update_first_recursive_chunk_data,
            update_second_rec_chunk_data=refresh.update_second_recursive_chunk_data,
            update_sparse_second_rec_chunk_data=(
                refresh.update_sparse_second_recursive_chunk_data
            ),
            update_summery_chunk_data=refresh.update_summary_chunk_data,
            update_prop_chunk_data=refresh.update_proposition_chunk_data,
            update_raptor_chunk_data=refresh.update_raptor_chunk_data,
            sudachi_mode=retrieval.sudachi_mode,
            sparse_bm25_k1=retrieval.sparse_bm25_k1,
            sparse_bm25_b=retrieval.sparse_bm25_b,
            sparse_use_normalized_form=retrieval.sparse_use_normalized_form,
            sparse_remove_symbols=retrieval.sparse_remove_symbols,
        )

    def _ensure_legacy_prompt_env_defaults(self) -> None:
        defaults = {
            "PROMPT_LLM_CHUNK_SYSTEM_PROMPT": (
                "You are a careful Japanese document chunking assistant."
            ),
            "PROMPT_PROPOSITION_CHUNK_TEMPLATE": (
                "次の本文を命題単位で分解し、JSON配列のみを出力してください。"
                "\\n- 1要素1命題"
                "\\n- 重複と空要素は禁止"
                "\\n\\n本文:\\n{text}"
            ),
            "PROMPT_SUMMERY_CHUNK_MESSAGES_TEMPLATE": (
                "次の会話ログを{target_characters}文字以内で要約してください。"
                "\\n重要な人物名・日時・数値は残してください。"
                "\\n\\n本文:\\n{text}"
            ),
            "PROMPT_SUMMERY_CHUNK_SHEETS_TEMPLATE": (
                "次の表データ由来本文を{target_characters}文字以内で要約してください。"
                "\\n元ファイル: {drive_path_display}"
                "\\n\\n本文:\\n{text}"
            ),
            "PROMPT_SUMMERY_CHUNK_DEFAULT_TEMPLATE": (
                "次の本文を{target_characters}文字以内で要約してください。"
                "\\n元ファイル: {drive_path_display}"
                "\\n\\n本文:\\n{text}"
            ),
            "PROMPT_RAPTOR_SUMMARY_SYSTEM_PROMPT": (
                "You summarize clustered Japanese chunks faithfully."
            ),
            "PROMPT_RAPTOR_SUMMARY_TEMPLATE": (
                "次の複数チャンクを統合し、重要情報を維持して要約してください。"
                "\\n目安トークン数: {target_tokens}"
                "\\n\\n本文:\\n{text}"
            ),
        }
        for key, value in defaults.items():
            if not os.getenv(key):
                os.environ[key] = value
        try:
            from kumc_agent.infra.indexing import config as legacy_config

            legacy_config.get_required_prompt_env.cache_clear()
        except Exception:
            logger.exception("Failed to clear legacy prompt env cache.")

    @staticmethod
    def _should_run_stage(*, stage_name: str, selected: set[str]) -> bool:
        if not selected:
            return True
        return stage_name in selected

    def _run_legacy_chunk_pipeline(
        self,
        *,
        legacy_cfg,
        selected: set[str],
        allow_cancel: bool,
        cancel_event: threading.Event | None,
    ) -> None:
        from kumc_agent.infra.indexing.chunking import (
            message_chunk_jsonl_dir,
            proposition_chunk_jsonl_dir,
            recursive_chunk_dir,
            recursive_chunk_jsonl_dir,
            sparse_chunk_jsonl_dir,
            summery_chunk_jsonl_dir,
        )
        from kumc_agent.infra.indexing.constants import (
            DOCS_SEPARATORS,
            MESSAGE_SEPARATORS,
            SHEETS_SEPARATORS,
        )
        from kumc_agent.infra.indexing.raptor import raptor_chunk_global

        refresh = self._runtime.indexing.refresh
        stages = self._runtime.indexing.stages
        chunking = self._runtime.indexing.chunking

        if self._should_run_stage(stage_name="first_recursive", selected=selected):
            if self._raw_messages_dir.exists():
                message_chunk_jsonl_dir(
                    raw_messages_dir=self._raw_messages_dir,
                    chunk_dir=self._first_rec_messages_dir,
                    chunk_size=chunking.first_recursive_chunk_size,
                    chunk_overlap=chunking.first_recursive_chunk_overlap,
                    stage="first_recursive",
                    skip_existing=not refresh.clear_first_recursive_chunk_data,
                    update_existing=refresh.update_first_recursive_chunk_data,
                    sync_deleted=refresh.update_first_recursive_chunk_data,
                )
            if self._raw_x_dir.exists():
                message_chunk_jsonl_dir(
                    raw_messages_dir=self._raw_x_dir,
                    chunk_dir=self._first_rec_x_dir,
                    chunk_size=chunking.first_recursive_chunk_size,
                    chunk_overlap=chunking.first_recursive_chunk_overlap,
                    stage="first_recursive",
                    skip_existing=not refresh.clear_first_recursive_chunk_data,
                    update_existing=refresh.update_first_recursive_chunk_data,
                    sync_deleted=refresh.update_first_recursive_chunk_data,
                )
            recursive_chunk_dir(
                raw_data_dir=self._raw_docs_dir,
                chunk_dir=self._first_rec_docs_dir,
                chunk_size=chunking.first_recursive_chunk_size,
                chunk_overlap=chunking.first_recursive_chunk_overlap,
                separators=DOCS_SEPARATORS,
                source_type="docs",
                stage="first_recursive",
                skip_existing=not refresh.clear_first_recursive_chunk_data,
                update_existing=refresh.update_first_recursive_chunk_data,
                sync_deleted=refresh.update_first_recursive_chunk_data,
            )
            recursive_chunk_dir(
                raw_data_dir=self._raw_sheets_dir,
                chunk_dir=self._first_rec_sheets_dir,
                chunk_size=chunking.first_recursive_chunk_size,
                chunk_overlap=chunking.first_recursive_chunk_overlap,
                separators=SHEETS_SEPARATORS,
                source_type="sheets",
                stage="first_recursive",
                file_extensions=(".csv",),
                skip_existing=not refresh.clear_first_recursive_chunk_data,
                update_existing=refresh.update_first_recursive_chunk_data,
                sync_deleted=refresh.update_first_recursive_chunk_data,
            )
            recursive_chunk_dir(
                raw_data_dir=self._raw_hatenablog_dir,
                chunk_dir=self._first_rec_hatenablog_dir,
                chunk_size=chunking.first_recursive_chunk_size,
                chunk_overlap=chunking.first_recursive_chunk_overlap,
                separators=DOCS_SEPARATORS,
                source_type="hatenablog",
                stage="first_recursive",
                file_extensions=(".md",),
                skip_existing=not refresh.clear_first_recursive_chunk_data,
                update_existing=refresh.update_first_recursive_chunk_data,
                sync_deleted=refresh.update_first_recursive_chunk_data,
            )
            recursive_chunk_dir(
                raw_data_dir=self._raw_crafters_colony_dir,
                chunk_dir=self._first_rec_crafters_colony_dir,
                chunk_size=chunking.first_recursive_chunk_size,
                chunk_overlap=chunking.first_recursive_chunk_overlap,
                separators=DOCS_SEPARATORS,
                source_type="crafters_colony",
                stage="first_recursive",
                file_extensions=(".md",),
                skip_existing=not refresh.clear_first_recursive_chunk_data,
                update_existing=refresh.update_first_recursive_chunk_data,
                sync_deleted=refresh.update_first_recursive_chunk_data,
            )
            recursive_chunk_dir(
                raw_data_dir=self._raw_notion_dir,
                chunk_dir=self._first_rec_notion_dir,
                chunk_size=chunking.first_recursive_chunk_size,
                chunk_overlap=chunking.first_recursive_chunk_overlap,
                separators=DOCS_SEPARATORS,
                source_type="notion",
                stage="first_recursive",
                file_extensions=(".md",),
                skip_existing=not refresh.clear_first_recursive_chunk_data,
                update_existing=refresh.update_first_recursive_chunk_data,
                sync_deleted=refresh.update_first_recursive_chunk_data,
            )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        if (
            stages.second_recursive_enabled
            and self._should_run_stage(stage_name="second_recursive", selected=selected)
        ):
            if self._first_rec_docs_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_docs_dir,
                    output_chunk_dir=self._second_rec_docs_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=DOCS_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_sheets_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_sheets_dir,
                    output_chunk_dir=self._second_rec_sheets_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=SHEETS_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_messages_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_messages_dir,
                    output_chunk_dir=self._second_rec_messages_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=MESSAGE_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_x_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_x_dir,
                    output_chunk_dir=self._second_rec_x_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=MESSAGE_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_hatenablog_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_hatenablog_dir,
                    output_chunk_dir=self._second_rec_hatenablog_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=DOCS_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_crafters_colony_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_crafters_colony_dir,
                    output_chunk_dir=self._second_rec_crafters_colony_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=DOCS_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_notion_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_notion_dir,
                    output_chunk_dir=self._second_rec_notion_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=DOCS_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._raw_vc_dir.exists():
                recursive_chunk_dir(
                    raw_data_dir=self._raw_vc_dir,
                    chunk_dir=self._second_rec_vc_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=MESSAGE_SEPARATORS,
                    source_type="vc_transcript",
                    stage="second_recursive",
                    file_extensions=(".txt",),
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        if (
            stages.sparse_second_recursive_enabled
            and self._should_run_stage(
                stage_name="sparse_second_recursive",
                selected=selected,
            )
        ):
            if stages.second_recursive_enabled:
                for input_dir, output_dir in (
                    (self._second_rec_docs_dir, self._sparse_second_rec_docs_dir),
                    (self._second_rec_sheets_dir, self._sparse_second_rec_sheets_dir),
                    (self._second_rec_messages_dir, self._sparse_second_rec_messages_dir),
                    (self._second_rec_x_dir, self._sparse_second_rec_x_dir),
                    (self._second_rec_hatenablog_dir, self._sparse_second_rec_hatenablog_dir),
                    (
                        self._second_rec_crafters_colony_dir,
                        self._sparse_second_rec_crafters_colony_dir,
                    ),
                    (self._second_rec_notion_dir, self._sparse_second_rec_notion_dir),
                ):
                    if not input_dir.exists():
                        continue
                    sparse_chunk_jsonl_dir(
                        input_chunk_dir=input_dir,
                        output_chunk_dir=output_dir,
                        config=legacy_cfg,
                        skip_existing=not refresh.clear_second_recursive_chunk_data,
                        update_existing=(
                            refresh.update_sparse_second_recursive_chunk_data
                        ),
                        sync_deleted=(
                            refresh.update_sparse_second_recursive_chunk_data
                        ),
                    )
            if self._second_rec_vc_dir.exists():
                sparse_chunk_jsonl_dir(
                    input_chunk_dir=self._second_rec_vc_dir,
                    output_chunk_dir=self._sparse_second_rec_vc_dir,
                    config=legacy_cfg,
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_sparse_second_recursive_chunk_data,
                    sync_deleted=refresh.update_sparse_second_recursive_chunk_data,
                )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        if stages.summary_enabled and self._should_run_stage(
            stage_name="summary",
            selected=selected,
        ):
            for input_dir, output_dir, second_dir in (
                (
                    self._first_rec_docs_dir,
                    self._summary_docs_dir,
                    self._second_rec_docs_dir if stages.second_recursive_enabled else None,
                ),
                (
                    self._first_rec_sheets_dir,
                    self._summary_sheets_dir,
                    self._second_rec_sheets_dir if stages.second_recursive_enabled else None,
                ),
                (
                    self._first_rec_messages_dir,
                    self._summary_messages_dir,
                    self._second_rec_messages_dir if stages.second_recursive_enabled else None,
                ),
                (
                    self._first_rec_x_dir,
                    self._summary_x_dir,
                    self._second_rec_x_dir if stages.second_recursive_enabled else None,
                ),
                (
                    self._first_rec_hatenablog_dir,
                    self._summary_hatenablog_dir,
                    self._second_rec_hatenablog_dir
                    if stages.second_recursive_enabled
                    else None,
                ),
                (
                    self._first_rec_crafters_colony_dir,
                    self._summary_crafters_colony_dir,
                    self._second_rec_crafters_colony_dir
                    if stages.second_recursive_enabled
                    else None,
                ),
                (
                    self._first_rec_notion_dir,
                    self._summary_notion_dir,
                    self._second_rec_notion_dir
                    if stages.second_recursive_enabled
                    else None,
                ),
            ):
                if not input_dir.exists():
                    continue
                summery_chunk_jsonl_dir(
                    input_chunk_dir=input_dir,
                    output_chunk_dir=output_dir,
                    second_chunk_dir=second_dir,
                    config=legacy_cfg,
                    skip_existing=not refresh.clear_summary_chunk_data,
                    update_existing=refresh.update_summary_chunk_data,
                    sync_deleted=refresh.update_summary_chunk_data,
                )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        if stages.proposition_enabled and self._should_run_stage(
            stage_name="proposition",
            selected=selected,
        ):
            if not stages.second_recursive_enabled:
                logger.warning(
                    "Proposition chunking is enabled but SECOND_REC is disabled. Skipping."
                )
            else:
                for input_dir, output_dir in (
                    (self._second_rec_docs_dir, self._prop_docs_dir),
                    (self._second_rec_sheets_dir, self._prop_sheets_dir),
                    (self._second_rec_hatenablog_dir, self._prop_hatenablog_dir),
                    (
                        self._second_rec_crafters_colony_dir,
                        self._prop_crafters_colony_dir,
                    ),
                    (self._second_rec_notion_dir, self._prop_notion_dir),
                ):
                    if not input_dir.exists():
                        continue
                    proposition_chunk_jsonl_dir(
                        input_chunk_dir=input_dir,
                        output_chunk_dir=output_dir,
                        config=legacy_cfg,
                        skip_existing=not refresh.clear_proposition_chunk_data,
                        update_existing=refresh.update_proposition_chunk_data,
                        sync_deleted=refresh.update_proposition_chunk_data,
                    )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        if stages.raptor_enabled and self._should_run_stage(
            stage_name="raptor",
            selected=selected,
        ):
            if stages.summary_enabled:
                raptor_input_dirs = [
                    self._summary_docs_dir,
                    self._summary_sheets_dir,
                    self._summary_hatenablog_dir,
                    self._summary_crafters_colony_dir,
                    self._summary_notion_dir,
                ]
            elif stages.second_recursive_enabled:
                raptor_input_dirs = [
                    self._second_rec_docs_dir,
                    self._second_rec_sheets_dir,
                    self._second_rec_hatenablog_dir,
                    self._second_rec_crafters_colony_dir,
                    self._second_rec_notion_dir,
                ]
            else:
                raptor_input_dirs = [
                    self._first_rec_docs_dir,
                    self._first_rec_sheets_dir,
                    self._first_rec_hatenablog_dir,
                    self._first_rec_crafters_colony_dir,
                    self._first_rec_notion_dir,
                ]
            raptor_chunk_global(
                input_chunk_dirs=raptor_input_dirs,
                output_chunk_dir=self._raptor_dir,
                config=legacy_cfg,
                skip_existing=not refresh.clear_raptor_chunk_data,
                update_existing=refresh.update_raptor_chunk_data,
                sync_deleted=refresh.update_raptor_chunk_data,
            )

    def _load_index_chunks_from_legacy_dirs(self, *, legacy_cfg) -> list[Chunk]:
        base_dirs: list[Path] = []
        if legacy_cfg.second_rec_enabled:
            base_dirs.extend(
                [
                    self._second_rec_docs_dir,
                    self._second_rec_sheets_dir,
                    self._second_rec_hatenablog_dir,
                    self._second_rec_crafters_colony_dir,
                    self._second_rec_notion_dir,
                ]
            )
            if self._second_rec_messages_dir.exists():
                base_dirs.append(self._second_rec_messages_dir)
            if self._second_rec_x_dir.exists():
                base_dirs.append(self._second_rec_x_dir)
        else:
            base_dirs.extend(
                [
                    self._first_rec_docs_dir,
                    self._first_rec_sheets_dir,
                    self._first_rec_hatenablog_dir,
                    self._first_rec_crafters_colony_dir,
                    self._first_rec_notion_dir,
                ]
            )
            if self._first_rec_messages_dir.exists():
                base_dirs.append(self._first_rec_messages_dir)
            if self._first_rec_x_dir.exists():
                base_dirs.append(self._first_rec_x_dir)

        chunks: list[Chunk] = []
        chunks.extend(self._load_legacy_chunks_from_dirs(base_dirs))
        if self._second_rec_vc_dir.exists():
            chunks.extend(self._load_legacy_chunks_from_dirs([self._second_rec_vc_dir]))
        if legacy_cfg.prop_enabled and legacy_cfg.second_rec_enabled:
            chunks.extend(
                self._load_legacy_chunks_from_dirs(
                    [
                        self._prop_docs_dir,
                        self._prop_sheets_dir,
                        self._prop_hatenablog_dir,
                        self._prop_crafters_colony_dir,
                        self._prop_notion_dir,
                    ]
                )
            )
        if legacy_cfg.raptor_enabled:
            chunks.extend(self._load_legacy_chunks_from_dirs([self._raptor_dir]))
        return chunks

    def _load_legacy_chunks_from_dirs(self, chunk_dirs: list[Path]) -> list[Chunk]:
        from kumc_agent.infra.indexing.chunks import load_chunks_from_dirs

        existing = [path for path in chunk_dirs if path.exists()]
        if not existing:
            return []
        legacy_chunks = load_chunks_from_dirs(existing)
        out: list[Chunk] = []
        for idx, legacy_chunk in enumerate(legacy_chunks):
            converted = self._legacy_chunk_to_domain_chunk(
                legacy_chunk=legacy_chunk,
                fallback_index=idx,
            )
            if converted is not None:
                out.append(converted)
        return out

    @staticmethod
    def _to_int(value: object, *, fallback: int) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return fallback
        return fallback

    def _legacy_chunk_uid(
        self,
        *,
        metadata: dict[str, object],
        text: str,
        fallback_index: int,
    ) -> str:
        source_type = str(metadata.get("source_type") or "").strip()
        source_name = str(
            metadata.get("drive_file_id")
            or metadata.get("source_file_name")
            or metadata.get("path")
            or ""
        ).strip()
        stage = str(metadata.get("chunk_stage") or "").strip()
        chunk_id = self._to_int(metadata.get("chunk_id"), fallback=fallback_index)
        raptor_level = str(metadata.get("raptor_level") or "")
        raptor_cluster = str(metadata.get("raptor_cluster_id") or "")
        return stable_hash(
            f"{source_type}|{source_name}|{stage}|{chunk_id}|{raptor_level}|"
            f"{raptor_cluster}|{text[:256]}"
        )

    def _legacy_chunk_to_domain_chunk(
        self,
        *,
        legacy_chunk,
        fallback_index: int,
    ) -> Chunk | None:
        text = str(getattr(legacy_chunk, "text", "") or "").strip()
        if not text:
            return None
        metadata = dict(getattr(legacy_chunk, "metadata", {}) or {})
        index = self._to_int(metadata.get("chunk_id"), fallback=fallback_index)
        source_type = str(metadata.get("source_type") or "").strip()
        source_name = str(
            metadata.get("drive_file_id")
            or metadata.get("source_file_name")
            or metadata.get("path")
            or ""
        ).strip()
        if not source_name:
            source_name = f"legacy:{source_type}:{index}"
        document_id = stable_hash(f"{source_type}:{source_name}")
        chunk_uid = self._legacy_chunk_uid(
            metadata=metadata,
            text=text,
            fallback_index=index,
        )
        metadata["chunk_uid"] = chunk_uid
        metadata.setdefault("chunk_id", index)
        return Chunk(
            id=chunk_uid,
            document_id=document_id,
            text=text,
            index=index,
            metadata=metadata,
        )

    def _chunk_embedding_text_for_dense(self, chunk: Chunk) -> str:
        from kumc_agent.infra.indexing.chunks import (
            Chunk as LegacyChunk,
            chunk_embedding_text,
        )

        try:
            return chunk_embedding_text(
                LegacyChunk(text=chunk.text, metadata=dict(chunk.metadata))
            )
        except Exception:
            logger.exception("Failed to build legacy embedding text. Falling back to chunk text.")
            return chunk.text

    def _build_keyword_inverted_indexes(self, *, legacy_cfg) -> None:
        try:
            from langchain_core.documents import Document as LangDocument
            from kumc_agent.infra.indexing.chunks import load_chunks_from_dirs
            from kumc_agent.infra.indexing.keyword_inverted_index import (
                KEYWORD_CORPUS_SECOND_REC_SPARSE,
                KEYWORD_CORPUS_SPARSE,
                KEYWORD_CORPUS_SPARSE_SECOND_REC,
                build_and_save_keyword_index,
                tokenize_sparse_doc,
            )
            from kumc_agent.infra.indexing.sparse_sources import (
                second_rec_chunk_dirs,
                sparse_chunk_dirs,
                sparse_second_rec_chunk_dirs,
            )
            from kumc_agent.infra.indexing.sparse_normalizer import (
                SparseNormalizer,
                SparseNormalizerConfig,
            )
        except Exception:
            logger.exception(
                "Legacy keyword inverted index dependencies are unavailable. "
                "Falling back to lightweight keyword payload."
            )
            self._build_keyword_indexes_payload(
                sparse_chunks=self._load_legacy_chunks_from_dirs(
                    [
                        self._sparse_second_rec_docs_dir,
                        self._sparse_second_rec_sheets_dir,
                        self._sparse_second_rec_messages_dir,
                        self._sparse_second_rec_x_dir,
                        self._sparse_second_rec_vc_dir,
                        self._sparse_second_rec_hatenablog_dir,
                        self._sparse_second_rec_crafters_colony_dir,
                        self._sparse_second_rec_notion_dir,
                    ]
                ),
                second_chunks=self._load_legacy_chunks_from_dirs(
                    [
                        self._second_rec_docs_dir,
                        self._second_rec_sheets_dir,
                        self._second_rec_messages_dir,
                        self._second_rec_x_dir,
                        self._second_rec_vc_dir,
                        self._second_rec_hatenablog_dir,
                        self._second_rec_crafters_colony_dir,
                        self._second_rec_notion_dir,
                    ]
                ),
            )
            return

        normalizer = SparseNormalizer(
            config=SparseNormalizerConfig(
                sudachi_mode=legacy_cfg.sudachi_mode,
                use_normalized_form=legacy_cfg.sparse_use_normalized_form,
                remove_symbols=legacy_cfg.sparse_remove_symbols,
                remove_stopwords=False,
            )
        )

        def _tokenize(doc: LangDocument) -> list[str]:
            return tokenize_sparse_doc(
                doc,
                sparse_stage="second_recursive_sparse",
                sudachi_tokenize=normalizer.normalize_tokens,
            )

        def _docs_for_dirs(chunk_dirs: list[Path]) -> list[LangDocument]:
            existing = [value for value in chunk_dirs if value.exists()]
            if not existing:
                return []
            chunks = load_chunks_from_dirs(existing)
            docs: list[LangDocument] = []
            for idx, chunk in enumerate(chunks):
                text = str(chunk.text or "").strip()
                if not text:
                    continue
                metadata = dict(chunk.metadata or {})
                metadata["chunk_uid"] = self._legacy_chunk_uid(
                    metadata=metadata,
                    text=text,
                    fallback_index=idx,
                )
                docs.append(LangDocument(page_content=text, metadata=metadata))
            return docs

        corpus_to_dirs = (
            (KEYWORD_CORPUS_SPARSE, sparse_chunk_dirs(legacy_cfg)),
            (KEYWORD_CORPUS_SPARSE_SECOND_REC, sparse_second_rec_chunk_dirs(legacy_cfg)),
            (KEYWORD_CORPUS_SECOND_REC_SPARSE, second_rec_chunk_dirs(legacy_cfg)),
        )
        for corpus_name, chunk_dirs in corpus_to_dirs:
            docs = _docs_for_dirs(chunk_dirs)
            build_and_save_keyword_index(
                index_dir=legacy_cfg.index_dir,
                corpus_name=corpus_name,
                docs=docs,
                tokenize_doc=_tokenize,
                k1=legacy_cfg.sparse_bm25_k1,
                b=legacy_cfg.sparse_bm25_b,
            )

    def _build_material_catalog_legacy(self, *, legacy_cfg) -> None:
        try:
            from kumc_agent.infra.indexing.material_catalog import (
                build_and_save_material_catalog,
            )

            build_and_save_material_catalog(legacy_cfg)
        except Exception:
            logger.exception(
                "Legacy material catalog build failed. Falling back to basic catalog."
            )
            self._build_material_catalog(documents=self._parse_documents_from_raw())

    def _load_or_build_first_chunks(
        self,
        *,
        documents: list[Document],
        chunk_size: int,
        chunk_overlap: int,
        should_update: bool,
        force: bool,
        selected: set[str],
        allow_cancel: bool,
        cancel_event: threading.Event | None,
    ) -> list[Chunk]:
        stage_name = "first_recursive"
        if not force and not should_update and self._stage_exists(self._first_rec_dir):
            return self._load_stage_chunks(self._first_rec_dir)
        if selected and stage_name not in selected and self._stage_exists(self._first_rec_dir):
            return self._load_stage_chunks(self._first_rec_dir)
        chunks: list[Chunk] = []
        for doc in documents:
            self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)
            pieces = self._split_text(
                doc.text,
                chunk_size=max(1, chunk_size),
                chunk_overlap=max(0, chunk_overlap),
            )
            for idx, piece in enumerate(pieces):
                text = piece.strip()
                if not text:
                    continue
                chunk_id = stable_hash(f"{doc.id}:first:{idx}:{text[:64]}")
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=doc.id,
                        text=text,
                        index=idx,
                        metadata={
                            "source_type": doc.source_type,
                            "source_name": doc.source_name,
                            "source_uri": doc.source_uri,
                            "source_file_name": doc.source_name,
                            "chunk_stage": "first_recursive",
                            "chunk_id": idx,
                            "updated_at": (
                                doc.updated_at.astimezone(timezone.utc).isoformat()
                                if doc.updated_at is not None
                                else ""
                            ),
                            **doc.metadata,
                        },
                    )
                )
        self._write_stage_chunks(self._first_rec_dir, chunks)
        return chunks

    def _load_or_build_second_chunks(
        self,
        *,
        first_chunks: list[Chunk],
        chunk_size: int,
        chunk_overlap: int,
        enabled: bool,
        should_update: bool,
        force: bool,
        selected: set[str],
        allow_cancel: bool,
        cancel_event: threading.Event | None,
    ) -> list[Chunk]:
        if not enabled:
            return []
        stage_name = "second_recursive"
        if not force and not should_update and self._stage_exists(self._second_rec_dir):
            return self._load_stage_chunks(self._second_rec_dir)
        if selected and stage_name not in selected and self._stage_exists(self._second_rec_dir):
            return self._load_stage_chunks(self._second_rec_dir)
        chunks: list[Chunk] = []
        out_index = 0
        for parent in first_chunks:
            self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)
            pieces = self._split_text(
                parent.text,
                chunk_size=max(1, chunk_size),
                chunk_overlap=max(0, chunk_overlap),
            )
            if len(pieces) == 1 and pieces[0].strip() == parent.text.strip():
                skip_parent_context = True
            else:
                skip_parent_context = False
            for piece in pieces:
                text = piece.strip()
                if not text:
                    continue
                metadata = dict(parent.metadata)
                metadata["chunk_stage"] = "second_recursive"
                metadata["parent_chunk_id"] = parent.metadata.get("chunk_id", parent.index)
                if skip_parent_context:
                    metadata["skip_parent_context"] = True
                metadata["chunk_id"] = out_index
                chunks.append(
                    Chunk(
                        id=stable_hash(f"{parent.id}:second:{out_index}:{text[:64]}"),
                        document_id=parent.document_id,
                        text=text,
                        index=out_index,
                        metadata=metadata,
                    )
                )
                out_index += 1
        self._write_stage_chunks(self._second_rec_dir, chunks)
        return chunks

    def _load_or_build_sparse_second_chunks(
        self,
        *,
        second_chunks: list[Chunk],
        enabled: bool,
        should_update: bool,
        force: bool,
        selected: set[str],
    ) -> list[Chunk]:
        if not enabled or not second_chunks:
            return []
        stage_name = "sparse_second_recursive"
        if not force and not should_update and self._stage_exists(self._sparse_second_rec_dir):
            return self._load_stage_chunks(self._sparse_second_rec_dir)
        if selected and stage_name not in selected and self._stage_exists(self._sparse_second_rec_dir):
            return self._load_stage_chunks(self._sparse_second_rec_dir)

        chunks: list[Chunk] = []
        for idx, chunk in enumerate(second_chunks):
            tokens = [token for token in (chunk.text or "").lower().split() if token]
            if not tokens:
                continue
            metadata = dict(chunk.metadata)
            metadata["chunk_stage"] = "second_recursive_sparse"
            metadata["chunk_id"] = idx
            text = " ".join(tokens)
            chunks.append(
                Chunk(
                    id=stable_hash(f"{chunk.id}:sparse:{idx}:{text[:64]}"),
                    document_id=chunk.document_id,
                    text=text,
                    index=idx,
                    metadata=metadata,
                )
            )
        self._write_stage_chunks(self._sparse_second_rec_dir, chunks)
        return chunks

    def _load_or_build_summary_chunks(
        self,
        *,
        first_chunks: list[Chunk],
        enabled: bool,
        target_characters: int,
        should_update: bool,
        force: bool,
        selected: set[str],
    ) -> list[Chunk]:
        if not enabled:
            return []
        stage_name = "summary"
        if not force and not should_update and self._stage_exists(self._summary_dir):
            return self._load_stage_chunks(self._summary_dir)
        if selected and stage_name not in selected and self._stage_exists(self._summary_dir):
            return self._load_stage_chunks(self._summary_dir)

        limit = max(32, target_characters)
        batch_size = max(1, int(self._runtime.indexing.chunking.summary_batch_size))
        total_batches = (len(first_chunks) + batch_size - 1) // batch_size
        if total_batches > 1:
            logger.info(
                "Generating summary chunks in %d batches (batch_size=%d).",
                total_batches,
                batch_size,
            )
        summaries = [""] * len(first_chunks)
        for batch_start in range(0, len(first_chunks), batch_size):
            batch = first_chunks[batch_start : batch_start + batch_size]
            max_workers = max(1, min(len(batch), batch_size))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: dict[Future[str], tuple[int, Chunk]] = {}
                for offset, source_chunk in enumerate(batch):
                    idx = batch_start + offset
                    futures[
                        executor.submit(
                            self._build_summary_text,
                            text=source_chunk.text or "",
                            target_characters=limit,
                        )
                    ] = (idx, source_chunk)
                for future in as_completed(futures):
                    idx, source_chunk = futures[future]
                    fallback = (source_chunk.text or "").strip()[:limit]
                    try:
                        summary_text = (future.result() or "").strip()
                    except Exception:
                        logger.exception(
                            "Summary chunk worker failed. Fallback to truncation."
                        )
                        summary_text = fallback
                    summaries[idx] = summary_text or fallback

        chunks: list[Chunk] = []
        for idx, chunk in enumerate(first_chunks):
            summary = summaries[idx]
            if not summary:
                continue
            metadata = dict(chunk.metadata)
            metadata["chunk_stage"] = "summary"
            metadata["parent_chunk_id"] = chunk.metadata.get("chunk_id", chunk.index)
            metadata["chunk_id"] = idx
            chunks.append(
                Chunk(
                    id=stable_hash(f"{chunk.id}:summary:{idx}:{summary[:64]}"),
                    document_id=chunk.document_id,
                    text=summary,
                    index=idx,
                    metadata=metadata,
                )
            )
        self._write_stage_chunks(self._summary_dir, chunks)
        return chunks

    def _build_summary_text(
        self,
        *,
        text: str,
        target_characters: int,
    ) -> str:
        source_text = (text or "").strip()
        if not source_text:
            return ""
        fallback = source_text[:target_characters]
        llm_provider = (
            self._runtime.indexing.chunking.summary_llm_provider or ""
        ).strip().lower()
        if llm_provider in {"", "none", "off", "disabled", "false", "0"}:
            return fallback
        if self._summary_llm is None:
            return fallback
        user_prompt = (
            f"次の本文を{target_characters}文字以内で要約してください。"
            "\n箇条書きにせず、重要な固有名詞・数値・日付は残してください。"
            f"\n\n本文:\n{source_text}"
        )
        try:
            summary = self._summary_llm.generate(
                system_prompt=(
                    "You summarize Japanese documents concisely and faithfully."
                ),
                user_prompt=user_prompt,
                temperature=self._runtime.indexing.chunking.summary_temperature,
                max_output_tokens=(
                    self._runtime.indexing.chunking.summary_max_output_tokens
                ),
            )
        except Exception:
            logger.exception(
                "Summary chunk generation failed. Fallback to truncation."
            )
            return fallback
        normalized = (summary or "").strip()
        if not normalized or self._is_llm_error_text(normalized):
            return fallback
        return normalized[:target_characters]

    @staticmethod
    def _is_llm_error_text(text: str) -> bool:
        return (
            "ローカルフォールバック回答" in text
            or text.endswith("回答生成に失敗しました。")
        )

    def _load_or_build_proposition_chunks(
        self,
        *,
        second_chunks: list[Chunk],
        enabled: bool,
        should_update: bool,
        force: bool,
        selected: set[str],
    ) -> list[Chunk]:
        if not enabled or not second_chunks:
            return []
        stage_name = "proposition"
        if not force and not should_update and self._stage_exists(self._prop_dir):
            return self._load_stage_chunks(self._prop_dir)
        if selected and stage_name not in selected and self._stage_exists(self._prop_dir):
            return self._load_stage_chunks(self._prop_dir)

        chunks: list[Chunk] = []
        out_index = 0
        for chunk in second_chunks:
            sentences = [
                sentence.strip()
                for sentence in re.split(r"[。\n]+", chunk.text or "")
                if sentence.strip()
            ]
            for sentence in sentences:
                metadata = dict(chunk.metadata)
                metadata["chunk_stage"] = "proposition"
                metadata["parent_chunk_id"] = chunk.metadata.get("chunk_id", chunk.index)
                metadata["chunk_id"] = out_index
                chunks.append(
                    Chunk(
                        id=stable_hash(f"{chunk.id}:prop:{out_index}:{sentence[:64]}"),
                        document_id=chunk.document_id,
                        text=sentence,
                        index=out_index,
                        metadata=metadata,
                    )
                )
                out_index += 1
        self._write_stage_chunks(self._prop_dir, chunks)
        return chunks

    def _load_or_build_raptor_chunks(
        self,
        *,
        source_chunks: list[Chunk],
        enabled: bool,
        should_update: bool,
        force: bool,
        selected: set[str],
    ) -> list[Chunk]:
        if not enabled or not source_chunks:
            return []
        stage_name = "raptor"
        if not force and not should_update and self._stage_exists(self._raptor_dir):
            return self._load_stage_chunks(self._raptor_dir)
        if selected and stage_name not in selected and self._stage_exists(self._raptor_dir):
            return self._load_stage_chunks(self._raptor_dir)

        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in source_chunks:
            grouped[chunk.document_id].append(chunk)

        out: list[Chunk] = []
        out_index = 0
        for document_id, chunks in grouped.items():
            merged = " ".join(chunk.text.strip() for chunk in chunks[:5] if chunk.text.strip())
            if not merged:
                continue
            summary = merged[:512]
            metadata = dict(chunks[0].metadata)
            metadata["chunk_stage"] = "raptor"
            metadata["chunk_id"] = out_index
            out.append(
                Chunk(
                    id=stable_hash(f"{document_id}:raptor:{out_index}:{summary[:64]}"),
                    document_id=document_id,
                    text=summary,
                    index=out_index,
                    metadata=metadata,
                )
            )
            out_index += 1
        self._write_stage_chunks(self._raptor_dir, out)
        return out

    @staticmethod
    def _compose_index_chunks(
        *,
        first_chunks: list[Chunk],
        second_chunks: list[Chunk],
        proposition_chunks: list[Chunk],
        raptor_chunks: list[Chunk],
        second_enabled: bool,
        proposition_enabled: bool,
        raptor_enabled: bool,
    ) -> list[Chunk]:
        chunks = list(second_chunks if second_enabled and second_chunks else first_chunks)
        if proposition_enabled and proposition_chunks:
            chunks.extend(proposition_chunks)
        if raptor_enabled and raptor_chunks:
            chunks.extend(raptor_chunks)
        return chunks

    def _build_keyword_indexes_payload(
        self,
        *,
        sparse_chunks: list[Chunk],
        second_chunks: list[Chunk],
    ) -> None:
        self._write_keyword_corpus(
            corpus_name=KEYWORD_CORPUS_SPARSE,
            chunks=sparse_chunks,
        )
        self._write_keyword_corpus(
            corpus_name=KEYWORD_CORPUS_SPARSE_SECOND_REC,
            chunks=sparse_chunks,
        )
        self._write_keyword_corpus(
            corpus_name=KEYWORD_CORPUS_SECOND_REC_SPARSE,
            chunks=second_chunks,
        )

    def _write_keyword_corpus(
        self,
        *,
        corpus_name: str,
        chunks: list[Chunk],
    ) -> None:
        keyword_dir = self._runtime.app.index_dir / "keyword"
        keyword_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_name": corpus_name,
            "docs": [
                {"page_content": chunk.text, "metadata": chunk.metadata}
                for chunk in chunks
            ],
        }
        path = keyword_dir / f"{corpus_name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    def _build_material_catalog(self, *, documents: list[Document]) -> None:
        materials: list[dict[str, object]] = []
        for doc in documents:
            source_key = str(doc.source_name or "").strip()
            if not source_key:
                continue
            canonical = Path(source_key).stem or source_key
            aliases = [canonical, source_key]
            materials.append(
                {
                    "material_id": f"{doc.source_type}:{source_key}",
                    "source_type": doc.source_type,
                    "source_key": source_key,
                    "canonical_name": canonical,
                    "aliases": aliases,
                    "raw_path": source_key,
                }
            )
        catalog = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "materials": materials,
        }
        path = self._runtime.app.index_dir / "material_catalog.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    def _parse_documents_from_raw(self) -> list[Document]:
        if not self._raw_dir.exists():
            return []

        source_specs: tuple[tuple[Path, set[str], str], ...] = (
            (self._raw_docs_dir, {".md"}, "docs"),
            (self._raw_sheets_dir, {".csv"}, "sheets"),
            (self._raw_messages_dir, {".jsonl"}, "messages"),
            (self._raw_x_dir, {".jsonl"}, "x_posts"),
            (self._raw_vc_dir, {".txt"}, "vc_transcript"),
            (self._raw_hatenablog_dir, {".md"}, "hatenablog"),
            (self._raw_crafters_colony_dir, {".md"}, "crafters_colony"),
            (self._raw_notion_dir, {".md"}, "notion"),
        )

        documents: list[Document] = []
        for root_dir, extensions, default_source_type in source_specs:
            if not root_dir.exists():
                continue
            for path in sorted(root_dir.rglob("*"), key=lambda value: str(value)):
                if not path.is_file():
                    continue
                if path.name.endswith((".meta.json", ".mtime.json", ".state.json")):
                    continue
                if path.suffix.lower() not in extensions:
                    continue
                text, extracted_meta, updated_at = self._read_raw_document(path)
                if not text.strip():
                    continue
                source_name = str(path.relative_to(self._raw_dir)).replace("\\", "/")
                source_type = str(extracted_meta.get("source_type") or "").strip().lower()
                if not source_type:
                    source_type = default_source_type
                doc_id = stable_hash(f"{source_type}:{source_name}")
                metadata = {"path": source_name, **extracted_meta}
                if "source_date" not in metadata:
                    inferred_source_date = self._derive_source_date(metadata)
                    if inferred_source_date:
                        metadata["source_date"] = inferred_source_date
                documents.append(
                    Document(
                        id=doc_id,
                        text=text,
                        source_type=source_type,
                        source_name=source_name,
                        source_uri="",
                        updated_at=updated_at,
                        metadata=metadata,
                    )
                )
        return documents

    @staticmethod
    def _read_raw_document(path: Path) -> tuple[str, dict[str, object], datetime | None]:
        sidecar_metadata = IndexingService._read_sidecar_metadata(path)
        file_updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            texts: list[str] = []
            latest_line_metadata: dict[str, object] = {}
            latest_line_updated_at: datetime | None = None
            with path.open("r", encoding="utf-8") as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(payload.get("text") or "").strip()
                    if text:
                        texts.append(text)
                    raw_metadata = payload.get("metadata")
                    if not isinstance(raw_metadata, dict):
                        continue
                    metadata = {
                        str(key): value
                        for key, value in raw_metadata.items()
                    }
                    candidate_updated_at = IndexingService._extract_updated_at(metadata)
                    if latest_line_updated_at is None:
                        latest_line_metadata = metadata
                        latest_line_updated_at = candidate_updated_at
                        continue
                    if (
                        candidate_updated_at is not None
                        and (
                            latest_line_updated_at is None
                            or candidate_updated_at >= latest_line_updated_at
                        )
                    ):
                        latest_line_metadata = metadata
                        latest_line_updated_at = candidate_updated_at

            merged_metadata = {
                **sidecar_metadata,
                **latest_line_metadata,
                "file_mtime": file_updated_at.isoformat(),
            }
            updated_at = latest_line_updated_at or IndexingService._extract_updated_at(merged_metadata) or file_updated_at
            merged_metadata["updated_at"] = updated_at.isoformat()
            return "\n".join(texts), merged_metadata, updated_at

        text = path.read_text(encoding="utf-8", errors="ignore")
        merged_metadata = {
            **sidecar_metadata,
            "file_mtime": file_updated_at.isoformat(),
        }
        updated_at = IndexingService._extract_updated_at(merged_metadata) or file_updated_at
        merged_metadata["updated_at"] = updated_at.isoformat()
        return text, merged_metadata, updated_at

    @staticmethod
    def _read_sidecar_metadata(path: Path) -> dict[str, object]:
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        if not sidecar.exists():
            return {}
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): value
            for key, value in payload.items()
        }

    @staticmethod
    def _extract_updated_at(metadata: dict[str, object]) -> datetime | None:
        for key in (
            "updated_at",
            "message_timestamp",
            "drive_modified_time",
            "hatenablog_updated_at",
            "hatenablog_created_at",
            "crafters_colony_published_at",
            "notion_last_edited_time",
            "created_at",
            "source_date",
            "first_message_date",
        ):
            value = str(metadata.get(key) or "").strip()
            if not value:
                continue
            parsed = IndexingService._parse_datetime(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _derive_source_date(metadata: dict[str, object]) -> str | None:
        updated_at = IndexingService._extract_updated_at(metadata)
        if updated_at is None:
            return None
        return updated_at.astimezone(timezone.utc).strftime("%Y/%m/%d")

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        raw = (value or "").strip()
        if not raw or raw == "不明":
            return None
        iso_value = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(iso_value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
            try:
                parsed = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                continue
        return None

    @staticmethod
    def _clear_dir_contents(target: Path) -> None:
        if not target.exists():
            return
        for path in target.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted((p for p in target.rglob("*") if p.is_dir()), reverse=True):
            try:
                path.rmdir()
            except OSError:
                continue

    @staticmethod
    def _stage_exists(stage_dir: Path) -> bool:
        return stage_dir.exists() and any(stage_dir.rglob("*.jsonl"))

    @staticmethod
    def _write_stage_chunks(stage_dir: Path, chunks: list[Chunk]) -> None:
        stage_dir.mkdir(parents=True, exist_ok=True)
        by_source: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            source_type = str(chunk.metadata.get("source_type") or "misc").strip().lower()
            if not source_type:
                source_type = "misc"
            by_source[source_type].append(chunk)
        for source_type, grouped in by_source.items():
            out_path = stage_dir / f"{source_type}.jsonl"
            with out_path.open("w", encoding="utf-8") as fw:
                for chunk in grouped:
                    fw.write(
                        json.dumps(
                            {
                                "id": chunk.id,
                                "document_id": chunk.document_id,
                                "text": chunk.text,
                                "index": chunk.index,
                                "metadata": chunk.metadata,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    @staticmethod
    def _load_stage_chunks(stage_dir: Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        if not stage_dir.exists():
            return chunks
        for path in sorted(stage_dir.rglob("*.jsonl")):
            with path.open("r", encoding="utf-8") as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    chunks.append(
                        Chunk(
                            id=str(payload["id"]),
                            document_id=str(payload["document_id"]),
                            text=str(payload["text"]),
                            index=int(payload["index"]),
                            metadata=dict(payload.get("metadata", {})),
                        )
                    )
        return chunks

    @staticmethod
    def _check_cancel(
        *,
        allow_cancel: bool,
        cancel_event: threading.Event | None,
    ) -> None:
        if not allow_cancel:
            return
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Index build cancelled.")

    @staticmethod
    def _split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
        value = (text or "").strip()
        if not value:
            return []
        if len(value) <= chunk_size:
            return [value]
        out: list[str] = []
        start = 0
        overlap = max(0, min(chunk_overlap, chunk_size - 1))
        while start < len(value):
            end = min(len(value), start + chunk_size)
            piece = value[start:end].strip()
            if piece:
                out.append(piece)
            if end >= len(value):
                break
            start = max(start + 1, end - overlap)
        return out
