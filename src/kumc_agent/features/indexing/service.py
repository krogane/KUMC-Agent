from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import logging
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

    def build(
        self,
        *,
        loaded_sources: int,
        full_rebuild: bool = False,
        stage_selection: tuple[str, ...] | None = None,
        allow_cancel: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> IndexBuildResult:
        selected = {value.strip() for value in (stage_selection or ()) if value.strip()}
        refresh = self._runtime.indexing.refresh
        chunking = self._runtime.indexing.chunking
        stages = self._runtime.indexing.stages

        self._apply_clear_flags(full_rebuild=full_rebuild)
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        documents = self._parse_documents_from_raw()
        self._storage.save_documents(documents)

        first_chunks = self._load_or_build_first_chunks(
            documents=documents,
            chunk_size=chunking.first_recursive_chunk_size,
            chunk_overlap=chunking.first_recursive_chunk_overlap,
            should_update=refresh.update_first_recursive_chunk_data,
            force=full_rebuild,
            selected=selected,
            allow_cancel=allow_cancel,
            cancel_event=cancel_event,
        )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        second_chunks = self._load_or_build_second_chunks(
            first_chunks=first_chunks,
            chunk_size=chunking.second_recursive_chunk_size,
            chunk_overlap=chunking.second_recursive_chunk_overlap,
            enabled=stages.second_recursive_enabled,
            should_update=refresh.update_second_recursive_chunk_data,
            force=full_rebuild,
            selected=selected,
            allow_cancel=allow_cancel,
            cancel_event=cancel_event,
        )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        sparse_second_chunks = self._load_or_build_sparse_second_chunks(
            second_chunks=second_chunks,
            enabled=stages.sparse_second_recursive_enabled,
            should_update=refresh.update_sparse_second_recursive_chunk_data,
            force=full_rebuild,
            selected=selected,
        )
        summary_chunks = self._load_or_build_summary_chunks(
            first_chunks=first_chunks,
            enabled=stages.summary_enabled,
            target_characters=chunking.summary_characters,
            should_update=refresh.update_summary_chunk_data,
            force=full_rebuild,
            selected=selected,
        )
        proposition_chunks = self._load_or_build_proposition_chunks(
            second_chunks=second_chunks,
            enabled=stages.proposition_enabled,
            should_update=refresh.update_proposition_chunk_data,
            force=full_rebuild,
            selected=selected,
        )
        raptor_chunks = self._load_or_build_raptor_chunks(
            source_chunks=(summary_chunks or second_chunks or first_chunks),
            enabled=stages.raptor_enabled,
            should_update=refresh.update_raptor_chunk_data,
            force=full_rebuild,
            selected=selected,
        )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        index_chunks = self._compose_index_chunks(
            first_chunks=first_chunks,
            second_chunks=second_chunks,
            proposition_chunks=proposition_chunks,
            raptor_chunks=raptor_chunks,
            second_enabled=stages.second_recursive_enabled,
            proposition_enabled=stages.proposition_enabled,
            raptor_enabled=stages.raptor_enabled,
        )
        self._storage.save_chunks(index_chunks)

        embeddings = self._embedder.embed_documents([chunk.text for chunk in index_chunks])
        self._faiss_index.build(chunks=index_chunks, embeddings=embeddings)
        self._bm25_index.build(index_chunks)

        self._build_keyword_indexes_payload(
            sparse_chunks=sparse_second_chunks,
            second_chunks=second_chunks,
        )
        self._build_material_catalog(documents=documents)

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
                thinking_level=self._runtime.indexing.chunking.summary_thinking_level,
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

        documents: list[Document] = []
        for path in sorted(self._raw_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".meta.json", ".mtime.json"}:
                continue
            text, extracted_meta, updated_at = self._read_raw_document(path)
            if not text.strip():
                continue
            source_type = path.parent.name
            source_name = str(path.relative_to(self._raw_dir))
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
