from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import base64
import html
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.operations import Asset, IndexingRun
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.infra.operations import OperationsRepository
from kumc_agent.utils.hashing import cosine_similarity_matrix, hashed_vector, stable_hash

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
_PUBLIC_SOURCE_KINDS = frozenset({"hatena", "hatenablog", "crafters_colony", "x", "x_posts"})
_PROTECTED_SOURCE_KINDS = frozenset({"discord", "google_drive", "drive"})
_DENIED_INDEX_STATUSES = frozenset({"deleted", "quarantined", "permission_lost"})
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"sk-[0-9A-Za-z_\-]{20,}"),
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*\bsrc=[\"'](?P<url>[^\"']+)[\"'][^>]*>", re.IGNORECASE)


@dataclass(frozen=True)
class ImageSearchConfig:
    limit: int = 8
    dense_top_k: int = 24
    feature_top_k: int = 16
    rrf_k: int = 60
    surrounding_text_char_limit: int = 1200
    ocr_text_char_limit: int = 800
    caption_model: str = ""
    ocr_model: str = ""
    feature_model: str = "openai/clip-vit-base-patch32"
    feature_dimensions: int = 512
    duplicate_group_limit: int = 1
    max_download_bytes: int = 8 * 1024 * 1024


@dataclass(frozen=True)
class ImageSearchRequest:
    query: str
    access_context: AccessContext = field(default_factory=AccessContext)
    limit: int | None = None
    source_filter: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageSearchResult:
    text: str
    detail_markdown: str
    assets: tuple[Asset, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ImageSourceCandidate:
    source_kind: str
    source_item_id: str
    title: str
    image_ref: str
    source_url: str = ""
    source_label: str = ""
    captured_at: datetime | None = None
    surrounding_text: str = ""
    image_index: int = 0
    access_scope: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _RankedAsset:
    asset: Asset
    rank: int
    score: float
    sources: tuple[str, ...]
    matched_fields: tuple[str, ...] = tuple()


class ImageAccessPolicy:
    def __init__(
        self,
        *,
        allowed_guild_ids: tuple[str, ...] = tuple(),
        admin_user_ids: tuple[str, ...] = tuple(),
    ) -> None:
        self._allowed_guild_ids = tuple(str(value) for value in allowed_guild_ids if str(value))
        self._admin_user_ids = tuple(str(value) for value in admin_user_ids if str(value))

    def allow_asset(self, asset: Asset, *, access: AccessContext | None) -> bool:
        metadata = dict(asset.metadata or {})
        if str(metadata.get("index_status") or "").strip().lower() in _DENIED_INDEX_STATUSES:
            return False
        return self.allow_scope(
            source_kind=asset.source_kind,
            access_scope=dict(asset.access_scope or {}),
            metadata=metadata,
            access=access,
        )

    def allow_scope(
        self,
        *,
        source_kind: str,
        access_scope: dict[str, Any],
        metadata: dict[str, Any],
        access: AccessContext | None,
    ) -> bool:
        normalized_source = _normalize_source_kind(source_kind)
        visibility = str(access_scope.get("visibility") or "").strip().lower()
        if not visibility:
            if normalized_source in _PUBLIC_SOURCE_KINDS:
                visibility = "public"
            elif normalized_source in _PROTECTED_SOURCE_KINDS:
                visibility = "guild"
            else:
                visibility = "admin"
        if visibility == "public":
            return True
        if access is None:
            return False

        allowed_guilds = set(self._allowed_guild_ids)
        admin_users = set(self._admin_user_ids)
        request_user_id = str(access.user_id or "").strip()
        is_admin = bool(access.is_admin) and (not admin_users or request_user_id in admin_users)
        if not is_admin and request_user_id and request_user_id in admin_users:
            is_admin = True
        if visibility == "admin":
            return is_admin
        if is_admin:
            return True

        request_guild_id = str(access.guild_id or "").strip()
        asset_guild_id = str(access_scope.get("guild_id") or metadata.get("guild_id") or "").strip()
        if visibility == "role":
            allowed_roles = {str(value) for value in access_scope.get("role_ids") or []}
            return bool(allowed_roles & set(access.role_ids))
        if visibility == "private":
            allowed_users = {str(value) for value in access_scope.get("user_ids") or []}
            return bool(request_user_id and request_user_id in allowed_users)
        if visibility != "guild":
            return False
        if request_guild_id:
            if allowed_guilds and request_guild_id not in allowed_guilds:
                return False
            return bool(asset_guild_id and asset_guild_id == request_guild_id)
        return False


class GeminiImageCaptioner:
    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "",
        prompt_path: Path | None = None,
        max_output_tokens: int = 512,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._prompt_path = prompt_path
        self._max_output_tokens = max_output_tokens

    def caption(self, *, image_path: Path, surrounding_text: str = "") -> tuple[str, dict[str, Any]]:
        if not self._api_key or not self._model or not image_path.exists():
            return "", {"caption_status": "fallback", "caption_error": "caption_model_unavailable"}
        try:
            from google import genai

            image_bytes = image_path.read_bytes()
            mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
            prompt = self._prompt_text().format(surrounding_text=surrounding_text[:2000])
            client = genai.Client(api_key=self._api_key)
            part = genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model=self._model,
                contents=[prompt, part],
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=self._max_output_tokens,
                ),
            )
            text = _mask_secret_text((getattr(response, "text", "") or "").strip())
            if not text:
                return "", {"caption_status": "fallback", "caption_error": "empty_caption"}
            return text, {"caption_status": "succeeded", "caption_model": self._model}
        except Exception as exc:  # pragma: no cover - depends on Gemini SDK/network
            logger.exception("Image caption generation failed: %s", image_path)
            return "", {"caption_status": "fallback", "caption_error": type(exc).__name__}

    def _prompt_text(self) -> str:
        if self._prompt_path and self._prompt_path.exists():
            return self._prompt_path.read_text(encoding="utf-8")
        return (
            "画像検索用の短い日本語説明文を作成してください。OCR結果は含めず、"
            "主対象、画面や資料の概要、検索に役立つ視覚特徴を簡潔に書いてください。\n\n"
            "周辺テキスト:\n{surrounding_text}"
        )


class LocalImageOcrExtractor:
    def __init__(self, *, model_path: str = "") -> None:
        self._model_path = model_path

    def extract(self, *, image_path: Path) -> tuple[str, dict[str, Any]]:
        if not self._model_path or not image_path.exists():
            return "", {"ocr_status": "skipped", "ocr_error": "ocr_model_unavailable"}
        try:
            from PIL import Image
            from kumc_agent.infra.loaders.google_drive_impl import (
                _extract_generated_text,
                _load_pdf_ocr_pipeline,
            )

            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
            try:
                ocr_pipeline = _load_pdf_ocr_pipeline(self._model_path)
                try:
                    result = ocr_pipeline(rgb_image, max_new_tokens=2048)
                except TypeError:
                    result = ocr_pipeline(rgb_image)
                text = _mask_secret_text(_extract_generated_text(result).strip())
                return text, {"ocr_status": "succeeded", "ocr_model": self._model_path}
            finally:
                rgb_image.close()
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            logger.exception("Image OCR failed: %s", image_path)
            return "", {"ocr_status": "failed", "ocr_error": type(exc).__name__}


class ImageFeatureExtractor:
    def __init__(self, *, model: str, dimensions: int) -> None:
        self._model = str(model or "").strip()
        self._dimensions = max(1, int(dimensions or 1))
        self._processor: Any | None = None
        self._model_obj: Any | None = None
        self._model_load_error = ""

    def vector_for_asset(self, asset: Asset) -> tuple[np.ndarray, dict[str, Any]]:
        metadata = dict(asset.metadata or {})
        image_path_raw = str(metadata.get("downloaded_image_path") or "").strip()
        image_path = Path(image_path_raw) if image_path_raw else None
        if image_path is not None and image_path.exists():
            external = self._external_vector(image_path)
            if external is not None:
                vector, model_name = external
                return vector, {
                    "feature_status": "succeeded",
                    "feature_model": model_name,
                    "feature_dimensions": self._dimensions,
                    "feature_vector_ref": str(metadata.get("feature_vector_ref") or f"image_search/features/{asset.id}"),
                }
            try:
                vector = _local_image_feature_vector(image_path=image_path, dimensions=self._dimensions)
                fallback_mode = self._model not in {"", "local_hash", "local_color"}
                return vector, {
                    "feature_status": "fallback" if fallback_mode else "succeeded",
                    "feature_model": self._model or "local_color",
                    "feature_fallback": "local_color" if fallback_mode else "",
                    "feature_error": (self._model_load_error or "external_feature_model_unavailable") if fallback_mode else "",
                    "feature_dimensions": self._dimensions,
                    "feature_vector_ref": str(metadata.get("feature_vector_ref") or f"image_search/features/{asset.id}"),
                }
            except Exception as exc:
                logger.debug("Local image feature extraction unavailable for %s: %s", image_path, exc)
                return self._hash_vector(asset, error=type(exc).__name__)
        return self._hash_vector(asset, error="image_path_unavailable")

    def _external_vector(self, image_path: Path) -> tuple[np.ndarray, str] | None:
        if not self._model or self._model in {"local_hash", "local_color"}:
            return None
        try:
            processor, model_obj = self._load_external_model()
            if processor is None or model_obj is None:
                return None
            from PIL import Image
            import torch

            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
            try:
                inputs = processor(images=rgb_image, return_tensors="pt")
                with torch.no_grad():
                    features = model_obj.get_image_features(**inputs)
                vector = features.detach().cpu().numpy()[0].astype(np.float32)
                return _resize_and_normalize_vector(vector, dimensions=self._dimensions), self._model
            finally:
                rgb_image.close()
        except Exception as exc:
            self._model_load_error = type(exc).__name__
            logger.debug("External image feature model unavailable: %s", exc)
            return None

    def _load_external_model(self) -> tuple[Any | None, Any | None]:
        if self._processor is not None and self._model_obj is not None:
            return self._processor, self._model_obj
        if self._model_load_error:
            return None, None
        try:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self._model, local_files_only=True)
            self._model_obj = CLIPModel.from_pretrained(self._model, local_files_only=True)
            self._model_obj.eval()
            return self._processor, self._model_obj
        except Exception as exc:
            self._model_load_error = type(exc).__name__
            logger.info(
                "Image feature model is not available locally; using fallback vectors: %s",
                self._model,
            )
            return None, None

    def _hash_vector(self, asset: Asset, *, error: str) -> tuple[np.ndarray, dict[str, Any]]:
        metadata = dict(asset.metadata or {})
        seed = "image-feature:" + ":".join(
            str(metadata.get(key) or "")
            for key in ("content_hash", "duplicate_group_id", "source_url", "source_label")
        )
        fallback_mode = self._model not in {"", "local_hash"}
        return hashed_vector(seed or asset.id, dimensions=self._dimensions), {
            "feature_status": "fallback" if fallback_mode else "succeeded",
            "feature_model": self._model or "local_hash",
            "feature_fallback": "metadata_hash" if fallback_mode else "",
            "feature_error": error if fallback_mode else "",
            "feature_dimensions": self._dimensions,
            "feature_vector_ref": str(metadata.get("feature_vector_ref") or f"image_search/features/{asset.id}"),
        }


class ImageAssetBuildService:
    def __init__(
        self,
        *,
        repository: OperationsRepository,
        raw_dir: Path,
        image_dir: Path,
        index_dir: Path,
        embedder: EmbedderPort,
        config: ImageSearchConfig,
        captioner: GeminiImageCaptioner | None = None,
        ocr: LocalImageOcrExtractor | None = None,
    ) -> None:
        self._repository = repository
        self._raw_dir = raw_dir
        self._image_dir = image_dir
        self._embedder = embedder
        self._index = _ImageSearchIndex(index_dir=index_dir, embedder=embedder, config=config)
        self._config = config
        self._captioner = captioner
        self._ocr = ocr

    def build_from_raw_sources(self, *, index_dir: Path | None = None) -> IndexingRun:
        target_index = (
            _ImageSearchIndex(index_dir=index_dir / "image_search", embedder=self._embedder, config=self._config)
            if index_dir is not None
            else self._index
        )
        seen = changed = skipped = failed = deleted = 0
        self._image_dir.mkdir(parents=True, exist_ok=True)
        current_asset_ids: set[str] = set()
        for candidate in self._scan_candidates():
            seen += 1
            try:
                asset = self._asset_from_candidate(candidate)
                current_asset_ids.add(asset.id)
                existing = self._repository.get_asset(asset.id)
                if existing and existing.metadata.get("source_fingerprint") == asset.metadata.get("source_fingerprint"):
                    skipped += 1
                    continue
                self._repository.save_asset(asset)
                changed += 1
            except Exception:
                failed += 1
                logger.exception("Failed to build image asset from %s", candidate.image_ref)
        for asset in self._repository.list_assets(query=""):
            if _normalize_source_kind(asset.source_kind) not in _all_image_source_kinds():
                continue
            if not asset.metadata.get("source_fingerprint") and asset.metadata.get("index_version") != "image-search-v1":
                continue
            if asset.id in current_asset_ids:
                continue
            if str(asset.metadata.get("index_status") or "active") in _DENIED_INDEX_STATUSES:
                continue
            self._repository.save_asset(
                replace(
                    asset,
                    metadata={
                        **asset.metadata,
                        "index_status": "deleted",
                        "deleted_reason": "source_image_missing",
                    },
                )
            )
            deleted += 1

        searchable_assets = [
            asset
            for asset in self._repository.list_assets(query="")
            if _normalize_source_kind(asset.source_kind) in _all_image_source_kinds()
            and str(asset.metadata.get("index_status") or "active") not in _DENIED_INDEX_STATUSES
        ]
        feature_metadata_by_asset = target_index.build(searchable_assets)
        for asset in searchable_assets:
            feature_metadata = feature_metadata_by_asset.get(asset.id)
            if not feature_metadata:
                continue
            metadata = {**dict(asset.metadata or {}), **feature_metadata}
            if metadata == dict(asset.metadata or {}):
                continue
            self._repository.save_asset(replace(asset, metadata=metadata))
        run = IndexingRun(
            id=stable_hash(f"image-search:{datetime.now(UTC).isoformat()}")[:32],
            source_kind="image_search",
            status="succeeded" if failed == 0 else "degraded",
            seen=seen,
            changed=changed,
            skipped=skipped,
            deleted=deleted,
            error="" if failed == 0 else f"{failed} image assets failed",
            metadata={
                "indexed_assets": len(searchable_assets),
                "failed": failed,
                "index_dir": str(target_index.index_dir),
            },
        )
        return self._repository.save_indexing_run(run)

    def _asset_from_candidate(self, candidate: _ImageSourceCandidate) -> Asset:
        local_path, content_hash, download_metadata = self._materialize_image(candidate)
        surrounding_text = _compact_text(candidate.surrounding_text, self._config.surrounding_text_char_limit)
        caption = ""
        caption_metadata: dict[str, Any] = {"caption_status": "fallback"}
        if self._captioner is not None and local_path is not None:
            caption, caption_metadata = self._captioner.caption(
                image_path=local_path,
                surrounding_text=surrounding_text,
            )
        if not caption:
            caption = surrounding_text[:240]
        ocr_text = ""
        ocr_metadata: dict[str, Any] = {"ocr_status": "skipped"}
        if self._ocr is not None and local_path is not None:
            ocr_text, ocr_metadata = self._ocr.extract(image_path=local_path)
        source_fingerprint = stable_hash(
            json.dumps(
                {
                    "image_ref": candidate.image_ref,
                    "content_hash": content_hash,
                    "surrounding_text": surrounding_text,
                    "caption": caption,
                    "ocr_text": ocr_text,
                    "access_scope": candidate.access_scope,
                    "source_url": candidate.source_url,
                    "source_label": candidate.source_label,
                    "source_created_at": candidate.captured_at.isoformat() if candidate.captured_at else "",
                    "metadata": candidate.metadata,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
        asset_id = stable_hash(
            f"asset:{candidate.source_kind}:{candidate.source_item_id}:{candidate.image_index}:{content_hash}"
        )[:32]
        metadata = {
            **candidate.metadata,
            **download_metadata,
            **caption_metadata,
            **ocr_metadata,
            "caption": caption,
            "ocr_text": _compact_text(ocr_text, 12000),
            "surrounding_text": surrounding_text,
            "source_url": candidate.source_url,
            "source_label": candidate.source_label,
            "source_created_at": candidate.captured_at.isoformat() if candidate.captured_at else "",
            "image_index": candidate.image_index,
            "content_hash": content_hash,
            "duplicate_group_id": f"image-content:{content_hash}",
            "feature_vector_ref": f"image_search/features/{asset_id}",
            "feature_model": self._config.feature_model,
            "feature_status": "pending",
            "source_fingerprint": source_fingerprint,
            "index_version": "image-search-v1",
            "index_status": "active",
        }
        return Asset(
            id=asset_id,
            source_kind=_normalize_source_kind(candidate.source_kind),
            source_item_id=candidate.source_item_id,
            title=candidate.title,
            description=caption,
            uri=str(local_path) if local_path is not None else candidate.image_ref,
            media_type=mimetypes.guess_type(candidate.image_ref)[0] or "image",
            captured_at=candidate.captured_at,
            access_scope=candidate.access_scope,
            rights_status="unknown",
            contains_people=False,
            metadata=_mask_metadata(metadata),
        )

    def _materialize_image(self, candidate: _ImageSourceCandidate) -> tuple[Path | None, str, dict[str, Any]]:
        ref = candidate.image_ref.strip()
        fallback_refs_raw = candidate.metadata.get("fallback_image_refs")
        fallback_refs = (
            [str(value).strip() for value in fallback_refs_raw if str(value).strip()]
            if isinstance(fallback_refs_raw, list)
            else []
        )
        refs = list(dict.fromkeys([ref, *fallback_refs]))
        metadata: dict[str, Any] = {"original_image_ref": ref}
        if any(_is_http_url(item) for item in refs):
            errors: list[dict[str, str]] = []
            for image_ref in refs:
                if not _is_http_url(image_ref):
                    continue
                downloaded = self._download_image_ref(
                    image_ref=image_ref,
                    source_kind=candidate.source_kind,
                )
                if downloaded is None:
                    errors.append({"ref": image_ref, "error": "download_failed"})
                    continue
                data, path = downloaded
                content_hash = stable_hash(data.hex())
                metadata.update(
                    {
                        "download_status": "succeeded",
                        "downloaded_image_path": str(path),
                        "downloaded_image_ref": image_ref,
                    }
                )
                if image_ref != ref:
                    metadata["download_fallback_used"] = True
                if errors:
                    metadata["download_attempt_errors"] = errors
                return path, content_hash, metadata
            logger.warning("Failed to download image refs %s", refs)
            metadata.update({"download_status": "failed", "download_attempt_errors": errors})
            return None, stable_hash(ref), metadata
        path = Path(ref)
        if path.exists():
            data = path.read_bytes()
            metadata.update({"download_status": "local", "downloaded_image_path": str(path)})
            return path, stable_hash(data.hex()), metadata
        metadata.update({"download_status": "missing"})
        return None, stable_hash(ref), metadata

    def _download_image_ref(self, *, image_ref: str, source_kind: str) -> tuple[bytes, Path] | None:
        try:
            data, content_type = _download_bytes(image_ref, max_bytes=self._config.max_download_bytes)
            content_hash = stable_hash(data.hex())
            suffix = _suffix_from_content_type(content_type) or Path(image_ref.split("?", 1)[0]).suffix or ".img"
            path = self._image_dir / _normalize_source_kind(source_kind) / f"{content_hash[:24]}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
            return data, path
        except Exception as exc:
            logger.warning("Failed to download image %s: %s", image_ref, exc)
            return None

    def _scan_candidates(self) -> list[_ImageSourceCandidate]:
        candidates: list[_ImageSourceCandidate] = []
        candidates.extend(_scan_discord_images(self._raw_dir))
        candidates.extend(_scan_google_drive_images(self._raw_dir))
        candidates.extend(_scan_x_images(self._raw_dir))
        candidates.extend(_scan_article_images(self._raw_dir / "hatenablog", source_kind="hatena"))
        candidates.extend(_scan_article_images(self._raw_dir / "crafters_colony", source_kind="crafters_colony"))
        return _dedupe_candidates(candidates)


class ImageSearchService:
    def __init__(
        self,
        *,
        repository: OperationsRepository,
        embedder: EmbedderPort,
        index_dir: Path,
        config: ImageSearchConfig | None = None,
        allowed_guild_ids: tuple[str, ...] = tuple(),
        admin_user_ids: tuple[str, ...] = tuple(),
    ) -> None:
        self._repository = repository
        self._config = config or ImageSearchConfig()
        self._policy = ImageAccessPolicy(
            allowed_guild_ids=allowed_guild_ids,
            admin_user_ids=admin_user_ids,
        )
        self._index = _ImageSearchIndex(index_dir=index_dir, embedder=embedder, config=self._config)

    def search(self, request: ImageSearchRequest) -> ImageSearchResult:
        query = request.query.strip()
        limit = max(1, int(request.limit or self._config.limit))
        source_filter = {
            normalized
            for value in request.source_filter
            if (normalized := _normalize_source_kind(value)) not in {"", "all", "image"}
        }
        all_assets = self._candidate_assets(
            access=request.access_context,
            source_filter=source_filter,
        )
        metadata: dict[str, Any] = {
            "route": "image_search",
            "query": query,
            "source_filter": sorted(source_filter),
            **dict(request.metadata or {}),
        }
        if not all_assets:
            metadata["candidate_count"] = 0
            return ImageSearchResult(
                text="画像候補は 0 件です。",
                detail_markdown="# Image Search\n\n該当する画像候補は登録されていません。",
                assets=tuple(),
                metadata=metadata,
            )

        degraded_reasons: list[str] = []
        dense = self._index.search_dense(
            query=query,
            allowed_asset_ids={asset.id for asset in all_assets},
            top_k=max(limit, self._config.dense_top_k),
        )
        if not dense:
            degraded_reasons.append("dense_index_unavailable")
            dense = _fallback_keyword_results(query=query, assets=all_assets)
        feature = self._index.search_similar_features(
            seed_asset_ids=tuple(asset_id for asset_id, _score in dense[: max(1, limit)]),
            allowed_asset_ids={asset.id for asset in all_assets},
            top_k=self._config.feature_top_k,
        )
        if not self._index.has_feature_index():
            metadata["feature_search"] = "unavailable"
            degraded_reasons.append("image_feature_unavailable")
        ranked_ids = _rrf_merge(
            ranked_lists=[
                ("dense", [asset_id for asset_id, _ in dense]),
                ("feature", [asset_id for asset_id, _ in feature]),
            ],
            k=self._config.rrf_k,
        )
        by_id = {asset.id: asset for asset in all_assets}
        dense_scores = dict(dense)
        feature_scores = dict(feature)
        ranked: list[_RankedAsset] = []
        duplicate_counts: dict[str, int] = {}
        for rank, (asset_id, score, source_names) in enumerate(ranked_ids, start=1):
            asset = by_id.get(asset_id)
            if asset is None:
                continue
            if not self._policy.allow_asset(asset, access=request.access_context):
                continue
            duplicate_group_id = _duplicate_group_id(asset)
            if self._config.duplicate_group_limit > 0:
                duplicate_count = duplicate_counts.get(duplicate_group_id, 0)
                if duplicate_count >= self._config.duplicate_group_limit:
                    continue
                duplicate_counts[duplicate_group_id] = duplicate_count + 1
            feature_status = str(asset.metadata.get("feature_status") or "").strip().lower()
            if feature_status and feature_status != "succeeded":
                degraded_reasons.append("image_feature_fallback")
            ranked.append(
                _RankedAsset(
                    asset=asset,
                    rank=rank,
                    score=score,
                    sources=source_names,
                    matched_fields=_matched_fields(query=query, asset=asset),
                )
            )
            if len(ranked) >= limit:
                break

        degraded_reasons = list(dict.fromkeys(degraded_reasons))
        assets = tuple(
            _asset_for_output(
                item.asset,
                rank=item.rank,
                score=item.score,
                sources=item.sources,
                matched_fields=item.matched_fields,
                dense_score=dense_scores.get(item.asset.id),
                feature_score=feature_scores.get(item.asset.id),
                config=self._config,
            )
            for item in ranked
        )
        metadata.update(
            {
                "candidate_count": len(assets),
                "degraded": bool(degraded_reasons),
                "degraded_reason": ",".join(degraded_reasons),
                "search_results": [
                    {
                        "asset_id": item.asset.id,
                        "rank": item.rank,
                        "score": item.score,
                        "sources": list(item.sources),
                        "matched_fields": list(item.matched_fields),
                    }
                    for item in ranked
                ],
            }
        )
        return ImageSearchResult(
            text=f"画像候補は {len(assets)} 件です。再利用可否はこの結果では判断しません。",
            detail_markdown=_format_image_search_detail(assets),
            assets=assets,
            metadata=metadata,
        )

    def _candidate_assets(
        self,
        *,
        access: AccessContext,
        source_filter: set[str],
    ) -> list[Asset]:
        assets = []
        for asset in self._repository.list_assets(query=""):
            source = _normalize_source_kind(asset.source_kind)
            if source_filter and source not in source_filter:
                continue
            if source not in _all_image_source_kinds():
                continue
            if self._policy.allow_asset(asset, access=access):
                assets.append(asset)
        return assets


class _ImageSearchIndex:
    def __init__(self, *, index_dir: Path, embedder: EmbedderPort, config: ImageSearchConfig) -> None:
        self.index_dir = index_dir
        self._embedder = embedder
        self._config = config
        self._feature_extractor = ImageFeatureExtractor(
            model=config.feature_model,
            dimensions=config.feature_dimensions,
        )
        self._text_vectors_path = self.index_dir / "image_text_vectors.npy"
        self._feature_vectors_path = self.index_dir / "image_feature_vectors.npy"
        self._items_path = self.index_dir / "image_assets.jsonl"
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def build(self, assets: list[Asset]) -> dict[str, dict[str, Any]]:
        texts = [_asset_embedding_text(asset) for asset in assets]
        text_vectors = self._embedder.embed_documents(texts) if texts else np.empty((0, 1), dtype=np.float32)
        feature_metadata_by_asset: dict[str, dict[str, Any]] = {}
        feature_rows: list[np.ndarray] = []
        for asset in assets:
            vector, feature_metadata = self._feature_extractor.vector_for_asset(asset)
            feature_rows.append(vector)
            feature_metadata_by_asset[asset.id] = feature_metadata
        feature_vectors = np.vstack(feature_rows) if feature_rows else np.empty((0, self._config.feature_dimensions), dtype=np.float32)
        np.save(self._text_vectors_path, text_vectors.astype(np.float32))
        np.save(self._feature_vectors_path, feature_vectors.astype(np.float32))
        with self._items_path.open("w", encoding="utf-8") as fw:
            for asset in assets:
                fw.write(
                    json.dumps(
                        {
                            "asset_id": asset.id,
                            "duplicate_group_id": _duplicate_group_id(asset),
                            **feature_metadata_by_asset.get(asset.id, {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return feature_metadata_by_asset

    def has_feature_index(self) -> bool:
        if not self._feature_vectors_path.exists() or not self._items_path.exists():
            return False
        asset_ids = self._load_asset_ids()
        if not asset_ids:
            return False
        try:
            matrix = np.load(self._feature_vectors_path)
        except Exception:
            return False
        return matrix.size > 0 and matrix.shape[0] >= len(asset_ids)

    def search_dense(
        self,
        *,
        query: str,
        allowed_asset_ids: set[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        asset_ids = self._load_asset_ids()
        if not asset_ids or not self._text_vectors_path.exists():
            return []
        matrix = np.load(self._text_vectors_path)
        if matrix.size == 0:
            return []
        query_vector = self._embedder.embed_query(query)
        scores = cosine_similarity_matrix(query_vector, matrix)
        order = np.argsort(-scores)[: max(0, top_k)]
        out: list[tuple[str, float]] = []
        for idx in order:
            pos = int(idx)
            if pos >= len(asset_ids):
                continue
            asset_id = asset_ids[pos]
            if asset_id in allowed_asset_ids:
                out.append((asset_id, float(scores[pos])))
        return out

    def search_similar_features(
        self,
        *,
        seed_asset_ids: tuple[str, ...],
        allowed_asset_ids: set[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        asset_ids = self._load_asset_ids()
        if not asset_ids or not seed_asset_ids or not self._feature_vectors_path.exists():
            return []
        matrix = np.load(self._feature_vectors_path)
        if matrix.size == 0:
            return []
        seed_positions = [asset_ids.index(asset_id) for asset_id in seed_asset_ids if asset_id in asset_ids]
        if not seed_positions:
            return []
        query_vector = np.mean(matrix[seed_positions], axis=0)
        scores = cosine_similarity_matrix(query_vector, matrix)
        order = np.argsort(-scores)[: max(0, top_k + len(seed_positions))]
        out: list[tuple[str, float]] = []
        for idx in order:
            pos = int(idx)
            if pos >= len(asset_ids):
                continue
            asset_id = asset_ids[pos]
            if asset_id in seed_asset_ids or asset_id not in allowed_asset_ids:
                continue
            out.append((asset_id, float(scores[pos])))
            if len(out) >= top_k:
                break
        return out

    def _load_asset_ids(self) -> list[str]:
        if not self._items_path.exists():
            return []
        out: list[str] = []
        with self._items_path.open("r", encoding="utf-8") as fr:
            for line in fr:
                if not line.strip():
                    continue
                payload = json.loads(line)
                out.append(str(payload.get("asset_id") or ""))
        return [asset_id for asset_id in out if asset_id]


def _scan_discord_images(raw_dir: Path) -> list[_ImageSourceCandidate]:
    candidates: list[_ImageSourceCandidate] = []
    root = raw_dir / "messages"
    if not root.exists():
        return candidates
    for path in root.glob("**/*.jsonl"):
        for payload in _read_jsonl(path):
            text = str(payload.get("text") or "")
            metadata = dict(payload.get("metadata") or {})
            attachments = metadata.get("attachments") or payload.get("attachments") or []
            if not isinstance(attachments, list):
                continue
            for index, item in enumerate(attachments):
                if not isinstance(item, dict):
                    continue
                primary_url = str(item.get("url") or "").strip()
                proxy_url = str(item.get("proxy_url") or "").strip()
                url = primary_url or proxy_url
                content_type = str(item.get("content_type") or "").lower()
                filename = str(item.get("filename") or "").strip()
                if not _looks_like_image(url, filename=filename, content_type=content_type):
                    continue
                message_id = str(metadata.get("message_id") or payload.get("id") or stable_hash(url)[:16])
                guild_id = str(metadata.get("guild_id") or "")
                channel_id = str(metadata.get("channel_id") or "")
                source_url = _discord_message_url(guild_id=guild_id, channel_id=channel_id, message_id=message_id)
                candidates.append(
                    _ImageSourceCandidate(
                        source_kind="discord",
                        source_item_id=message_id,
                        title=filename or f"Discord image {message_id}",
                        image_ref=url,
                        source_url=source_url,
                        source_label=str(metadata.get("channel_name") or "Discord"),
                        captured_at=_dt_from(metadata.get("message_timestamp")),
                        surrounding_text=_join_nonempty(
                            text,
                            str(metadata.get("author_name") or ""),
                            str(metadata.get("channel_name") or ""),
                        ),
                        image_index=index,
                        access_scope={"visibility": "guild", "guild_id": guild_id},
                        metadata={
                            "guild_id": guild_id,
                            "channel_id": channel_id,
                            "message_id": message_id,
                            "source_type": "discord_message",
                            "source_kind": "discord",
                            "attachment_id": str(item.get("id") or ""),
                            "fallback_image_refs": [
                                ref
                                for ref in (proxy_url, primary_url)
                                if ref and ref != url
                            ],
                        },
                    )
                )
    return candidates


def _scan_google_drive_images(raw_dir: Path) -> list[_ImageSourceCandidate]:
    candidates: list[_ImageSourceCandidate] = []
    root = raw_dir / "images" / "google_drive"
    if root.exists():
        for path in root.glob("**/*"):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            metadata = _read_sidecar(path)
            source_item_id = str(metadata.get("drive_file_id") or path.stem)
            candidates.append(
                _ImageSourceCandidate(
                    source_kind="google_drive",
                    source_item_id=source_item_id,
                    title=str(metadata.get("drive_name") or path.name),
                    image_ref=str(path),
                    source_url=str(metadata.get("drive_url") or ""),
                    source_label=str(metadata.get("drive_path") or "Google Drive"),
                    captured_at=_dt_from(metadata.get("drive_modified_time")),
                    surrounding_text=str(
                        metadata.get("surrounding_text")
                        or metadata.get("drive_path")
                        or metadata.get("drive_name")
                        or ""
                    ),
                    image_index=int(metadata.get("image_index") or 0),
                    access_scope={"visibility": "guild", "guild_id": str(metadata.get("guild_id") or "")},
                    metadata={"source_type": "docs", "source_kind": "google_drive", **metadata},
                )
            )
    for path in (raw_dir / "docs").glob("*.md") if (raw_dir / "docs").exists() else []:
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata = _read_sidecar(path)
        for index, image_ref, context in _extract_image_refs(text, base_url=str(metadata.get("drive_url") or "")):
            candidates.append(
                _ImageSourceCandidate(
                    source_kind="google_drive",
                    source_item_id=str(metadata.get("drive_file_id") or path.stem),
                    title=str(metadata.get("drive_name") or path.name),
                    image_ref=image_ref,
                    source_url=str(metadata.get("drive_url") or ""),
                    source_label=str(metadata.get("drive_path") or "Google Drive"),
                    captured_at=_dt_from(metadata.get("drive_modified_time")),
                    surrounding_text=context,
                    image_index=index,
                    access_scope={"visibility": "guild", "guild_id": str(metadata.get("guild_id") or "")},
                    metadata={"source_type": "docs", "source_kind": "google_drive", **metadata},
                )
            )
    return candidates


def _scan_x_images(raw_dir: Path) -> list[_ImageSourceCandidate]:
    candidates: list[_ImageSourceCandidate] = []
    path = raw_dir / "x" / "posts.jsonl"
    if not path.exists():
        return candidates
    for payload in _read_jsonl(path):
        record = dict(payload.get("record") or payload)
        text = str(record.get("text") or payload.get("text") or "")
        metadata = dict(record.get("metadata") or payload.get("metadata") or {})
        media_urls = metadata.get("x_media_urls") or record.get("media_urls") or []
        if isinstance(media_urls, str):
            media_urls = [media_urls]
        if not isinstance(media_urls, list):
            continue
        post_id = str(metadata.get("x_post_id") or payload.get("id") or "")
        for index, url in enumerate(media_urls):
            image_ref = str(url).strip()
            if not _looks_like_image(image_ref):
                continue
            candidates.append(
                _ImageSourceCandidate(
                    source_kind="x",
                    source_item_id=post_id or stable_hash(image_ref)[:16],
                    title=f"X post {post_id}" if post_id else "X image",
                    image_ref=image_ref,
                    source_url=str(metadata.get("x_post_url") or ""),
                    source_label=str(metadata.get("x_author_handle") or "X"),
                    captured_at=_dt_from(metadata.get("message_timestamp")),
                    surrounding_text=text,
                    image_index=index,
                    access_scope={"visibility": "public"},
                    metadata={"source_type": "x_posts", "source_kind": "x", **metadata},
                )
            )
    return candidates


def _scan_article_images(root: Path, *, source_kind: str) -> list[_ImageSourceCandidate]:
    candidates: list[_ImageSourceCandidate] = []
    if not root.exists():
        return candidates
    for path in root.glob("*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata = _read_sidecar(path)
        source_url = _article_source_url(source_kind=source_kind, metadata=metadata)
        title = str(
            metadata.get("hatenablog_title")
            or metadata.get("crafters_colony_title")
            or _first_heading(text)
            or path.stem
        )
        for index, image_ref, context in _extract_image_refs(text, base_url=source_url):
            candidates.append(
                _ImageSourceCandidate(
                    source_kind=source_kind,
                    source_item_id=str(
                        metadata.get("hatenablog_entry_id")
                        or metadata.get("crafters_colony_article_url")
                        or source_url
                        or path.stem
                    ),
                    title=title,
                    image_ref=image_ref,
                    source_url=source_url,
                    source_label=title,
                    captured_at=_dt_from(
                        metadata.get("hatenablog_updated_at")
                        or metadata.get("hatenablog_published_at")
                        or metadata.get("crafters_colony_published_at")
                    ),
                    surrounding_text=context,
                    image_index=index,
                    access_scope={"visibility": "public"},
                    metadata={"source_type": source_kind, "source_kind": source_kind, **metadata},
                )
            )
    return candidates


def _extract_image_refs(text: str, *, base_url: str = "") -> list[tuple[int, str, str]]:
    refs: list[tuple[int, str, str]] = []
    for match in _MARKDOWN_IMAGE_RE.finditer(text):
        url = html.unescape(match.group("url").strip())
        if url:
            refs.append((len(refs), urljoin(base_url, url), _context_around(text, match.start())))
    for match in _HTML_IMAGE_RE.finditer(text):
        url = html.unescape(match.group("url").strip())
        if url:
            refs.append((len(refs), urljoin(base_url, url), _context_around(text, match.start())))
    return refs


def _dedupe_candidates(candidates: list[_ImageSourceCandidate]) -> list[_ImageSourceCandidate]:
    seen: set[tuple[str, str, int, str]] = set()
    out: list[_ImageSourceCandidate] = []
    for item in candidates:
        key = (
            _normalize_source_kind(item.source_kind),
            item.source_item_id,
            item.image_index,
            item.image_ref,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _fallback_keyword_results(*, query: str, assets: list[Asset]) -> list[tuple[str, float]]:
    needle = query.strip().lower()
    scored: list[tuple[str, float]] = []
    for asset in assets:
        text = _asset_embedding_text(asset).lower()
        if not needle:
            score = 0.1
        elif needle in text:
            score = 1.0 + text.count(needle) * 0.05
        else:
            terms = [term for term in re.split(r"\s+", needle) if term]
            hits = sum(1 for term in terms if term in text)
            score = hits / max(1, len(terms))
        if score > 0:
            scored.append((asset.id, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def _rrf_merge(
    *,
    ranked_lists: list[tuple[str, list[str]]],
    k: int,
) -> list[tuple[str, float, tuple[str, ...]]]:
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for name, values in ranked_lists:
        for rank, asset_id in enumerate(values, start=1):
            scores[asset_id] = scores.get(asset_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(asset_id, []).append(name)
    return [
        (asset_id, score, tuple(dict.fromkeys(sources.get(asset_id, []))))
        for asset_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def _asset_for_output(
    asset: Asset,
    *,
    rank: int,
    score: float,
    sources: tuple[str, ...],
    matched_fields: tuple[str, ...],
    dense_score: float | None,
    feature_score: float | None,
    config: ImageSearchConfig,
) -> Asset:
    metadata = _mask_metadata(dict(asset.metadata or {}))
    ocr_text = str(metadata.get("ocr_text") or "")
    surrounding_text = str(metadata.get("surrounding_text") or "")
    if ocr_text:
        metadata["ocr_text"] = _compact_text(ocr_text, config.ocr_text_char_limit)
    if surrounding_text:
        metadata["surrounding_text"] = _compact_text(surrounding_text, config.surrounding_text_char_limit)
    for key in (
        "downloaded_image_path",
        "original_image_ref",
        "fallback_image_refs",
        "download_attempt_errors",
        "downloaded_image_ref",
    ):
        metadata.pop(key, None)
    metadata["search"] = {
        "rank": rank,
        "score": score,
        "sources": list(sources),
        "matched_fields": list(matched_fields),
        "dense_score": dense_score,
        "feature_score": feature_score,
    }
    return replace(asset, metadata=metadata)


def _format_image_search_detail(assets: tuple[Asset, ...]) -> str:
    lines = ["# Image Search", ""]
    if not assets:
        lines.append("該当する画像候補は登録されていません。")
        return "\n".join(lines)
    for asset in assets:
        metadata = dict(asset.metadata or {})
        source_label = str(metadata.get("source_label") or asset.source_kind)
        source_url = str(metadata.get("source_url") or asset.uri or "")
        description = asset.description or str(metadata.get("caption") or "")
        lines.append(f"- `{asset.id}` {asset.title or 'untitled'}")
        if description:
            lines.append(f"  - 説明: {_compact_text(description, 160)}")
        lines.append(f"  - 出典: {source_label}{(' / ' + source_url) if source_url else ''}")
    lines.append("")
    lines.append("この結果は画像候補の提示のみで、外部公開・転載・再利用の可否は判断しません。")
    return "\n".join(lines)


def _asset_embedding_text(asset: Asset) -> str:
    metadata = dict(asset.metadata or {})
    return "\n".join(
        line
        for line in (
            f"タイトル: {asset.title}",
            f"投稿媒体: {_normalize_source_kind(asset.source_kind)}",
            f"出典: {metadata.get('source_label') or metadata.get('source_url') or asset.uri}",
            f"画像説明: {asset.description or metadata.get('caption') or ''}",
            f"OCR: {metadata.get('ocr_text') or ''}",
            f"周辺テキスト: {metadata.get('surrounding_text') or ''}",
        )
        if line.strip()
    )


def _resize_and_normalize_vector(vector: np.ndarray, *, dimensions: int) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float32).reshape(-1)
    if out.size < dimensions:
        out = np.pad(out, (0, dimensions - out.size))
    elif out.size > dimensions:
        out = out[:dimensions]
    norm = np.linalg.norm(out)
    if norm > 0:
        out = out / norm
    return out.astype(np.float32)


def _local_image_feature_vector(*, image_path: Path, dimensions: int) -> np.ndarray:
    from PIL import Image

    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((32, 32))
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0.0, 1.0))
        hist_parts.append(hist.astype(np.float32))
    vector = np.concatenate([means, stds, *hist_parts]).astype(np.float32)
    if vector.size < dimensions:
        vector = np.pad(vector, (0, dimensions - vector.size))
    elif vector.size > dimensions:
        vector = vector[:dimensions]
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def _duplicate_group_id(asset: Asset) -> str:
    metadata = dict(asset.metadata or {})
    return str(metadata.get("duplicate_group_id") or metadata.get("content_hash") or asset.id)


def _matched_fields(*, query: str, asset: Asset) -> tuple[str, ...]:
    needle = query.strip().lower()
    terms = [term for term in re.split(r"\s+", needle) if term]
    metadata = dict(asset.metadata or {})
    field_texts = {
        "title": asset.title,
        "caption": asset.description or str(metadata.get("caption") or ""),
        "ocr_text": str(metadata.get("ocr_text") or ""),
        "surrounding_text": str(metadata.get("surrounding_text") or ""),
        "source_label": str(metadata.get("source_label") or ""),
        "source_kind": _normalize_source_kind(asset.source_kind),
    }
    matched: list[str] = []
    for field_name, text in field_texts.items():
        lowered = str(text or "").lower()
        if not lowered:
            continue
        if needle and needle in lowered:
            matched.append(field_name)
            continue
        if terms and any(term in lowered for term in terms):
            matched.append(field_name)
    return tuple(matched)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fr:
        for line in fr:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                out.append(payload)
    return out


def _read_sidecar(path: Path) -> dict[str, Any]:
    for meta_path in (path.with_suffix(path.suffix + ".meta.json"), path.with_suffix(".meta.json")):
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _download_bytes(url: str, *, max_bytes: int) -> tuple[bytes, str]:
    request = Request(url=url, headers={"User-Agent": "KUMC-Agent/1.0"})
    with urlopen(request, timeout=20) as response:  # noqa: S310 - configured source URLs.
        content_type = str(response.headers.get("content-type") or "")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("image exceeds max download size")
    return data, content_type


def _mask_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            out[key] = _mask_secret_text(value)
        elif isinstance(value, list):
            out[key] = [_mask_secret_text(item) if isinstance(item, str) else item for item in value]
        else:
            out[key] = value
    return out


def _mask_secret_text(text: str) -> str:
    masked = text
    for pattern in _SECRET_PATTERNS:
        masked = pattern.sub(lambda m: m.group(0).split(m.group(2), 1)[0] + "[REDACTED]" if len(m.groups()) >= 2 else "[REDACTED]", masked)
    return masked


def _compact_text(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if limit <= 0 or len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def _context_around(text: str, position: int, *, radius: int = 700) -> str:
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    context = text[start:end]
    context = _HTML_IMAGE_RE.sub("", context)
    context = _MARKDOWN_IMAGE_RE.sub("", context)
    context = re.sub(r"<[^>]+>", " ", context)
    return html.unescape(_compact_text(context, radius * 2))


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _article_source_url(*, source_kind: str, metadata: dict[str, Any]) -> str:
    if source_kind == "hatena":
        return str(metadata.get("hatenablog_url") or "").strip()
    return str(metadata.get("crafters_colony_article_url") or "").strip()


def _discord_message_url(*, guild_id: str, channel_id: str, message_id: str) -> str:
    if guild_id and channel_id and message_id:
        return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
    return ""


def _looks_like_image(url: str, *, filename: str = "", content_type: str = "") -> bool:
    lowered_type = content_type.lower()
    if lowered_type.startswith("image/"):
        return True
    suffix = Path((filename or url).split("?", 1)[0]).suffix.lower()
    return suffix in _IMAGE_EXTENSIONS


def _is_http_url(value: str) -> bool:
    return value.lower().startswith(("http://", "https://"))


def _suffix_from_content_type(content_type: str) -> str:
    lowered = content_type.split(";", 1)[0].strip().lower()
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }.get(lowered, "")


def _dt_from(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _join_nonempty(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _normalize_source_kind(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized == "x_posts":
        return "x"
    if normalized == "hatenablog":
        return "hatena"
    if normalized == "drive":
        return "google_drive"
    return normalized


def _all_image_source_kinds() -> set[str]:
    return {"discord", "google_drive", "x", "hatena", "crafters_colony"}
