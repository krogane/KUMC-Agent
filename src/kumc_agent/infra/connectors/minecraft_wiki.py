from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import html
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote, urlparse
from urllib.request import Request, urlopen

from kumc_agent.domain.models.source import (
    AccessScope,
    BackfillScope,
    NormalizedDocument,
    SourceDeleteItem,
    SourceRawItem,
    SyncCursor,
)
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class MinecraftWikiConnector:
    supports_incremental = False

    ingestion_dir: Path
    page_titles: tuple[str, ...]
    api_url: str
    page_url_base: str
    max_pages: int
    rate_limit_per_minute: int = 30
    request_interval_seconds: float = 1.0
    namespaces: tuple[int, ...] = (0,)
    full_backfill_enabled: bool = False
    max_request_retries: int = 3
    retry_initial_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 8.0
    max_redirect_depth: int = 5
    acquisition_mode: str = "configured"
    category_sample_categories: dict[str, str] = field(default_factory=dict)
    category_sample_per_category: int = 20

    source_kind: str = "minecraft_wiki"
    _last_request_monotonic: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_japanese_minecraft_wiki_url(self.api_url, field_name="api_url")
        _validate_japanese_minecraft_wiki_url(
            self.page_url_base,
            field_name="page_url_base",
        )

    async def backfill(self, scope: BackfillScope) -> AsyncIterator[SourceRawItem]:
        self.ingestion_dir.mkdir(parents=True, exist_ok=True)
        titles = self._resolve_backfill_titles(scope)
        if scope.limit is not None:
            titles = titles[: max(0, scope.limit)]
        if self.max_pages > 0:
            titles = titles[: self.max_pages]
        seen_page_ids: set[str] = set()
        for title in titles:
            raw = self._fetch_page(title, force=scope.force)
            if raw is None:
                continue
            page_id = str(raw.metadata.get("minecraft_wiki_page_id") or raw.external_id).strip()
            if page_id and page_id in seen_page_ids:
                continue
            if page_id:
                seen_page_ids.add(page_id)
            yield raw

    async def poll_changes(
        self,
        cursor: SyncCursor,
    ) -> AsyncIterator[SourceRawItem | SourceDeleteItem]:
        async for item in self.backfill(BackfillScope()):
            yield item

    async def fetch_item(self, external_id: str) -> SourceRawItem:
        raw = self._fetch_page(external_id, force=True)
        if raw is None:
            raise KeyError(f"Minecraft Wiki page not found: {external_id}")
        return raw

    async def normalize(self, raw: SourceRawItem) -> NormalizedDocument:
        source_item_id = stable_hash(f"{raw.source_kind}:{raw.external_id}")
        return NormalizedDocument(
            id=stable_hash(f"document:{source_item_id}:{raw.checksum}"),
            source_item_id=source_item_id,
            source_kind=raw.source_kind,
            external_id=raw.external_id,
            version=1,
            title=raw.title,
            normalized_text=raw.text,
            normalized_format="wiki_markdown",
            language="ja",
            access_scope=raw.access_scope,
            checksum=raw.checksum,
            metadata=dict(raw.metadata),
        )

    def _fetch_page(self, title: str, *, force: bool = False) -> SourceRawItem | None:
        clean_title = (title or "").strip()
        if not clean_title:
            return None
        path = self._cache_path_for_title(clean_title)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if path.exists() and not force:
            text = path.read_text(encoding="utf-8", errors="ignore")
            metadata = _read_json(meta_path)
            cached_revision = str(
                metadata.get("minecraft_wiki_revision_id") or ""
            ).strip()
            try:
                remote_metadata = self._revision_metadata(
                    title=clean_title,
                    resolve_redirects=True,
                )
            except Exception:
                remote_metadata = {}
            remote_revision = str(remote_metadata.get("revid") or "").strip()
            if not remote_revision or remote_revision == cached_revision:
                path = self._consolidate_cached_page(
                    text=text,
                    metadata=metadata,
                    current_path=path,
                    fallback_title=clean_title,
                )
                return self._raw_item(
                    title=_raw_item_title(clean_title, metadata),
                    text=text,
                    metadata=metadata,
                    raw_path=path,
                )

        try:
            text, metadata = self._download_page(clean_title)
        except Exception:
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore")
                metadata = _read_json(meta_path)
                path = self._consolidate_cached_page(
                    text=text,
                    metadata=metadata,
                    current_path=path,
                    fallback_title=clean_title,
                )
                return self._raw_item(
                    title=_raw_item_title(clean_title, metadata),
                    text=text,
                    metadata=metadata,
                    raw_path=path,
                )
            raise
        if not text.strip():
            return None
        metadata["checksum"] = stable_hash(text)
        path = self._cache_path_for_metadata(metadata=metadata, fallback_title=clean_title)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        metadata["minecraft_wiki_cache_title"] = _raw_item_title(clean_title, metadata)
        metadata["minecraft_wiki_cache_file"] = path.name
        path.write_text(text, encoding="utf-8")
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        self._remove_duplicate_cache_files(metadata=metadata, keep_path=path)
        self._write_manifest_entry(metadata=metadata, path=path)
        return self._raw_item(
            title=_raw_item_title(clean_title, metadata),
            text=text,
            metadata=metadata,
            raw_path=path,
        )

    def _resolve_backfill_titles(self, scope: BackfillScope) -> list[str]:
        explicit_titles = list(scope.source_ids or ())
        if explicit_titles:
            return [title for title in explicit_titles if str(title).strip()]
        configured_titles = list(self.page_titles)
        mode = (self.acquisition_mode or "configured").strip().lower()
        if mode == "configured" and configured_titles:
            return [title for title in configured_titles if str(title).strip()]
        if mode == "category_sample":
            return self._list_category_sample_titles()
        if configured_titles and not self.full_backfill_enabled:
            return [title for title in configured_titles if str(title).strip()]
        if not self.full_backfill_enabled:
            return []
        return self._list_all_page_titles()

    def _list_category_sample_titles(self) -> list[str]:
        categories = {
            key: value
            for key, value in (self.category_sample_categories or {}).items()
            if str(key).strip() and str(value).strip()
        }
        if not categories:
            return []
        limit = max(0, int(self.category_sample_per_category))
        titles: list[str] = []
        seen: set[str] = set()
        for category_title in categories.values():
            for title in self._list_category_titles(category_title, limit=limit):
                key = title.strip().replace("_", " ")
                if not key or key in seen:
                    continue
                seen.add(key)
                titles.append(title)
        return titles

    def _list_category_titles(self, category_title: str, *, limit: int) -> list[str]:
        titles: list[str] = []
        cmcontinue = ""
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category_title,
                "cmnamespace": "|".join(str(int(value)) for value in (self.namespaces or (0,))),
                "cmlimit": "max" if limit <= 0 else str(min(500, max(1, limit - len(titles)))),
                "format": "json",
                "formatversion": "2",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            payload = self._request_json(params)
            query = payload.get("query") if isinstance(payload, dict) else None
            members = query.get("categorymembers") if isinstance(query, dict) else None
            if isinstance(members, list):
                for member in members:
                    if not isinstance(member, dict):
                        continue
                    title = str(member.get("title") or "").strip()
                    if title:
                        titles.append(title)
                        if limit > 0 and len(titles) >= limit:
                            return titles
            cont = payload.get("continue") if isinstance(payload, dict) else None
            cmcontinue = (
                str(cont.get("cmcontinue") or "").strip()
                if isinstance(cont, dict)
                else ""
            )
            if not cmcontinue:
                break
        return titles

    def _list_all_page_titles(self) -> list[str]:
        titles: list[str] = []
        namespaces = self.namespaces or (0,)
        for namespace in namespaces:
            apcontinue = ""
            while True:
                params = {
                    "action": "query",
                    "list": "allpages",
                    "apnamespace": str(int(namespace)),
                    "aplimit": "max",
                    "format": "json",
                    "formatversion": "2",
                }
                if apcontinue:
                    params["apcontinue"] = apcontinue
                payload = self._request_json(params)
                query = payload.get("query") if isinstance(payload, dict) else None
                pages = query.get("allpages") if isinstance(query, dict) else None
                if isinstance(pages, list):
                    for page in pages:
                        if not isinstance(page, dict):
                            continue
                        title = str(page.get("title") or "").strip()
                        if title:
                            titles.append(title)
                            if self.max_pages > 0 and len(titles) >= self.max_pages:
                                return titles
                cont = payload.get("continue") if isinstance(payload, dict) else None
                apcontinue = (
                    str(cont.get("apcontinue") or "").strip()
                    if isinstance(cont, dict)
                    else ""
                )
                if not apcontinue:
                    break
        return titles

    def _download_page(self, title: str) -> tuple[str, dict[str, object]]:
        parse, redirect_chain = self._parse_page_resolving_redirects(title)
        if not isinstance(parse, dict):
            return "", {}
        resolved_title = str(parse.get("title") or title).strip() or title
        revision_metadata = self._revision_metadata(
            title=resolved_title,
            resolve_redirects=True,
        )
        text = _normalize_wikitext(str(parse.get("wikitext") or ""))
        page_id = str(parse.get("pageid") or revision_metadata.get("pageid") or title)
        is_redirect = bool(redirect_chain)
        redirect_from = str(redirect_chain[0]["from"]) if redirect_chain else ""
        redirect_to = str(redirect_chain[-1]["to"]) if redirect_chain else ""
        metadata = {
            "minecraft_wiki_title": resolved_title,
            "minecraft_wiki_requested_title": title,
            "minecraft_wiki_page_id": page_id,
            "minecraft_wiki_revision_id": str(
                revision_metadata.get("revid") or parse.get("revid") or ""
            ),
            "minecraft_wiki_is_redirect": is_redirect,
            "minecraft_wiki_redirect_from": redirect_from,
            "minecraft_wiki_redirect_to": redirect_to,
            "minecraft_wiki_resolved_title": resolved_title,
            "minecraft_wiki_resolved_page_id": page_id,
            "minecraft_wiki_aliases": [redirect_from] if redirect_from else [],
            "minecraft_wiki_redirect_chain": redirect_chain,
            "canonical_url": str(
                revision_metadata.get("canonicalurl")
                or self.page_url_base.rstrip("/") + "/" + quote(resolved_title.replace(" ", "_"))
            ),
            "source_kind": "minecraft_wiki",
            "source_type": "minecraft_wiki",
            "access_scope": {"visibility": "public"},
            "visibility": "public",
        }
        if revision_metadata.get("timestamp"):
            metadata["updated_at"] = str(revision_metadata["timestamp"])
        return text, metadata

    def _parse_page_resolving_redirects(
        self,
        title: str,
    ) -> tuple[dict[str, object], list[dict[str, str]]]:
        current_title = title
        redirect_chain: list[dict[str, str]] = []
        max_depth = max(0, int(self.max_redirect_depth))
        for _ in range(max_depth + 1):
            payload = self._request_json(
                {
                    "action": "parse",
                    "page": current_title,
                    "prop": "wikitext|revid|displaytitle",
                    "redirects": "1",
                    "format": "json",
                    "formatversion": "2",
                }
            )
            parse = payload.get("parse") if isinstance(payload, dict) else None
            if not isinstance(parse, dict):
                return {}, redirect_chain
            redirect_chain.extend(
                _redirect_chain_from_payload(
                    payload=payload,
                    current_title=current_title,
                )
            )
            wikitext_redirect = _extract_redirect_target(str(parse.get("wikitext") or ""))
            if wikitext_redirect:
                redirect_chain.append({"from": current_title, "to": wikitext_redirect})
                current_title = wikitext_redirect
                continue
            parsed_title = str(parse.get("title") or current_title).strip()
            if (
                parsed_title
                and not _same_wiki_title(parsed_title, current_title)
                and not redirect_chain
            ):
                redirect_chain.append({"from": current_title, "to": parsed_title})
            return parse, _dedupe_redirect_chain(redirect_chain)
        raise RuntimeError(f"Minecraft Wiki redirect depth exceeded for {title}")

    def _revision_metadata(
        self,
        *,
        title: str,
        resolve_redirects: bool = False,
    ) -> dict[str, object]:
        params = {
            "action": "query",
            "titles": title,
            "prop": "revisions|info",
            "rvprop": "ids|timestamp",
            "inprop": "url",
            "format": "json",
            "formatversion": "2",
        }
        if resolve_redirects:
            params["redirects"] = "1"
        payload = self._request_json(
            params
        )
        query = payload.get("query") if isinstance(payload, dict) else None
        pages = query.get("pages") if isinstance(query, dict) else None
        if not isinstance(pages, list) or not pages:
            return {}
        page = pages[0]
        if not isinstance(page, dict):
            return {}
        revisions = page.get("revisions")
        revision = revisions[0] if isinstance(revisions, list) and revisions else {}
        if not isinstance(revision, dict):
            revision = {}
        metadata: dict[str, object] = {
            "pageid": page.get("pageid"),
            "title": page.get("title"),
            "canonicalurl": page.get("canonicalurl") or page.get("fullurl"),
            "revid": revision.get("revid"),
            "timestamp": revision.get("timestamp"),
        }
        return {key: value for key, value in metadata.items() if value}

    def _request_json(self, params: dict[str, str]) -> dict[str, object]:
        query = urlencode(params)
        retries = max(0, int(self.max_request_retries))
        for attempt in range(retries + 1):
            self._wait_for_rate_limit()
            request = Request(
                f"{self.api_url}?{query}",
                headers={
                    "User-Agent": "KUMC-Agent Minecraft Wiki RAG/1.0",
                    "Accept": "application/json",
                },
            )
            try:
                with urlopen(request, timeout=20) as response:  # nosec B310
                    payload = json.loads(response.read().decode("utf-8"))
                return payload if isinstance(payload, dict) else {}
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                    raise
            except (URLError, TimeoutError, json.JSONDecodeError):
                if attempt >= retries:
                    raise
            wait_seconds = min(
                float(self.retry_max_delay_seconds),
                float(self.retry_initial_delay_seconds) * (2 ** attempt),
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        return {}

    def _wait_for_rate_limit(self) -> None:
        interval = max(0.0, float(self.request_interval_seconds))
        if self.rate_limit_per_minute > 0:
            interval = max(interval, 60.0 / float(self.rate_limit_per_minute))
        if interval <= 0.0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_monotonic
        wait_seconds = interval - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.monotonic()
        object.__setattr__(self, "_last_request_monotonic", now)

    def _raw_item(
        self,
        *,
        title: str,
        text: str,
        metadata: dict[str, object],
        raw_path: Path | None = None,
    ) -> SourceRawItem:
        external_id = str(metadata.get("minecraft_wiki_page_id") or title)
        canonical_url = str(
            metadata.get("canonical_url")
            or self.page_url_base.rstrip("/") + "/" + quote(title.replace(" ", "_"))
        )
        return SourceRawItem(
            source_kind=self.source_kind,
            external_id=external_id,
            title=title,
            text=text,
            canonical_url=canonical_url,
            access_scope=AccessScope(visibility="public"),
            raw_path=str(raw_path or self.ingestion_dir / f"{_safe_name(title)}.md"),
            checksum=stable_hash(text),
            metadata={
                **metadata,
                "source_kind": "minecraft_wiki",
                "source_type": "minecraft_wiki",
                "visibility": "public",
            },
        )

    def _cache_path_for_title(self, title: str) -> Path:
        return self.ingestion_dir / f"{_safe_name(title)}.md"

    def _cache_path_for_metadata(
        self,
        *,
        metadata: dict[str, object],
        fallback_title: str,
    ) -> Path:
        title = _raw_item_title(fallback_title, metadata)
        page_id = str(metadata.get("minecraft_wiki_page_id") or "").strip()
        base = self._cache_path_for_title(title)
        existing_metadata = _read_json(base.with_suffix(base.suffix + ".meta.json"))
        existing_page_id = str(existing_metadata.get("minecraft_wiki_page_id") or "").strip()
        if not base.exists() or not page_id or not existing_page_id or existing_page_id == page_id:
            return base

        stem = _safe_name(f"{title}_{page_id}")
        candidate = self.ingestion_dir / f"{stem}.md"
        counter = 2
        while candidate.exists():
            candidate_metadata = _read_json(candidate.with_suffix(candidate.suffix + ".meta.json"))
            candidate_page_id = str(
                candidate_metadata.get("minecraft_wiki_page_id") or ""
            ).strip()
            if not candidate_page_id or candidate_page_id == page_id:
                return candidate
            candidate = self.ingestion_dir / f"{stem}_{counter}.md"
            counter += 1
        return candidate

    def _consolidate_cached_page(
        self,
        *,
        text: str,
        metadata: dict[str, object],
        current_path: Path,
        fallback_title: str,
    ) -> Path:
        target_path = self._cache_path_for_metadata(
            metadata=metadata,
            fallback_title=fallback_title,
        )
        metadata["checksum"] = stable_hash(text)
        metadata["minecraft_wiki_cache_title"] = _raw_item_title(fallback_title, metadata)
        metadata["minecraft_wiki_cache_file"] = target_path.name
        if target_path != current_path:
            target_path.write_text(text, encoding="utf-8")
            target_path.with_suffix(target_path.suffix + ".meta.json").write_text(
                json.dumps(metadata, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            current_path.with_suffix(current_path.suffix + ".meta.json").write_text(
                json.dumps(metadata, ensure_ascii=False),
                encoding="utf-8",
            )
        self._remove_duplicate_cache_files(metadata=metadata, keep_path=target_path)
        self._write_manifest_entry(metadata=metadata, path=target_path)
        return target_path

    def _remove_duplicate_cache_files(
        self,
        *,
        metadata: dict[str, object],
        keep_path: Path,
    ) -> None:
        page_id = str(metadata.get("minecraft_wiki_page_id") or "").strip()
        if not page_id:
            return
        keep_path = keep_path.resolve()
        for meta_path in self.ingestion_dir.glob("*.md.meta.json"):
            cached_metadata = _read_json(meta_path)
            cached_page_id = str(
                cached_metadata.get("minecraft_wiki_page_id") or ""
            ).strip()
            if cached_page_id != page_id:
                continue
            raw_path = Path(str(meta_path)[: -len(".meta.json")])
            if raw_path.resolve() == keep_path:
                continue
            try:
                raw_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
            except OSError:
                continue

    def _write_manifest_entry(
        self,
        *,
        metadata: dict[str, object],
        path: Path,
    ) -> None:
        manifest_path = self.ingestion_dir / "manifest.json"
        current = _read_json(manifest_path)
        entries = current.get("pages")
        pages = entries if isinstance(entries, list) else []
        page_id = str(metadata.get("minecraft_wiki_page_id") or "").strip()
        requested_title = str(metadata.get("minecraft_wiki_requested_title") or "").strip()
        filtered: list[object] = []
        for entry in pages:
            if not isinstance(entry, dict):
                continue
            if page_id and str(entry.get("minecraft_wiki_page_id") or "") == page_id:
                continue
            if requested_title and str(entry.get("minecraft_wiki_requested_title") or "") == requested_title:
                continue
            filtered.append(entry)
        filtered.append(
            {
                "minecraft_wiki_page_id": page_id,
                "minecraft_wiki_requested_title": requested_title,
                "minecraft_wiki_resolved_title": str(
                    metadata.get("minecraft_wiki_resolved_title") or ""
                ),
                "minecraft_wiki_revision_id": str(
                    metadata.get("minecraft_wiki_revision_id") or ""
                ),
                "canonical_url": str(metadata.get("canonical_url") or ""),
                "file": path.name,
            }
        )
        manifest_path.write_text(
            json.dumps({"pages": filtered}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "page"


def _validate_japanese_minecraft_wiki_url(value: str, *, field_name: str) -> None:
    parsed = urlparse(str(value or ""))
    host = (parsed.hostname or "").strip().lower()
    if parsed.scheme != "https" or host != "ja.minecraft.wiki":
        raise ValueError(
            f"Minecraft Wiki {field_name} must point to https://ja.minecraft.wiki"
        )


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}


def _raw_item_title(fallback: str, metadata: dict[str, object]) -> str:
    for key in ("minecraft_wiki_resolved_title", "minecraft_wiki_title"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return fallback


def _same_wiki_title(left: str, right: str) -> bool:
    def _normalize(value: str) -> str:
        return value.strip().replace("_", " ").lower()

    return _normalize(left) == _normalize(right)


def _redirect_chain_from_payload(
    *,
    payload: dict[str, object],
    current_title: str,
) -> list[dict[str, str]]:
    candidates: list[object] = []
    parse = payload.get("parse")
    if isinstance(parse, dict):
        raw = parse.get("redirects")
        if isinstance(raw, list):
            candidates.extend(raw)
    query = payload.get("query")
    if isinstance(query, dict):
        raw = query.get("redirects")
        if isinstance(raw, list):
            candidates.extend(raw)

    chain: list[dict[str, str]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        source = str(item.get("from") or current_title).strip()
        target = str(item.get("to") or item.get("*") or "").strip()
        if source and target:
            chain.append({"from": source, "to": target})
    return _dedupe_redirect_chain(chain)


def _dedupe_redirect_chain(chain: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in chain:
        source = str(item.get("from") or "").strip()
        target = str(item.get("to") or "").strip()
        key = (source, target)
        if not source or not target or key in seen:
            continue
        seen.add(key)
        deduped.append({"from": source, "to": target})
    return deduped


def _extract_redirect_target(text: str) -> str | None:
    match = re.match(
        r"(?is)^\s*#(?:転送|redirect)\s*:?\s*\[\[([^\]|#]+)",
        text or "",
    )
    if not match:
        return None
    target = match.group(1).strip()
    return target or None


def _normalize_wikitext(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"(?is)<ref[^>/]*/>", "", cleaned)
    cleaned = re.sub(r"(?is)<ref[^>]*>.*?</ref>", "", cleaned)
    cleaned = re.sub(r"(?is)<!--.*?-->", "", cleaned)
    cleaned = re.sub(r"(?is)<gallery[^>]*>(.*?)</gallery>", _gallery_to_text, cleaned)
    cleaned = re.sub(r"(?is)<code[^>]*>(.*?)</code>", lambda m: m.group(1), cleaned)
    cleaned = re.sub(r"(?is)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?is)</(?:div|p|li|tr|table|section|blockquote)>", "\n", cleaned)
    cleaned = re.sub(r"(?is)<(?:div|p|ul|ol|li|table|tbody|thead|tr|td|th|span|section|blockquote)[^>]*>", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\[\[カテゴリ:[^\]]+\]\]\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\[\[Category:[^\]]+\]\]\s*$", "", cleaned)
    cleaned = _templates_to_text(cleaned)
    cleaned = _tables_to_text(cleaned)
    cleaned = re.sub(r"(?m)^(={2,6})\s*(.*?)\s*\1\s*$", _heading_to_markdown, cleaned)
    cleaned = re.sub(
        r"\[\[((?:ファイル|File|Image|画像):[^\]]+)\]\]",
        _file_link_to_text,
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", cleaned)
    cleaned = re.sub(r"\[\[([^\]]+)\]\]", r"\1", cleaned)
    cleaned = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", cleaned)
    cleaned = re.sub(r"'''([^']+)'''", r"\1", cleaned)
    cleaned = re.sub(r"''([^']+)''", r"\1", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", "", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"(?m)^[!|].*$", lambda m: _table_line_to_text(m.group(0)), cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _gallery_to_text(match: re.Match[str]) -> str:
    lines: list[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"(?i)^(?:ファイル|File|Image|画像):", line):
            caption = _caption_from_file_parts(line.split("|"))
            if caption:
                lines.append(caption)
            continue
        lines.append(line)
    return "\n".join(lines)


def _file_link_to_text(match: re.Match[str]) -> str:
    return _caption_from_file_parts(match.group(1).split("|"))


def _caption_from_file_parts(parts: list[str]) -> str:
    for part in reversed(parts[1:]):
        candidate = part.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in {"thumb", "thumbnail", "frame", "right", "left", "center", "none"}:
            continue
        if lowered.endswith("px") or lowered.startswith("link="):
            continue
        if "=" in candidate and candidate.split("=", 1)[0].strip().lower() in {
            "alt",
            "class",
            "style",
            "link",
        }:
            continue
        return candidate
    return ""


def _tables_to_text(text: str) -> str:
    return re.sub(r"(?ms)^\{\|.*?^\|\}\s*", _table_block_to_text, text)


def _table_block_to_text(match: re.Match[str]) -> str:
    headers: list[str] = []
    rows: list[str] = []
    for raw_line in match.group(0).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("{|") or line.startswith("|}") or line.startswith("|-"):
            continue
        if line.startswith("!"):
            cells = [_clean_table_cell(cell) for cell in _split_table_cells(line, header=True)]
            cells = [cell for cell in cells if cell]
            if cells:
                headers = cells
                rows.append(" | ".join(cells))
            continue
        if line.startswith("|"):
            cells = [_clean_table_cell(cell) for cell in _split_table_cells(line, header=False)]
            cells = [cell for cell in cells if cell]
            if not cells:
                continue
            if headers and len(headers) == len(cells):
                rows.append("; ".join(f"{header}: {cell}" for header, cell in zip(headers, cells)))
            else:
                rows.append(" | ".join(cells))
    if not rows:
        return "\n"
    return "\n".join(f"- {row}" for row in rows) + "\n"


def _split_table_cells(line: str, *, header: bool) -> list[str]:
    stripped = line.lstrip("!|" if header else "|").strip()
    delimiter = "!!" if header and "!!" in stripped else "||"
    if delimiter in stripped:
        return stripped.split(delimiter)
    return [stripped]


def _clean_table_cell(value: str) -> str:
    cleaned = value.strip()
    if "|" in cleaned:
        prefix, rest = cleaned.split("|", 1)
        if "=" in prefix or prefix.strip().lower().startswith(("style", "class", "scope")):
            cleaned = rest.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _templates_to_text(text: str) -> str:
    previous = None
    cleaned = text
    pattern = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
    while previous != cleaned:
        previous = cleaned
        cleaned = pattern.sub(lambda m: _template_to_text(m.group(0)), cleaned)
    return cleaned


def _template_to_text(value: str) -> str:
    body = value.strip()[2:-2].strip()
    if not body:
        return ""
    parts = [part.strip() for part in body.split("|")]
    if len(parts) <= 1:
        return ""
    preserved: list[str] = []
    for raw in parts[1:]:
        if not raw:
            continue
        if "=" in raw:
            key, item = raw.split("=", 1)
            key = key.strip().lower()
            item = item.strip()
            if key in {"image", "画像", "file", "ファイル", "alt", "class", "style"}:
                continue
        else:
            item = raw.strip()
        if item:
            preserved.append(item)
    return " ".join(preserved)


def _heading_to_markdown(match: re.Match[str]) -> str:
    level = max(1, min(6, len(match.group(1)) - 1))
    title = match.group(2).strip()
    return f"{'#' * level} {title}"


def _table_line_to_text(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("|-"):
        return ""
    stripped = stripped.lstrip("!|").strip()
    stripped = stripped.replace("!!", " | ").replace("||", " | ")
    stripped = re.sub(r"^[^|=]+=[^|]+\|", "", stripped).strip()
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped
