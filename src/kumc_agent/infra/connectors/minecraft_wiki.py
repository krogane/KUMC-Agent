from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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
    raw_dir: Path
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

    source_kind: str = "minecraft_wiki"
    _last_request_monotonic: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_japanese_minecraft_wiki_url(self.api_url, field_name="api_url")
        _validate_japanese_minecraft_wiki_url(
            self.page_url_base,
            field_name="page_url_base",
        )

    async def backfill(self, scope: BackfillScope) -> AsyncIterator[SourceRawItem]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        titles = self._resolve_backfill_titles(scope)
        if scope.limit is not None:
            titles = titles[: max(0, scope.limit)]
        if self.max_pages > 0:
            titles = titles[: self.max_pages]
        for title in titles:
            raw = self._fetch_page(title, force=scope.force)
            if raw is None:
                continue
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
        path = self.raw_dir / f"{_safe_name(clean_title)}.md"
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if path.exists() and not force:
            text = path.read_text(encoding="utf-8", errors="ignore")
            metadata = _read_json(meta_path)
            cached_revision = str(
                metadata.get("minecraft_wiki_revision_id") or ""
            ).strip()
            try:
                remote_metadata = self._revision_metadata(title=clean_title)
            except Exception:
                remote_metadata = {}
            remote_revision = str(remote_metadata.get("revid") or "").strip()
            if not remote_revision or remote_revision == cached_revision:
                return self._raw_item(title=clean_title, text=text, metadata=metadata)

        try:
            text, metadata = self._download_page(clean_title)
        except Exception:
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore")
                metadata = _read_json(meta_path)
                return self._raw_item(title=clean_title, text=text, metadata=metadata)
            raise
        if not text.strip():
            return None
        metadata["checksum"] = stable_hash(text)
        path.write_text(text, encoding="utf-8")
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        return self._raw_item(title=clean_title, text=text, metadata=metadata)

    def _resolve_backfill_titles(self, scope: BackfillScope) -> list[str]:
        explicit_titles = list(scope.source_ids or ())
        if explicit_titles:
            return [title for title in explicit_titles if str(title).strip()]
        configured_titles = list(self.page_titles)
        if configured_titles:
            return [title for title in configured_titles if str(title).strip()]
        if not self.full_backfill_enabled:
            return []
        return self._list_all_page_titles()

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
        payload = self._request_json(
            {
                "action": "parse",
                "page": title,
                "prop": "wikitext|revid|displaytitle",
                "format": "json",
                "formatversion": "2",
            }
        )
        parse = payload.get("parse") if isinstance(payload, dict) else None
        if not isinstance(parse, dict):
            return "", {}
        revision_metadata = self._revision_metadata(title=title)
        text = _normalize_wikitext(str(parse.get("wikitext") or ""))
        page_id = str(parse.get("pageid") or title)
        metadata = {
            "minecraft_wiki_title": str(parse.get("title") or title),
            "minecraft_wiki_page_id": page_id,
            "minecraft_wiki_revision_id": str(
                revision_metadata.get("revid") or parse.get("revid") or ""
            ),
            "canonical_url": str(
                revision_metadata.get("canonicalurl")
                or self.page_url_base.rstrip("/") + "/" + quote(title.replace(" ", "_"))
            ),
            "source_kind": "minecraft_wiki",
            "source_type": "minecraft_wiki",
            "access_scope": {"visibility": "public"},
            "visibility": "public",
        }
        if revision_metadata.get("timestamp"):
            metadata["updated_at"] = str(revision_metadata["timestamp"])
        return text, metadata

    def _revision_metadata(self, *, title: str) -> dict[str, object]:
        payload = self._request_json(
            {
                "action": "query",
                "titles": title,
                "prop": "revisions|info",
                "rvprop": "ids|timestamp",
                "inprop": "url",
                "format": "json",
                "formatversion": "2",
            }
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
            raw_path=str(self.raw_dir / f"{_safe_name(title)}.md"),
            checksum=stable_hash(text),
            metadata={
                **metadata,
                "source_kind": "minecraft_wiki",
                "source_type": "minecraft_wiki",
                "visibility": "public",
            },
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


def _normalize_wikitext(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"(?is)<ref[^>/]*/>", "", cleaned)
    cleaned = re.sub(r"(?is)<ref[^>]*>.*?</ref>", "", cleaned)
    cleaned = re.sub(r"(?is)<!--.*?-->", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\[\[カテゴリ:[^\]]+\]\]\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\[\[Category:[^\]]+\]\]\s*$", "", cleaned)
    cleaned = _templates_to_text(cleaned)
    cleaned = re.sub(r"(?m)^(={2,6})\s*(.*?)\s*\1\s*$", _heading_to_markdown, cleaned)
    cleaned = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", cleaned)
    cleaned = re.sub(r"\[\[([^\]]+)\]\]", r"\1", cleaned)
    cleaned = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", cleaned)
    cleaned = re.sub(r"'''([^']+)'''", r"\1", cleaned)
    cleaned = re.sub(r"''([^']+)''", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^\{\|.*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\|\}.*$", "", cleaned)
    cleaned = re.sub(r"(?m)^[!|].*$", lambda m: _table_line_to_text(m.group(0)), cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
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
