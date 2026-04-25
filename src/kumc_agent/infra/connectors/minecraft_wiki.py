from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
from pathlib import Path
from urllib.parse import urlencode, quote
from urllib.request import urlopen

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

    source_kind: str = "minecraft_wiki"

    async def backfill(self, scope: BackfillScope) -> AsyncIterator[SourceRawItem]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        titles = list(scope.source_ids or self.page_titles)
        if scope.limit is not None:
            titles = titles[: max(0, scope.limit)]
        if self.max_pages > 0:
            titles = titles[: self.max_pages]
        for title in titles:
            raw = self._fetch_page(title)
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
        raw = self._fetch_page(external_id)
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

    def _fetch_page(self, title: str) -> SourceRawItem | None:
        clean_title = (title or "").strip()
        if not clean_title:
            return None
        path = self.raw_dir / f"{_safe_name(clean_title)}.md"
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            metadata = _read_json(meta_path)
            return self._raw_item(title=clean_title, text=text, metadata=metadata)

        text, metadata = self._download_page(clean_title)
        if not text.strip():
            return None
        path.write_text(text, encoding="utf-8")
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        return self._raw_item(title=clean_title, text=text, metadata=metadata)

    def _download_page(self, title: str) -> tuple[str, dict[str, object]]:
        query = urlencode(
            {
                "action": "parse",
                "page": title,
                "prop": "wikitext|revid|displaytitle",
                "format": "json",
                "formatversion": "2",
            }
        )
        with urlopen(f"{self.api_url}?{query}", timeout=20) as response:  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
        parse = payload.get("parse") if isinstance(payload, dict) else None
        if not isinstance(parse, dict):
            return "", {}
        text = str(parse.get("wikitext") or "")
        page_id = str(parse.get("pageid") or title)
        metadata = {
            "minecraft_wiki_title": str(parse.get("title") or title),
            "minecraft_wiki_page_id": page_id,
            "minecraft_wiki_revision_id": str(parse.get("revid") or ""),
            "canonical_url": self.page_url_base.rstrip("/") + "/" + quote(title.replace(" ", "_")),
            "source_type": "minecraft_wiki",
            "visibility": "public",
        }
        return text, metadata

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
            metadata={**metadata, "source_type": "minecraft_wiki"},
        )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "page"


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
