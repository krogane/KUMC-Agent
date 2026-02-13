from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from indexing.constants import FILE_ID_SEPARATOR
from indexing.utils import ensure_dir, sanitize_filename

logger = logging.getLogger(__name__)

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_REQUEST_TIMEOUT_SECONDS = 30
_MAX_FEED_PAGES = 200


@dataclass(frozen=True)
class HatenablogEntry:
    entry_id: str
    title: str
    url: str
    created_at: str
    updated_at: str
    content_html: str


class _ArticleTextExtractor(HTMLParser):
    _SKIP_TAGS = {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "img",
        "picture",
        "source",
        "figure",
        "figcaption",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
    }
    _BLOCK_TAGS = {
        "article",
        "section",
        "div",
        "p",
        "br",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "tr",
        "td",
        "th",
        "blockquote",
        "pre",
        "hr",
    }
    _VOID_SKIP_TAGS = {"img", "source"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        if tag_name in self._SKIP_TAGS or self._has_comment_marker(attrs):
            if tag_name not in self._VOID_SKIP_TAGS:
                self._skip_stack.append(tag_name)
            return
        if self._skip_stack:
            return
        if tag_name in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if self._skip_stack:
            if tag_name == self._skip_stack[-1]:
                self._skip_stack.pop()
            return
        if tag_name in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        if not data:
            return
        self._parts.append(data)

    @staticmethod
    def _has_comment_marker(attrs: list[tuple[str, str | None]]) -> bool:
        for key, value in attrs:
            if key not in {"class", "id"}:
                continue
            if not value:
                continue
            if "comment" in value.lower():
                return True
        return False

    def text(self) -> str:
        return _normalize_text("".join(self._parts))


def _normalize_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if previous_blank:
                continue
            cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False
    return "\n".join(cleaned_lines).strip()


def _http_get_text(url: str) -> str:
    request = Request(
        url=url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; KUMC-Agent/1.0; +https://kumc.hatenablog.com/)"
            )
        },
    )
    with urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        payload = response.read()
    return payload.decode(charset, errors="replace")


def _extract_entry_link(entry: ET.Element) -> str:
    for link in entry.findall(f"{_ATOM_NS}link"):
        href = (link.get("href") or "").strip()
        if not href:
            continue
        rel = (link.get("rel") or "").strip().lower()
        if rel in {"alternate", ""}:
            return href
    first_link = entry.find(f"{_ATOM_NS}link")
    if first_link is not None:
        return (first_link.get("href") or "").strip()
    return ""


def _text_or_empty(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _parse_feed_page(feed_xml: str, *, feed_url: str) -> tuple[list[HatenablogEntry], str | None]:
    root = ET.fromstring(feed_xml)
    entries: list[HatenablogEntry] = []

    for entry in root.findall(f"{_ATOM_NS}entry"):
        title = _text_or_empty(entry.find(f"{_ATOM_NS}title"))
        entry_id = _text_or_empty(entry.find(f"{_ATOM_NS}id"))
        url = _extract_entry_link(entry)
        created_at = _text_or_empty(entry.find(f"{_ATOM_NS}published"))
        updated_at = _text_or_empty(entry.find(f"{_ATOM_NS}updated"))
        content_html = _text_or_empty(entry.find(f"{_ATOM_NS}content"))
        if not url:
            continue
        if not entry_id:
            entry_id = url
        if not created_at:
            created_at = updated_at
        if not updated_at:
            updated_at = created_at
        entries.append(
            HatenablogEntry(
                entry_id=entry_id,
                title=title,
                url=urljoin(feed_url, url),
                created_at=created_at,
                updated_at=updated_at,
                content_html=content_html,
            )
        )

    next_url: str | None = None
    for link in root.findall(f"{_ATOM_NS}link"):
        rel = (link.get("rel") or "").strip().lower()
        href = (link.get("href") or "").strip()
        if rel == "next" and href:
            next_url = urljoin(feed_url, href)
            break

    return entries, next_url


def _collect_entries(blog_url: str) -> list[HatenablogEntry]:
    feed_url = urljoin(blog_url.rstrip("/") + "/", "feed")
    queue_url: str | None = feed_url
    visited_feed_urls: set[str] = set()
    seen_entry_ids: set[str] = set()
    collected: list[HatenablogEntry] = []
    page_count = 0

    while queue_url and queue_url not in visited_feed_urls:
        if page_count >= _MAX_FEED_PAGES:
            logger.warning("Feed pagination exceeded max pages (%d).", _MAX_FEED_PAGES)
            break
        visited_feed_urls.add(queue_url)
        page_count += 1
        feed_xml = _http_get_text(queue_url)
        entries, next_url = _parse_feed_page(feed_xml, feed_url=queue_url)
        for entry in entries:
            if entry.entry_id in seen_entry_ids:
                continue
            seen_entry_ids.add(entry.entry_id)
            collected.append(entry)
        queue_url = next_url

    return collected


def _entry_key(entry: HatenablogEntry) -> str:
    source = entry.entry_id or entry.url
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
    return digest[:16]


def _extract_entry_key(filename: str) -> str | None:
    if FILE_ID_SEPARATOR not in filename:
        return None
    prefix, _ = filename.split(FILE_ID_SEPARATOR, 1)
    return prefix or None


def _metadata_sidecar_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".meta.json")


def _read_entry_metadata(out_path: Path) -> dict[str, str]:
    meta_path = _metadata_sidecar_path(out_path)
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Failed to read Hatenablog metadata sidecar %s: %s",
            meta_path.name,
            exc,
        )
        return {}
    if not isinstance(data, dict):
        return {}

    metadata: dict[str, str] = {}
    for key in (
        "hatenablog_entry_id",
        "hatenablog_title",
        "hatenablog_url",
        "hatenablog_created_at",
        "hatenablog_updated_at",
    ):
        value = data.get(key)
        if isinstance(value, str):
            metadata[key] = value
    return metadata


def _write_entry_metadata(out_path: Path, entry: HatenablogEntry) -> None:
    metadata = {
        "hatenablog_entry_id": entry.entry_id,
        "hatenablog_title": entry.title,
        "hatenablog_url": entry.url,
        "hatenablog_created_at": entry.created_at,
        "hatenablog_updated_at": entry.updated_at,
    }
    _metadata_sidecar_path(out_path).write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )


def _is_entry_up_to_date(out_path: Path, entry: HatenablogEntry) -> bool:
    if not out_path.exists():
        return False

    metadata = _read_entry_metadata(out_path)
    if not metadata:
        return False

    entry_id = metadata.get("hatenablog_entry_id") or metadata.get("hatenablog_url")
    if entry_id != (entry.entry_id or entry.url):
        return False

    current_revision = entry.updated_at or entry.created_at
    stored_revision = metadata.get("hatenablog_updated_at") or metadata.get(
        "hatenablog_created_at"
    )
    if not current_revision or not stored_revision:
        return False
    return stored_revision == current_revision


def _cleanup_entry_duplicates(*, output_dir: Path, entry: HatenablogEntry, keep_path: Path) -> None:
    key = _entry_key(entry)
    keep_meta = _metadata_sidecar_path(keep_path)
    for path in output_dir.glob(f"{key}{FILE_ID_SEPARATOR}*"):
        if path == keep_path or path == keep_meta:
            continue
        if path.is_dir():
            continue
        try:
            path.unlink()
            logger.info("Removed stale Hatenablog export %s", path.name)
        except Exception as exc:
            logger.warning(
                "Failed to remove stale Hatenablog export %s: %s",
                path.name,
                exc,
            )


def _cleanup_missing_entries(*, output_dir: Path, valid_entry_keys: set[str]) -> None:
    for path in output_dir.glob("*.txt"):
        entry_key = _extract_entry_key(path.name)
        if not entry_key:
            continue
        if entry_key in valid_entry_keys:
            continue
        try:
            path.unlink()
            logger.info("Removed deleted Hatenablog export %s", path.name)
        except Exception as exc:
            logger.warning(
                "Failed to remove deleted Hatenablog export %s: %s",
                path.name,
                exc,
            )
            continue

        meta_path = _metadata_sidecar_path(path)
        if not meta_path.exists():
            continue
        try:
            meta_path.unlink()
            logger.info("Removed deleted Hatenablog metadata %s", meta_path.name)
        except Exception as exc:
            logger.warning(
                "Failed to remove deleted Hatenablog metadata %s: %s",
                meta_path.name,
                exc,
            )


def _build_output_filename(entry: HatenablogEntry) -> str:
    key = _entry_key(entry)
    normalized_url_path = re.sub(r"^https?://", "", entry.url, flags=re.IGNORECASE)
    safe_slug = sanitize_filename(normalized_url_path.replace("/", "__"))
    return f"{key}{FILE_ID_SEPARATOR}{safe_slug}.txt"


def _extract_article_text(entry: HatenablogEntry) -> str:
    raw_html = entry.content_html
    if not raw_html:
        raw_html = _http_get_text(entry.url)
    extractor = _ArticleTextExtractor()
    extractor.feed(raw_html)
    extractor.close()
    return extractor.text()


def download_hatenablog_articles(
    *,
    blog_url: str,
    output_dir: Path,
    skip_existing: bool = False,
    update_existing: bool = True,
    sync_deleted: bool = False,
) -> int:
    ensure_dir(output_dir)
    normalized_blog_url = (blog_url or "").strip()
    if not normalized_blog_url:
        logger.warning("Hatenablog URL is empty. Skipping.")
        return 0

    entries = _collect_entries(normalized_blog_url)
    if not entries:
        logger.warning("No Hatenablog entries found: %s", normalized_blog_url)
        return 0

    if sync_deleted:
        valid_entry_keys = {_entry_key(entry) for entry in entries}
        _cleanup_missing_entries(
            output_dir=output_dir,
            valid_entry_keys=valid_entry_keys,
        )

    downloaded_count = 0
    for entry in entries:
        out_path = output_dir / _build_output_filename(entry)
        _cleanup_entry_duplicates(
            output_dir=output_dir,
            entry=entry,
            keep_path=out_path,
        )
        if skip_existing and out_path.exists():
            if not update_existing:
                logger.info("Skip Hatenablog download (exists): %s", out_path.name)
                continue
            if _is_entry_up_to_date(out_path, entry):
                logger.info(
                    "Skip Hatenablog download (up-to-date): %s",
                    out_path.name,
                )
                continue

        try:
            text = _extract_article_text(entry)
            if not text:
                logger.warning("Empty Hatenablog article body: %s", entry.url)
                continue
            out_path.write_text(text, encoding="utf-8")
            _write_entry_metadata(out_path, entry)
            downloaded_count += 1
            logger.info("Downloaded Hatenablog article: %s", entry.url)
        except Exception:
            logger.exception("Failed to download Hatenablog article: %s", entry.url)

    logger.info("Downloaded %d Hatenablog articles", downloaded_count)
    return downloaded_count
