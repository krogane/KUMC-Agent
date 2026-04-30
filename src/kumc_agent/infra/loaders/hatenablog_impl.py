from __future__ import annotations

import hashlib
import html
from html.parser import HTMLParser
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from kumc_agent.infra.loaders.common import FILE_ID_SEPARATOR, ensure_dir, sanitize_filename

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


@dataclass(frozen=True)
class NormalizedHatenablogArticle:
    markdown: str
    metadata: dict[str, object]


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


def _read_entry_metadata(out_path: Path) -> dict[str, object]:
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

    metadata: dict[str, object] = {}
    for key in (
        "hatenablog_entry_id",
        "hatenablog_title",
        "hatenablog_url",
        "hatenablog_created_at",
        "hatenablog_updated_at",
        "source_kind",
        "source_type",
        "normalized_format",
    ):
        value = data.get(key)
        if isinstance(value, str):
            metadata[key] = value
    return metadata


def _write_entry_metadata(
    out_path: Path,
    entry: HatenablogEntry,
    *,
    normalized_metadata: dict[str, object] | None = None,
) -> None:
    metadata = {
        "source_kind": "hatenablog",
        "source_type": "hatenablog",
        "normalized_format": "markdown",
        "hatenablog_entry_id": entry.entry_id,
        "hatenablog_title": entry.title,
        "hatenablog_url": entry.url,
        "hatenablog_created_at": entry.created_at,
        "hatenablog_updated_at": entry.updated_at,
    }
    if normalized_metadata:
        metadata.update(normalized_metadata)
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
    for pattern in ("*.md", "*.txt"):
        for path in output_dir.glob(pattern):
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
    return f"{key}{FILE_ID_SEPARATOR}{safe_slug}.md"


def _extract_article_markdown(entry: HatenablogEntry) -> str:
    return _extract_article(entry).markdown


def _extract_article(entry: HatenablogEntry) -> NormalizedHatenablogArticle:
    raw_html = entry.content_html
    if not raw_html:
        raw_html = _http_get_text(entry.url)
    return _normalize_hatenablog_html(raw_html)


def _normalize_hatenablog_html(raw_html: str) -> NormalizedHatenablogArticle:
    parser = _HatenaHTMLToMarkdownParser()
    parser.feed(raw_html or "")
    parser.close()
    return parser.article()


class _HatenaHTMLToMarkdownParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._blocks: list[str] = []
        self._contexts: list[dict[str, object]] = []
        self._skip_stack: list[str] = []
        self._related_stack: list[str] = []
        self._related_links: list[dict[str, str]] = []
        self._images: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_map = {key.lower(): value or "" for key, value in attrs}
        if self._skip_stack:
            self._skip_stack.append(tag)
            return
        classes = _split_classes(attrs_map.get("class", ""))
        element_id = attrs_map.get("id", "")
        if tag in {"script", "style"} or _is_table_of_contents(classes, element_id):
            self._skip_stack.append(tag)
            return
        if tag in {"div", "section", "aside"} and _is_related_container(classes):
            self._related_stack.append(tag)
            self._contexts.append(
                {
                    "tag": tag,
                    "kind": "related",
                    "parts": [],
                    "href": attrs_map.get("href", ""),
                    "class": " ".join(classes),
                }
            )
            return
        if tag in {"h2", "h3", "h4"}:
            self._flush_inline_contexts()
            self._contexts.append(
                {
                    "tag": tag,
                    "kind": "heading",
                    "level": int(tag[1]),
                    "parts": [],
                }
            )
            return
        if tag in {"p", "li", "figcaption"}:
            self._contexts.append({"tag": tag, "kind": tag, "parts": []})
            return
        if tag == "a":
            self._contexts.append(
                {
                    "tag": tag,
                    "kind": "link",
                    "parts": [],
                    "href": attrs_map.get("href", ""),
                    "is_keyword": "keyword" in classes,
                    "is_related": bool(self._related_stack) or _is_related_container(classes),
                }
            )
            return
        if tag == "img":
            self._append_image(attrs_map)
            return
        if tag == "iframe":
            src = attrs_map.get("src", "").strip()
            title = attrs_map.get("title", "").strip()
            if src:
                self._append_related_link(url=src, title=title)
            return
        if tag == "br":
            self._append_text("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_stack:
            self._skip_stack.pop()
            return
        if self._related_stack and tag == self._related_stack[-1]:
            self._related_stack.pop()
        index = self._find_context_index(tag)
        if index is None:
            return
        context = self._contexts.pop(index)
        rendered = self._render_context(context)
        if rendered:
            self._append_rendered(rendered, context=context)

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        self._append_text(data)

    def article(self) -> NormalizedHatenablogArticle:
        self._flush_inline_contexts()
        blocks = [block for block in self._blocks if block.strip()]
        if self._related_links:
            blocks.append("## 関連リンク")
            seen: set[str] = set()
            for link in self._related_links:
                url = link.get("url", "").strip()
                title = link.get("title", "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                blocks.append(f"- {title}: {url}" if title else f"- {url}")
        markdown = "\n\n".join(block.strip() for block in blocks if block.strip())
        markdown = _normalize_markdown(markdown)
        return NormalizedHatenablogArticle(
            markdown=markdown,
            metadata={
                "hatenablog_html_normalized": True,
                "hatenablog_image_count": len(self._images),
                "hatenablog_images": self._images,
                "hatenablog_related_link_count": len(self._related_links),
                "hatenablog_related_links": self._related_links,
            },
        )

    def _append_text(self, text: str) -> None:
        value = html.unescape(text or "").replace("\xa0", " ")
        if not value:
            return
        if self._contexts:
            parts = self._contexts[-1].setdefault("parts", [])
            if isinstance(parts, list):
                parts.append(value)
            return
        cleaned = _normalize_inline_text(value)
        if cleaned:
            self._blocks.append(cleaned)

    def _append_image(self, attrs: dict[str, str]) -> None:
        src = attrs.get("src", "").strip()
        if not src:
            return
        caption = (
            attrs.get("alt", "").strip()
            or attrs.get("title", "").strip()
            or attrs.get("aria-label", "").strip()
        )
        self._images.append({"url": src, "caption": caption})
        rendered = f"![{caption}]({src})" if caption else f"![]({src})"
        self._append_rendered(rendered, context=None)

    def _append_related_link(self, *, url: str, title: str = "") -> None:
        self._related_links.append({"url": url.strip(), "title": title.strip()})

    def _append_rendered(
        self,
        rendered: str,
        *,
        context: dict[str, object] | None,
    ) -> None:
        if not rendered.strip():
            return
        if self._contexts:
            parts = self._contexts[-1].setdefault("parts", [])
            if isinstance(parts, list):
                parts.append(rendered)
            return
        if context and context.get("kind") == "related":
            text = _normalize_inline_text(rendered)
            if text and not any(link.get("title") == text for link in self._related_links):
                self._append_related_link(url="", title=text)
            return
        self._blocks.append(rendered.strip())

    def _render_context(self, context: dict[str, object]) -> str:
        kind = str(context.get("kind") or "")
        text = _normalize_inline_text("".join(str(part) for part in context.get("parts", [])))
        if not text:
            return ""
        if kind == "heading":
            level = int(context.get("level") or 2)
            level = max(2, min(4, level))
            return f"{'#' * level} {text}"
        if kind == "li":
            return f"- {text}"
        if kind == "link":
            href = str(context.get("href") or "").strip()
            if context.get("is_related") and href:
                self._append_related_link(url=href, title=text)
                return ""
            if context.get("is_keyword") or not href or href == text:
                return text
            return f"{text} ({href})"
        if kind == "related":
            href = str(context.get("href") or "").strip()
            if href:
                self._append_related_link(url=href, title=text)
            return ""
        return text

    def _flush_inline_contexts(self) -> None:
        while self._contexts:
            context = self._contexts.pop()
            rendered = self._render_context(context)
            if rendered:
                self._blocks.append(rendered)

    def _find_context_index(self, tag: str) -> int | None:
        for index in range(len(self._contexts) - 1, -1, -1):
            if self._contexts[index].get("tag") == tag:
                return index
        return None


def _split_classes(value: str) -> set[str]:
    return {part.strip().lower() for part in str(value or "").split() if part.strip()}


def _is_table_of_contents(classes: set[str], element_id: str) -> bool:
    tokens = set(classes)
    if element_id:
        tokens.add(element_id.strip().lower())
    return any(token in {"table-of-contents", "toc"} for token in tokens)


def _is_related_container(classes: set[str]) -> bool:
    return any(
        token in {"hatena-citation", "embed-card", "hatena-asin-detail"}
        for token in classes
    )


def _normalize_inline_text(value: str) -> str:
    lines = [" ".join(line.split()) for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _normalize_markdown(value: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", str(value or ""))
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


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
            article = _extract_article(entry)
            if not article.markdown.strip():
                logger.warning("Empty Hatenablog article body: %s", entry.url)
                continue
            out_path.write_text(article.markdown, encoding="utf-8")
            _write_entry_metadata(
                out_path,
                entry,
                normalized_metadata=article.metadata,
            )
            downloaded_count += 1
            logger.info("Downloaded Hatenablog article: %s", entry.url)
        except Exception:
            logger.exception("Failed to download Hatenablog article: %s", entry.url)

    logger.info("Downloaded %d Hatenablog articles", downloaded_count)
    return downloaded_count
