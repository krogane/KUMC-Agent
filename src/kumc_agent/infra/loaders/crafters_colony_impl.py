from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import ParseResult, urljoin, urlparse
from urllib.request import Request, urlopen

from kumc_agent.infra.loaders.common import FILE_ID_SEPARATOR, ensure_dir, sanitize_filename

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_PAGES = 100
_ARTICLE_PATH_RE = re.compile(r"^/\d+/?$")
_META_TITLE_RE = re.compile(
    r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)
_META_PUBLISHED_RE = re.compile(
    r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\'](.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TIME_RE = re.compile(
    r"<time[^>]+datetime=[\"'](.*?)[\"'][^>]*>",
    re.IGNORECASE | re.DOTALL,
)
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class CraftersColonyEntry:
    article_url: str
    title: str
    published_at: str


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value.strip())
                return


class _ArticleBodyParser(HTMLParser):
    _TARGET_CLASS_KEYWORDS = (
        "entry-content",
        "post-content",
        "article-content",
        "single-content",
        "the-content",
        "entry__content",
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._capturing_depth = 0
        self._started = False
        self._parts: list[str] = []

    @property
    def html(self) -> str:
        return "".join(self._parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        class_value = attrs_dict.get("class", "").lower()
        should_start = False

        if self._capturing_depth == 0 and not self._started:
            if lower_tag == "article":
                should_start = True
            elif lower_tag in {"div", "section", "main"} and any(
                key in class_value for key in self._TARGET_CLASS_KEYWORDS
            ):
                should_start = True

        if should_start:
            self._started = True
            self._capturing_depth = 1
            self._parts.append(_start_tag_text(tag=tag, attrs=attrs))
            return

        if self._capturing_depth > 0:
            self._capturing_depth += 1
            self._parts.append(_start_tag_text(tag=tag, attrs=attrs))

    def handle_endtag(self, tag: str) -> None:
        if self._capturing_depth <= 0:
            return
        self._parts.append(f"</{tag}>")
        self._capturing_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._capturing_depth > 0:
            self._parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._capturing_depth > 0:
            self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._capturing_depth > 0:
            self._parts.append(f"&#{name};")


def _start_tag_text(*, tag: str, attrs: list[tuple[str, str | None]]) -> str:
    if not attrs:
        return f"<{tag}>"
    parts: list[str] = []
    for key, value in attrs:
        if value is None:
            parts.append(key)
        else:
            escaped = html.escape(value, quote=True)
            parts.append(f'{key}="{escaped}"')
    return f"<{tag} {' '.join(parts)}>"


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


def _canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    path = parsed.path or "/"
    if path != "/" and not path.endswith("/"):
        path += "/"
    normalized = ParseResult(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=path,
        params="",
        query="",
        fragment="",
    )
    return normalized.geturl()


def _extract_links(html_text: str, *, page_url: str) -> list[str]:
    parser = _HrefParser()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        logger.exception("Failed to parse link tags from %s", page_url)
    links: list[str] = []
    for href in parser.hrefs:
        resolved = _canonicalize_url(urljoin(page_url, href))
        if resolved:
            links.append(resolved)
    return links


def _is_article_url(*, candidate: str, host: str) -> bool:
    parsed = urlparse(candidate)
    if parsed.netloc.lower() != host:
        return False
    return bool(_ARTICLE_PATH_RE.fullmatch(parsed.path or ""))


def _is_author_page_url(*, candidate: str, host: str, author_prefix: str) -> bool:
    parsed = urlparse(candidate)
    if parsed.netloc.lower() != host:
        return False
    path = parsed.path or "/"
    if not path.startswith(author_prefix):
        return False
    suffix = path[len(author_prefix) :]
    return not suffix or bool(re.fullmatch(r"page/\d+/?", suffix))


def _collect_article_urls(
    *,
    author_url: str,
    max_pages: int,
    max_articles: int,
) -> list[str]:
    normalized_author_url = _canonicalize_url(author_url)
    parsed_author = urlparse(normalized_author_url)
    host = parsed_author.netloc.lower()
    author_prefix = parsed_author.path
    if not author_prefix.endswith("/"):
        author_prefix += "/"

    pending: deque[str] = deque([normalized_author_url])
    seen_pages: set[str] = set()
    queued_pages: set[str] = {normalized_author_url}
    seen_articles: set[str] = set()
    article_urls: list[str] = []

    while pending:
        current_page = pending.popleft()
        queued_pages.discard(current_page)
        if current_page in seen_pages:
            continue
        if len(seen_pages) >= max_pages:
            logger.warning(
                "Crafters Colony pagination exceeded max pages (%d).", max_pages
            )
            break

        seen_pages.add(current_page)
        try:
            page_html = _http_get_text(current_page)
        except Exception:
            logger.exception("Failed to fetch Crafters Colony author page: %s", current_page)
            continue

        for link in _extract_links(page_html, page_url=current_page):
            if _is_article_url(candidate=link, host=host):
                if link in seen_articles:
                    continue
                seen_articles.add(link)
                article_urls.append(link)
                if max_articles > 0 and len(article_urls) >= max_articles:
                    return article_urls
                continue

            if _is_author_page_url(
                candidate=link,
                host=host,
                author_prefix=author_prefix,
            ):
                if link in seen_pages or link in queued_pages:
                    continue
                pending.append(link)
                queued_pages.add(link)

    return article_urls


def _extract_title(html_text: str) -> str:
    match = _META_TITLE_RE.search(html_text)
    if match:
        return html.unescape(match.group(1)).strip()

    title_match = _TITLE_RE.search(html_text)
    if not title_match:
        return ""

    raw = html.unescape(title_match.group(1)).strip()
    if " | " in raw:
        return raw.split(" | ", 1)[0].strip()
    if " - " in raw:
        return raw.split(" - ", 1)[0].strip()
    return raw


def _extract_published_at(html_text: str) -> str:
    match = _META_PUBLISHED_RE.search(html_text)
    if match:
        return html.unescape(match.group(1)).strip()

    time_match = _TIME_RE.search(html_text)
    if time_match:
        return html.unescape(time_match.group(1)).strip()

    return ""


def _extract_article_body_html(html_text: str) -> str:
    parser = _ArticleBodyParser()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        logger.exception("Failed to parse article body HTML.")

    body_html = parser.html.strip()
    if body_html:
        return body_html
    return html_text


def _html_to_markdown(html_fragment: str) -> str:
    text = _SCRIPT_STYLE_RE.sub("", html_fragment)

    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li\b[^>]*>", "\n- ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"<h([1-6])\b[^>]*>",
        lambda m: "\n" + ("#" * int(m.group(1))) + " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"</(p|div|section|article|ul|ol|li|h[1-6]|table|tr|blockquote)>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)

    lines = [line.strip() for line in text.splitlines()]
    markdown = "\n".join(line for line in lines if line)
    markdown = _MULTI_NEWLINE_RE.sub("\n\n", markdown)
    return markdown.strip()


def _build_markdown(entry: CraftersColonyEntry, *, body_markdown: str) -> str:
    lines = [f"# {entry.title or entry.article_url}", ""]
    lines.append(f"- URL: {entry.article_url}")
    if entry.published_at:
        lines.append(f"- Published: {entry.published_at}")
    lines.append("")
    lines.append(body_markdown.strip())
    return "\n".join(lines).strip() + "\n"


def _article_key(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return digest[:16]


def _extract_article_key(filename: str) -> str | None:
    if FILE_ID_SEPARATOR not in filename:
        return None
    prefix, _ = filename.split(FILE_ID_SEPARATOR, 1)
    return prefix or None


def _build_output_filename(article_url: str) -> str:
    key = _article_key(article_url)
    normalized_url_path = re.sub(r"^https?://", "", article_url, flags=re.IGNORECASE)
    safe_slug = sanitize_filename(normalized_url_path.replace("/", "__"))
    return f"{key}{FILE_ID_SEPARATOR}{safe_slug}.md"


def _metadata_sidecar_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".meta.json")


def _write_metadata(out_path: Path, entry: CraftersColonyEntry, *, author_url: str) -> None:
    metadata = {
        "crafters_colony_article_url": entry.article_url,
        "crafters_colony_title": entry.title,
        "crafters_colony_published_at": entry.published_at,
        "crafters_colony_author_url": author_url,
    }
    _metadata_sidecar_path(out_path).write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_metadata(out_path: Path) -> dict[str, str]:
    meta_path = _metadata_sidecar_path(out_path)
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Failed to read Crafters Colony metadata sidecar %s: %s",
            meta_path.name,
            exc,
        )
        return {}
    if not isinstance(data, dict):
        return {}

    metadata: dict[str, str] = {}
    for key in (
        "crafters_colony_article_url",
        "crafters_colony_title",
        "crafters_colony_published_at",
        "crafters_colony_author_url",
    ):
        value = data.get(key)
        if isinstance(value, str):
            metadata[key] = value
    return metadata


def _is_entry_up_to_date(out_path: Path, entry: CraftersColonyEntry) -> bool:
    if not out_path.exists():
        return False

    metadata = _read_metadata(out_path)
    if not metadata:
        return False

    if metadata.get("crafters_colony_article_url") != entry.article_url:
        return False

    current_revision = entry.published_at
    stored_revision = metadata.get("crafters_colony_published_at")
    if not current_revision or not stored_revision:
        return False

    return current_revision == stored_revision


def _cleanup_entry_duplicates(*, output_dir: Path, article_url: str, keep_path: Path) -> None:
    key = _article_key(article_url)
    keep_meta = _metadata_sidecar_path(keep_path)
    for path in output_dir.glob(f"{key}{FILE_ID_SEPARATOR}*"):
        if path == keep_path or path == keep_meta:
            continue
        if path.is_dir():
            continue
        try:
            path.unlink()
            logger.info("Removed stale Crafters Colony export %s", path.name)
        except Exception as exc:
            logger.warning(
                "Failed to remove stale Crafters Colony export %s: %s",
                path.name,
                exc,
            )


def _cleanup_missing_entries(*, output_dir: Path, valid_article_keys: set[str]) -> None:
    for pattern in ("*.md", "*.txt"):
        for path in output_dir.glob(pattern):
            article_key = _extract_article_key(path.name)
            if not article_key:
                continue
            if article_key in valid_article_keys:
                continue
            try:
                path.unlink()
                logger.info("Removed deleted Crafters Colony export %s", path.name)
            except Exception as exc:
                logger.warning(
                    "Failed to remove deleted Crafters Colony export %s: %s",
                    path.name,
                    exc,
                )
                continue

            meta_path = _metadata_sidecar_path(path)
            if not meta_path.exists():
                continue
            try:
                meta_path.unlink()
                logger.info(
                    "Removed deleted Crafters Colony metadata %s", meta_path.name
                )
            except Exception as exc:
                logger.warning(
                    "Failed to remove deleted Crafters Colony metadata %s: %s",
                    meta_path.name,
                    exc,
                )


def _fetch_entry(*, article_url: str) -> tuple[CraftersColonyEntry, str]:
    article_html = _http_get_text(article_url)
    title = _extract_title(article_html)
    published_at = _extract_published_at(article_html)
    body_html = _extract_article_body_html(article_html)
    body_markdown = _html_to_markdown(body_html)
    entry = CraftersColonyEntry(
        article_url=article_url,
        title=title,
        published_at=published_at,
    )
    markdown = _build_markdown(entry, body_markdown=body_markdown)
    return entry, markdown


def download_crafters_colony_articles(
    *,
    author_url: str,
    output_dir: Path,
    max_pages: int = _DEFAULT_MAX_PAGES,
    max_articles: int = 0,
    skip_existing: bool = False,
    update_existing: bool = True,
    sync_deleted: bool = False,
) -> int:
    ensure_dir(output_dir)
    normalized_author_url = _canonicalize_url((author_url or "").strip())
    if not normalized_author_url:
        logger.warning("Crafters Colony author URL is empty. Skipping.")
        return 0

    page_limit = max(1, int(max_pages))
    article_limit = max(0, int(max_articles))

    article_urls = _collect_article_urls(
        author_url=normalized_author_url,
        max_pages=page_limit,
        max_articles=article_limit,
    )
    if not article_urls:
        logger.warning(
            "No Crafters Colony articles found for author: %s",
            normalized_author_url,
        )
        return 0

    if sync_deleted:
        valid_article_keys = {_article_key(url) for url in article_urls}
        _cleanup_missing_entries(
            output_dir=output_dir,
            valid_article_keys=valid_article_keys,
        )

    downloaded_count = 0
    for article_url in article_urls:
        expected_out_path = output_dir / _build_output_filename(article_url)
        _cleanup_entry_duplicates(
            output_dir=output_dir,
            article_url=article_url,
            keep_path=expected_out_path,
        )
        if skip_existing and expected_out_path.exists() and not update_existing:
            logger.info(
                "Skip Crafters Colony download (exists): %s",
                expected_out_path.name,
            )
            continue

        try:
            entry, markdown = _fetch_entry(article_url=article_url)
            out_path = output_dir / _build_output_filename(entry.article_url)
            _cleanup_entry_duplicates(
                output_dir=output_dir,
                article_url=entry.article_url,
                keep_path=out_path,
            )
            if skip_existing and out_path.exists():
                if not update_existing:
                    logger.info("Skip Crafters Colony download (exists): %s", out_path.name)
                    continue
                if _is_entry_up_to_date(out_path, entry):
                    logger.info("Skip Crafters Colony download (up-to-date): %s", out_path.name)
                    continue

            if not markdown.strip():
                logger.warning("Empty Crafters Colony article body: %s", entry.article_url)
                continue

            out_path.write_text(markdown, encoding="utf-8")
            _write_metadata(out_path, entry, author_url=normalized_author_url)
            downloaded_count += 1
            logger.info("Downloaded Crafters Colony article: %s", entry.article_url)
        except Exception:
            logger.exception("Failed to download Crafters Colony article: %s", article_url)

    logger.info("Downloaded %d Crafters Colony articles", downloaded_count)
    return downloaded_count
