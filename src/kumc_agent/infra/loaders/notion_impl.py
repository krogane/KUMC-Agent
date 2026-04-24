from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from kumc_agent.infra.loaders.common import FILE_ID_SEPARATOR, ensure_dir, sanitize_filename

logger = logging.getLogger(__name__)

_NOTION_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"
_REQUEST_TIMEOUT_SECONDS = 30
_MAX_BLOCK_RECURSION_DEPTH = 20
_NOTION_ID_HEX_RE = re.compile(r"[0-9a-fA-F]{32}")


@dataclass(frozen=True)
class NotionPage:
    page_id: str
    title: str
    url: str
    last_edited_time: str
    created_time: str


@dataclass(frozen=True)
class NotionSyncStats:
    databases: int = 0
    pages_seen: int = 0
    pages_updated: int = 0
    pages_skipped: int = 0
    pages_deleted: int = 0


def download_notion_database_pages(
    *,
    api_token: str,
    database_ids: list[str],
    output_dir: Path,
    skip_existing: bool,
    update_existing: bool,
    sync_deleted: bool,
) -> int:
    ensure_dir(output_dir)
    token = (api_token or "").strip()
    if not token:
        logger.warning("Notion API token is empty. Skipping Notion sync.")
        return 0

    stats = NotionSyncStats()
    for raw_database_id in database_ids:
        database_id = _normalize_notion_id(raw_database_id)
        if not database_id:
            logger.warning("Skip invalid Notion database id: %s", raw_database_id)
            continue

        db_dir = output_dir / database_id
        ensure_dir(db_dir)
        try:
            pages = _list_database_pages(
                api_token=token,
                database_id=database_id,
            )
        except Exception:
            logger.exception("Failed to list Notion database pages: %s", database_id)
            continue

        pages_seen = 0
        pages_updated = 0
        pages_skipped = 0
        valid_page_ids: set[str] = set()

        for page in pages:
            page_id = _normalize_notion_id(page.page_id)
            if not page_id:
                continue
            valid_page_ids.add(page_id)
            pages_seen += 1

            output_path = _page_output_path(db_dir=db_dir, page=page)
            _cleanup_page_duplicates(db_dir=db_dir, page_id=page_id, keep_path=output_path)

            if _should_skip_page(
                output_path=output_path,
                page=page,
                skip_existing=skip_existing,
                update_existing=update_existing,
            ):
                pages_skipped += 1
                continue

            try:
                markdown = _render_page_markdown(
                    api_token=token,
                    page=page,
                )
            except Exception:
                logger.exception("Failed to render Notion page markdown: %s", page_id)
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
            _write_page_metadata(output_path=output_path, page=page, database_id=database_id)
            pages_updated += 1

        deleted = 0
        if sync_deleted:
            deleted = _cleanup_deleted_pages(db_dir=db_dir, valid_page_ids=valid_page_ids)

        logger.info(
            "Synced Notion database %s (seen=%d updated=%d skipped=%d deleted=%d)",
            database_id,
            pages_seen,
            pages_updated,
            pages_skipped,
            deleted,
        )
        stats = NotionSyncStats(
            databases=stats.databases + 1,
            pages_seen=stats.pages_seen + pages_seen,
            pages_updated=stats.pages_updated + pages_updated,
            pages_skipped=stats.pages_skipped + pages_skipped,
            pages_deleted=stats.pages_deleted + deleted,
        )

    logger.info(
        "Notion sync completed (databases=%d seen=%d updated=%d skipped=%d deleted=%d)",
        stats.databases,
        stats.pages_seen,
        stats.pages_updated,
        stats.pages_skipped,
        stats.pages_deleted,
    )
    return stats.pages_updated


def _notion_request_json(
    *,
    api_token: str,
    method: str,
    path: str,
    query: dict[str, object] | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    encoded_query = ""
    if query:
        encoded_query = urlencode(
            {key: value for key, value in query.items() if value is not None}
        )
    url = f"{_NOTION_API_BASE}{path}"
    if encoded_query:
        url = f"{url}?{encoded_query}"

    body: bytes | None = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = Request(
        url=url,
        method=method,
        data=body,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            content = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Notion API request failed: {exc.code} {method} {path} {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Notion API request failed: {method} {path} {exc}") from exc

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Notion API response is not valid JSON: {method} {path}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Notion API response root is not object: {method} {path}")
    return parsed


def _list_database_pages(*, api_token: str, database_id: str) -> list[NotionPage]:
    pages: list[NotionPage] = []
    start_cursor: str | None = None
    while True:
        body: dict[str, object] = {"page_size": 100}
        if start_cursor:
            body["start_cursor"] = start_cursor

        response = _notion_request_json(
            api_token=api_token,
            method="POST",
            path=f"/databases/{database_id}/query",
            payload=body,
        )

        results = response.get("results")
        if isinstance(results, list):
            for row in results:
                if not isinstance(row, dict):
                    continue
                if str(row.get("object") or "") != "page":
                    continue
                if bool(row.get("archived", False)) or bool(row.get("in_trash", False)):
                    continue
                page_id = str(row.get("id") or "").strip()
                if not page_id:
                    continue
                pages.append(
                    NotionPage(
                        page_id=page_id,
                        title=_extract_page_title(row),
                        url=str(row.get("url") or "").strip(),
                        last_edited_time=str(row.get("last_edited_time") or "").strip(),
                        created_time=str(row.get("created_time") or "").strip(),
                    )
                )

        has_more = bool(response.get("has_more", False))
        next_cursor = response.get("next_cursor")
        if not has_more:
            break
        if not isinstance(next_cursor, str) or not next_cursor.strip():
            break
        start_cursor = next_cursor.strip()

    return pages


def _extract_page_title(page_payload: dict[str, object]) -> str:
    properties = page_payload.get("properties")
    if not isinstance(properties, dict):
        return "Untitled"
    for value in properties.values():
        if not isinstance(value, dict):
            continue
        if str(value.get("type") or "") != "title":
            continue
        title_parts = value.get("title")
        if isinstance(title_parts, list):
            text = _rich_text_plain(title_parts)
            if text:
                return text
    return "Untitled"


def _rich_text_plain(values: list[object]) -> str:
    parts: list[str] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        plain = str(value.get("plain_text") or "")
        if plain:
            parts.append(plain)
    return "".join(parts).strip()


def _render_page_markdown(*, api_token: str, page: NotionPage) -> str:
    lines: list[str] = []
    title = (page.title or "Untitled").strip()
    if title:
        lines.append(f"# {title}")

    body_lines = _collect_block_lines(
        api_token=api_token,
        block_id=page.page_id,
        depth=0,
        visited=set(),
    )
    if body_lines:
        if lines:
            lines.append("")
        lines.extend(body_lines)

    if not lines:
        return ""
    return _normalize_markdown_lines(lines)


def _collect_block_lines(
    *,
    api_token: str,
    block_id: str,
    depth: int,
    visited: set[str],
) -> list[str]:
    if depth > _MAX_BLOCK_RECURSION_DEPTH:
        return []
    normalized_block_id = _normalize_notion_id(block_id)
    if not normalized_block_id:
        return []
    if normalized_block_id in visited:
        return []
    visited.add(normalized_block_id)

    lines: list[str] = []
    start_cursor: str | None = None
    while True:
        query = {"page_size": 100}
        if start_cursor:
            query["start_cursor"] = start_cursor

        response = _notion_request_json(
            api_token=api_token,
            method="GET",
            path=f"/blocks/{normalized_block_id}/children",
            query=query,
        )

        results = response.get("results")
        if isinstance(results, list):
            for row in results:
                if not isinstance(row, dict):
                    continue
                block_lines = _block_to_lines(row)
                if block_lines:
                    lines.extend(block_lines)
                if bool(row.get("has_children", False)):
                    child_lines = _collect_block_lines(
                        api_token=api_token,
                        block_id=str(row.get("id") or ""),
                        depth=depth + 1,
                        visited=visited,
                    )
                    if child_lines:
                        lines.extend(child_lines)

        has_more = bool(response.get("has_more", False))
        next_cursor = response.get("next_cursor")
        if not has_more:
            break
        if not isinstance(next_cursor, str) or not next_cursor.strip():
            break
        start_cursor = next_cursor.strip()

    return lines


def _block_to_lines(block: dict[str, object]) -> list[str]:
    block_type = str(block.get("type") or "").strip()
    if not block_type:
        return []

    payload = block.get(block_type)
    if not isinstance(payload, dict):
        payload = {}

    text = _rich_text_plain(payload.get("rich_text") if isinstance(payload.get("rich_text"), list) else [])

    if block_type == "paragraph":
        return [text] if text else []
    if block_type == "heading_1":
        return [f"# {text}" if text else "#"]
    if block_type == "heading_2":
        return [f"## {text}" if text else "##"]
    if block_type == "heading_3":
        return [f"### {text}" if text else "###"]
    if block_type == "bulleted_list_item":
        return [f"- {text}" if text else "-"]
    if block_type == "numbered_list_item":
        return [f"1. {text}" if text else "1."]
    if block_type == "to_do":
        checked = bool(payload.get("checked", False))
        marker = "x" if checked else " "
        return [f"- [{marker}] {text}".rstrip()]
    if block_type == "quote":
        return [f"> {text}" if text else ">"]
    if block_type == "callout":
        return [f"> {text}" if text else ">"]
    if block_type == "toggle":
        return [f"Toggle: {text}" if text else "Toggle"]
    if block_type == "code":
        language = str(payload.get("language") or "").strip()
        fenced_header = f"```{language}" if language else "```"
        code_text = text if text else ""
        return [fenced_header, code_text, "```"]
    if block_type == "child_page":
        title = str(payload.get("title") or "").strip()
        return [f"## {title}" if title else "## Child Page"]
    if block_type == "bookmark":
        url = str(payload.get("url") or "").strip()
        return [url] if url else []
    if block_type == "link_preview":
        url = str(payload.get("url") or "").strip()
        return [url] if url else []
    if block_type == "equation":
        expression = str(payload.get("expression") or "").strip()
        return [expression] if expression else []
    if block_type == "table_row":
        cells = payload.get("cells")
        if not isinstance(cells, list):
            return []
        cell_values: list[str] = []
        for cell in cells:
            if not isinstance(cell, list):
                cell_values.append("")
                continue
            cell_values.append(_rich_text_plain(cell))
        row = " | ".join(cell_values).strip()
        return [row] if row else []
    if block_type == "divider":
        return ["---"]

    if text:
        return [text]
    return []


def _normalize_markdown_lines(lines: list[str]) -> str:
    normalized: list[str] = []
    previous_blank = False
    for raw in lines:
        line = str(raw or "").rstrip()
        is_blank = line == ""
        if is_blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = is_blank
    return "\n".join(normalized).strip() + "\n"


def _page_output_path(*, db_dir: Path, page: NotionPage) -> Path:
    page_id = _normalize_notion_id(page.page_id)
    safe_title = sanitize_filename((page.title or "").strip())
    if not safe_title:
        safe_title = "notion_page"
    return db_dir / f"{page_id}{FILE_ID_SEPARATOR}{safe_title}.md"


def _metadata_sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".meta.json")


def _read_page_metadata(output_path: Path) -> dict[str, object]:
    meta_path = _metadata_sidecar_path(output_path)
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_page_metadata(*, output_path: Path, page: NotionPage, database_id: str) -> None:
    metadata = {
        "source_type": "notion",
        "notion_database_id": _normalize_notion_id(database_id),
        "notion_page_id": _normalize_notion_id(page.page_id),
        "notion_title": page.title,
        "notion_url": page.url,
        "notion_created_time": page.created_time,
        "notion_last_edited_time": page.last_edited_time,
        "source_date": _iso_to_source_date(page.last_edited_time),
        "updated_at": page.last_edited_time,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
    _metadata_sidecar_path(output_path).write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )


def _should_skip_page(
    *,
    output_path: Path,
    page: NotionPage,
    skip_existing: bool,
    update_existing: bool,
) -> bool:
    if not skip_existing or not output_path.exists():
        return False
    if not update_existing:
        logger.info("Skip Notion page (exists): %s", output_path.name)
        return True
    if _is_page_up_to_date(output_path=output_path, page=page):
        logger.info("Skip Notion page (up-to-date): %s", output_path.name)
        return True
    return False


def _is_page_up_to_date(*, output_path: Path, page: NotionPage) -> bool:
    if not output_path.exists():
        return False
    metadata = _read_page_metadata(output_path)
    if not metadata:
        return False
    stored_page_id = _normalize_notion_id(str(metadata.get("notion_page_id") or ""))
    if stored_page_id != _normalize_notion_id(page.page_id):
        return False
    stored_last_edited = str(metadata.get("notion_last_edited_time") or "").strip()
    current_last_edited = (page.last_edited_time or "").strip()
    if not stored_last_edited or not current_last_edited:
        return False
    return stored_last_edited == current_last_edited


def _cleanup_page_duplicates(*, db_dir: Path, page_id: str, keep_path: Path) -> None:
    normalized_page_id = _normalize_notion_id(page_id)
    if not normalized_page_id:
        return
    keep_meta = _metadata_sidecar_path(keep_path)
    for path in db_dir.glob(f"{normalized_page_id}{FILE_ID_SEPARATOR}*.md"):
        if path == keep_path:
            continue
        try:
            path.unlink()
            logger.info("Removed stale Notion page file: %s", path.name)
        except Exception as exc:
            logger.warning("Failed to remove stale Notion page file %s: %s", path.name, exc)
            continue
        meta_path = _metadata_sidecar_path(path)
        if not meta_path.exists() or meta_path == keep_meta:
            continue
        try:
            meta_path.unlink()
            logger.info("Removed stale Notion metadata file: %s", meta_path.name)
        except Exception as exc:
            logger.warning("Failed to remove stale Notion metadata file %s: %s", meta_path.name, exc)


def _cleanup_deleted_pages(*, db_dir: Path, valid_page_ids: set[str]) -> int:
    deleted = 0
    for path in sorted(db_dir.glob("*.md"), key=lambda value: value.name):
        page_id = _extract_page_id_from_filename(path.name)
        if not page_id:
            continue
        if page_id in valid_page_ids:
            continue
        try:
            path.unlink()
            deleted += 1
            logger.info("Removed deleted Notion page file: %s", path.name)
        except Exception as exc:
            logger.warning("Failed to remove deleted Notion page file %s: %s", path.name, exc)
            continue
        meta_path = _metadata_sidecar_path(path)
        if not meta_path.exists():
            continue
        try:
            meta_path.unlink()
            logger.info("Removed deleted Notion metadata file: %s", meta_path.name)
        except Exception as exc:
            logger.warning("Failed to remove deleted Notion metadata file %s: %s", meta_path.name, exc)
    return deleted


def _extract_page_id_from_filename(file_name: str) -> str | None:
    if FILE_ID_SEPARATOR not in file_name:
        return None
    prefix, _ = file_name.split(FILE_ID_SEPARATOR, 1)
    normalized = _normalize_notion_id(prefix)
    return normalized or None


def _normalize_notion_id(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""

    candidate = raw
    if raw.lower().startswith(("http://", "https://")):
        candidate = _extract_id_from_url(raw)

    compact = re.sub(r"[^0-9a-fA-F]", "", candidate)
    if len(compact) == 32 and _NOTION_ID_HEX_RE.fullmatch(compact):
        return compact.lower()

    match = _NOTION_ID_HEX_RE.search(candidate)
    if match:
        return match.group(0).lower()
    return ""


def _extract_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = (parsed.path or "").strip("/")
    if not path:
        return ""
    return path.split("-")[-1]


def _iso_to_source_date(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    iso = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y/%m/%d")
