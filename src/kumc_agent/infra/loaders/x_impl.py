from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_JST = ZoneInfo("Asia/Tokyo")
_TWEET_ASSIGN_HEAD_RE = re.compile(r"window\.YTD\.tweets\.part\d+\s*=", re.MULTILINE)
_STATUS_URL_RE = re.compile(
    r"https?://(?:x|twitter)\.com/(?P<handle>[A-Za-z0-9_]+)/status/(?P<id>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class XConvertStats:
    files: int = 0
    posts: int = 0
    skipped_invalid: int = 0


def convert_x_tweets_js_to_jsonl(
    *,
    raw_x_dir: Path,
    output_path: Path,
    skip_existing: bool,
    update_existing: bool,
    sync_deleted: bool,
) -> XConvertStats:
    raw_x_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(raw_x_dir.rglob("tweets*.js"), key=lambda path: str(path))
    if not input_files:
        if sync_deleted and output_path.exists():
            output_path.unlink()
            logger.info("Removed stale X posts output: %s", output_path.name)
        logger.info("No tweets*.js found under %s. Skipping X conversion.", raw_x_dir)
        return XConvertStats(files=0, posts=0, skipped_invalid=0)

    output_exists = output_path.exists()
    if output_exists and skip_existing and not update_existing:
        logger.info("Skip X conversion (exists): %s", output_path.name)
        return XConvertStats(files=len(input_files), posts=0, skipped_invalid=0)

    if output_exists and update_existing:
        latest_input_mtime = max(path.stat().st_mtime for path in input_files)
        output_mtime = output_path.stat().st_mtime
        if output_mtime >= latest_input_mtime:
            logger.info("Skip X conversion (up-to-date): %s", output_path.name)
            return XConvertStats(files=len(input_files), posts=0, skipped_invalid=0)

    posts_by_id: dict[str, tuple[datetime, dict[str, object]]] = {}
    skipped_invalid = 0
    for path in input_files:
        try:
            entries = _load_tweet_entries(path)
        except Exception:
            logger.warning("Failed to parse X archive file %s", path, exc_info=True)
            continue
        for entry in entries:
            tweet = entry.get("tweet")
            if not isinstance(tweet, dict):
                skipped_invalid += 1
                continue
            parsed = _convert_tweet(tweet)
            if parsed is None:
                skipped_invalid += 1
                continue
            post_id = str(parsed["id"])
            current = posts_by_id.get(post_id)
            if current is None or parsed["created_at"] >= current[0]:
                posts_by_id[post_id] = (parsed["created_at"], parsed["record"])

    ordered = sorted(
        posts_by_id.values(),
        key=lambda item: (item[0], str(item[1].get("metadata", {}).get("message_id", ""))),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fw:
        for _, record in ordered:
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = XConvertStats(
        files=len(input_files),
        posts=len(ordered),
        skipped_invalid=skipped_invalid,
    )
    logger.info(
        "Converted X archive -> %s (files=%d posts=%d skipped=%d)",
        output_path,
        stats.files,
        stats.posts,
        stats.skipped_invalid,
    )
    return stats


def _load_tweet_entries(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    entries: list[dict[str, object]] = []
    for match in _TWEET_ASSIGN_HEAD_RE.finditer(text):
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text) or text[start] != "[":
            continue
        try:
            data, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    entries.append(item)
    return entries


def _convert_tweet(tweet: dict[str, object]) -> dict[str, object] | None:
    tweet_id = str(tweet.get("id_str") or tweet.get("id") or "").strip()
    if not tweet_id:
        return None
    text = str(tweet.get("full_text") or tweet.get("text") or "").strip()
    media_urls = _collect_media_image_urls(tweet)
    if not text and not media_urls:
        return None
    created_at_raw = str(tweet.get("created_at") or "").strip()
    created_at = _parse_created_at(created_at_raw)
    if created_at is None:
        return None

    handle = _extract_author_handle(tweet)
    author_name = f"@{handle}" if handle else "unknown"
    source_date = created_at.astimezone(_JST).strftime("%Y/%m/%d")
    canonical_url = (
        f"https://x.com/{handle}/status/{tweet_id}"
        if handle
        else f"https://x.com/i/web/status/{tweet_id}"
    )
    metadata: dict[str, object] = {
        "source_type": "x_posts",
        "source_file_name": "x/posts",
        "source_date": source_date,
        "guild_id": "x",
        "guild_name": "X",
        "category_id": "",
        "category_name": "",
        "channel_id": "posts",
        "channel_name": "X posts",
        "message_id": tweet_id,
        "message_timestamp": created_at.astimezone(timezone.utc).isoformat(),
        "author_id": "",
        "author_name": author_name,
        "x_post_id": tweet_id,
        "x_post_url": canonical_url,
        "x_author_handle": handle or "",
        "x_media_urls": media_urls,
    }
    return {
        "id": tweet_id,
        "created_at": created_at,
        "record": {"text": text, "metadata": metadata},
    }


def _parse_created_at(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return None


def _extract_author_handle(tweet: dict[str, object]) -> str | None:
    for url in _collect_candidate_urls(tweet):
        match = _STATUS_URL_RE.search(url)
        if not match:
            continue
        handle = (match.group("handle") or "").strip()
        status_id = (match.group("id") or "").strip()
        tweet_id = str(tweet.get("id_str") or tweet.get("id") or "").strip()
        if handle and status_id and status_id == tweet_id:
            return handle
    return None


def _collect_candidate_urls(tweet: dict[str, object]) -> list[str]:
    urls: list[str] = []
    entities = tweet.get("entities")
    if isinstance(entities, dict):
        urls.extend(_extract_urls_from_entity(entities))
    ext_entities = tweet.get("extended_entities")
    if isinstance(ext_entities, dict):
        urls.extend(_extract_urls_from_entity(ext_entities))
    return urls


def _extract_urls_from_entity(entity: dict[str, object]) -> list[str]:
    found: list[str] = []
    for key in ("urls", "media"):
        values = entity.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            expanded = str(item.get("expanded_url") or "").strip()
            if expanded:
                found.append(expanded)
            url = str(item.get("url") or "").strip()
            if url:
                found.append(url)
    return found


def _collect_media_image_urls(tweet: dict[str, object]) -> list[str]:
    urls: list[str] = []
    for entity_key in ("extended_entities", "entities"):
        entity = tweet.get(entity_key)
        if not isinstance(entity, dict):
            continue
        media = entity.get("media")
        if not isinstance(media, list):
            continue
        for item in media:
            if not isinstance(item, dict):
                continue
            media_type = str(item.get("type") or "").lower()
            if media_type and media_type not in {"photo", "animated_gif"}:
                continue
            for key in ("media_url_https", "media_url"):
                url = str(item.get(key) or "").strip()
                if url:
                    urls.append(url)
                    break
    return list(dict.fromkeys(urls))
