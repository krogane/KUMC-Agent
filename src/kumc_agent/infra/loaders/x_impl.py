from __future__ import annotations

import json
import logging
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_JST = ZoneInfo("Asia/Tokyo")
_TWEET_ASSIGN_HEAD_RE = re.compile(r"window\.YTD\.tweets\.part\d+\s*=", re.MULTILINE)
_ACCOUNT_ASSIGN_HEAD_RE = re.compile(
    r"window\.YTD\.account\.part\d+\s*=",
    re.MULTILINE,
)
_STATUS_URL_RE = re.compile(
    r"https?://(?:x|twitter)\.com/(?P<handle>[A-Za-z0-9_]+)/status/(?P<id>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class XConvertStats:
    files: int = 0
    posts: int = 0
    skipped_invalid: int = 0


@dataclass(frozen=True)
class _XArchiveAccount:
    account_id: str = ""
    username: str = ""
    display_name: str = ""


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

    account = _load_archive_account(raw_x_dir)
    local_media_by_post_id = _build_local_media_index(raw_x_dir)
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
            parsed = _convert_tweet(
                tweet,
                account=account,
                local_media_by_post_id=local_media_by_post_id,
                local_media_root=raw_x_dir,
            )
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
    return _load_assigned_entries(path, _TWEET_ASSIGN_HEAD_RE)


def _load_assigned_entries(path: Path, head_re: re.Pattern[str]) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    entries: list[dict[str, object]] = []
    for match in head_re.finditer(text):
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


def _load_archive_account(raw_x_dir: Path) -> _XArchiveAccount:
    for path in sorted(raw_x_dir.rglob("account.js"), key=lambda value: str(value)):
        try:
            entries = _load_assigned_entries(path, _ACCOUNT_ASSIGN_HEAD_RE)
        except Exception:
            logger.warning("Failed to parse X account file %s", path, exc_info=True)
            continue
        for entry in entries:
            account = entry.get("account")
            if not isinstance(account, dict):
                continue
            return _XArchiveAccount(
                account_id=str(account.get("accountId") or "").strip(),
                username=str(account.get("username") or "").strip().lstrip("@"),
                display_name=str(account.get("accountDisplayName") or "").strip(),
            )
    return _XArchiveAccount()


def _build_local_media_index(raw_x_dir: Path) -> dict[str, list[Path]]:
    media_dir = raw_x_dir / "data" / "tweets_media"
    if not media_dir.exists():
        return {}
    by_post_id: dict[str, list[Path]] = {}
    for path in sorted(media_dir.iterdir(), key=lambda value: str(value)):
        if not path.is_file():
            continue
        post_id = path.name.split("-", maxsplit=1)[0].strip()
        if not post_id:
            continue
        by_post_id.setdefault(post_id, []).append(path)
    return by_post_id


def _convert_tweet(
    tweet: dict[str, object],
    *,
    account: _XArchiveAccount | None = None,
    local_media_by_post_id: dict[str, list[Path]] | None = None,
    local_media_root: Path | None = None,
) -> dict[str, object] | None:
    tweet_id = str(tweet.get("id_str") or tweet.get("id") or "").strip()
    if not tweet_id:
        return None
    raw_text = str(tweet.get("full_text") or tweet.get("text") or "").strip()
    media = _collect_media_records(
        tweet,
        tweet_id=tweet_id,
        local_media_by_post_id=local_media_by_post_id or {},
        local_media_root=local_media_root,
    )
    media_urls = _legacy_media_image_urls(media)
    text = _normalize_tweet_text(raw_text, tweet=tweet)
    if not text and not media:
        return None
    created_at_raw = str(tweet.get("created_at") or "").strip()
    created_at = _parse_created_at(created_at_raw)
    if created_at is None:
        return None

    account = account or _XArchiveAccount()
    handle = _extract_author_handle(tweet) or account.username
    author_name = f"@{handle}" if handle else "unknown"
    if account.display_name:
        author_name = account.display_name
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
        "author_id": account.account_id or handle or "",
        "author_name": author_name,
        "x_post_id": tweet_id,
        "x_post_url": canonical_url,
        "x_author_handle": handle or "",
        "x_media_urls": media_urls,
        "x_media": media,
        "x_expanded_urls": _collect_expanded_urls(tweet),
        "x_account_id": account.account_id,
        "x_account_username": account.username,
        "x_account_display_name": account.display_name,
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


def _normalize_tweet_text(text: str, *, tweet: dict[str, object]) -> str:
    normalized = text or ""
    for token in _collect_media_tco_urls(tweet):
        normalized = normalized.replace(token, "")
    for token, replacement in _collect_url_replacements(tweet):
        normalized = normalized.replace(token, replacement)
    normalized = re.sub(r"https://t\.co/\S+", "", normalized)
    lines = [
        re.sub(r"[ \t]{2,}", " ", line).rstrip()
        for line in normalized.splitlines()
    ]
    compacted: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        compacted.append(line)
        previous_blank = blank
    return "\n".join(compacted).strip()


def _collect_url_replacements(tweet: dict[str, object]) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for entity_key in ("entities", "extended_entities"):
        entity = tweet.get(entity_key)
        if not isinstance(entity, dict):
            continue
        values = entity.get("urls")
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            token = str(item.get("url") or "").strip()
            expanded = str(
                item.get("expanded_url") or item.get("display_url") or ""
            ).strip()
            if token and expanded:
                replacements.append((token, expanded))
    return list(dict.fromkeys(replacements))


def _collect_expanded_urls(tweet: dict[str, object]) -> list[str]:
    urls: list[str] = []
    for _, expanded in _collect_url_replacements(tweet):
        if expanded:
            urls.append(expanded)
    return list(dict.fromkeys(urls))


def _collect_media_tco_urls(tweet: dict[str, object]) -> list[str]:
    urls: list[str] = []
    for item in _iter_media_entities(tweet):
        url = str(item.get("url") or "").strip()
        if url:
            urls.append(url)
    return list(dict.fromkeys(urls))


def _iter_media_entities(tweet: dict[str, object]) -> list[dict[str, object]]:
    media_items: list[dict[str, object]] = []
    seen: set[str] = set()
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
            key = str(item.get("id_str") or item.get("id") or item.get("url") or "")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            media_items.append(item)
    return media_items


def _collect_media_records(
    tweet: dict[str, object],
    *,
    tweet_id: str,
    local_media_by_post_id: dict[str, list[Path]],
    local_media_root: Path | None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    candidates = local_media_by_post_id.get(tweet_id, [])
    used_local_paths: set[Path] = set()
    for item in _iter_media_entities(tweet):
        media_type = str(item.get("type") or "").strip().lower()
        if media_type not in {"photo", "video", "animated_gif"}:
            media_type = "photo" if not media_type else media_type
        remote_url = _media_remote_url(item, media_type=media_type)
        thumbnail_remote_url = _media_thumbnail_url(item)
        local_path = _match_local_media_path(
            item=item,
            media_type=media_type,
            candidates=candidates,
            used_local_paths=used_local_paths,
        )
        if local_path is not None:
            used_local_paths.add(local_path)
        local_relative_path = (
            str(local_path.relative_to(local_media_root)).replace("\\", "/")
            if local_path is not None and local_media_root is not None
            else ""
        )
        record = {
            "type": media_type,
            "remote_url": remote_url,
            "local_relative_path": local_relative_path,
            "content_hash": _file_sha256(local_path) if local_path is not None else "",
            "thumbnail_remote_url": thumbnail_remote_url,
        }
        if any(str(value).strip() for value in record.values()):
            records.append(record)
    return records


def _media_remote_url(item: dict[str, object], *, media_type: str) -> str:
    if media_type in {"video", "animated_gif"}:
        video_info = item.get("video_info")
        if isinstance(video_info, dict):
            variants = video_info.get("variants")
            if isinstance(variants, list):
                mp4_variants = [
                    variant
                    for variant in variants
                    if isinstance(variant, dict)
                    and str(variant.get("content_type") or "").lower() == "video/mp4"
                    and str(variant.get("url") or "").strip()
                ]
                if mp4_variants:
                    best = max(
                        mp4_variants,
                        key=lambda variant: _safe_int(variant.get("bitrate")),
                    )
                    return str(best.get("url") or "").strip()
    for key in ("media_url_https", "media_url"):
        url = str(item.get(key) or "").strip()
        if url:
            return url
    return ""


def _media_thumbnail_url(item: dict[str, object]) -> str:
    for key in ("media_url_https", "media_url"):
        url = str(item.get(key) or "").strip()
        if url:
            return url
    return ""


def _match_local_media_path(
    *,
    item: dict[str, object],
    media_type: str,
    candidates: list[Path],
    used_local_paths: set[Path],
) -> Path | None:
    if not candidates:
        return None
    tokens = [
        _url_basename(_media_remote_url(item, media_type=media_type)),
        _url_basename(_media_thumbnail_url(item)),
    ]
    tokens = [token for token in tokens if token]
    for token in tokens:
        for path in candidates:
            if path in used_local_paths:
                continue
            if path.name.endswith(token) or Path(token).stem in path.stem:
                return path
    for path in candidates:
        if path not in used_local_paths:
            return path
    return None


def _legacy_media_image_urls(media: list[dict[str, object]]) -> list[str]:
    urls: list[str] = []
    for item in media:
        media_type = str(item.get("type") or "").strip().lower()
        remote_url = str(item.get("remote_url") or "").strip()
        thumbnail = str(item.get("thumbnail_remote_url") or "").strip()
        if media_type == "photo" and remote_url:
            urls.append(remote_url)
        elif thumbnail:
            urls.append(thumbnail)
    return list(dict.fromkeys(urls))


def _url_basename(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return Path(parsed.path).name


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as fr:
        for chunk in iter(lambda: fr.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_int(value: object) -> int:
    try:
        return int(str(value or "0"))
    except ValueError:
        return 0
