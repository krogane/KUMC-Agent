from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from kumc_agent.domain.models.source import AccessScope, SourceRawItem
from kumc_agent.utils.hashing import stable_hash


def read_sidecar_metadata(path: Path) -> dict[str, object]:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not sidecar.exists():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}


def iter_raw_files(
    *,
    source_kind: str,
    root_dir: Path,
    extensions: set[str],
    default_visibility: str = "admin",
    page_url_base: str = "",
) -> list[SourceRawItem]:
    if not root_dir.exists():
        return []
    out: list[SourceRawItem] = []
    for path in sorted(root_dir.rglob("*"), key=lambda value: str(value)):
        if not path.is_file():
            continue
        if path.name.endswith((".meta.json", ".mtime.json", ".state.json")):
            continue
        if path.suffix.lower() not in extensions:
            continue
        metadata = read_sidecar_metadata(path)
        text = _read_text(path)
        if not text.strip():
            continue
        rel = str(path.relative_to(root_dir)).replace("\\", "/")
        title = _title_for(source_kind=source_kind, path=path, metadata=metadata)
        external_id = _external_id(source_kind=source_kind, rel=rel, metadata=metadata)
        canonical_url = _canonical_url(
            source_kind=source_kind,
            external_id=external_id,
            metadata=metadata,
            page_url_base=page_url_base,
        )
        updated_at = _parse_datetime(
            str(
                metadata.get("updated_at")
                or metadata.get("drive_modified_time")
                or metadata.get("notion_last_edited_time")
                or metadata.get("hatenablog_updated_at")
                or metadata.get("hatenablog_created_at")
                or metadata.get("crafters_colony_published_at")
                or metadata.get("source_date")
                or ""
            )
        ) or datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        out.append(
            SourceRawItem(
                source_kind=source_kind,
                external_id=external_id,
                title=title,
                text=text,
                canonical_url=canonical_url,
                author_id=str(metadata.get("author_id") or metadata.get("x_author_handle") or ""),
                created_at=_parse_datetime(
                    str(
                        metadata.get("created_at")
                        or metadata.get("notion_created_time")
                        or metadata.get("hatenablog_created_at")
                        or metadata.get("source_date")
                        or ""
                    )
                ),
                updated_at=updated_at,
                access_scope=_access_scope(
                    source_kind=source_kind,
                    metadata=metadata,
                    default_visibility=default_visibility,
                ),
                raw_path=str(path),
                checksum=stable_hash(text),
                metadata={
                    "raw_relative_path": rel,
                    **metadata,
                },
            )
        )
    return out


def iter_structured_jsonl_records(
    *,
    source_kind: str,
    root_dir: Path,
    default_visibility: str = "admin",
) -> list[SourceRawItem]:
    if not root_dir.exists():
        return []
    out: list[SourceRawItem] = []
    for path in sorted(root_dir.rglob("*.jsonl"), key=lambda value: str(value)):
        if not path.is_file() or path.name.endswith((".meta.json", ".mtime.json")):
            continue
        file_metadata = read_sidecar_metadata(path)
        rel = str(path.relative_to(root_dir)).replace("\\", "/")
        with path.open("r", encoding="utf-8") as fr:
            for line_no, line in enumerate(fr, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                text = str(payload.get("text") or "").strip()
                if not text:
                    continue
                record_metadata = payload.get("metadata")
                if not isinstance(record_metadata, dict):
                    record_metadata = {}
                metadata = {
                    **file_metadata,
                    **{str(key): value for key, value in record_metadata.items()},
                    "raw_relative_path": rel,
                    "structured_record_line": line_no,
                }
                record_id = str(
                    metadata.get("normalized_record_id")
                    or metadata.get("page_number")
                    or metadata.get("slide_number")
                    or line_no
                )
                drive_file_id = str(metadata.get("drive_file_id") or path.stem).strip()
                external_id = f"{drive_file_id}:{record_id}"
                title = _title_for(
                    source_kind=source_kind,
                    path=path,
                    metadata=metadata,
                )
                out.append(
                    SourceRawItem(
                        source_kind=source_kind,
                        external_id=external_id,
                        title=title,
                        text=text,
                        canonical_url=_canonical_url(
                            source_kind=source_kind,
                            external_id=external_id,
                            metadata=metadata,
                            page_url_base="",
                        ),
                        updated_at=_parse_datetime(
                            str(
                                metadata.get("updated_at")
                                or metadata.get("drive_modified_time")
                                or metadata.get("source_date")
                                or ""
                            )
                        )
                        or datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
                        access_scope=_access_scope(
                            source_kind=source_kind,
                            metadata=metadata,
                            default_visibility=default_visibility,
                        ),
                        raw_path=f"{path}#{line_no}",
                        checksum=stable_hash(
                            json.dumps(
                                {
                                    "text": text,
                                    "metadata": metadata,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            )
                        ),
                        metadata=metadata,
                    )
                )
    return out


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        texts: list[str] = []
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(payload.get("text") or "").strip()
                if text:
                    texts.append(text)
        return "\n".join(texts)
    return path.read_text(encoding="utf-8", errors="ignore")


def _title_for(*, source_kind: str, path: Path, metadata: dict[str, object]) -> str:
    for key in (
        "drive_file_name",
        "notion_title",
        "hatenablog_title",
        "crafters_colony_title",
        "minecraft_wiki_title",
        "title",
    ):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return path.stem


def _external_id(*, source_kind: str, rel: str, metadata: dict[str, object]) -> str:
    for key in (
        "drive_file_id",
        "notion_page_id",
        "hatenablog_entry_id",
        "crafters_colony_article_id",
        "minecraft_wiki_page_id",
        "x_post_id",
    ):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return f"{source_kind}:{rel}"


def _canonical_url(
    *,
    source_kind: str,
    external_id: str,
    metadata: dict[str, object],
    page_url_base: str,
) -> str:
    for key in (
        "canonical_url",
        "notion_url",
        "hatenablog_url",
        "crafters_colony_article_url",
        "url",
    ):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    if source_kind == "google_drive":
        drive_file_id = str(metadata.get("drive_file_id") or "").strip()
        if drive_file_id:
            return f"https://drive.google.com/file/d/{drive_file_id}/view"
    if source_kind == "minecraft_wiki" and page_url_base:
        return page_url_base.rstrip("/") + "/" + quote(external_id.replace(" ", "_"))
    return ""


def _access_scope(
    *,
    source_kind: str,
    metadata: dict[str, object],
    default_visibility: str,
) -> AccessScope:
    scope = metadata.get("access_scope")
    scope = scope if isinstance(scope, dict) else {}
    visibility = str(
        scope.get("visibility") or metadata.get("visibility") or default_visibility
    ).strip().lower()
    if visibility not in {"public", "guild", "role", "private", "admin"}:
        visibility = default_visibility
    guild_id = str(scope.get("guild_id") or metadata.get("guild_id") or "").strip() or None
    role_ids_raw = scope.get("role_ids") or metadata.get("role_ids") or []
    role_ids = (
        tuple(str(item) for item in role_ids_raw if str(item).strip())
        if isinstance(role_ids_raw, list)
        else tuple()
    )
    user_ids_raw = scope.get("user_ids") or metadata.get("user_ids") or []
    user_ids = (
        tuple(str(item) for item in user_ids_raw if str(item).strip())
        if isinstance(user_ids_raw, list)
        else tuple()
    )
    return AccessScope(
        visibility=visibility,  # type: ignore[arg-type]
        guild_id=guild_id,
        role_ids=role_ids,
        user_ids=user_ids,
        source_acl_hash=stable_hash(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )


def _parse_datetime(value: str) -> datetime | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
