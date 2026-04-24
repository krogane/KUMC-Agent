from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kumc_agent.infra.indexing.config import AppConfig

logger = logging.getLogger(__name__)

MATERIAL_CATALOG_FILENAME = "material_catalog.json"
MATERIAL_CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MaterialCatalogEntry:
    material_id: str
    source_type: str
    source_key: str
    canonical_name: str
    aliases: tuple[str, ...]
    raw_path: str

    def to_payload(self) -> dict[str, object]:
        return {
            "material_id": self.material_id,
            "source_type": self.source_type,
            "source_key": self.source_key,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "raw_path": self.raw_path,
        }


def material_catalog_path(index_dir: Path) -> Path:
    return index_dir / MATERIAL_CATALOG_FILENAME


def save_material_catalog(
    *,
    index_dir: Path,
    entries: list[MaterialCatalogEntry],
) -> Path:
    path = material_catalog_path(index_dir)
    payload = {
        "schema_version": MATERIAL_CATALOG_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "materials": [entry.to_payload() for entry in entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fw:
        json.dump(payload, fw, ensure_ascii=False, indent=2)
    return path


def load_material_catalog(index_dir: Path) -> list[MaterialCatalogEntry]:
    path = material_catalog_path(index_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to load material catalog from %s", path, exc_info=True)
        return []
    if not isinstance(payload, dict):
        return []
    materials_raw = payload.get("materials")
    if not isinstance(materials_raw, list):
        return []
    entries: list[MaterialCatalogEntry] = []
    for row in materials_raw:
        if not isinstance(row, dict):
            continue
        material_id = str(row.get("material_id") or "").strip()
        source_type = str(row.get("source_type") or "").strip().lower()
        source_key = str(row.get("source_key") or "").strip()
        canonical_name = str(row.get("canonical_name") or "").strip()
        raw_path = str(row.get("raw_path") or "").strip()
        aliases_raw = row.get("aliases")
        aliases: list[str] = []
        if isinstance(aliases_raw, list):
            for item in aliases_raw:
                if not isinstance(item, str):
                    continue
                value = item.strip()
                if value:
                    aliases.append(value)
        if (
            not material_id
            or not source_type
            or not source_key
            or not canonical_name
            or not raw_path
        ):
            continue
        merged_aliases = _dedupe_aliases([canonical_name, *aliases])
        entries.append(
            MaterialCatalogEntry(
                material_id=material_id,
                source_type=source_type,
                source_key=source_key,
                canonical_name=canonical_name,
                aliases=tuple(merged_aliases),
                raw_path=raw_path,
            )
        )
    return entries


def build_and_save_material_catalog(cfg: AppConfig) -> Path:
    entries = build_material_catalog(cfg)
    path = save_material_catalog(index_dir=cfg.index_dir, entries=entries)
    logger.info("Saved material catalog: entries=%d path=%s", len(entries), path)
    return path


def build_material_catalog(cfg: AppConfig) -> list[MaterialCatalogEntry]:
    merged: dict[str, dict[str, object]] = {}

    def _upsert(entry: MaterialCatalogEntry) -> None:
        current = merged.get(entry.material_id)
        if current is None:
            merged[entry.material_id] = {
                "source_type": entry.source_type,
                "source_key": entry.source_key,
                "canonical_name": entry.canonical_name,
                "aliases": list(entry.aliases),
                "raw_path": entry.raw_path,
            }
            return
        aliases = _dedupe_aliases(
            [*list(current.get("aliases") or []), *list(entry.aliases)]
        )
        current["aliases"] = aliases
        if not str(current.get("canonical_name") or "").strip():
            current["canonical_name"] = entry.canonical_name
        if not str(current.get("raw_path") or "").strip():
            current["raw_path"] = entry.raw_path

    _collect_drive_like_entries(
        cfg=cfg,
        raw_dir=cfg.raw_data_dir / "docs",
        source_type="docs",
        file_extensions=(".md",),
        title_key=None,
        on_entry=_upsert,
    )
    _collect_drive_like_entries(
        cfg=cfg,
        raw_dir=cfg.raw_data_dir / "sheets",
        source_type="sheets",
        file_extensions=(".csv",),
        title_key=None,
        on_entry=_upsert,
    )
    _collect_drive_like_entries(
        cfg=cfg,
        raw_dir=cfg.raw_data_dir / "hatenablog",
        source_type="hatenablog",
        file_extensions=(".md",),
        title_key="hatenablog_title",
        on_entry=_upsert,
    )
    _collect_drive_like_entries(
        cfg=cfg,
        raw_dir=cfg.raw_data_dir / "crafters_colony",
        source_type="crafters_colony",
        file_extensions=(".md",),
        title_key="crafters_colony_title",
        on_entry=_upsert,
    )
    _collect_drive_like_entries(
        cfg=cfg,
        raw_dir=cfg.raw_data_dir / "notion",
        source_type="notion",
        file_extensions=(".md",),
        title_key="notion_title",
        source_key_fields=("notion_page_id",),
        alias_extra_fields=("notion_url",),
        on_entry=_upsert,
    )
    _collect_message_entries(
        cfg=cfg,
        raw_dir=cfg.raw_data_dir / "messages",
        default_source_type="messages",
        on_entry=_upsert,
    )
    _collect_message_entries(
        cfg=cfg,
        raw_dir=cfg.raw_data_dir / "x",
        default_source_type="x_posts",
        on_entry=_upsert,
    )
    _collect_vc_entries(cfg=cfg, on_entry=_upsert)

    entries: list[MaterialCatalogEntry] = []
    for material_id, row in merged.items():
        source_type = str(row.get("source_type") or "").strip().lower()
        source_key = str(row.get("source_key") or "").strip()
        canonical_name = str(row.get("canonical_name") or "").strip()
        raw_path = str(row.get("raw_path") or "").strip()
        aliases = _dedupe_aliases(
            [canonical_name, *list(row.get("aliases") or [])]
        )
        if (
            not source_type
            or not source_key
            or not canonical_name
            or not raw_path
            or not aliases
        ):
            continue
        entries.append(
            MaterialCatalogEntry(
                material_id=material_id,
                source_type=source_type,
                source_key=source_key,
                canonical_name=canonical_name,
                aliases=tuple(aliases),
                raw_path=raw_path,
            )
        )
    entries.sort(key=lambda item: (item.source_type, item.canonical_name.lower()))
    return entries


def _collect_drive_like_entries(
    *,
    cfg: AppConfig,
    raw_dir: Path,
    source_type: str,
    file_extensions: tuple[str, ...],
    title_key: str | None,
    source_key_fields: tuple[str, ...] = ("drive_file_id",),
    alias_extra_fields: tuple[str, ...] = (),
    on_entry,
) -> None:
    if not raw_dir.exists():
        return
    files: list[Path] = []
    for ext in file_extensions:
        files.extend(raw_dir.rglob(f"*{ext}"))
    for path in sorted(set(files), key=lambda value: str(value)):
        sidecar = _load_sidecar(path)
        source_file_name = path.name
        source_key = ""
        for field_name in source_key_fields:
            candidate = str(sidecar.get(field_name) or "").strip()
            if candidate:
                source_key = candidate
                break
        if not source_key:
            source_key = source_file_name
        canonical_name = ""
        if title_key:
            canonical_name = str(sidecar.get(title_key) or "").strip()
        if not canonical_name:
            canonical_name = str(sidecar.get("drive_file_name") or "").strip()
        if not canonical_name:
            canonical_name = path.stem
        aliases = [
            canonical_name,
            path.stem,
            str(sidecar.get("drive_file_name") or "").strip(),
            str(sidecar.get("drive_path") or "").strip(),
            str(sidecar.get("drive_file_path") or "").strip(),
            source_file_name,
        ]
        if title_key:
            aliases.append(str(sidecar.get(title_key) or "").strip())
        for field_name in alias_extra_fields:
            aliases.append(str(sidecar.get(field_name) or "").strip())
        entry = MaterialCatalogEntry(
            material_id=f"{source_type}:{source_key}",
            source_type=source_type,
            source_key=source_key,
            canonical_name=canonical_name,
            aliases=tuple(_dedupe_aliases(aliases)),
            raw_path=_to_base_relative(path, base_dir=cfg.base_dir),
        )
        on_entry(entry)


def _collect_message_entries(
    *,
    cfg: AppConfig,
    raw_dir: Path,
    default_source_type: str,
    on_entry,
) -> None:
    if not raw_dir.exists():
        return
    for path in sorted(raw_dir.rglob("*.jsonl"), key=lambda value: str(value)):
        if path.name.endswith(".state.json"):
            continue
        metadata = _first_message_metadata(path)
        if not metadata:
            continue
        guild = str(metadata.get("guild_name") or "").strip()
        category = str(metadata.get("category_name") or "").strip()
        channel = str(metadata.get("channel_name") or "").strip()
        guild_id = str(metadata.get("guild_id") or "").strip()
        channel_id = str(metadata.get("channel_id") or "").strip()
        source_type = str(metadata.get("source_type") or "").strip().lower()
        if not source_type:
            source_type = default_source_type
        source_file_name = str(metadata.get("source_file_name") or "").strip()
        if not source_file_name and guild_id and channel_id:
            if source_type in {"messages", "discord_message"}:
                source_file_name = f"discord/{guild_id}/{channel_id}"
            elif source_type == "x_posts":
                source_file_name = "x/posts"
            else:
                source_file_name = f"{guild_id}/{channel_id}"
        if not source_file_name:
            source_file_name = path.stem
        canonical = " / ".join(
            part for part in [category, channel] if part
        ) or channel or source_file_name
        if source_type == "x_posts":
            x_handle = str(metadata.get("x_author_handle") or "").strip()
            if x_handle:
                canonical = f"X @{x_handle}"
        aliases = [
            canonical,
            channel,
            " / ".join(part for part in [guild, category, channel] if part),
            source_file_name,
            path.stem,
        ]
        source_key = source_file_name
        entry = MaterialCatalogEntry(
            material_id=f"{source_type}:{source_key}",
            source_type=source_type,
            source_key=source_key,
            canonical_name=canonical,
            aliases=tuple(_dedupe_aliases(aliases)),
            raw_path=_to_base_relative(path, base_dir=cfg.base_dir),
        )
        on_entry(entry)


def _collect_vc_entries(*, cfg: AppConfig, on_entry) -> None:
    raw_dir = cfg.raw_data_dir / "vc"
    if not raw_dir.exists():
        return
    for path in sorted(raw_dir.rglob("*.txt"), key=lambda value: str(value)):
        rel = path.relative_to(raw_dir)
        rel_posix = str(rel).replace("\\", "/")
        source_file_name = f"vc/{rel_posix}"
        meeting_date, meeting_label = _meeting_labels_for_vc(path)
        canonical = meeting_label or meeting_date or path.stem
        aliases = [canonical, meeting_label, meeting_date, path.stem, source_file_name]
        source_key = source_file_name
        entry = MaterialCatalogEntry(
            material_id=f"vc_transcript:{source_key}",
            source_type="vc_transcript",
            source_key=source_key,
            canonical_name=canonical,
            aliases=tuple(_dedupe_aliases(aliases)),
            raw_path=_to_base_relative(path, base_dir=cfg.base_dir),
        )
        on_entry(entry)


def _load_sidecar(path: Path) -> dict[str, object]:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not sidecar.exists():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _first_message_metadata(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                metadata = payload.get("metadata")
                if isinstance(metadata, dict):
                    return metadata
    except Exception:
        logger.warning("Failed to read message metadata from %s", path, exc_info=True)
    return {}


def _meeting_labels_for_vc(path: Path) -> tuple[str, str]:
    parent = path.parent.name
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})_\d+$", parent)
    if not match:
        return "", ""
    meeting_date = f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
    return meeting_date, f"{meeting_date} VC meeting"


def _to_base_relative(path: Path, *, base_dir: Path) -> str:
    try:
        relative = path.relative_to(base_dir)
    except ValueError:
        relative = path
    return str(relative).replace("\\", "/")


def _dedupe_aliases(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped
