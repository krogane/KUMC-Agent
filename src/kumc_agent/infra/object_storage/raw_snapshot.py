from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import urlparse

from kumc_agent.config.schema import ObjectStorageSection
from kumc_agent.domain.models.source import SourceRawItem
from kumc_agent.infra.object_storage.s3 import S3ObjectStorageClient
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class RawSnapshotStore:
    config: ObjectStorageSection
    local_root: Path
    s3: S3ObjectStorageClient

    def put(self, raw: SourceRawItem) -> str:
        key = self._key(raw)
        data = raw.text.encode("utf-8")
        if self.s3.is_configured():
            self.s3.client().put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=data,
                ContentType="text/plain; charset=utf-8",
            )
            return key
        path = self.local_root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path.relative_to(self.local_root)).replace("\\", "/")

    def _key(self, raw: SourceRawItem) -> str:
        prefix = (self.config.prefix or "kumc-agent").strip("/ ")
        source_kind = _safe_key_segment(raw.source_kind)
        external_path = _external_key_path(raw)
        return (
            f"{prefix}/raw/{source_kind}/{external_path}/"
            f"{raw.checksum[:16]}.txt"
        )


def _external_key_path(raw: SourceRawItem) -> str:
    if raw.source_kind == "hatenablog":
        url = str(raw.metadata.get("hatenablog_url") or raw.canonical_url or "").strip()
        parsed = urlparse(url)
        if parsed.path.startswith("/entry/"):
            return _safe_key_path(parsed.path.lstrip("/"))
        raw_id = str(raw.external_id or "")
        entry_index = raw_id.find("/entry/")
        if entry_index >= 0:
            return _safe_key_path(raw_id[entry_index + 1 :])
    return _safe_key_path(raw.external_id)


def _safe_key_path(value: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    raw = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*:/*", "", raw)
    parts = [_safe_key_segment(part) for part in raw.split("/") if part.strip()]
    if parts:
        return "/".join(parts)
    return stable_hash(raw or "empty")[:16]


def _safe_key_segment(value: str) -> str:
    raw = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._=-]+", "-", raw).strip(".-")
    return cleaned or stable_hash(raw or "empty")[:12]
