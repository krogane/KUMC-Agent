from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.config.schema import ObjectStorageSection
from kumc_agent.domain.models.source import SourceRawItem
from kumc_agent.infra.object_storage.s3 import S3ObjectStorageClient


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
        return (
            f"{prefix}/raw/{raw.source_kind}/{raw.external_id}/"
            f"{raw.checksum[:16]}.txt"
        )
