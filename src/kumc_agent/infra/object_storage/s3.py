from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from kumc_agent.config.schema import ObjectStorageSection
from kumc_agent.domain.models.health import ComponentHealth


@dataclass(frozen=True)
class S3ObjectStorageClient:
    config: ObjectStorageSection

    def is_configured(self) -> bool:
        return bool((self.config.bucket or "").strip())

    def client(self) -> Any:
        if not self.is_configured():
            raise RuntimeError("Object storage bucket is not configured.")
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on deployment env
            raise RuntimeError("boto3 is not installed.") from exc

        kwargs: dict[str, object] = {
            "service_name": "s3",
            "region_name": self.config.region or None,
            "use_ssl": bool(self.config.use_ssl),
        }
        if self.config.endpoint_url:
            kwargs["endpoint_url"] = self.config.endpoint_url
        if self.config.access_key_id:
            kwargs["aws_access_key_id"] = self.config.access_key_id
        if self.config.secret_access_key:
            kwargs["aws_secret_access_key"] = self.config.secret_access_key
        return boto3.client(**kwargs)

    def check(self) -> ComponentHealth:
        if not self.is_configured():
            return ComponentHealth(
                name="object_storage",
                status="not_configured",
                detail="KUMC_OBJECT_STORAGE_BUCKET is empty.",
            )
        started = perf_counter()
        try:
            self.client().head_bucket(Bucket=self.config.bucket)
        except Exception as exc:
            return ComponentHealth(
                name="object_storage",
                status="unhealthy",
                detail=str(exc),
                latency_ms=round((perf_counter() - started) * 1000, 2),
            )
        return ComponentHealth(
            name="object_storage",
            status="healthy",
            latency_ms=round((perf_counter() - started) * 1000, 2),
        )
