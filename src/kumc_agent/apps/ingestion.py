from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.features.ingestion.chunking import ChunkingSettings, IngestionChunker
from kumc_agent.features.ingestion.service import IngestionService
from kumc_agent.infra.connectors import build_source_connectors
from kumc_agent.infra.ingestion import build_ingestion_repository
from kumc_agent.infra.object_storage.raw_snapshot import RawSnapshotStore
from kumc_agent.infra.secret_finding import SecretFindingDetector


@dataclass(frozen=True)
class IngestionAppContext:
    service: IngestionService
    repository: object
    config: object


def build_ingestion_app_context(*, base_dir: Path | None = None) -> IngestionAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    repository = build_ingestion_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "ingestion",
    )
    raw_snapshots = RawSnapshotStore(
        config=foundation.config.infrastructure.object_storage,
        local_root=foundation.config.base_dir / "data" / "object_storage",
        s3=foundation.object_storage,
    )
    service = IngestionService(
        connectors=build_source_connectors(foundation.config),
        repository=repository,
        raw_snapshots=raw_snapshots,
        chunker=IngestionChunker(
            ChunkingSettings(
                max_characters=foundation.config.indexing.chunking.second_recursive_chunk_size * 4,
                overlap_characters=foundation.config.indexing.chunking.second_recursive_chunk_overlap * 4,
            )
        ),
        secret_detector=SecretFindingDetector(),
        audit_log=foundation.audit_log,
    )
    return IngestionAppContext(
        service=service,
        repository=repository,
        config=foundation.config,
    )
