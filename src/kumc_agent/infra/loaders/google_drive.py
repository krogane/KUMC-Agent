from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GoogleDriveLoader:
    def __init__(
        self,
        *,
        folder_id: str,
        credentials_path: str,
        ingestion_dir: Path,
        max_files: int,
        batch_size: int,
        download_max_retries: int,
        download_retry_initial_delay_seconds: float,
        download_retry_max_delay_seconds: float,
        download_retry_backoff_multiplier: float,
        pdf_ocr_model_path: str,
    ) -> None:
        self._folder_id = folder_id
        self._credentials_path = credentials_path
        self._ingestion_dir = ingestion_dir
        self._max_files = max_files
        self._batch_size = batch_size
        self._download_max_retries = download_max_retries
        self._download_retry_initial_delay_seconds = (
            download_retry_initial_delay_seconds
        )
        self._download_retry_max_delay_seconds = download_retry_max_delay_seconds
        self._download_retry_backoff_multiplier = (
            download_retry_backoff_multiplier
        )
        self._pdf_ocr_model_path = pdf_ocr_model_path

    def load(self) -> int:
        if not self._folder_id:
            return 0
        from kumc_agent.infra.loaders.google_drive_impl import (
            download_drive_markdown,
        )

        docs_dir = self._ingestion_dir / "docs"
        sheets_dir = self._ingestion_dir / "sheets"
        sheets_structured_dir = self._ingestion_dir / "sheets_structured"
        docs_dir.mkdir(parents=True, exist_ok=True)
        sheets_dir.mkdir(parents=True, exist_ok=True)
        sheets_structured_dir.mkdir(parents=True, exist_ok=True)
        docs_count, sheets_count = download_drive_markdown(
            drive_folder_id=self._folder_id,
            docs_dir=docs_dir,
            sheets_dir=sheets_dir,
            sheets_structured_dir=sheets_structured_dir,
            google_application_credentials=self._credentials_path,
            pdf_ocr_model_path=self._pdf_ocr_model_path,
            drive_max_files=self._max_files,
            drive_batch_size=self._batch_size,
            drive_download_max_retries=self._download_max_retries,
            drive_download_retry_initial_delay_seconds=(
                self._download_retry_initial_delay_seconds
            ),
            drive_download_retry_max_delay_seconds=(
                self._download_retry_max_delay_seconds
            ),
            drive_download_retry_backoff_multiplier=(
                self._download_retry_backoff_multiplier
            ),
            skip_existing=True,
            update_existing=True,
            sync_deleted=True,
        )
        try:
            from kumc_agent.infra.loaders.sheets_profile import write_sheets_profile

            write_sheets_profile(
                sheets_dir=sheets_dir,
                structured_sheets_dir=sheets_structured_dir,
                output_path=self._ingestion_dir / "sheets_profile.json",
            )
        except Exception:
            logger.exception("Failed to write Sheets raw profile.")
        return int(docs_count) + int(sheets_count)
