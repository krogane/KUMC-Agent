from __future__ import annotations

from pathlib import Path


class GoogleDriveLoader:
    def __init__(
        self,
        *,
        folder_id: str,
        credentials_path: str,
        raw_dir: Path,
        max_files: int,
    ) -> None:
        self._folder_id = folder_id
        self._credentials_path = credentials_path
        self._raw_dir = raw_dir
        self._max_files = max_files

    def load(self) -> int:
        if not self._folder_id:
            return 0
        from kumc_agent.infra.legacy.indexing.drive_loader import (
            download_drive_markdown,
        )

        docs_dir = self._raw_dir / "docs"
        sheets_dir = self._raw_dir / "sheets"
        docs_dir.mkdir(parents=True, exist_ok=True)
        sheets_dir.mkdir(parents=True, exist_ok=True)
        download_drive_markdown(
            drive_folder_id=self._folder_id,
            docs_dir=docs_dir,
            sheets_dir=sheets_dir,
            google_application_credentials=self._credentials_path,
            drive_max_files=self._max_files,
            skip_existing=False,
            update_existing=True,
            sync_deleted=True,
        )
        return 1
