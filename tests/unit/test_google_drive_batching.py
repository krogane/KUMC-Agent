from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.infra.loaders.google_drive_impl import DriveFile, _split_batches


def _drive_file(index: int) -> DriveFile:
    return DriveFile(
        file_id=f"id-{index}",
        name=f"file-{index}",
        mime_type="application/vnd.google-apps.document",
        path=f"path/file-{index}",
        modified_time="2026-03-08T00:00:00.000Z",
    )


class GoogleDriveBatchingTests(unittest.TestCase):
    def test_split_batches_uses_requested_batch_size(self) -> None:
        files = [_drive_file(i) for i in range(45)]
        batches = _split_batches(files, batch_size=20)
        self.assertEqual([20, 20, 5], [len(batch) for batch in batches])

    def test_split_batches_non_positive_is_single_batch(self) -> None:
        files = [_drive_file(i) for i in range(7)]
        batches = _split_batches(files, batch_size=0)
        self.assertEqual(1, len(batches))
        self.assertEqual(7, len(batches[0]))

    def test_split_batches_none_is_single_batch(self) -> None:
        files = [_drive_file(i) for i in range(3)]
        batches = _split_batches(files, batch_size=None)
        self.assertEqual(1, len(batches))
        self.assertEqual(3, len(batches[0]))


if __name__ == "__main__":
    unittest.main()
