from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import kumc_agent.infra.loaders.google_drive_impl as drive_impl
from kumc_agent.infra.loaders.common import DRIVE_DOC_MIME
from kumc_agent.infra.loaders.google_drive_impl import DriveFile


class GoogleDriveDocsLoadingTests(unittest.TestCase):
    def test_download_drive_markdown_writes_extended_metadata_and_normalized_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs_dir = root / "docs"
            normalized_dir = root / "docs_normalized"
            sheets_dir = root / "sheets"
            docs_dir.mkdir()
            normalized_dir.mkdir()
            sheets_dir.mkdir()
            doc_file = DriveFile(
                file_id="doc-id",
                name="議事録",
                mime_type=DRIVE_DOC_MIME,
                path="2026/04: 例会/議事録",
                modified_time="2026-04-29T00:00:00Z",
            )

            with (
                patch.object(drive_impl, "_build_google_credentials", return_value=object()),
                patch.object(drive_impl, "_build_drive_service", return_value=object()),
                patch.object(drive_impl, "_list_drive_files", return_value=[doc_file]),
                patch.object(
                    drive_impl,
                    "_download_export_bytes",
                    return_value=(
                        "# 議事録\n\n"
                        "例会では新歓企画とサーバー運用を確認しました。"
                        "参加者の担当、次回までの作業、資料更新の手順を整理し、"
                        "検索時に根拠として利用できる十分な本文量を持つ議事録として保存します。"
                        "この段落は品質判定で短文扱いにならないよう、具体的な説明を含めています。"
                        "また、Driveのパスからsource_dateを推定できることも確認します。"
                        "議題、決定事項、未解決事項、担当者、期限、関連資料、補足説明を本文に残し、"
                        "RAGの引用で利用者がどの資料のどの内容を参照したのか判断できる状態にします。"
                        "この記録は例会後の作業確認、イベント準備、サーバー管理、告知文作成にも再利用します。"
                    ).encode("utf-8"),
                ),
            ):
                docs_count, sheets_count = drive_impl.download_drive_markdown(
                    drive_folder_id="folder-id",
                    docs_dir=docs_dir,
                    docs_normalized_dir=normalized_dir,
                    sheets_dir=sheets_dir,
                    google_application_credentials="",
                    pdf_ocr_model_path="",
                )

            self.assertEqual(1, docs_count)
            self.assertEqual(0, sheets_count)
            raw_output = next(docs_dir.glob("*.md"))
            metadata = json.loads(
                raw_output.with_suffix(".md.meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual("google_docs_markdown", metadata["extraction_method"])
            self.assertIn("content_sha256", metadata)
            self.assertEqual("2026/04/01", metadata["source_date"])
            self.assertEqual("active", metadata["index_status"])
            normalized_output = next(normalized_dir.glob("*.jsonl"))
            record = json.loads(normalized_output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual("docs", record["metadata"]["source_type"])
            self.assertEqual("議事録", record["metadata"]["heading_path"][0])
            self.assertEqual("doc-id", record["metadata"]["drive_file_id"])


if __name__ == "__main__":
    unittest.main()
