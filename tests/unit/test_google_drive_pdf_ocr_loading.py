from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import kumc_agent.infra.loaders.google_drive_impl as drive_impl


class _FakeProcessor:
    class _ImageProcessor:
        min_pixels = 64

    def __init__(self) -> None:
        self.image_processor = self._ImageProcessor()
        self.decoded_ids: torch.Tensor | None = None

    def apply_chat_template(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        del args
        del kwargs
        return {
            "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        }

    def batch_decode(self, generated_outputs: torch.Tensor, **kwargs) -> list[str]:
        del kwargs
        self.decoded_ids = generated_outputs
        return ["decoded text"]


class _FakeModel:
    def __init__(self) -> None:
        self.device: str | None = None
        self.generate_kwargs: dict[str, object] | None = None

    def to(self, device: str) -> "_FakeModel":
        self.device = device
        return self

    def eval(self) -> "_FakeModel":
        return self

    def generate(self, **kwargs) -> torch.Tensor:
        self.generate_kwargs = kwargs
        return torch.tensor([[10, 11, 90, 91]], dtype=torch.long)


class GoogleDrivePdfOcrLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        drive_impl._load_pdf_ocr_pipeline.cache_clear()

    def tearDown(self) -> None:
        drive_impl._load_pdf_ocr_pipeline.cache_clear()

    def test_pipeline_failure_falls_back_to_direct_runner(self) -> None:
        sentinel_runner = object()

        with (
            patch(
                "transformers.pipeline",
                side_effect=[ValueError("task1 failed"), ValueError("task2 failed")],
            ) as pipeline_mock,
            patch.object(
                drive_impl,
                "_load_pdf_ocr_direct_runner",
                return_value=sentinel_runner,
            ) as direct_runner_mock,
        ):
            runner = drive_impl._load_pdf_ocr_pipeline("model/path")

        self.assertEqual(2, pipeline_mock.call_count)
        direct_runner_mock.assert_called_once_with("model/path")
        self.assertIs(sentinel_runner, runner)

    def test_direct_runner_decodes_generated_tokens_without_prompt(self) -> None:
        fake_processor = _FakeProcessor()
        fake_model = _FakeModel()
        runner = drive_impl._DirectPdfOcrRunner(
            model=fake_model,
            processor=fake_processor,
            device="cpu",
        )

        result = runner(object(), max_new_tokens=123)

        self.assertEqual("decoded text", result[0]["generated_text"])
        self.assertIsNotNone(fake_model.generate_kwargs)
        self.assertEqual(123, fake_model.generate_kwargs["max_new_tokens"])
        self.assertIsNotNone(fake_processor.decoded_ids)
        self.assertTrue(
            torch.equal(
                fake_processor.decoded_ids,
                torch.tensor([[90, 91]], dtype=torch.long),
            )
        )

    def test_error_includes_dependency_name_when_direct_load_fails(self) -> None:
        with (
            patch(
                "transformers.pipeline",
                side_effect=[ValueError("task1 failed"), ValueError("task2 failed")],
            ),
            patch.object(
                drive_impl,
                "_load_pdf_ocr_direct_runner",
                side_effect=ImportError("No module named 'einops'"),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                drive_impl._load_pdf_ocr_pipeline("model/path")

        self.assertIn("einops", str(ctx.exception))

    def test_pp_ocr_v5_path_prefers_paddle_runner(self) -> None:
        sentinel_runner = object()
        with (
            patch.object(
                drive_impl,
                "_load_pdf_ocr_paddle_runner",
                return_value=sentinel_runner,
            ) as paddle_runner_mock,
            patch("transformers.pipeline") as pipeline_mock,
        ):
            runner = drive_impl._load_pdf_ocr_pipeline(
                "model/ocr/PaddlePaddle/PP-OCRv5_mobile"
            )

        paddle_runner_mock.assert_called_once_with(
            "model/ocr/PaddlePaddle/PP-OCRv5_mobile"
        )
        pipeline_mock.assert_not_called()
        self.assertIs(sentinel_runner, runner)

    def test_resolve_pp_ocr_v5_mobile_dirs_from_det_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            det_dir = base / "PP-OCRv5_mobile_det"
            rec_dir = base / "PP-OCRv5_mobile_rec"
            det_dir.mkdir()
            rec_dir.mkdir()
            (det_dir / "inference.pdiparams").write_text("", encoding="utf-8")
            (det_dir / "inference.yml").write_text("", encoding="utf-8")
            (rec_dir / "inference.pdiparams").write_text("", encoding="utf-8")
            (rec_dir / "inference.yml").write_text("", encoding="utf-8")

            resolved = drive_impl._resolve_pp_ocr_v5_mobile_dirs(str(det_dir))

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(det_dir, resolved[0])
        self.assertEqual(rec_dir, resolved[1])
        self.assertIsNone(resolved[2])

    def test_extract_generated_text_from_paddle_payload(self) -> None:
        payload = {
            "res": {
                "rec_texts": ["line-1", "line-2"],
            }
        }
        self.assertEqual("line-1\nline-2", drive_impl._extract_generated_text(payload))

        old_style_payload = [
            [
                [[0, 0], [1, 0], [1, 1], [0, 1]],
                ("legacy-line", 0.99),
            ]
        ]
        self.assertEqual(
            "legacy-line",
            drive_impl._extract_generated_text(old_style_payload),
        )


if __name__ == "__main__":
    unittest.main()
