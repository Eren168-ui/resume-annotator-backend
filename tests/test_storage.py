import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services import storage


class DownloadManifestTests(unittest.TestCase):
    def test_build_download_manifest_only_includes_existing_generated_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            task_id = "task-123"
            output_dir = data_dir / "outputs" / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "annotated.pdf").write_bytes(b"pdf")
            (output_dir / "report.md").write_text("# report", encoding="utf-8")
            (output_dir / "report.html").write_text("<html></html>", encoding="utf-8")
            (output_dir / "result.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(storage, "DATA_DIR", data_dir), mock.patch.object(
                storage, "OUTPUTS_DIR", data_dir / "outputs"
            ):
                manifest = storage.build_download_manifest(task_id)

        self.assertEqual(
            manifest,
            {
                "annotatedPdf": "annotated_task-123.pdf",
                "markdownReport": "report_task-123.md",
                "report": "report_task-123.html",
                "jsonResult": "result_task-123.json",
                "pageImages": [],
            },
        )

    def test_get_download_path_supports_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            task_id = "task-456"
            output_dir = data_dir / "outputs" / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            md_path = output_dir / "report.md"
            json_path = output_dir / "result.json"
            md_path.write_text("# report", encoding="utf-8")
            json_path.write_text("{}", encoding="utf-8")

            with mock.patch.object(storage, "DATA_DIR", data_dir), mock.patch.object(
                storage, "OUTPUTS_DIR", data_dir / "outputs"
            ):
                self.assertEqual(storage.get_download_path(task_id, "markdown_report"), md_path)
                self.assertEqual(storage.get_download_path(task_id, "json_result"), json_path)


if __name__ == "__main__":
    unittest.main()
