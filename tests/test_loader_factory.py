from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.ingest.loader_factory import find_source_file_for_metadata
from app.models.metadata import PolicyMetadata


def build_metadata(*, source_format: str) -> PolicyMetadata:
    return PolicyMetadata(
        doc_id="BJ999",
        title="示例政策",
        region="北京",
        level="市级",
        issuer="示例机关",
        publish_date="2025/1/1",
        policy_type="若干措施",
        theme="人工智能",
        tier="core",
        status="official_text",
        source_format=source_format,
    )


class LoaderFactoryTests(unittest.TestCase):
    def test_find_source_file_for_metadata_prefers_matching_source_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            txt_path = root / "beijing" / "txt" / "BJ999.txt"
            pdf_path = root / "beijing" / "pdf" / "BJ999.pdf"
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text("示例文本", encoding="utf-8")
            pdf_path.write_bytes(b"%PDF-1.4")

            resolved_txt = find_source_file_for_metadata(
                build_metadata(source_format="txt"),
                raw_root=root,
            )
            resolved_pdf = find_source_file_for_metadata(
                build_metadata(source_format="pdf"),
                raw_root=root,
            )

            self.assertEqual(resolved_txt, txt_path)
            self.assertEqual(resolved_pdf, pdf_path)


if __name__ == "__main__":
    unittest.main()
