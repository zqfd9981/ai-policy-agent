from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.ingest.source_audit import build_audit_row, find_doc_sources
from app.models.metadata import PolicyMetadata


def build_metadata(*, doc_id: str = "BJ002", source_format: str = "pdf") -> PolicyMetadata:
    return PolicyMetadata(
        doc_id=doc_id,
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


class SourceAuditTests(unittest.TestCase):
    def test_find_doc_sources_returns_pdf_and_txt_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "beijing" / "pdf").mkdir(parents=True, exist_ok=True)
            (root / "beijing" / "txt").mkdir(parents=True, exist_ok=True)
            (root / "beijing" / "pdf" / "BJ002.pdf").write_bytes(b"%PDF-1.4")
            (root / "beijing" / "txt" / "BJ002.txt").write_text("正文", encoding="utf-8")

            pdf_paths, txt_paths = find_doc_sources("BJ002", root)

            self.assertEqual(pdf_paths, ("beijing\\pdf\\BJ002.pdf",))
            self.assertEqual(txt_paths, ("beijing\\txt\\BJ002.txt",))

    def test_build_audit_row_marks_pdf_doc_with_txt_backup_as_migrate_ready(self) -> None:
        row = build_audit_row(
            build_metadata(source_format="pdf"),
            pdf_paths=("beijing\\pdf\\BJ002.pdf",),
            txt_paths=("beijing\\txt\\BJ002.txt",),
        )

        self.assertEqual(row.status, "migrate_ready")
        self.assertIn("source_format", row.suggestion)


if __name__ == "__main__":
    unittest.main()
