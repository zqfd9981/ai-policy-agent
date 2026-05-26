from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# 兼容直接运行 `python app\ingest\source_audit.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.ingest.loader_factory import DEFAULT_RAW_ROOT
from app.ingest.metadata_loader import DEFAULT_METADATA_PATH, load_metadata
from app.models.metadata import PolicyMetadata


@dataclass(frozen=True, slots=True)
class SourceAuditRow:
    """表示一条政策文档的原始源文件核对结果。"""

    doc_id: str
    title: str
    metadata_source_format: str
    pdf_count: int
    txt_count: int
    preferred_exists: bool
    status: str
    suggestion: str
    pdf_paths: tuple[str, ...]
    txt_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "metadata_source_format": self.metadata_source_format,
            "pdf_count": self.pdf_count,
            "txt_count": self.txt_count,
            "preferred_exists": self.preferred_exists,
            "status": self.status,
            "suggestion": self.suggestion,
            "pdf_paths": list(self.pdf_paths),
            "txt_paths": list(self.txt_paths),
        }


def audit_sources(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> list[SourceAuditRow]:
    """扫描 metadata 和 raw 目录，输出每个 doc_id 的源文件状态。"""

    metadata_list = load_metadata(metadata_path)
    rows: list[SourceAuditRow] = []

    for metadata in metadata_list:
        pdf_paths, txt_paths = find_doc_sources(metadata.doc_id, raw_root)
        rows.append(
            build_audit_row(
                metadata,
                pdf_paths=pdf_paths,
                txt_paths=txt_paths,
                raw_root=raw_root,
            )
        )

    return rows


def find_doc_sources(doc_id: str, raw_root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """返回某个 doc_id 在 raw 目录下对应的 pdf/txt 文件列表。"""

    pdf_paths: list[str] = []
    txt_paths: list[str] = []

    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        if not path.stem.startswith(doc_id):
            continue

        relative_path = str(path.relative_to(raw_root))
        if path.suffix.lower() == ".pdf":
            pdf_paths.append(relative_path)
        if path.suffix.lower() == ".txt":
            txt_paths.append(relative_path)

    return tuple(sorted(pdf_paths)), tuple(sorted(txt_paths))


def has_nonempty_text_files(paths: tuple[str, ...], raw_root: Path) -> bool:
    """判断 txt 候选里是否至少有一份非空正文。"""

    for relative_path in paths:
        file_path = raw_root / relative_path
        if not file_path.exists():
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-8-sig")
        if text.strip():
            return True
    return False


def build_audit_row(
    metadata: PolicyMetadata,
    *,
    pdf_paths: tuple[str, ...],
    txt_paths: tuple[str, ...],
    raw_root: Path = DEFAULT_RAW_ROOT,
) -> SourceAuditRow:
    """根据 metadata 与真实文件列表推导状态与建议。"""

    metadata_source_format = metadata.source_format.lower()
    nonempty_txt_exists = has_nonempty_text_files(txt_paths, raw_root) if txt_paths else False
    effective_txt_paths = txt_paths if nonempty_txt_exists else ()

    preferred_paths = effective_txt_paths if metadata_source_format == "txt" else pdf_paths
    preferred_exists = bool(preferred_paths)

    status = "ok"
    suggestion = "当前 metadata 与主源文件一致。"

    if not pdf_paths and not txt_paths:
        status = "missing_all"
        suggestion = "raw 目录下未找到任何原始文件，需补齐主数据源。"
    elif metadata_source_format == "pdf" and effective_txt_paths and pdf_paths:
        status = "migrate_ready"
        suggestion = "已存在 txt，可把 metadata.source_format 从 pdf 改为 txt。"
    elif metadata_source_format == "pdf" and effective_txt_paths and not pdf_paths:
        status = "metadata_mismatch"
        suggestion = "metadata 仍指向 pdf，但只找到 txt，需同步 source_format。"
    elif metadata_source_format == "txt" and pdf_paths and not effective_txt_paths:
        status = "metadata_mismatch"
        suggestion = "metadata 指向 txt，但当前只有 pdf，需要补 txt 或改回 pdf。"
    elif metadata_source_format == "txt" and effective_txt_paths and pdf_paths:
        status = "ok_with_backup_pdf"
        suggestion = "当前已用 txt 主源，pdf 可继续保留作备份。"
    elif metadata_source_format == "pdf" and pdf_paths and not effective_txt_paths:
        status = "pdf_only"
        suggestion = "当前仍是 pdf 主源；如能拿到官方正文，建议补 txt。"
    elif metadata_source_format == "txt" and effective_txt_paths and not pdf_paths:
        status = "txt_only"
        suggestion = "当前仅有 txt 主源，可继续补留原 pdf 作为备份。"
    elif txt_paths and not nonempty_txt_exists:
        status = "empty_txt"
        suggestion = "已存在 txt 文件，但内容为空；暂不应切换 metadata.source_format。"

    if len(preferred_paths) > 1:
        status = "duplicate_preferred"
        suggestion = "存在多个与 metadata.source_format 一致的候选文件，需要人工去重。"

    return SourceAuditRow(
        doc_id=metadata.doc_id,
        title=metadata.title,
        metadata_source_format=metadata_source_format,
        pdf_count=len(pdf_paths),
        txt_count=len(txt_paths),
        preferred_exists=preferred_exists,
        status=status,
        suggestion=suggestion,
        pdf_paths=pdf_paths,
        txt_paths=txt_paths,
    )


def write_report(rows: list[SourceAuditRow], output_path: Path) -> Path:
    """把核对结果导出成 csv，便于人工筛选与批量迁移。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = (
        "doc_id",
        "title",
        "metadata_source_format",
        "pdf_count",
        "txt_count",
        "preferred_exists",
        "status",
        "suggestion",
        "pdf_paths",
        "txt_paths",
    )

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            payload = row.to_dict()
            payload["pdf_paths"] = " | ".join(row.pdf_paths)
            payload["txt_paths"] = " | ".join(row.txt_paths)
            writer.writerow(payload)

    return output_path


def print_summary(rows: list[SourceAuditRow]) -> None:
    """在终端输出简短统计，帮助快速判断迁移进度。"""

    counter = Counter(row.status for row in rows)
    print("=" * 72)
    print("Source Audit Summary")
    print("=" * 72)
    print(f"总文档数: {len(rows)}")
    for status, count in sorted(counter.items()):
        print(f"{status:<22} {count}")

    print("\n仍建议优先处理的文档：")
    focus_rows = [
        row
        for row in rows
        if row.status in {"pdf_only", "migrate_ready", "metadata_mismatch", "missing_all"}
    ]
    for row in focus_rows[:20]:
        print(
            f"- {row.doc_id} | {row.metadata_source_format} | "
            f"pdf={row.pdf_count} txt={row.txt_count} | {row.suggestion}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 raw 源文件与 metadata.source_format 的一致性。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "source_audit.csv",
        help="导出的核对表路径，默认 outputs/source_audit.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = audit_sources()
    report_path = write_report(rows, args.output)
    print_summary(rows)
    print(f"\n[OK] 已导出核对表: {report_path}")


if __name__ == "__main__":
    main()
