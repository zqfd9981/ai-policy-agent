from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

# 兼容直接运行 `python app\ingest\metadata_loader.py` 的场景。
# 如果当前文件是被直接执行，Python 默认不会把项目根目录放进 sys.path，
# 这时绝对导入 `from app...` 会失败，所以这里补一次项目根目录。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.models.metadata import PolicyMetadata


# 默认的 metadata 文件路径。后续如果要换文件，只需要在调用时传入新路径。
DEFAULT_METADATA_PATH = Path(__file__).resolve().parents[2] / "data" / "metadata" / "policies.csv"

# 当前项目已经确认的 metadata 表头。
EXPECTED_HEADERS = (
    "doc_id",
    "title",
    "region",
    "level",
    "issuer",
    "publish_date",
    "policy_type",
    "theme",
    "tier",
    "status",
    "source_format",
    "doc_no",
    "source_url",
    "notes",
)

# 必填字段先保持克制，优先保证基础链路稳定。
REQUIRED_FIELDS = (
    "doc_id",
    "title",
    "region",
    "level",
    "issuer",
    "publish_date",
    "policy_type",
    "theme",
    "tier",
    "status",
    "source_format",
)

# 目前仓库中真实存在的原始文件格式。
ALLOWED_SOURCE_FORMATS = {"pdf", "txt"}


class MetadataLoaderError(ValueError):
    """metadata 加载或校验失败时抛出的统一异常。"""


# 读取全部 metadata，返回 list[PolicyMetadata]
def load_metadata(path: str | Path = DEFAULT_METADATA_PATH) -> list[PolicyMetadata]:
    """读取 metadata 文件，并返回结构化的元数据列表。"""

    metadata_path = Path(path)
    rows = _read_rows(metadata_path)

    metadata_list: list[PolicyMetadata] = []
    seen_doc_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        cleaned_row = _clean_row(row)
        _validate_required_fields(cleaned_row, row_number=row_number)
        _validate_source_format(cleaned_row["source_format"], row_number=row_number)
        _validate_doc_id(cleaned_row["doc_id"], seen_doc_ids, row_number=row_number)
        metadata_list.append(_build_metadata(cleaned_row))

    return metadata_list


def load_metadata_map(path: str | Path = DEFAULT_METADATA_PATH) -> dict[str, PolicyMetadata]:
    """读取 metadata，并构建 doc_id 到元数据对象的映射。"""

    metadata_list = load_metadata(path)
    return {item.doc_id: item for item in metadata_list}


def _read_rows(path: Path) -> list[dict[str, str]]:
    """以 UTF-8 BOM 兼容模式读取 TSV 文件。"""

    if not path.exists():
        raise MetadataLoaderError(f"metadata 文件不存在: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        _validate_headers(reader.fieldnames)
        return list(reader)


def _validate_headers(fieldnames: Iterable[str] | None) -> None:
    """校验表头是否满足当前项目约定。"""

    if fieldnames is None:
        raise MetadataLoaderError("metadata 文件为空，无法读取表头。")

    actual_headers = tuple(fieldnames)
    if actual_headers != EXPECTED_HEADERS:
        raise MetadataLoaderError(
            "metadata 表头不符合预期。\n"
            f"期望: {EXPECTED_HEADERS}\n"
            f"实际: {actual_headers}"
        )


def _clean_row(row: dict[str, str | None]) -> dict[str, str]:
    """统一去除首尾空白，避免后续校验被脏值干扰。"""

    return {key: (value or "").strip() for key, value in row.items()}


def _validate_required_fields(row: dict[str, str], *, row_number: int) -> None:
    """校验必填字段是否为空。"""

    missing_fields = [field for field in REQUIRED_FIELDS if not row.get(field)]
    if missing_fields:
        raise MetadataLoaderError(
            f"metadata 第 {row_number} 行缺少必填字段: {', '.join(missing_fields)}"
        )


def _validate_source_format(source_format: str, *, row_number: int) -> None:
    """限制 source_format 的取值范围，避免后续 loader 走到未知分支。"""

    if source_format not in ALLOWED_SOURCE_FORMATS:
        allowed_formats = ", ".join(sorted(ALLOWED_SOURCE_FORMATS))
        raise MetadataLoaderError(
            f"metadata 第 {row_number} 行的 source_format 非法: {source_format}，"
            f"允许值为: {allowed_formats}"
        )


def _validate_doc_id(doc_id: str, seen_doc_ids: set[str], *, row_number: int) -> None:
    """确保每条政策文档的 doc_id 唯一。"""

    if doc_id in seen_doc_ids:
        raise MetadataLoaderError(f"metadata 第 {row_number} 行出现重复 doc_id: {doc_id}")
    seen_doc_ids.add(doc_id)


def _build_metadata(row: dict[str, str]) -> PolicyMetadata:
    """把一行原始数据转换成统一的 PolicyMetadata 对象。"""

    return PolicyMetadata(
        doc_id=row["doc_id"],
        title=row["title"],
        region=row["region"],
        level=row["level"],
        issuer=row["issuer"],
        publish_date=row["publish_date"],
        policy_type=row["policy_type"],
        theme=row["theme"],
        tier=row["tier"],
        status=row["status"],
        source_format=row["source_format"],
        doc_no=row.get("doc_no", ""),
        source_url=row.get("source_url", ""),
        notes=row.get("notes", ""),
    )



# ======================
# 测试用 main 函数（输出全部属性）
# ======================
def main():
    print("=" * 80)
    print("开始测试 metadata 加载器（输出全部属性）")
    print("=" * 80)

    try:
        metadata_list = load_metadata()
        print(f"[OK] 成功加载 {len(metadata_list)} 条数据\n")

        # 遍历每一条，打印全部属性
        """
        for idx, item in enumerate(metadata_list, 1):
            print(f"第 {idx} 条政策元数据：")
            print(f"  doc_id:         {item.doc_id}")
            print(f"  title:          {item.title}")
            print(f"  region:         {item.region}")
            print(f"  level:          {item.level}")
            print(f"  issuer:         {item.issuer}")
            print(f"  publish_date:   {item.publish_date}")
            print(f"  policy_type:    {item.policy_type}")
            print(f"  theme:          {item.theme}")
            print(f"  tier:           {item.tier}")
            print(f"  status:         {item.status}")
            print(f"  source_format:  {item.source_format}")
            print(f"  doc_no:         {item.doc_no}")
            print(f"  source_url:     {item.source_url}")
            print(f"  notes:          {item.notes}")
            print("-" * 80)
        """
        print(load_metadata_map())
        print("\n[OK] 全部数据打印完成！")

    except MetadataLoaderError as e:
        print("\n[ERROR] 出错：", e)

if __name__ == "__main__":
    main()
