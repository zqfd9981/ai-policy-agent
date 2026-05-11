from __future__ import annotations

import sys
from pathlib import Path

# 兼容直接运行 `python app\ingest\txt_loader.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.ingest.loader_factory import find_source_file_for_metadata
from app.ingest.metadata_loader import load_metadata_map
from app.models.document import Document
from app.models.metadata import PolicyMetadata


# txt 文本读取时优先尝试的编码顺序。
# 先试 UTF-8，再回退到更常见的中文编码，尽量减少手工改编码的成本。
TXT_READ_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


class TxtLoaderError(ValueError):
    """读取 txt 原始文件失败时抛出的统一异常。"""


def load_txt_document(
    metadata: PolicyMetadata,
    source_path: str | Path | None = None,
) -> Document:
    """
    根据 metadata 读取一份 txt 政策文件，并返回统一的 Document 对象。

    如果没有显式传入 source_path，就根据 metadata 自动去 data/raw 中查找。
    """

    resolved_source_path = _resolve_source_path(metadata, source_path)
    raw_text = _read_txt_file(resolved_source_path)

    return Document(
        metadata=metadata,
        source_path=resolved_source_path,
        source_format="txt",
        raw_text=raw_text,
    )


def _resolve_source_path(
    metadata: PolicyMetadata,
    source_path: str | Path | None,
) -> Path:
    """
    统一解析 txt 文件路径。

    - 如果调用方传了 source_path，就直接使用它
    - 如果没传，就根据 metadata 去 raw 目录中自动查找
    """

    if source_path is None:
        resolved_path = find_source_file_for_metadata(metadata)
    else:
        resolved_path = Path(source_path)

    _validate_txt_source_path(metadata, resolved_path)
    return resolved_path


def _validate_txt_source_path(metadata: PolicyMetadata, source_path: Path) -> None:
    """确保当前路径确实是该 metadata 对应的 txt 文件。"""

    if not source_path.exists():
        raise TxtLoaderError(f"txt 文件不存在: {source_path}")

    if not source_path.is_file():
        raise TxtLoaderError(f"txt 路径不是文件: {source_path}")

    if source_path.suffix.lower() != ".txt":
        raise TxtLoaderError(f"txt loader 只支持 .txt 文件，当前路径为: {source_path}")

    if metadata.source_format.lower() != "txt":
        raise TxtLoaderError(
            f"doc_id={metadata.doc_id} 的 metadata.source_format 不是 txt，"
            f"当前值为: {metadata.source_format}"
        )


def _read_txt_file(source_path: Path) -> str:
    """
    尝试用多种常见编码读取 txt 文件。

    当前阶段先尽量保留原始读取结果，只做去掉首尾空白这样的轻量处理，
    后续更复杂的标准化和去噪留给 clean 模块。
    """

    last_error: UnicodeDecodeError | None = None

    for encoding in TXT_READ_ENCODINGS:
        try:
            text = source_path.read_text(encoding=encoding)
            return text.strip()
        except UnicodeDecodeError as error:
            last_error = error

    raise TxtLoaderError(
        f"无法识别 txt 文件编码: {source_path}"
        if last_error is None
        else f"无法识别 txt 文件编码: {source_path}，最后一次解码错误: {last_error}"
    )


def main() -> None:
    """对 txt loader 做一个简单联调测试。"""

    print("=" * 60)
    print("开始测试 txt loader...")
    print("=" * 60)

    metadata_map = load_metadata_map()
    metadata = metadata_map["JS002"]
    document = load_txt_document(metadata)

    print(f"[OK] 成功读取: {document.doc_id}")
    print(f"title: {document.title}")
    print(f"source_path: {document.source_path}")
    print(f"source_format: {document.source_format}")
    print(f"text_length: {document.text_length}")
    print("\n正文前 200 字：")
    print(document.raw_text[:200])

    print("\n[OK] txt loader 测试通过！")


if __name__ == "__main__":
    main()
