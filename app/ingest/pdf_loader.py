from __future__ import annotations

import sys
from pathlib import Path

# 兼容直接运行 `python app\ingest\pdf_loader.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from pypdf import PdfReader

from app.ingest.loader_factory import find_source_file_for_metadata
from app.ingest.metadata_loader import load_metadata_map
from app.models.document import Document
from app.models.metadata import PolicyMetadata


class PdfLoaderError(ValueError):
    """读取 pdf 原始文件失败时抛出的统一异常。"""


def load_pdf_document(
    metadata: PolicyMetadata,
    source_path: str | Path | None = None,
) -> Document:
    """
    根据 metadata 读取一份 pdf 政策文件，并返回统一的 Document 对象。

    如果没有显式传入 source_path，就根据 metadata 自动去 data/raw 中查找。
    """

    resolved_source_path = _resolve_source_path(metadata, source_path)
    raw_text = _extract_pdf_text(resolved_source_path)

    return Document(
        metadata=metadata,
        source_path=resolved_source_path,
        source_format="pdf",
        raw_text=raw_text,
    )


def _resolve_source_path(
    metadata: PolicyMetadata,
    source_path: str | Path | None,
) -> Path:
    """
    统一解析 pdf 文件路径。

    - 如果调用方传了 source_path，就直接使用它
    - 如果没传，就根据 metadata 去 raw 目录中自动查找
    """

    if source_path is None:
        resolved_path = find_source_file_for_metadata(metadata)
    else:
        resolved_path = Path(source_path)

    _validate_pdf_source_path(metadata, resolved_path)
    return resolved_path


def _validate_pdf_source_path(metadata: PolicyMetadata, source_path: Path) -> None:
    """确保当前路径确实是该 metadata 对应的 pdf 文件。"""

    if not source_path.exists():
        raise PdfLoaderError(f"pdf 文件不存在: {source_path}")

    if not source_path.is_file():
        raise PdfLoaderError(f"pdf 路径不是文件: {source_path}")

    if source_path.suffix.lower() != ".pdf":
        raise PdfLoaderError(f"pdf loader 只支持 .pdf 文件，当前路径为: {source_path}")

    if metadata.source_format.lower() != "pdf":
        raise PdfLoaderError(
            f"doc_id={metadata.doc_id} 的 metadata.source_format 不是 pdf，"
            f"当前值为: {metadata.source_format}"
        )


def _extract_pdf_text(source_path: Path) -> str:
    """
    从 pdf 中提取全文文本。

    当前阶段先尽量忠实保留原始提取结果：
    - 按页提取
    - 页与页之间用空行分隔
    更复杂的页眉页脚去除、断行合并等处理，后续交给 clean 模块。
    """

    try:
        reader = PdfReader(str(source_path))
    except Exception as error:  # pragma: no cover - 依赖底层库报错类型
        raise PdfLoaderError(f"打开 pdf 失败: {source_path}，原因: {error}") from error

    page_texts: list[str] = []

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:  # pragma: no cover - 依赖底层库报错类型
            raise PdfLoaderError(
                f"提取 pdf 第 {page_index} 页文本失败: {source_path}，原因: {error}"
            ) from error

        # 只保留非空页，避免大量空白页把正文冲淡。
        stripped_text = text.strip()
        if stripped_text:
            page_texts.append(stripped_text)

    if not page_texts:
        raise PdfLoaderError(f"pdf 未提取到任何正文文本: {source_path}")

    return "\n\n".join(page_texts)


def main() -> None:
    """对 pdf loader 做一个简单联调测试。"""

    print("=" * 60)
    print("开始测试 pdf loader...")
    print("=" * 60)

    metadata_map = load_metadata_map()
    metadata = metadata_map["SH001"]
    document = load_pdf_document(metadata)

    print(f"[OK] 成功读取: {document.doc_id}")
    print(f"title: {document.title}")
    print(f"source_path: {document.source_path}")
    print(f"source_format: {document.source_format}")
    print(f"text_length: {document.text_length}")
    print("\n正文前 200 字：")
    print(document.raw_text[:200])

    print("\n[OK] pdf loader 测试通过！")


if __name__ == "__main__":
    main()
