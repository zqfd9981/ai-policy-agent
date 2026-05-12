from __future__ import annotations

import sys
from pathlib import Path

# 兼容直接运行 `python app\clean\cleaner.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.clean.normalizer import (
    collapse_redundant_blank_lines,
    normalize_text,
    remove_page_number_lines,
)
from app.ingest.loader_factory import load_document
from app.ingest.metadata_loader import load_metadata_map
from app.models.document import CleanDocument, Document


def clean_document(document: Document) -> CleanDocument:
    """
    对单篇 Document 做第一版基础清洗。

    当前第一版遵循“先稳再细”的原则：
    - 不做激进的内容改写
    - 只做基础标准化
    - 对 PDF 额外去掉明显页码
    """

    clean_text = clean_text_content(document.raw_text, source_format=document.source_format)
    return CleanDocument(document=document, clean_text=clean_text)


def clean_text_content(text: str, *, source_format: str) -> str:
    """
    清洗正文文本的纯文本入口。

    之所以单独拆这个函数，是为了以后测试规则时可以不依赖 Document 对象。
    """

    normalized = normalize_text(text)

    if source_format.lower() == "pdf":
        normalized = remove_page_number_lines(normalized)
        normalized = collapse_redundant_blank_lines(normalized)

    return normalized.strip()


def main() -> None:
    """对 cleaner 第一版做一个简单联调测试。"""

    print("=" * 60)
    print("开始测试 cleaner 第一版...")
    print("=" * 60)

    metadata_map = load_metadata_map()

    for doc_id in ("JS002", "SH001"):
        raw_document = load_document(metadata_map[doc_id])
        clean_document_result = clean_document(raw_document)

        print(f"\n[OK] {doc_id} 清洗完成")
        print(f"source_format: {clean_document_result.source_format}")
        print(f"raw_length: {raw_document.text_length}")
        print(f"clean_length: {clean_document_result.text_length}")
        print("清洗后前 200 字：")
        print(clean_document_result.clean_text[:200])

    print("\n[OK] cleaner 第一版测试通过！")


if __name__ == "__main__":
    main()
