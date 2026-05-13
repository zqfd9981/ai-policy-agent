from __future__ import annotations

import sys
from pathlib import Path

# 兼容直接运行 `python app\chunk\chunker.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.chunk.splitters import DEFAULT_MAX_CHARS, split_text_by_length
from app.chunk.title_parser import TitleSection, parse_title_sections
from app.clean.cleaner import clean_document
from app.ingest.loader_factory import load_document
from app.ingest.metadata_loader import load_metadata_map
from app.models.chunk import Chunk
from app.models.document import CleanDocument


def chunk_document(clean_document: CleanDocument, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """
    把一篇 CleanDocument 切成最终的 Chunk 列表。

    当前策略：
    1. 先按 title_parser 解析出的结构段切
    2. 再对超长结构段做长度兜底切分
    """

    sections = parse_title_sections(clean_document)
    chunks: list[Chunk] = []
    chunk_index = 1

    for section in sections:
        section_chunks, chunk_index = chunk_section(
            section,
            start_chunk_index=chunk_index,
            max_chars=max_chars,
        )
        chunks.extend(section_chunks)

    return chunks


def chunk_section(
    section: TitleSection,
    *,
    start_chunk_index: int,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[list[Chunk], int]:
    """
    把单个结构段切成一个或多个 Chunk。

    如果结构段本身不长，就直接变成一个 chunk；
    如果过长，就交给 splitters 做兜底切分。
    """

    parts = split_text_by_length(section.text, max_chars=max_chars)
    chunks: list[Chunk] = []
    next_chunk_index = start_chunk_index

    for part in parts:
        chunk = Chunk(
            clean_document=section.clean_document,
            chunk_id=build_chunk_id(section.doc_id, next_chunk_index),
            chunk_index=next_chunk_index,
            title_path=section.title_path,
            text=part,
        )
        chunks.append(chunk)
        next_chunk_index += 1

    return chunks, next_chunk_index


def build_chunk_id(doc_id: str, chunk_index: int) -> str:
    """统一生成 chunk_id，例如 `JS002_0001`。"""

    return f"{doc_id}_{chunk_index:04d}"


def main() -> None:
    """对 chunker 第一版做一个简单联调测试。"""

    print("=" * 60)
    print("开始测试 chunker 第一版...")
    print("=" * 60)

    metadata_map = load_metadata_map()

    for doc_id in ("JS002", "SH001"):
        clean_doc = clean_document(load_document(metadata_map[doc_id]))
        chunks = chunk_document(clean_doc)

        print(f"\n[OK] {doc_id} 共生成 {len(chunks)} 个 chunk")
        print("前 8 个 chunk：")
        for chunk in chunks[:8]:
            print(
                f"chunk_id={chunk.chunk_id} | "
                f"path={chunk.title_path or ('<前置信息>',)} | "
                f"text_length={chunk.text_length}"
            )
            print(chunk.text[:80])
            print("-" * 40)

    print("\n[OK] chunker 第一版测试通过！")


if __name__ == "__main__":
    main()
