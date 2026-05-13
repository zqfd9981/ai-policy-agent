from __future__ import annotations

import json
import sys
from pathlib import Path

# 兼容直接运行 `python app\chunk\chunk_builder.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.chunk.chunker import chunk_document
from app.clean.cleaner import clean_document
from app.ingest.loader_factory import load_document
from app.ingest.metadata_loader import load_metadata, load_metadata_map
from app.models.chunk import Chunk
from app.models.metadata import PolicyMetadata


# 默认把批量切片结果导出到 outputs/chunks 目录下。
DEFAULT_CHUNK_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "outputs" / "chunks" / "policy_chunks.jsonl"
)


def build_document_chunks(metadata: PolicyMetadata) -> list[Chunk]:
    """
    为单篇政策文档构建完整 chunk 列表。

    这一步把我们前面已经打通的链路串起来：
    metadata -> 原始文件 -> Document -> CleanDocument -> Chunk
    """

    document = load_document(metadata)
    clean_doc = clean_document(document)
    return chunk_document(clean_doc)


def build_chunks_for_metadata_list(metadata_list: list[PolicyMetadata]) -> list[Chunk]:
    """按给定 metadata 列表批量构建 chunk。"""

    all_chunks: list[Chunk] = []
    for metadata in metadata_list:
        all_chunks.extend(build_document_chunks(metadata))
    return all_chunks


def build_chunks_for_doc_ids(doc_ids: list[str]) -> list[Chunk]:
    """按 doc_id 列表批量构建 chunk，便于局部调试。"""

    metadata_map = load_metadata_map()
    metadata_list = [metadata_map[doc_id] for doc_id in doc_ids]
    return build_chunks_for_metadata_list(metadata_list)


def serialize_chunk(chunk: Chunk) -> dict[str, object]:
    """把 Chunk 转成适合写入 jsonl 的字典。"""

    return chunk.to_dict()


def export_chunks_to_jsonl(
    chunks: list[Chunk],
    output_path: Path | str = DEFAULT_CHUNK_OUTPUT_PATH,
) -> Path:
    """
    把 chunk 列表导出成 jsonl。

    jsonl 很适合后续做：
    - 人工抽查
    - 向量化入库
    - 检索调试
    """

    normalized_output_path = Path(output_path)
    normalized_output_path.parent.mkdir(parents=True, exist_ok=True)

    with normalized_output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            payload = serialize_chunk(chunk)
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return normalized_output_path


def build_and_export_chunks(
    doc_ids: list[str] | None = None,
    output_path: Path | str = DEFAULT_CHUNK_OUTPUT_PATH,
) -> tuple[list[Chunk], Path]:
    """
    一步完成“批量构建 + 导出”。

    - 不传 doc_ids 时：导出全部政策文档
    - 传 doc_ids 时：只导出指定文档
    """

    if doc_ids is None:
        metadata_list = load_metadata()
        chunks = build_chunks_for_metadata_list(metadata_list)
    else:
        chunks = build_chunks_for_doc_ids(doc_ids)

    exported_path = export_chunks_to_jsonl(chunks, output_path=output_path)
    return chunks, exported_path


def main() -> None:
    """给 chunk 构建与导出做一个简单联调测试。"""

    print("=" * 60)
    print("开始测试 chunk_builder 第一版...")
    print("=" * 60)

    sample_doc_ids = ["JS002", "SH001"]
    sample_output_path = (
        Path(__file__).resolve().parents[2] / "outputs" / "chunks" / "sample_chunks.jsonl"
    )

    chunks, exported_path = build_and_export_chunks(
        doc_ids=sample_doc_ids,
        output_path=sample_output_path,
    )

    print(f"[OK] 已为 {len(sample_doc_ids)} 篇文档生成 {len(chunks)} 个 chunk")
    print(f"[OK] 导出文件: {exported_path}")
    print("前 5 个 chunk 预览:")

    for chunk in chunks[:5]:
        print(
            f"chunk_id={chunk.chunk_id} | "
            f"path={chunk.title_path or ('<前置信息>',)} | "
            f"text_length={chunk.text_length}"
        )
        print(chunk.text[:80])
        print("-" * 40)

    print("[OK] chunk_builder 第一版测试通过")


if __name__ == "__main__":
    main()
