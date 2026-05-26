from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunk.chunk_builder import DEFAULT_CHUNK_OUTPUT_PATH, build_and_export_chunks
from app.retrieval.retriever import (
    DEFAULT_FAISS_INDEX_PATH,
    DEFAULT_RETRIEVAL_MANIFEST_PATH,
    DEFAULT_RETRIEVAL_PAYLOAD_PATH,
    build_and_save_default_retriever,
)


@dataclass(frozen=True, slots=True)
class RebuildPaths:
    """表示一次重建任务实际会写入的产物路径。"""

    chunk_output_path: Path
    index_path: Path
    payload_path: Path
    manifest_path: Path
    is_partial: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="重建 chunk 与检索索引产物。默认执行全量正式重建。"
    )
    parser.add_argument(
        "--doc-ids",
        nargs="+",
        help="只为指定 doc_id 构建 chunk。默认会写入 partial 产物，避免覆盖正式索引。",
    )
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="只重建 chunk jsonl，不重建检索索引。",
    )
    parser.add_argument(
        "--retriever-only",
        action="store_true",
        help="只基于已有 chunk jsonl 重建检索索引。",
    )
    parser.add_argument(
        "--chunk-output",
        type=Path,
        help="自定义 chunk jsonl 输出路径。",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        help="自定义 FAISS 索引路径。",
    )
    parser.add_argument(
        "--payload-path",
        type=Path,
        help="自定义 retrieval payload jsonl 路径。",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="自定义 retriever manifest 路径。",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> RebuildPaths:
    """根据参数决定本次重建写入正式产物还是 partial 调试产物。"""

    use_partial_defaults = bool(args.doc_ids) and not any(
        (
            args.chunk_output,
            args.index_path,
            args.payload_path,
            args.manifest_path,
        )
    )

    if use_partial_defaults:
        chunk_output_path = PROJECT_ROOT / "outputs" / "chunks" / "policy_chunks.partial.jsonl"
        index_path = PROJECT_ROOT / "outputs" / "retrieval" / "policy_chunks.partial.faiss"
        payload_path = PROJECT_ROOT / "outputs" / "retrieval" / "policy_payloads.partial.jsonl"
        manifest_path = (
            PROJECT_ROOT / "outputs" / "retrieval" / "policy_retriever_manifest.partial.json"
        )
    else:
        chunk_output_path = args.chunk_output or DEFAULT_CHUNK_OUTPUT_PATH
        index_path = args.index_path or DEFAULT_FAISS_INDEX_PATH
        payload_path = args.payload_path or DEFAULT_RETRIEVAL_PAYLOAD_PATH
        manifest_path = args.manifest_path or DEFAULT_RETRIEVAL_MANIFEST_PATH

    return RebuildPaths(
        chunk_output_path=Path(chunk_output_path),
        index_path=Path(index_path),
        payload_path=Path(payload_path),
        manifest_path=Path(manifest_path),
        is_partial=use_partial_defaults,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.chunks_only and args.retriever_only:
        raise ValueError("--chunks-only 与 --retriever-only 不能同时使用。")

    if args.retriever_only and args.doc_ids:
        raise ValueError("--retriever-only 不接受 --doc-ids；请先生成 chunk jsonl。")


def rebuild_chunks(*, doc_ids: list[str] | None, output_path: Path) -> tuple[int, Path]:
    """执行 chunk 重建并返回 chunk 数量与输出路径。"""

    chunks, exported_path = build_and_export_chunks(doc_ids=doc_ids, output_path=output_path)
    return len(chunks), Path(exported_path)


def rebuild_retriever(*, chunk_output_path: Path, paths: RebuildPaths) -> dict[str, Path]:
    """基于给定 chunk jsonl 重建 retriever 产物。"""

    _, saved_paths = build_and_save_default_retriever(
        chunk_jsonl_path=chunk_output_path,
        index_path=paths.index_path,
        payload_path=paths.payload_path,
        manifest_path=paths.manifest_path,
    )
    return {name: Path(path) for name, path in saved_paths.items()}


def print_header(paths: RebuildPaths, args: argparse.Namespace) -> None:
    print("=" * 72)
    print("Rebuild Data Artifacts")
    print("=" * 72)
    print(f"模式: {'partial 调试重建' if paths.is_partial else '正式重建'}")
    print(f"doc_ids: {args.doc_ids or 'ALL'}")
    print(f"chunk_output: {paths.chunk_output_path}")
    if not args.chunks_only:
        print(f"retriever.index: {paths.index_path}")
        print(f"retriever.payload: {paths.payload_path}")
        print(f"retriever.manifest: {paths.manifest_path}")
    print()


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = resolve_paths(args)
    print_header(paths, args)

    started_at = time.perf_counter()
    chunk_output_path = paths.chunk_output_path

    if not args.retriever_only:
        chunk_count, chunk_output_path = rebuild_chunks(
            doc_ids=args.doc_ids,
            output_path=paths.chunk_output_path,
        )
        print(f"[OK] chunk 已重建: {chunk_output_path}")
        print(f"[OK] chunk_count = {chunk_count}")

    if not args.chunks_only:
        saved_paths = rebuild_retriever(
            chunk_output_path=chunk_output_path,
            paths=paths,
        )
        print(f"[OK] retriever 已重建: {saved_paths['index_path']}")
        print(f"[OK] payload 已导出: {saved_paths['payload_path']}")
        print(f"[OK] manifest 已导出: {saved_paths['manifest_path']}")

    elapsed = time.perf_counter() - started_at
    print(f"\n[OK] 重建完成，用时 {elapsed:.2f}s")


if __name__ == "__main__":
    main()
