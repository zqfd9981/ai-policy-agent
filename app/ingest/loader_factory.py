from __future__ import annotations

import sys
from pathlib import Path

# 兼容直接运行 `python app\ingest\loader_factory.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.ingest.metadata_loader import load_metadata, load_metadata_map
from app.models.metadata import PolicyMetadata


# raw 原始文件的根目录。
DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"

# 当前阶段仓库中真正支持的原始文件格式。
ALLOWED_RAW_SUFFIXES = {".pdf", ".txt"}


class SourceFileLookupError(FileNotFoundError):
    """根据 metadata 查找 raw 文件失败时抛出的异常。"""


def find_source_file(doc_id: str, raw_root: str | Path = DEFAULT_RAW_ROOT) -> Path:
    """
    根据 doc_id 在 data/raw 目录中查找真实文件。

    这一步故意不依赖 region 或 source_format 拼路径，
    而是直接在 raw 目录递归查找 doc_id 前缀，避免后续路径规则变动时出错。
    """

    root_path = Path(raw_root)
    _validate_raw_root(root_path)

    candidates = _find_candidates(doc_id, root_path)

    if not candidates:
        raise SourceFileLookupError(f"未找到 doc_id={doc_id} 对应的 raw 文件。")

    if len(candidates) > 1:
        candidate_text = "\n".join(f"- {path}" for path in candidates)
        raise SourceFileLookupError(
            f"doc_id={doc_id} 匹配到了多个 raw 文件，无法唯一确定：\n{candidate_text}"
        )

    return candidates[0]


def find_source_file_for_metadata(
    metadata: PolicyMetadata,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    *,
    validate_source_format: bool = True,
) -> Path:
    """
    根据一条 metadata 查找对应原始文件。

    默认会顺手校验：metadata 记录的 source_format 是否和真实文件扩展名一致。
    """

    source_path = find_source_file(metadata.doc_id, raw_root=raw_root)

    if validate_source_format:
        actual_source_format = source_path.suffix.lstrip(".").lower()
        if actual_source_format != metadata.source_format:
            raise SourceFileLookupError(
                f"doc_id={metadata.doc_id} 的 source_format 不一致："
                f"metadata={metadata.source_format}, raw={actual_source_format}"
            )

    return source_path


def build_source_file_map(
    metadata_list: list[PolicyMetadata],
    raw_root: str | Path = DEFAULT_RAW_ROOT,
) -> dict[str, Path]:
    """批量构建 doc_id 到 raw 文件路径的映射。"""

    return {
        metadata.doc_id: find_source_file_for_metadata(metadata, raw_root=raw_root)
        for metadata in metadata_list
    }


def _validate_raw_root(raw_root: Path) -> None:
    """确保 raw 根目录存在。"""

    if not raw_root.exists():
        raise SourceFileLookupError(f"raw 根目录不存在: {raw_root}")

    if not raw_root.is_dir():
        raise SourceFileLookupError(f"raw 根路径不是目录: {raw_root}")


def _find_candidates(doc_id: str, raw_root: Path) -> list[Path]:
    """
    找出所有可能属于该 doc_id 的候选文件。

    当前文件命名有两种典型形式：
    - BJ001.pdf
    - JS001_省政府关于......pdf
    所以这里只判断“文件名是否以 doc_id 开头”，并限制扩展名必须是当前支持的格式。
    """

    candidates: list[Path] = []

    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in ALLOWED_RAW_SUFFIXES:
            continue

        if path.stem.startswith(doc_id):
            candidates.append(path)

    return sorted(candidates)


def main() -> None:
    """对“根据 metadata 查找 raw 文件”做一个简单联调测试。"""

    print("=" * 60)
    print("开始测试 raw 文件查找...")
    print("=" * 60)

    metadata_list = load_metadata()
    metadata_map = load_metadata_map()

    source_file_map = build_source_file_map(metadata_list)
    print(f"[OK] 成功为 {len(source_file_map)} 条 metadata 找到 raw 文件")

    # 打印前几条结果，确认映射关系是否符合预期。
    print("\n前 5 条映射结果：")
    for metadata in metadata_list[:5]:
        source_path = source_file_map[metadata.doc_id]
        print(f"{metadata.doc_id} -> {source_path.relative_to(DEFAULT_RAW_ROOT)}")

    # 额外单独验证一次指定 doc_id，方便肉眼检查。
    print("\n单条校验：")
    sh003 = metadata_map["SH003"]
    sh003_path = find_source_file_for_metadata(sh003)
    print(f"SH003 -> {sh003_path}")

    print("\n[OK] raw 文件查找测试通过！")


if __name__ == "__main__":
    main()
