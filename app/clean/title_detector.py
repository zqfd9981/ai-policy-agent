from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# 兼容直接运行 `python app\clean\title_detector.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.clean.cleaner import clean_document
from app.ingest.loader_factory import load_document
from app.ingest.metadata_loader import load_metadata_map


# 第一版只覆盖当前政策文本里最常见、最稳定的几类标题样式。
# 后续如果遇到“第X章”“第X条”之类结构，再在这里继续扩展。
TITLE_PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
    # 一级标题，例如：一、总体要求
    (1, re.compile(r"^(?P<marker>[一二三四五六七八九十百千]+、)(?P<title>.*)$")),
    # 二级标题，例如：（一）工作目标
    (2, re.compile(r"^(?P<marker>（[一二三四五六七八九十百千]+）)(?P<title>.*)$")),
    # 三级标题，例如：1. 基本要求 / 1．基本要求 / 1、基本要求
    (3, re.compile(r"^(?P<marker>\d+[\.．、])(?P<title>.*)$")),
    # 四级标题，例如：（1）申报条件
    (4, re.compile(r"^(?P<marker>（\d+）)(?P<title>.*)$")),
)


@dataclass(frozen=True, slots=True)
class TitleDetectionResult:
    """表示某一行被识别为标题后的结构化结果。"""

    # 标题原始文本，保持清洗后的原样。
    line: str
    # 标题层级，数值越小层级越高。
    level: int
    # 标题前缀标记，例如“一、”“（一）”“1.”。
    marker: str
    # 去掉前缀后的标题主体。
    title_text: str


def detect_title(line: str) -> TitleDetectionResult | None:
    """
    识别单行文本是否为标题。

    如果命中已知规则，就返回结构化结果；
    否则返回 None。
    """

    normalized_line = normalize_title_line(line)
    if not normalized_line:
        return None

    for level, pattern in TITLE_PATTERNS:
        match = pattern.fullmatch(normalized_line)
        if not match:
            continue

        marker = match.group("marker").strip()
        title_text = match.group("title").strip()
        return TitleDetectionResult(
            line=normalized_line,
            level=level,
            marker=marker,
            title_text=title_text,
        )

    return None


def is_title_line(line: str) -> bool:
    """判断一行文本是否是标题。"""

    return detect_title(line) is not None


def detect_title_level(line: str) -> int | None:
    """只返回标题层级，方便后续轻量判断。"""

    result = detect_title(line)
    if result is None:
        return None
    return result.level


def extract_titles(text: str) -> list[TitleDetectionResult]:
    """从整篇文本中提取所有识别到的标题行。"""

    results: list[TitleDetectionResult] = []

    for line in text.split("\n"):
        title = detect_title(line)
        if title is not None:
            results.append(title)

    return results


def normalize_title_line(line: str) -> str:
    """
    对标题行做轻量规范化，避免因为前后空白导致漏检。

    这里故意不做激进改写，只做 strip。
    """

    return line.strip()


def main() -> None:
    """对标题识别第一版做一个简单联调测试。"""

    print("=" * 60)
    print("开始测试 title_detector 第一版...")
    print("=" * 60)

    metadata_map = load_metadata_map()

    for doc_id in ("JS002", "SH001"):
        clean_doc = clean_document(load_document(metadata_map[doc_id]))
        titles = extract_titles(clean_doc.clean_text)

        print(f"\n[OK] {doc_id} 共识别到 {len(titles)} 个标题")
        print("前 10 个标题：")
        for title in titles[:10]:
            print(
                f"level={title.level} | marker={title.marker} | "
                f"title={title.title_text or '<空标题>'}"
            )

    print("\n[OK] title_detector 第一版测试通过！")


if __name__ == "__main__":
    main()
