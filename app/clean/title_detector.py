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


@dataclass(frozen=True, slots=True)
class TitlePatternRule:
    """表示一种标题样式识别规则。"""

    # 样式名称。后续如果做“文档内层级推断”，会优先基于这个字段而不是写死层级。
    pattern_name: str
    # 该样式在第一版中的默认层级。
    # 注意：这是默认值，不等于永远的最终层级。
    default_level: int
    # 对应的正则表达式。
    pattern: re.Pattern[str]


# 第一版只覆盖当前政策文本里最常见、最稳定的几类标题样式。
# 后续如果遇到“第X章”“第X条”“一是、二是”“1.1”之类结构，再在这里继续扩展。
TITLE_PATTERN_RULES: tuple[TitlePatternRule, ...] = (
    TitlePatternRule(
        pattern_name="cn_section",
        default_level=1,
        pattern=re.compile(r"^(?P<marker>[一二三四五六七八九十百千]+、)(?P<title>.*)$"),
    ),
    TitlePatternRule(
        pattern_name="cn_paren",
        default_level=2,
        pattern=re.compile(r"^(?P<marker>（[一二三四五六七八九十百千]+）)(?P<title>.*)$"),
    ),
    TitlePatternRule(
        pattern_name="arabic_dot",
        default_level=3,
        pattern=re.compile(r"^(?P<marker>\d+[\.．、])(?P<title>.*)$"),
    ),
    TitlePatternRule(
        pattern_name="arabic_paren",
        default_level=4,
        pattern=re.compile(r"^(?P<marker>（\d+）)(?P<title>.*)$"),
    ),
)


@dataclass(frozen=True, slots=True)
class TitleDetectionResult:
    """表示某一行被识别为标题后的结构化结果。"""

    # 标题原始文本，保持清洗后的原样。
    line: str
    # 标题样式名称，例如 cn_section、cn_paren、arabic_dot、arabic_paren。
    pattern_name: str
    # 标题默认层级，数值越小层级越高。
    # 这里先保留“规则默认值”，后续可在文档级推断阶段再覆盖为最终层级。
    default_level: int
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

    for rule in TITLE_PATTERN_RULES:
        match = rule.pattern.fullmatch(normalized_line)
        if not match:
            continue

        marker = match.group("marker").strip()
        title_text = match.group("title").strip()
        return TitleDetectionResult(
            line=normalized_line,
            pattern_name=rule.pattern_name,
            default_level=rule.default_level,
            marker=marker,
            title_text=title_text,
        )

    return None


def is_title_line(line: str) -> bool:
    """判断一行文本是否是标题。"""

    return detect_title(line) is not None


def detect_title_level(line: str) -> int | None:
    """只返回默认层级，方便后续轻量判断。"""

    result = detect_title(line)
    if result is None:
        return None
    return result.default_level


def group_titles_by_pattern(text: str) -> dict[str, list[TitleDetectionResult]]:
    """按标题样式分组，便于后续做文档内层级推断。"""

    grouped: dict[str, list[TitleDetectionResult]] = {}

    for title in extract_titles(text):
        grouped.setdefault(title.pattern_name, []).append(title)

    return grouped


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
                f"pattern={title.pattern_name} | "
                f"default_level={title.default_level} | "
                f"marker={title.marker} | "
                f"title={title.title_text or '<空标题>'}"
            )

    print("\n[OK] title_detector 第一版测试通过！")


if __name__ == "__main__":
    main()
