from __future__ import annotations

import re


# 常见的 PDF 页码样式，例如：
# - — 1 —
# - - 2 -
# - 第 3 页
PAGE_NUMBER_PATTERNS = (
    re.compile(r"^[—\-]+\s*\d+\s*[—\-]+$"),
    re.compile(r"^第\s*\d+\s*页$"),
)


def normalize_text(text: str) -> str:
    """
    对正文做第一版基础标准化。

    当前阶段只做“低风险、可解释”的处理：
    - 统一换行
    - 统一常见空白字符
    - 去除每行首尾多余空白
    - 压缩连续空行
    """

    normalized = normalize_newlines(text)
    normalized = normalize_unicode_spaces(normalized)
    normalized = strip_line_whitespace(normalized)
    normalized = collapse_redundant_blank_lines(normalized)
    return normalized.strip()


def normalize_newlines(text: str) -> str:
    """把不同平台的换行统一成 \\n。"""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_unicode_spaces(text: str) -> str:
    """
    统一常见空白字符。

    这里只处理最常见、风险较低的几种：
    - 全角空格 `\\u3000`
    - 不间断空格 `\\xa0`
    - 制表符 `\\t`
    """

    return text.replace("\u3000", " ").replace("\xa0", " ").replace("\t", " ")


def strip_line_whitespace(text: str) -> str:
    """去掉每一行首尾多余空白，但保留正文换行结构。"""

    return "\n".join(line.strip() for line in text.split("\n"))


def collapse_redundant_blank_lines(text: str, max_blank_lines: int = 1) -> str:
    """
    压缩多余空行。
    `max_blank_lines=1` 表示最多只保留一个空白行，也就是相邻段落之间保留一行间隔。
    """

    blank_streak = 0
    cleaned_lines: list[str] = []

    for line in text.split("\n"):
        if line == "":
            blank_streak += 1
            if blank_streak <= max_blank_lines:
                cleaned_lines.append(line)
            continue

        blank_streak = 0
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_page_number_lines(text: str) -> str:
    """删除明显属于页码的整行文本。"""

    kept_lines: list[str] = []

    for line in text.split("\n"):
        if is_page_number_line(line):
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines)


def is_page_number_line(line: str) -> bool:
    """判断某一行是否像 PDF 页码。"""

    stripped = line.strip()
    if not stripped:
        return False

    return any(pattern.fullmatch(stripped) for pattern in PAGE_NUMBER_PATTERNS)
