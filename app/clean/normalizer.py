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

# 有些 PDF 提取结果会把页码嵌进正文中间，例如：
# - 大模型+疾— 3 —控监测
# 这里单独清理这种行内页码片段。
INLINE_PAGE_NUMBER_PATTERN = re.compile(r"\s*[—\-]+\s*\d+\s*[—\-]+\s*")

# 用于识别行内出现的“小层级标题”，例如：
# - （1）
# - （2）
# - （9）
# 这里只先限制 1~2 位数字，避免把年份之类内容误拆开。
INLINE_SUBTITLE_PATTERN = re.compile(r"(?<!^)(?=（\d{1,2}）)")

# 标题起始样式的轻量识别规则。
# 这里不追求完整层级推断，只为清洗阶段判断“下一行是不是一个新标题”。
TITLE_START_PATTERNS = (
    re.compile(r"^[一二三四五六七八九十百千]+、"),
    re.compile(r"^（[一二三四五六七八九十百千]+）"),
    re.compile(r"^\d+[\.．、]"),
    re.compile(r"^（\d{1,2}）"),
)

# 需要做“标题续行合并”的小层级标题样式。
MERGEABLE_SUBTITLE_PATTERNS = (
    re.compile(r"^（\d{1,2}）"),
    re.compile(r"^\d+[\.．、]"),
)

# 当合并后的标题首句里已经出现这些标点时，就说明至少拼回了一个完整的短句边界。
TITLE_SENTENCE_PUNCTUATION_PATTERN = re.compile(r"[，。；：!?！？]")


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
    normalized = remove_inline_page_number_fragments(normalized)
    normalized = strip_line_whitespace(normalized)
    normalized = split_inline_subtitles(normalized)
    normalized = merge_broken_subsection_titles(normalized)
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


def remove_inline_page_number_fragments(text: str) -> str:
    """
    删除嵌入到正文中的页码片段。

    典型场景来自 PDF 跨页提取，把页码夹到了词语中间：
    - 大模型+疾— 3 —控监测
    处理后会变成：
    - 大模型+疾控监测
    """

    return INLINE_PAGE_NUMBER_PATTERN.sub("", text)


def split_inline_subtitles(text: str) -> str:
    """
    把同一行里的行内小标题拆成独立行。

    典型目标形态：
    - （一）医疗服务：（1）大模型+诊疗服务……
    - ……平台；（2）大模型+患者服务……

    处理后会变成：
    - （一）医疗服务：
    - （1）大模型+诊疗服务……
    - ……平台；
    - （2）大模型+患者服务……

    这一版先只处理最常见的 `（1）`、`（2）` 这类数字小点，
    后续如果需要，再扩展到更多样式。
    """

    normalized_lines: list[str] = []

    for line in text.split("\n"):
        stripped_line = line.strip()
        if not stripped_line:
            normalized_lines.append(stripped_line)
            continue

        if "（" not in stripped_line or "）" not in stripped_line:
            normalized_lines.append(stripped_line)
            continue

        split_line = INLINE_SUBTITLE_PATTERN.sub("\n", stripped_line)
        normalized_lines.extend(part.strip() for part in split_line.split("\n") if part.strip())

    return "\n".join(normalized_lines)


def merge_broken_subsection_titles(text: str) -> str:
    """
    合并被 PDF 强行断开的“小层级标题首句”。

    典型问题：
    - （2）大模型+患者
      服务，支持打造……
    - （5）大模型+疾
      控监测，支持打造……

    处理目标：
    - （2）大模型+患者服务，支持打造……
    - （5）大模型+疾控监测，支持打造……

    注意：
    - 这里只处理 `（1）`、`1.` 这类小层级标题
    - 不处理 `一、`、`（一）` 这种大标题，避免误把正文并进去
    """

    lines = text.split("\n")
    merged_lines: list[str] = []
    index = 0

    while index < len(lines):
        current_line = lines[index].strip()

        if not current_line:
            merged_lines.append(current_line)
            index += 1
            continue

        if not is_mergeable_subtitle_line(current_line):
            merged_lines.append(current_line)
            index += 1
            continue

        merged_line = current_line
        look_ahead = index + 1

        while look_ahead < len(lines):
            next_line = lines[look_ahead].strip()

            # 跳过空行，继续往后找真正的续行。
            if not next_line:
                look_ahead += 1
                continue

            # 一旦遇到新的标题行，就停止合并。
            if is_title_start_line(next_line):
                break

            merged_line += next_line
            look_ahead += 1

            # 只要拼回了一个常见句读边界，就先收手，避免把整段正文全吃进来。
            if TITLE_SENTENCE_PUNCTUATION_PATTERN.search(merged_line):
                break

        merged_lines.append(merged_line)
        index = look_ahead

    return "\n".join(merged_lines)


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


def is_title_start_line(line: str) -> bool:
    """判断一行文本是否像某种标题起始行。"""

    stripped = line.strip()
    if not stripped:
        return False

    return any(pattern.match(stripped) for pattern in TITLE_START_PATTERNS)


def is_mergeable_subtitle_line(line: str) -> bool:
    """判断当前行是否属于需要做续行合并的小层级标题。"""

    stripped = line.strip()
    if not stripped:
        return False

    return any(pattern.match(stripped) for pattern in MERGEABLE_SUBTITLE_PATTERNS)
