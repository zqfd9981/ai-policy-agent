from __future__ import annotations

import re


# 默认的单块最大字符数。
# 第一版先用一个偏保守的值，后续可以再做配置化。
DEFAULT_MAX_CHARS = 500


def split_text_by_length(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """
    对超长正文做长度兜底切分。

    当前策略尽量保持“结构优先”：
    1. 优先按已有换行段落拼接
    2. 如果单段仍然过长，再按句号/分号等句读切
    3. 如果还过长，最后再按固定长度硬切
    """

    normalized_text = text.strip()
    if not normalized_text:
        return []

    if len(normalized_text) <= max_chars:
        return [normalized_text]

    paragraphs = [paragraph.strip() for paragraph in normalized_text.split("\n") if paragraph.strip()]
    if not paragraphs:
        return _split_long_unit(normalized_text, max_chars=max_chars)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current_parts:
                chunks.append("\n".join(current_parts))
                current_parts = []
                current_length = 0

            chunks.extend(_split_long_unit(paragraph, max_chars=max_chars))
            continue

        projected_length = current_length + len(paragraph)
        if current_parts:
            projected_length += 1  # 预留一个换行符

        if projected_length > max_chars:
            chunks.append("\n".join(current_parts))
            current_parts = [paragraph]
            current_length = len(paragraph)
        else:
            current_parts.append(paragraph)
            current_length = projected_length

    if current_parts:
        chunks.append("\n".join(current_parts))

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _split_long_unit(text: str, *, max_chars: int) -> list[str]:
    """
    对单个超长单元继续细分。

    先尝试按句读切；
    如果句子本身也过长，再退化到固定长度切分。
    """

    sentences = split_by_sentences(text)
    if len(sentences) <= 1:
        return split_by_hard_limit(text, max_chars=max_chars)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current_parts:
                chunks.append("".join(current_parts).strip())
                current_parts = []
                current_length = 0

            chunks.extend(split_by_hard_limit(sentence, max_chars=max_chars))
            continue

        projected_length = current_length + len(sentence)
        if projected_length > max_chars and current_parts:
            chunks.append("".join(current_parts).strip())
            current_parts = [sentence]
            current_length = len(sentence)
        else:
            current_parts.append(sentence)
            current_length = projected_length

    if current_parts:
        chunks.append("".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk]


def split_by_sentences(text: str) -> list[str]:
    """
    按句读符号切成较小语义单元，并尽量保留句读本身。

    例如：
    - 句号
    - 分号
    - 问号
    - 感叹号
    """

    parts = re.split(r"(?<=[。！？；])", text)
    return [part.strip() for part in parts if part.strip()]


def split_by_hard_limit(text: str, *, max_chars: int) -> list[str]:
    """最后兜底：按固定字符长度硬切。"""

    normalized_text = text.strip()
    return [
        normalized_text[index:index + max_chars].strip()
        for index in range(0, len(normalized_text), max_chars)
        if normalized_text[index:index + max_chars].strip()
    ]
