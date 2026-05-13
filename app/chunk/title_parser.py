from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# 兼容直接运行 `python app\chunk\title_parser.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.clean.cleaner import clean_document
from app.clean.title_detector import TitleDetectionResult, detect_title
from app.ingest.loader_factory import load_document
from app.ingest.metadata_loader import load_metadata_map
from app.models.document import CleanDocument


SUBTITLE_SPLIT_PATTERN = re.compile(r"^(?P<label>[^，。；：:]{1,30})[，。；：:](?P<rest>.+)$")


@dataclass(frozen=True, slots=True)
class TitleSection:
    """表示一段已经挂到标题路径下、但还未进一步切成 chunk 的结构段。"""

    # 结构段来源的清洗后文档。
    clean_document: CleanDocument
    # 结构段在当前文档中的顺序编号，从 1 开始。
    section_index: int
    # 当前结构段对应的标题路径。
    # 如果是导语或前置说明，没有进入正式标题层级，就可能为空元组。
    title_path: tuple[str, ...]
    # 当前结构段正文。
    text: str
    # 触发这个结构段的当前标题行。
    # 比如“一、总体要求”或“（一）重点任务”，前置信息段则为 None。
    heading_line: str | None = None
    # 当前标题的样式名称，例如 cn_section、cn_paren、arabic_paren。
    heading_pattern_name: str | None = None
    # 当前标题的默认层级。
    heading_default_level: int | None = None

    def __post_init__(self) -> None:
        """统一字段格式并做基础校验。"""

        normalized_text = self.text.strip()
        normalized_title_path = tuple(title.strip() for title in self.title_path if title.strip())
        normalized_heading_line = self.heading_line.strip() if self.heading_line else None
        normalized_pattern_name = self.heading_pattern_name.strip() if self.heading_pattern_name else None

        object.__setattr__(self, "text", normalized_text)
        object.__setattr__(self, "title_path", normalized_title_path)
        object.__setattr__(self, "heading_line", normalized_heading_line)
        object.__setattr__(self, "heading_pattern_name", normalized_pattern_name)

        if self.section_index <= 0:
            raise ValueError(f"TitleSection.section_index 必须大于 0，当前值为: {self.section_index}")

        if not normalized_text:
            raise ValueError(f"TitleSection.text 不能为空，section_index={self.section_index}")

    @property
    def doc_id(self) -> str:
        """直接暴露 doc_id。"""

        return self.clean_document.doc_id

    @property
    def title(self) -> str:
        """直接暴露源文档标题。"""

        return self.clean_document.title

    @property
    def source_path(self) -> Path:
        """直接暴露源文件路径。"""

        return self.clean_document.source_path

    @property
    def text_length(self) -> int:
        """返回结构段正文长度。"""

        return len(self.text)


def parse_title_sections(clean_document: CleanDocument) -> list[TitleSection]:
    """
    把 CleanDocument 解析成“挂在标题路径下的结构段”。

    当前版本采用“顺序扫描 + 标题栈”思路：
    - 遇到标题时更新标题路径
    - 遇到普通行时挂到当前标题路径下
    - 首个标题之前的内容作为前置信息段保留
    """

    sections: list[TitleSection] = []
    title_stack: list[tuple[int, str]] = []
    current_heading: TitleDetectionResult | None = None
    body_lines: list[str] = []
    section_index = 1

    for line in clean_document.clean_text.split("\n"):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        title = detect_title(stripped_line)
        if title is not None:
            section_index = _flush_section(
                sections=sections,
                clean_document=clean_document,
                section_index=section_index,
                title_path=_extract_title_path(title_stack),
                body_lines=body_lines,
                heading=current_heading,
            )

            title_stack, title_body_prefix = _update_title_stack(title_stack, title)
            current_heading = title
            body_lines = [title_body_prefix] if title_body_prefix else []
            continue

        body_lines.append(stripped_line)

    _flush_section(
        sections=sections,
        clean_document=clean_document,
        section_index=section_index,
        title_path=_extract_title_path(title_stack),
        body_lines=body_lines,
        heading=current_heading,
    )

    return sections


def _flush_section(
    *,
    sections: list[TitleSection],
    clean_document: CleanDocument,
    section_index: int,
    title_path: tuple[str, ...],
    body_lines: list[str],
    heading: TitleDetectionResult | None,
) -> int:
    """
    把当前累计的正文行刷成一个结构段。

    如果当前没有正文内容，就不生成空 section。
    """

    if not body_lines:
        return section_index

    section = TitleSection(
        clean_document=clean_document,
        section_index=section_index,
        title_path=title_path,
        text="\n".join(body_lines).strip(),
        heading_line=heading.line if heading else None,
        heading_pattern_name=heading.pattern_name if heading else None,
        heading_default_level=heading.default_level if heading else None,
    )
    sections.append(section)
    return section_index + 1


def _update_title_stack(
    title_stack: list[tuple[int, str]],
    title: TitleDetectionResult,
) -> tuple[list[tuple[int, str]], str | None]:
    """
    根据标题默认层级更新标题路径栈。

    当前先使用 default_level 维护结构路径。
    后续如果接入“文档内层级推断”，只需要替换这里的层级来源即可。
    """

    level = title.default_level
    new_stack = list(title_stack)

    # 这里不能只按“当前路径长度”判断，因为文档可能跳级：
    # 例如二级标题下面直接出现四级标题 `（1）`。
    # 所以要看“栈顶标题的层级”，只要栈顶层级大于等于当前层级，就应该弹出。
    while new_stack and new_stack[-1][0] >= level:
        new_stack.pop()

    normalized_title_line, title_body_prefix = normalize_heading_line_for_stack(title)
    new_stack.append((level, normalized_title_line))
    return new_stack, title_body_prefix


def _extract_title_path(title_stack: list[tuple[int, str]]) -> tuple[str, ...]:
    """从带层级的标题栈中提取纯标题路径。"""

    return tuple(title for _, title in title_stack)


def normalize_heading_line_for_stack(title: TitleDetectionResult) -> tuple[str, str | None]:
    """
    规范化用于标题路径的 heading_line，并在必要时拆出正文前缀。

    当前只对小点标题做轻量拆分，例如：
    - （1）大模型+诊疗服务，支持打造……

    会变成：
    - 标题路径中的当前标题：`（1）大模型+诊疗服务`
    - 正文前缀：`支持打造……`
    """

    if title.pattern_name not in {"arabic_paren", "arabic_dot"}:
        return title.line, None

    match = SUBTITLE_SPLIT_PATTERN.fullmatch(title.title_text)
    if not match:
        return title.line, None

    subtitle_label = match.group("label").strip()
    remaining_text = match.group("rest").strip()

    if not subtitle_label or not remaining_text:
        return title.line, None

    normalized_heading_line = f"{title.marker}{subtitle_label}"
    return normalized_heading_line, remaining_text


def main() -> None:
    """对 title_parser 第一版做一个简单联调测试。"""

    print("=" * 60)
    print("开始测试 title_parser 第一版...")
    print("=" * 60)

    metadata_map = load_metadata_map()

    for doc_id in ("JS002", "SH001"):
        parsed_sections = parse_title_sections(
            clean_document(load_document(metadata_map[doc_id]))
        )

        print(f"\n[OK] {doc_id} 共解析出 {len(parsed_sections)} 个结构段")
        print("前 8 个结构段：")
        for section in parsed_sections[:8]:
            print(
                f"section={section.section_index} | "
                f"path={section.title_path or ('<前置信息>',)} | "
                f"text_length={section.text_length}"
            )
            print(section.text[:80])
            print("-" * 40)

    print("\n[OK] title_parser 第一版测试通过！")


if __name__ == "__main__":
    main()
