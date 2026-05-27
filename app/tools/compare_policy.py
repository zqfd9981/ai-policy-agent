from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.ingest.metadata_loader import load_metadata_map
from app.models.metadata import PolicyMetadata
from app.tools.retrieve_policy import RetrievePolicyOutput, RetrievePolicyTool
from app.tools.summarize_policy import (
    SECTION_DEFINITIONS,
    PolicySummaryOutput,
    SummaryEvidence,
    choose_doc_id_from_retrieval,
    extract_doc_id_from_text,
    match_doc_id_by_title,
    summarize_policy,
)


@dataclass(frozen=True, slots=True)
class PolicyComparisonSection:
    """表示某一个固定分区下的双文档对比结果。"""

    key: str
    label: str
    left_points: tuple[str, ...]
    right_points: tuple[str, ...]
    comparison_note: str

    def to_dict(self) -> dict[str, Any]:
        """把分区对比结果转换成普通字典。"""

        return {
            "key": self.key,
            "label": self.label,
            "left_points": list(self.left_points),
            "right_points": list(self.right_points),
            "comparison_note": self.comparison_note,
        }


@dataclass(frozen=True, slots=True)
class PolicyCompareOutput:
    """
    表示一次政策对比工具调用的统一输出。

    当前设计尽量复用 summarize 结果：
    - 先各自得到两篇政策的结构化摘要
    - 再把同名分区并排组织成对比结果
    """

    query: str
    selection_reason: str
    left_summary: PolicySummaryOutput
    right_summary: PolicySummaryOutput
    sections: tuple[PolicyComparisonSection, ...]

    @property
    def citation_count(self) -> int:
        """返回当前对比结果里的总引用条数。"""

        return len(self.all_citations)

    @property
    def all_citations(self) -> tuple[dict[str, Any], ...]:
        """按去重顺序返回两篇政策的全部引用。"""

        seen: set[tuple[str, str, str]] = set()
        citations: list[dict[str, Any]] = []

        for item in (*self.left_summary.all_citations, *self.right_summary.all_citations):
            key = (item.doc_id, item.chunk_id, item.text)
            if key in seen:
                continue
            seen.add(key)
            citations.append(item.to_dict())

        return tuple(citations)

    def to_dict(self) -> dict[str, Any]:
        """把对比结果转换成适合 JSON 序列化的字典。"""

        return {
            "query": self.query,
            "selection_reason": self.selection_reason,
            "left_summary": self.left_summary.to_dict(),
            "right_summary": self.right_summary.to_dict(),
            "sections": [section.to_dict() for section in self.sections],
            "citation_count": self.citation_count,
            "citations": [dict(item) for item in self.all_citations],
        }


class PolicyCompareResolutionError(ValueError):
    """无法稳定定位两篇待对比政策时抛出的异常。"""


def compare_policy(
    query: str,
    *,
    left_doc_id: str | None = None,
    right_doc_id: str | None = None,
    top_k: int = 6,
    max_points_per_section: int = 3,
    retrieve_tool: RetrievePolicyTool | None = None,
) -> PolicyCompareOutput:
    """
    对两篇政策做固定分区的结构化对比。

    第一版采用“摘要复用型 compare”：
    1. 先确定要对比的两篇政策
    2. 再分别调用 summarize_policy 得到结构化摘要
    3. 最后把两边的固定分区并排组织成对比结果
    """

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空。")

    metadata_map = load_metadata_map()
    resolved_left_doc_id, resolved_right_doc_id, selection_reason = resolve_compare_doc_ids(
        normalized_query,
        metadata_map=metadata_map,
        left_doc_id=left_doc_id,
        right_doc_id=right_doc_id,
        top_k=top_k,
        retrieve_tool=retrieve_tool,
    )

    left_summary = summarize_policy(
        normalized_query,
        doc_id=resolved_left_doc_id,
        top_k=top_k,
        max_points_per_section=max_points_per_section,
        retrieve_tool=retrieve_tool,
    )
    right_summary = summarize_policy(
        normalized_query,
        doc_id=resolved_right_doc_id,
        top_k=top_k,
        max_points_per_section=max_points_per_section,
        retrieve_tool=retrieve_tool,
    )

    sections = build_compare_sections(left_summary, right_summary)
    return PolicyCompareOutput(
        query=normalized_query,
        selection_reason=selection_reason,
        left_summary=left_summary,
        right_summary=right_summary,
        sections=sections,
    )


def resolve_compare_doc_ids(
    query: str,
    *,
    metadata_map: dict[str, PolicyMetadata],
    left_doc_id: str | None = None,
    right_doc_id: str | None = None,
    top_k: int = 6,
    retrieve_tool: RetrievePolicyTool | None = None,
) -> tuple[str, str, str]:
    """确定当前 compare 请求对应的两篇目标政策。"""

    if left_doc_id and right_doc_id:
        normalized_left_doc_id = left_doc_id.strip().upper()
        normalized_right_doc_id = right_doc_id.strip().upper()
        validate_doc_id(normalized_left_doc_id, metadata_map)
        validate_doc_id(normalized_right_doc_id, metadata_map)
        if normalized_left_doc_id == normalized_right_doc_id:
            raise PolicyCompareResolutionError("compare 需要两篇不同的政策文档。")
        return (
            normalized_left_doc_id,
            normalized_right_doc_id,
            f"根据显式 doc_id 对定位到 {normalized_left_doc_id} 与 {normalized_right_doc_id}。",
        )

    matched_doc_ids = match_compare_doc_ids_from_query(query, metadata_map)
    if len(matched_doc_ids) >= 2:
        return (
            matched_doc_ids[0],
            matched_doc_ids[1],
            f"根据 query 中的政策标题线索匹配到 {matched_doc_ids[0]} 与 {matched_doc_ids[1]}。",
        )

    # 对“苏州 vs 杭州”“北京 vs 上海”这类地区型 compare，
    # 仅靠标题匹配不够稳，所以这里单独补一层地区感知的文档定位。
    mentioned_regions = extract_regions_from_query(query, metadata_map)
    if len(mentioned_regions) >= 2:
        retrieval_output = (retrieve_tool or RetrievePolicyTool()).run(query, top_k=max(4, int(top_k)))
        region_doc_ids = resolve_compare_doc_ids_by_regions(
            query,
            mentioned_regions=mentioned_regions,
            metadata_map=metadata_map,
            retrieval_output=retrieval_output,
        )
        if len(region_doc_ids) >= 2:
            return (
                region_doc_ids[0],
                region_doc_ids[1],
                (
                    f"根据 query 中的地区线索 {mentioned_regions[0]} 与 {mentioned_regions[1]}，"
                    f"定位到 {region_doc_ids[0]} 与 {region_doc_ids[1]}。"
                ),
            )

    retrieval_output = (retrieve_tool or RetrievePolicyTool()).run(query, top_k=max(4, int(top_k)))
    resolved_doc_ids = choose_compare_doc_ids_from_retrieval(retrieval_output)
    if len(resolved_doc_ids) < 2:
        raise PolicyCompareResolutionError(
            build_compare_followup_hint(
                query,
                retrieval_output=retrieval_output,
            )
        )

    return (
        resolved_doc_ids[0],
        resolved_doc_ids[1],
        f"根据 compare query 的检索结果推断目标政策为 {resolved_doc_ids[0]} 与 {resolved_doc_ids[1]}。",
    )


def validate_doc_id(doc_id: str, metadata_map: dict[str, PolicyMetadata]) -> None:
    """校验 doc_id 是否存在。"""

    if doc_id not in metadata_map:
        raise PolicyCompareResolutionError(f"doc_id 不存在: {doc_id}")


def match_compare_doc_ids_from_query(
    query: str,
    metadata_map: dict[str, PolicyMetadata],
) -> tuple[str, ...]:
    """尝试从 query 中直接匹配出两篇候选政策。"""

    matched_doc_ids: list[str] = []

    explicit_doc_id = extract_doc_id_from_text(query)
    if explicit_doc_id and explicit_doc_id in metadata_map:
        matched_doc_ids.append(explicit_doc_id)

    # 这里不做复杂 NER，而是利用现有 title matcher 逐步从 query 里找可能的标题。
    # 这样第一版 compare 能在不新增重依赖的情况下稳定工作。
    normalized_query = query.strip()
    remaining_query = normalized_query
    for _ in range(2):
        matched_doc_id = match_doc_id_by_title(remaining_query, metadata_map)
        if matched_doc_id is None or matched_doc_id in matched_doc_ids:
            break

        matched_doc_ids.append(matched_doc_id)
        title = metadata_map[matched_doc_id].title
        remaining_query = remaining_query.replace(title, " ")

    if len(matched_doc_ids) >= 2:
        return tuple(matched_doc_ids[:2])

    # 再做一层全文标题包含匹配，方便“对比北京和上海人工智能政策”这类 query。
    for metadata in metadata_map.values():
        if metadata.doc_id in matched_doc_ids:
            continue
        if metadata.title and metadata.title in normalized_query:
            matched_doc_ids.append(metadata.doc_id)
        if len(matched_doc_ids) >= 2:
            break

    return tuple(matched_doc_ids[:2])


def extract_regions_from_query(
    query: str,
    metadata_map: dict[str, PolicyMetadata],
) -> tuple[str, ...]:
    """从 query 中按出现顺序提取地区线索。"""

    region_positions: list[tuple[int, str]] = []
    seen_regions: set[str] = set()

    for region in {item.region for item in metadata_map.values()}:
        start_index = query.find(region)
        if start_index < 0 or region in seen_regions:
            continue
        region_positions.append((start_index, region))
        seen_regions.add(region)

    region_positions.sort(key=lambda item: item[0])
    return tuple(region for _, region in region_positions)


def resolve_compare_doc_ids_by_regions(
    query: str,
    *,
    mentioned_regions: tuple[str, ...],
    metadata_map: dict[str, PolicyMetadata],
    retrieval_output: RetrievePolicyOutput,
) -> tuple[str, ...]:
    """
    根据 query 中出现的地区线索，为每个地区各选一篇最合适的政策。

    设计原则：
    - 优先复用 retrieval 已经命中的文档，避免纯 metadata 猜测太飘
    - 如果某个地区在 retrieval 里完全没命中，再退回 metadata 启发式选择
    - 第一版只取 query 里前两个地区，保持 compare 输出稳定
    """

    selected_doc_ids: list[str] = []
    for region in mentioned_regions[:2]:
        doc_id = select_best_doc_for_region(
            query,
            region=region,
            metadata_map=metadata_map,
            retrieval_output=retrieval_output,
        )
        if doc_id is None or doc_id in selected_doc_ids:
            continue
        selected_doc_ids.append(doc_id)

    return tuple(selected_doc_ids)


def select_best_doc_for_region(
    query: str,
    *,
    region: str,
    metadata_map: dict[str, PolicyMetadata],
    retrieval_output: RetrievePolicyOutput,
) -> str | None:
    """从指定地区的候选政策里选出最适合当前 compare query 的一篇。"""

    retrieval_doc_scores: dict[str, float] = {}
    for item in retrieval_output.results:
        item_region = str(item.metadata.get("region", ""))
        if item_region != region:
            continue
        retrieval_doc_scores[item.doc_id] = retrieval_doc_scores.get(item.doc_id, 0.0) + float(item.score)

    if retrieval_doc_scores:
        # 先看 retrieval 在该地区已经稳定命中了哪些文档。
        # 这样可以最大化复用检索层给出的主题相关性信号。
        return max(
            retrieval_doc_scores.items(),
            key=lambda pair: (
                pair[1],
                score_metadata_candidate(query, metadata_map[pair[0]]),
            ),
        )[0]

    region_candidates = [item for item in metadata_map.values() if item.region == region]
    if not region_candidates:
        return None

    return max(
        region_candidates,
        key=lambda item: score_metadata_candidate(query, item),
    ).doc_id


def score_metadata_candidate(query: str, metadata: PolicyMetadata) -> tuple[int, str]:
    """
    为地区内候选政策打一个轻量启发式分数。

    第一版不追求“绝对最准”，而是优先保证这类规则清晰、可解释：
    - 与 query 关键词越贴近，分越高
    - core 文档优先
    - 同分时默认让更新、更像总纲/配套的政策排在前面
    """

    score = 0
    normalized_query = query.lower()
    title_text = metadata.title.lower()
    theme_text = metadata.theme.lower()
    notes_text = metadata.notes.lower()

    for keyword in extract_compare_query_keywords(query):
        normalized_keyword = keyword.lower()
        if normalized_keyword in title_text:
            score += 5
        if normalized_keyword in theme_text:
            score += 4
        if normalized_keyword in notes_text:
            score += 2

    if metadata.tier == "core":
        score += 6
    else:
        score -= 2

    if any(marker in metadata.title for marker in ("行动方案", "行动计划", "实施方案", "若干措施")):
        score += 2

    if "人工智能" in metadata.title or "人工智能" in metadata.theme:
        score += 2

    # 对地区级 compare，优先选能代表地区“主干政策画像”的文档，而不是过窄的专题补充政策。
    score += score_policy_breadth(metadata)

    return score, parse_publish_date_key(metadata.publish_date)


def score_policy_breadth(metadata: PolicyMetadata) -> int:
    """Estimate how representative a policy is for region-level comparison."""

    score = 0
    normalized_theme = metadata.theme.strip().lower()
    normalized_title = metadata.title.strip().lower()
    normalized_notes = metadata.notes.strip().lower()

    broad_theme_markers = (
        "人工智能+应用",
        "ai总纲",
        "通用人工智能",
    )
    if any(marker in normalized_theme for marker in broad_theme_markers):
        score += 5

    narrow_theme_markers = (
        "ai+science",
        "ai+制造",
        "制造",
        "医疗",
        "science",
    )
    if any(marker in normalized_theme for marker in narrow_theme_markers):
        score -= 3

    broad_title_markers = (
        "行动计划",
        "若干措施",
        "实施方案",
    )
    if any(marker in normalized_title for marker in broad_title_markers):
        score += 2

    if any(marker in normalized_notes for marker in ("总纲", "核心支持政策", "主干政策")):
        score += 3

    if any(marker in normalized_notes for marker in ("专题", "偏", "补充")):
        score -= 2

    return score


def extract_compare_query_keywords(query: str) -> tuple[str, ...]:
    """从 compare query 中提炼一批轻量关键词，供 metadata 打分使用。"""

    candidates = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9+\-]{2,}", query)
    stopwords = {
        "对比",
        "比较",
        "差异",
        "区别",
        "不同",
        "相关政策",
        "政策",
        "一下",
    }

    keywords: list[str] = []
    for item in candidates:
        normalized_item = item.strip()
        if not normalized_item or normalized_item in stopwords:
            continue
        if normalized_item not in keywords:
            keywords.append(normalized_item)

    return tuple(keywords)


def parse_publish_date_key(publish_date: str) -> tuple[int, int, int]:
    """把发布日期字符串转成可稳定比较的日期键。"""

    parts = publish_date.split("/")
    if len(parts) != 3:
        return (0, 0, 0)

    try:
        year, month, day = (int(part) for part in parts)
    except ValueError:
        return (0, 0, 0)

    return (year, month, day)


def choose_compare_doc_ids_from_retrieval(
    retrieval_output: RetrievePolicyOutput,
) -> tuple[str, ...]:
    """从检索结果中选出两篇最值得对比的政策。"""

    doc_scores: dict[str, float] = {}
    for item in retrieval_output.results:
        doc_scores[item.doc_id] = doc_scores.get(item.doc_id, 0.0) + float(item.score)

    if len(doc_scores) < 2:
        # 如果总共只有一篇命中，就没法形成真正的 compare。
        return tuple(doc_scores.keys())

    ranked_doc_ids = [
        doc_id
        for doc_id, _ in sorted(
            doc_scores.items(),
            key=lambda pair: (-pair[1], pair[0]),
        )
    ]
    return tuple(ranked_doc_ids[:2])


def build_compare_followup_hint(
    query: str,
    retrieval_output: RetrievePolicyOutput | None = None,
) -> str:
    """为 compare 定位失败场景生成更具体的错误提示。"""

    if retrieval_output is not None and retrieval_output.result_count > 0:
        doc_ids = []
        titles = []
        for item in retrieval_output.results:
            if item.doc_id not in doc_ids:
                doc_ids.append(item.doc_id)
                titles.append(item.title)
            if len(doc_ids) >= 2:
                break

        if len(titles) == 1:
            return f"当前只稳定定位到 1 篇政策：{titles[0]}。请再补充另一篇想比较的政策。"

    return f"未能从“{query}”中稳定定位到两篇政策，请补充两个明确的比较对象或地区范围。"


def build_compare_sections(
    left_summary: PolicySummaryOutput,
    right_summary: PolicySummaryOutput,
) -> tuple[PolicyComparisonSection, ...]:
    """把两份摘要按固定分区拼装成结构化对比结果。"""

    sections: list[PolicyComparisonSection] = []

    for section_key, section_label, _ in SECTION_DEFINITIONS:
        left_points = tuple(item.text for item in getattr(left_summary, section_key))
        right_points = tuple(item.text for item in getattr(right_summary, section_key))
        sections.append(
            PolicyComparisonSection(
                key=section_key,
                label=section_label,
                left_points=left_points,
                right_points=right_points,
                comparison_note=build_section_comparison_note(left_points, right_points),
            )
        )

    return tuple(sections)


def build_section_comparison_note(
    left_points: tuple[str, ...],
    right_points: tuple[str, ...],
) -> str:
    """为单个分区生成一句轻量对比说明。"""

    if left_points and right_points:
        return "两篇政策在该维度都有明确表述，可结合左右要点继续比较侧重点。"
    if left_points and not right_points:
        return "左侧政策在该维度有更明确的信息，右侧暂未抽取到稳定要点。"
    if right_points and not left_points:
        return "右侧政策在该维度有更明确的信息，左侧暂未抽取到稳定要点。"
    return "两篇政策在该维度都暂未抽取到稳定要点。"


def render_policy_comparison(output: PolicyCompareOutput) -> str:
    """把结构化对比结果渲染成适合终端展示的多行文本。"""

    left_metadata = output.left_summary.metadata
    right_metadata = output.right_summary.metadata
    lines = [
        "政策对比：",
        f"A. {output.left_summary.title} ({output.left_summary.doc_id})",
        (
            f"   地区：{left_metadata.get('region', '')} | "
            f"发布日期：{left_metadata.get('publish_date', '')} | "
            f"类型：{left_metadata.get('policy_type', '')}"
        ),
        f"B. {output.right_summary.title} ({output.right_summary.doc_id})",
        (
            f"   地区：{right_metadata.get('region', '')} | "
            f"发布日期：{right_metadata.get('publish_date', '')} | "
            f"类型：{right_metadata.get('policy_type', '')}"
        ),
        f"定位依据：{output.selection_reason}",
    ]

    for section in output.sections:
        lines.append("")
        lines.append(f"{section.label}:")
        lines.append("A:")
        if not section.left_points:
            lines.append("1. 暂未从当前文本中提取到稳定要点。")
        else:
            for index, item in enumerate(section.left_points, start=1):
                lines.append(f"{index}. {item}")

        lines.append("B:")
        if not section.right_points:
            lines.append("1. 暂未从当前文本中提取到稳定要点。")
        else:
            for index, item in enumerate(section.right_points, start=1):
                lines.append(f"{index}. {item}")

        lines.append(f"对比提示：{section.comparison_note}")

    return "\n".join(lines)


class ComparePolicyTool:
    """给 Agent 或上层流程调用的政策对比工具。"""

    name = "compare_policy"
    description = "对两篇政策做固定分区的结构化对比。"

    def __init__(
        self,
        *,
        retrieve_tool: RetrievePolicyTool | None = None,
        default_top_k: int = 6,
        max_points_per_section: int = 3,
    ) -> None:
        self.retrieve_tool = retrieve_tool
        self.default_top_k = max(2, int(default_top_k))
        self.max_points_per_section = max(1, int(max_points_per_section))

    def run(
        self,
        query: str,
        *,
        left_doc_id: str | None = None,
        right_doc_id: str | None = None,
        top_k: int | None = None,
    ) -> PolicyCompareOutput:
        """执行一次政策对比。"""

        effective_top_k = self.default_top_k if top_k is None else top_k
        return compare_policy(
            query,
            left_doc_id=left_doc_id,
            right_doc_id=right_doc_id,
            top_k=effective_top_k,
            max_points_per_section=self.max_points_per_section,
            retrieve_tool=self.retrieve_tool,
        )
