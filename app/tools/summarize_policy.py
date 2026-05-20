from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from app.chunk.chunk_builder import DEFAULT_CHUNK_OUTPUT_PATH, build_and_export_chunks
from app.ingest.metadata_loader import load_metadata_map
from app.models.metadata import PolicyMetadata
from app.retrieval.retriever import (
    DEFAULT_CHUNK_OUTPUT_PATH as RETRIEVAL_DEFAULT_CHUNK_OUTPUT_PATH,
    load_chunk_payloads_from_jsonl,
)
from app.tools.retrieve_policy import RetrievePolicyTool


DOC_ID_PATTERN = re.compile(r"\b[A-Za-z]{2}\d{3}\b")

SECTION_DEFINITIONS = (
    (
        "overview",
        "政策概览",
        ("目标", "总体", "思路", "概览", "基本信息", "原则"),
    ),
    (
        "support_points",
        "支持重点",
        ("支持", "重点", "任务", "举措", "方向", "行动", "工程", "措施"),
    ),
    (
        "target_audiences",
        "适用对象",
        ("对象", "企业", "单位", "机构", "高校", "医院", "主体", "面向", "适用"),
    ),
    (
        "application_conditions",
        "申报条件",
        ("条件", "要求", "申报", "资格", "标准", "遴选", "程序", "应当", "截止"),
    ),
)

TEMPLATE_NOISE_KEYWORDS = (
    "不超过",
    "包括但不限于",
    "负责人",
    "联系人",
    "传真",
    "e-mail",
    "单位性质",
    "真实性承诺",
    "公章",
    "附件",
    "承诺书",
    "组织机构代码",
    "手机",
    "填写",
)

SECTION_NEGATIVE_KEYWORDS = {
    "overview": ("申报", "补贴", "奖励", "模型券", "算力券", "语料券"),
    "support_points": ("通知如下", "制定本措施", "总体目标", "工作目标"),
    "target_audiences": ("支持力度", "补贴", "奖励", "申报流程"),
    "application_conditions": ("总体目标", "工作目标", "支持力度", "人才奖励"),
}

OCR_NOISE_MARKERS = ("/g", "北京市人民政府公报", "上 海 市", "文件", "通知如下")

SECTION_STRICT_TITLE_PATH = {"overview", "target_audiences", "application_conditions"}

OVERVIEW_TITLE_KEYWORDS = ("工作目标", "总体目标", "总体要求", "发展思路", "政策概览", "主要目标")
TARGET_AUDIENCE_TITLE_KEYWORDS = ("主体", "对象", "适用", "申报主体", "支持对象")
APPLICATION_CONDITION_TITLE_KEYWORDS = ("申报", "条件", "要求", "资格", "流程")

SUPPORT_POINT_NEGATIVE_PHRASES = (
    "为贯彻落实",
    "现印发给你们",
    "制定本措施",
    "通知如下",
    "请认真贯彻执行",
)
TARGET_AUDIENCE_TEXT_KEYWORDS = ("企业", "机构", "单位", "主体", "高校", "医院", "园区", "适用")
APPLICATION_CONDITION_TEXT_KEYWORDS = ("申报", "符合条件", "资格", "流程", "要求", "标准")
OVERVIEW_TEXT_KEYWORDS = ("实施", "推动", "加快", "围绕", "目标", "打造", "建设")


@dataclass(frozen=True, slots=True)
class SummaryEvidence:
    """表示摘要中的一条结构化证据。"""

    section: str
    text: str
    chunk_id: str
    doc_id: str
    title_path_str: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """把证据对象转换成适合序列化的字典。"""

        return {
            "section": self.section,
            "text": self.text,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title_path_str": self.title_path_str,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PolicySummaryOutput:
    """表示单篇政策的结构化摘要输出。"""

    query: str
    doc_id: str
    title: str
    metadata: dict[str, Any]
    selection_reason: str
    overview: tuple[SummaryEvidence, ...]
    support_points: tuple[SummaryEvidence, ...]
    target_audiences: tuple[SummaryEvidence, ...]
    application_conditions: tuple[SummaryEvidence, ...]

    @property
    def citation_count(self) -> int:
        """返回摘要中总引用条数。"""

        return len(self.all_citations)

    @property
    def all_citations(self) -> tuple[SummaryEvidence, ...]:
        """按去重后顺序返回全部引用证据。"""

        seen: set[tuple[str, str]] = set()
        citations: list[SummaryEvidence] = []

        for item in (
            *self.overview,
            *self.support_points,
            *self.target_audiences,
            *self.application_conditions,
        ):
            key = (item.chunk_id, item.text)
            if key in seen:
                continue
            seen.add(key)
            citations.append(item)

        return tuple(citations)

    def to_dict(self) -> dict[str, Any]:
        """把摘要结果转换成适合 JSON 序列化的字典。"""

        return {
            "query": self.query,
            "doc_id": self.doc_id,
            "title": self.title,
            "metadata": dict(self.metadata),
            "selection_reason": self.selection_reason,
            "overview": [item.to_dict() for item in self.overview],
            "support_points": [item.to_dict() for item in self.support_points],
            "target_audiences": [item.to_dict() for item in self.target_audiences],
            "application_conditions": [item.to_dict() for item in self.application_conditions],
            "citation_count": self.citation_count,
            "citations": [item.to_dict() for item in self.all_citations],
        }


class PolicySummaryResolutionError(ValueError):
    """无法确定要摘要的政策文档时抛出的异常。"""


@lru_cache(maxsize=1)
def get_chunk_payload_map(
    chunk_jsonl_path: str = str(RETRIEVAL_DEFAULT_CHUNK_OUTPUT_PATH),
) -> dict[str, tuple[dict[str, Any], ...]]:
    """读取并缓存 doc_id -> chunk payload 列表的映射。"""

    normalized_path = str(chunk_jsonl_path)
    payload_path = _ensure_chunk_jsonl_exists(normalized_path)
    payloads = load_chunk_payloads_from_jsonl(payload_path)

    payload_map: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        payload_map.setdefault(str(payload.get("doc_id", "")), []).append(payload)

    return {doc_id: tuple(items) for doc_id, items in payload_map.items()}


@lru_cache(maxsize=1)
def get_cached_metadata_map() -> dict[str, PolicyMetadata]:
    """读取并缓存 metadata 映射。"""

    return load_metadata_map()


def summarize_policy(
    query: str,
    *,
    doc_id: str | None = None,
    policy_title: str | None = None,
    top_k: int = 5,
    max_points_per_section: int = 3,
    retrieve_tool: RetrievePolicyTool | None = None,
) -> PolicySummaryOutput:
    """
    对单篇政策输出结构化摘要。

    第一版不依赖 LLM，采用：
    - metadata 解析
    - chunk 载入
    - 关键词 + 标题路径的轻量规则提取
    """

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空。")

    metadata_map = get_cached_metadata_map()
    chunk_payload_map = get_chunk_payload_map()
    resolved_doc_id, selection_reason = resolve_summary_doc_id(
        normalized_query,
        metadata_map=metadata_map,
        doc_id=doc_id,
        policy_title=policy_title,
        top_k=top_k,
        retrieve_tool=retrieve_tool,
    )

    if resolved_doc_id not in metadata_map:
        raise PolicySummaryResolutionError(f"未找到 doc_id={resolved_doc_id} 对应的 metadata。")

    payloads = chunk_payload_map.get(resolved_doc_id, ())
    if not payloads:
        raise PolicySummaryResolutionError(f"未找到 doc_id={resolved_doc_id} 对应的 chunk 数据。")

    metadata = metadata_map[resolved_doc_id]
    sections = build_summary_sections(
        payloads,
        max_points_per_section=max_points_per_section,
    )

    return PolicySummaryOutput(
        query=normalized_query,
        doc_id=resolved_doc_id,
        title=metadata.title,
        metadata=_serialize_metadata(metadata),
        selection_reason=selection_reason,
        overview=sections["overview"],
        support_points=sections["support_points"],
        target_audiences=sections["target_audiences"],
        application_conditions=sections["application_conditions"],
    )


def resolve_summary_doc_id(
    query: str,
    *,
    metadata_map: dict[str, PolicyMetadata],
    doc_id: str | None = None,
    policy_title: str | None = None,
    top_k: int = 5,
    retrieve_tool: RetrievePolicyTool | None = None,
) -> tuple[str, str]:
    """确定当前摘要请求对应的目标政策文档。"""

    if doc_id:
        normalized_doc_id = doc_id.strip().upper()
        if normalized_doc_id not in metadata_map:
            raise PolicySummaryResolutionError(f"doc_id 不存在: {normalized_doc_id}")
        return normalized_doc_id, f"根据显式 doc_id={normalized_doc_id} 定位政策。"

    extracted_doc_id = extract_doc_id_from_text(query)
    if extracted_doc_id and extracted_doc_id in metadata_map:
        return extracted_doc_id, f"根据 query 中的 doc_id={extracted_doc_id} 定位政策。"

    if policy_title:
        matched_doc_id = match_doc_id_by_title(policy_title, metadata_map)
        if matched_doc_id is None:
            raise PolicySummaryResolutionError(f"未找到与标题最匹配的政策: {policy_title}")
        return matched_doc_id, f"根据显式政策标题匹配到 {matched_doc_id}。"

    matched_from_query = match_doc_id_by_title(query, metadata_map)
    if matched_from_query is not None:
        return matched_from_query, f"根据 query 中的政策标题线索匹配到 {matched_from_query}。"

    active_tool = retrieve_tool or RetrievePolicyTool()
    retrieval_output = active_tool.run(query, top_k=top_k)
    if retrieval_output.result_count == 0:
        raise PolicySummaryResolutionError("未能通过检索定位到可摘要的政策文档。")

    best_doc_id = choose_doc_id_from_retrieval(retrieval_output.to_dict()["results"])
    if best_doc_id not in metadata_map:
        raise PolicySummaryResolutionError("检索结果未能稳定定位到有效政策文档。")

    return best_doc_id, f"根据摘要 query 的检索结果推断目标政策为 {best_doc_id}。"


def build_summary_sections(
    payloads: tuple[dict[str, Any], ...],
    *,
    max_points_per_section: int,
) -> dict[str, tuple[SummaryEvidence, ...]]:
    """基于 chunk payload 列表构建结构化摘要分区。"""

    sections: dict[str, tuple[SummaryEvidence, ...]] = {}

    for section_key, _, keywords in SECTION_DEFINITIONS:
        sections[section_key] = extract_section_evidence(
            payloads,
            section=section_key,
            keywords=keywords,
            limit=max_points_per_section,
        )

    if not sections["overview"]:
        sections["overview"] = extract_fallback_overview(payloads, limit=max_points_per_section)

    if not sections["support_points"]:
        sections["support_points"] = extract_fallback_overview(payloads, limit=max_points_per_section)

    return sections


def extract_section_evidence(
    payloads: tuple[dict[str, Any], ...],
    *,
    section: str,
    keywords: tuple[str, ...],
    limit: int,
) -> tuple[SummaryEvidence, ...]:
    """从 payload 列表中提取某个摘要分区的证据。"""

    candidates: list[tuple[int, SummaryEvidence]] = []
    seen_texts: set[str] = set()

    for payload in payloads:
        if is_noise_payload(payload):
            continue

        title_path_str = str(payload.get("title_path_str", ""))
        if section in SECTION_STRICT_TITLE_PATH and not title_path_str.strip():
            continue
        text = str(payload.get("text", ""))
        units = extract_text_units(text)

        for unit in units:
            normalized_unit = unit.strip()
            if len(normalized_unit) < 8:
                continue
            if is_noise_unit(normalized_unit):
                continue
            if not passes_section_gate(
                normalized_unit,
                title_path_str=title_path_str,
                section=section,
            ):
                continue

            score = score_summary_unit(
                normalized_unit,
                title_path_str=title_path_str,
                keywords=keywords,
                section=section,
            )
            if score <= 0:
                continue

            dedup_key = normalize_for_match(normalized_unit)
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)

            candidates.append(
                (
                    score,
                    SummaryEvidence(
                        section=section,
                        text=normalized_unit,
                        chunk_id=str(payload.get("chunk_id", "")),
                        doc_id=str(payload.get("doc_id", "")),
                        title_path_str=title_path_str,
                        metadata=dict(payload.get("metadata", {})),
                    ),
                )
            )

    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1].chunk_id,
            item[1].text,
        )
    )
    return tuple(item for _, item in candidates[: max(1, limit)])


def extract_fallback_overview(
    payloads: tuple[dict[str, Any], ...],
    *,
    limit: int,
) -> tuple[SummaryEvidence, ...]:
    """在没有命中规则时，从前几段 chunk 中抽取一个兜底概览。"""

    evidences: list[SummaryEvidence] = []

    for payload in payloads:
        if is_noise_payload(payload):
            continue
        title_path_str = str(payload.get("title_path_str", ""))
        if title_path_str.strip():
            continue

        units = extract_text_units(str(payload.get("text", "")))
        if not units:
            continue
        first_unit = next(
            (
                unit
                for unit in units
                if not is_noise_unit(unit)
                and any(keyword in unit for keyword in OVERVIEW_TEXT_KEYWORDS)
                and "通知" not in unit
                and "印发" not in unit
                and len(unit) <= 120
            ),
            None,
        )
        if first_unit is None:
            continue

        evidences.append(
            SummaryEvidence(
                section="overview",
                text=first_unit,
                chunk_id=str(payload.get("chunk_id", "")),
                doc_id=str(payload.get("doc_id", "")),
                title_path_str=str(payload.get("title_path_str", "")),
                metadata=dict(payload.get("metadata", {})),
            )
        )
        if len(evidences) >= max(1, limit):
            break

    return tuple(evidences[: max(1, limit)])


def render_policy_summary(summary: PolicySummaryOutput) -> str:
    """把结构化摘要渲染成适合终端展示的多行文本。"""

    metadata = summary.metadata
    lines = [
        f"政策摘要：{summary.title} ({summary.doc_id})",
        f"地区：{metadata.get('region', '')} | 发布日期：{metadata.get('publish_date', '')} | 类型：{metadata.get('policy_type', '')}",
        f"定位依据：{summary.selection_reason}",
    ]

    for section_key, section_label, _ in SECTION_DEFINITIONS:
        items = getattr(summary, section_key)
        lines.append("")
        lines.append(f"{section_label}:")
        if not items:
            lines.append("1. 暂未从当前文本中提取到稳定要点。")
            continue

        for index, item in enumerate(items, start=1):
            lines.append(f"{index}. {item.text}")

    return "\n".join(lines)


class SummarizePolicyTool:
    """给 Agent 或上层流程调用的政策摘要工具。"""

    name = "summarize_policy"
    description = "对单篇政策输出结构化摘要。"

    def __init__(
        self,
        *,
        retrieve_tool: RetrievePolicyTool | None = None,
        default_top_k: int = 5,
        max_points_per_section: int = 3,
    ) -> None:
        self.retrieve_tool = retrieve_tool
        self.default_top_k = max(1, int(default_top_k))
        self.max_points_per_section = max(1, int(max_points_per_section))

    def run(
        self,
        query: str,
        *,
        doc_id: str | None = None,
        policy_title: str | None = None,
        top_k: int | None = None,
    ) -> PolicySummaryOutput:
        """执行一次政策摘要。"""

        effective_top_k = self.default_top_k if top_k is None else top_k
        return summarize_policy(
            query,
            doc_id=doc_id,
            policy_title=policy_title,
            top_k=effective_top_k,
            max_points_per_section=self.max_points_per_section,
            retrieve_tool=self.retrieve_tool,
        )


def extract_doc_id_from_text(text: str) -> str | None:
    """从文本中尝试提取 doc_id。"""

    match = DOC_ID_PATTERN.search(text)
    if match is None:
        return None
    return match.group(0).upper()


def match_doc_id_by_title(
    text: str,
    metadata_map: dict[str, PolicyMetadata],
) -> str | None:
    """根据标题相似度或包含关系匹配最可能的 doc_id。"""

    normalized_text = normalize_for_match(text)
    if not normalized_text:
        return None

    best_doc_id: str | None = None
    best_score = 0.0

    for metadata in metadata_map.values():
        normalized_title = normalize_for_match(metadata.title)
        if not normalized_title:
            continue

        if normalized_title in normalized_text:
            return metadata.doc_id

        score = SequenceMatcher(None, normalized_text, normalized_title).ratio()
        if score > best_score:
            best_score = score
            best_doc_id = metadata.doc_id

    if best_score >= 0.45:
        return best_doc_id
    return None


def choose_doc_id_from_retrieval(results: list[dict[str, Any]]) -> str:
    """从检索结果中选出最可能要摘要的目标 doc_id。"""

    scores: dict[str, float] = {}
    for item in results:
        doc_id = str(item.get("doc_id", ""))
        score = float(item.get("score", 0.0))
        scores[doc_id] = scores.get(doc_id, 0.0) + score

    if not scores:
        return ""

    return max(scores.items(), key=lambda pair: pair[1])[0]


def extract_text_units(text: str) -> list[str]:
    """把 chunk 文本切成较适合摘要抽取的短单元。"""

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return []

    normalized_text = trim_policy_preamble(normalized_text)

    merged_lines = merge_wrapped_lines(normalized_text)
    units: list[str] = []
    for line in merged_lines:
        stripped_line = line.strip(" -\u2022\t")
        if not stripped_line:
            continue

        parts = re.split(r"(?<=[。；])", stripped_line)
        units.extend(part.strip() for part in parts if part.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for unit in units:
        normalized_unit = normalize_for_match(unit)
        if not normalized_unit or normalized_unit in seen:
            continue
        seen.add(normalized_unit)
        deduped.append(unit)

    return deduped


def merge_wrapped_lines(text: str) -> list[str]:
    """把 PDF 抽取后被强制断开的行尽量合并回完整句子。"""

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    merged_lines: list[str] = []
    current = ""

    for line in lines:
        if not current:
            current = line
            continue

        if line.startswith(("□", "（", "(", "1.", "2.", "3.", "4.", "5.")) and current:
            merged_lines.append(current)
            current = line
            continue

        if current.endswith(("。", "！", "？", "；", "：", ":", "。")):
            merged_lines.append(current)
            current = line
            continue

        current += line

    if current:
        merged_lines.append(current)

    return merged_lines


def trim_policy_preamble(text: str) -> str:
    """Trim file headers and keep the readable policy introduction."""

    intro_markers = ("为贯彻落实", "为深入贯彻", "围绕", "为加快")
    for marker in intro_markers:
        marker_index = text.find(marker)
        if marker_index >= 0:
            return text[marker_index:]
    return text


def score_summary_unit(
    text: str,
    *,
    title_path_str: str,
    keywords: tuple[str, ...],
    section: str,
) -> int:
    """根据标题路径和正文命中情况给摘要单元打分。"""

    score = 0
    normalized_title_path = title_path_str.lower()

    if is_bad_fit_for_section(text, title_path_str=title_path_str, section=section):
        return -1

    if section in SECTION_STRICT_TITLE_PATH and not title_path_str.strip():
        return -1

    for keyword in keywords:
        if keyword in title_path_str:
            score += 4
        if keyword in text:
            score += 1

    if normalized_title_path:
        score += 1

    if section == "overview" and any(
        keyword in title_path_str for keyword in ("工作目标", "总体目标", "总体要求", "发展思路")
    ):
        score += 6
    elif section == "overview":
        score -= 2

    if section == "support_points" and any(
        keyword in title_path_str for keyword in ("配套支持", "征集范围", "重点任务", "支持")
    ):
        score += 4

    if section == "target_audiences" and any(
        keyword in title_path_str for keyword in ("主体", "对象", "申报主体")
    ):
        score += 5
    elif section == "target_audiences":
        score -= 2

    if section == "application_conditions" and any(
        keyword in title_path_str for keyword in ("申报", "要求", "条件", "流程")
    ):
        score += 5
    elif section == "application_conditions":
        score -= 2

    if any(marker in text for marker in ("□", "支持", "推进", "鼓励", "加快", "建设")):
        score += 1

    return score


def normalize_for_match(text: str) -> str:
    """对字符串做轻量归一化，方便标题匹配和去重。"""

    return re.sub(r"[\s\W_]+", "", text).lower()


def is_noise_payload(payload: dict[str, Any]) -> bool:
    """判断一个 chunk payload 是否更像附件/模板噪声。"""

    title_path_str = str(payload.get("title_path_str", "")).lower()
    text = str(payload.get("text", "")).lower()

    noise_hits = sum(keyword in title_path_str or keyword in text for keyword in TEMPLATE_NOISE_KEYWORDS)
    if noise_hits >= 2:
        return True

    if "基本信息" in title_path_str and noise_hits >= 1:
        return True

    if looks_like_ocr_noise(text):
        return True

    return False


def is_noise_unit(text: str) -> bool:
    """判断一个摘要单元是否更像模板字段或附件说明。"""

    normalized_text = text.lower()
    if sum(keyword in normalized_text for keyword in TEMPLATE_NOISE_KEYWORDS) >= 1:
        return True

    if normalized_text.startswith("(") and "不超过" in normalized_text:
        return True

    if len(normalized_text) <= 12 and any(keyword in normalized_text for keyword in ("签字", "公章", "电话")):
        return True

    if looks_like_ocr_noise(normalized_text):
        return True

    if len(normalized_text) > 180 and "通知" in normalized_text and "印发" in normalized_text:
        return True

    return False


def is_bad_fit_for_section(text: str, *, title_path_str: str, section: str) -> bool:
    """Filter obvious section mismatches before scoring."""

    section_negative_keywords = SECTION_NEGATIVE_KEYWORDS.get(section, ())
    combined_text = f"{title_path_str} {text}".lower()
    return any(keyword.lower() in combined_text for keyword in section_negative_keywords)


def passes_section_gate(text: str, *, title_path_str: str, section: str) -> bool:
    """Apply section-specific inclusion gates before scoring."""

    normalized_title_path = title_path_str.strip()

    if section == "overview":
        if any(keyword in normalized_title_path for keyword in OVERVIEW_TITLE_KEYWORDS):
            return True
        return any(keyword in text for keyword in ("为贯彻落实", "为深入贯彻", "推动", "加快", "打造", "建设"))

    if section == "support_points":
        if any(marker in text for marker in SUPPORT_POINT_NEGATIVE_PHRASES):
            return False
        return bool(normalized_title_path) or any(keyword in text for keyword in ("支持", "推进", "鼓励", "建设", "补贴", "奖励"))

    if section == "target_audiences":
        if any(keyword in normalized_title_path for keyword in TARGET_AUDIENCE_TITLE_KEYWORDS):
            return any(keyword in text for keyword in TARGET_AUDIENCE_TEXT_KEYWORDS)
        return any(keyword in text for keyword in ("申报主体为", "面向", "适用于"))

    if section == "application_conditions":
        if any(keyword in normalized_title_path for keyword in APPLICATION_CONDITION_TITLE_KEYWORDS):
            return any(keyword in text for keyword in APPLICATION_CONDITION_TEXT_KEYWORDS)
        return any(keyword in text for keyword in ("符合条件", "申报主体应", "申报流程", "遴选", "评审"))

    return True


def looks_like_ocr_noise(text: str) -> bool:
    """Detect heavily garbled OCR or header boilerplate text."""

    normalized_text = text.lower()
    if any(marker in normalized_text for marker in OCR_NOISE_MARKERS):
        return True

    alpha_count = sum(char.isalpha() for char in normalized_text)
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in normalized_text)
    if alpha_count >= 20 and cjk_count <= 5:
        return True

    slash_g_count = normalized_text.count("/g")
    if slash_g_count >= 3:
        return True

    return False


def _ensure_chunk_jsonl_exists(chunk_jsonl_path: str) -> str:
    """确保摘要所需的 chunk jsonl 已存在。"""

    path = RETRIEVAL_DEFAULT_CHUNK_OUTPUT_PATH.__class__(chunk_jsonl_path)
    if path.exists():
        return str(path)

    build_and_export_chunks(output_path=path)
    return str(path)


def _serialize_metadata(metadata: PolicyMetadata) -> dict[str, Any]:
    """把 metadata 对象转换成普通字典。"""

    return {
        "doc_id": metadata.doc_id,
        "title": metadata.title,
        "region": metadata.region,
        "level": metadata.level,
        "issuer": metadata.issuer,
        "publish_date": metadata.publish_date,
        "policy_type": metadata.policy_type,
        "theme": metadata.theme,
        "tier": metadata.tier,
        "status": metadata.status,
        "source_format": metadata.source_format,
        "doc_no": metadata.doc_no,
        "source_url": metadata.source_url,
        "notes": metadata.notes,
    }
