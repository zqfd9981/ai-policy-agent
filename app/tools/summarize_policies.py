from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.strategy import aggregate_retrieval_documents
from app.tools.retrieve_policy import RetrievePolicyOutput, RetrievePolicyTool
from app.tools.summarize_policy import (
    SECTION_DEFINITIONS,
    PolicySummaryOutput,
    SummaryEvidence,
    summarize_policy,
)


@dataclass(frozen=True, slots=True)
class MultiPolicySectionSummary:
    """Aggregated summary for one section across multiple policies."""

    key: str
    label: str
    highlights: tuple[str, ...]
    evidence: tuple[SummaryEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "highlights": list(self.highlights),
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class MultiPolicySummaryOutput:
    """Structured topic / region-level summary across multiple policies."""

    query: str
    doc_ids: tuple[str, ...]
    policy_titles: tuple[str, ...]
    selection_reason: str
    policy_summaries: tuple[PolicySummaryOutput, ...]
    sections: tuple[MultiPolicySectionSummary, ...]

    @property
    def citation_count(self) -> int:
        return len(self.all_citations)

    @property
    def all_citations(self) -> tuple[SummaryEvidence, ...]:
        seen: set[tuple[str, str]] = set()
        citations: list[SummaryEvidence] = []

        for summary in self.policy_summaries:
            for item in summary.all_citations:
                key = (item.chunk_id, item.text)
                if key in seen:
                    continue
                seen.add(key)
                citations.append(item)

        return tuple(citations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "doc_ids": list(self.doc_ids),
            "policy_titles": list(self.policy_titles),
            "selection_reason": self.selection_reason,
            "policy_summaries": [item.to_dict() for item in self.policy_summaries],
            "sections": [item.to_dict() for item in self.sections],
            "citation_count": self.citation_count,
            "citations": [item.to_dict() for item in self.all_citations],
        }


def summarize_policies(
    query: str,
    *,
    retrieval_output: RetrievePolicyOutput | None = None,
    top_k: int = 6,
    max_docs: int = 3,
    max_points_per_section: int = 2,
    retrieve_tool: RetrievePolicyTool | None = None,
) -> MultiPolicySummaryOutput:
    """Build a multi-policy summary for region/topic-level requests."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空。")

    active_retrieval_output = retrieval_output
    if active_retrieval_output is None:
        active_tool = retrieve_tool or RetrievePolicyTool()
        active_retrieval_output = active_tool.run(normalized_query, top_k=max(4, top_k))

    candidates = aggregate_retrieval_documents(active_retrieval_output)
    selected_doc_ids = tuple(item.doc_id for item in candidates[: max(1, max_docs)])
    if not selected_doc_ids:
        raise ValueError("当前检索结果不足以生成多文档摘要。")

    summaries = tuple(
        summarize_policy(
            normalized_query,
            doc_id=doc_id,
            top_k=top_k,
            max_points_per_section=max_points_per_section,
            retrieve_tool=retrieve_tool,
        )
        for doc_id in selected_doc_ids
    )

    sections = tuple(build_multi_policy_section(section_key, section_label, summaries) for section_key, section_label, _ in SECTION_DEFINITIONS)
    selection_reason = (
        f"根据检索结果聚合出 {len(selected_doc_ids)} 篇高相关政策，"
        f"分别为：{'；'.join(item.title for item in summaries)}。"
    )
    return MultiPolicySummaryOutput(
        query=normalized_query,
        doc_ids=selected_doc_ids,
        policy_titles=tuple(item.title for item in summaries),
        selection_reason=selection_reason,
        policy_summaries=summaries,
        sections=sections,
    )


def build_multi_policy_section(
    section_key: str,
    section_label: str,
    policy_summaries: tuple[PolicySummaryOutput, ...],
) -> MultiPolicySectionSummary:
    """Combine one summary section across multiple policy summaries."""

    highlights: list[str] = []
    evidence: list[SummaryEvidence] = []
    seen_texts: set[tuple[str, str]] = set()

    for summary in policy_summaries:
        items = getattr(summary, section_key)
        for item in items[:2]:
            highlight = f"{summary.title}：{item.text}"
            if highlight not in highlights:
                highlights.append(highlight)

            key = (item.chunk_id, item.text)
            if key in seen_texts:
                continue
            seen_texts.add(key)
            evidence.append(item)

    return MultiPolicySectionSummary(
        key=section_key,
        label=section_label,
        highlights=tuple(highlights),
        evidence=tuple(evidence),
    )


def render_multi_policy_summary(output: MultiPolicySummaryOutput) -> str:
    """Render a readable region/topic-level summary."""

    lines = [
        "政策汇总：",
        f"问题：{output.query}",
        f"覆盖政策数：{len(output.policy_summaries)}",
        f"纳入范围：{'；'.join(output.policy_titles)}",
        f"选择依据：{output.selection_reason}",
    ]

    for section in output.sections:
        lines.append("")
        lines.append(f"{section.label}:")
        if not section.highlights:
            lines.append("1. 暂未从当前政策集合中抽取到稳定要点。")
            continue
        for index, item in enumerate(section.highlights, start=1):
            lines.append(f"{index}. {item}")

    return "\n".join(lines)


class SummarizePoliciesTool:
    """Tool for region/topic-level policy aggregation summary."""

    name = "summarize_policies"
    description = "对多篇相关政策做聚合汇总，适合地区/主题级摘要请求。"

    def __init__(
        self,
        *,
        retrieve_tool: RetrievePolicyTool | None = None,
        default_top_k: int = 6,
        max_docs: int = 3,
        max_points_per_section: int = 2,
    ) -> None:
        self.retrieve_tool = retrieve_tool
        self.default_top_k = max(2, int(default_top_k))
        self.max_docs = max(1, int(max_docs))
        self.max_points_per_section = max(1, int(max_points_per_section))

    def run(
        self,
        query: str,
        *,
        retrieval_output: RetrievePolicyOutput | None = None,
        top_k: int | None = None,
    ) -> MultiPolicySummaryOutput:
        effective_top_k = self.default_top_k if top_k is None else top_k
        return summarize_policies(
            query,
            retrieval_output=retrieval_output,
            top_k=effective_top_k,
            max_docs=self.max_docs,
            max_points_per_section=self.max_points_per_section,
            retrieve_tool=self.retrieve_tool,
        )
