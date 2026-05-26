from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm.client import OpenAILLMClient
from app.memory.session import SessionMemory, WorkingMemory


MEMORY_COMPLETION_SYSTEM_PROMPT = """
你是 Policy Agent 的多轮上下文补全器。
你的职责不是回答问题，而是结合当前用户短追问和会话上下文，把它补全成一个完整、明确、适合后续 planner / rewrite 使用的问题。

输出要求：
1. 只输出结构化 JSON。
2. 如果当前 query 已经完整，就原样返回。
3. 不要编造不存在的政策名称或地区。
4. 优先结合 working memory 里的地区、主题、当前策略、候选政策标题、compare 对象。
""".strip()


class ContextCompletionModel(BaseModel):
    contextualized_query: str = Field(description="补全后的完整 query")
    reason: str = Field(description="一句中文说明为什么这样补全")
    resolved_action: str | None = Field(
        default=None,
        description="可选：retrieve / summarize / compare",
    )
    resolved_entities: list[str] = Field(
        default_factory=list,
        description="可选：本轮识别到的关键对象标签",
    )


@dataclass(frozen=True, slots=True)
class ContextCompletionDecision:
    contextualized_query: str
    reason: str
    source: str
    resolved_action: str | None = None
    resolved_entities: tuple[str, ...] = ()


class MemoryContextCompleter:
    def __init__(self, *, client: OpenAILLMClient | None = None) -> None:
        self.client = client or OpenAILLMClient()

    @property
    def is_available(self) -> bool:
        return self.client.is_available

    def complete(
        self,
        *,
        user_query: str,
        session_memory: SessionMemory,
    ) -> ContextCompletionDecision:
        recent_turns = session_memory.turns[-4:]
        turns_text = "\n".join(
            f"{item.role}: {item.content}"
            for item in recent_turns
        )
        user_prompt = (
            f"当前用户问题：{user_query}\n\n"
            f"working_memory:\n{session_memory.working_memory.to_dict()}\n\n"
            f"最近对话:\n{turns_text}"
        )

        parsed = self.client.parse_structured_response(
            system_prompt=MEMORY_COMPLETION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=ContextCompletionModel,
            model=os.getenv("CONTEXT_MODEL") or os.getenv("REWRITE_MODEL"),
        )
        return ContextCompletionDecision(
            contextualized_query=parsed.contextualized_query.strip(),
            reason=parsed.reason.strip(),
            source="llm",
            resolved_action=(parsed.resolved_action.strip().lower() if parsed.resolved_action else None),
            resolved_entities=tuple(item.strip() for item in parsed.resolved_entities if item.strip()),
        )


def resolve_context_query(
    user_query: str,
    *,
    session_memory: SessionMemory | None,
    completer: MemoryContextCompleter | None = None,
) -> ContextCompletionDecision:
    """
    Resolve the current user query against multi-turn memory and return a structured decision.

    Strategy:
    1. Prefer LLM-based context resolution when an LLM is available.
    2. Fall back to deterministic rules only when LLM is unavailable or fails.
    3. Otherwise return the original query unchanged.
    """

    normalized_query = user_query.strip()
    if not normalized_query or session_memory is None:
        return ContextCompletionDecision(
            contextualized_query=normalized_query,
            reason="无可用 session memory，直接使用原始 query。",
            source="none",
        )

    active_completer = completer or MemoryContextCompleter()
    if active_completer.is_available:
        try:
            llm_decision = active_completer.complete(
                user_query=normalized_query,
                session_memory=session_memory,
            )
            if llm_decision.contextualized_query:
                return llm_decision
        except Exception:
            pass

    working_memory = session_memory.working_memory
    decision = (
        complete_region_switch_query(normalized_query, working_memory)
        or complete_group_compare_query(normalized_query, working_memory)
        or complete_focus_dimension_query(normalized_query, working_memory)
        or complete_candidate_reference_query(normalized_query, working_memory)
    )
    if decision:
        return decision

    return ContextCompletionDecision(
        contextualized_query=normalized_query,
        reason="未使用有效上下文补全，保留原始 query。",
        source="none",
    )


def contextualize_query(
    user_query: str,
    *,
    session_memory: SessionMemory | None,
) -> str:
    """Backward-compatible wrapper that only returns the final contextualized query string."""

    return resolve_context_query(user_query, session_memory=session_memory).contextualized_query


def complete_region_switch_query(
    query: str,
    working_memory: WorkingMemory,
) -> ContextCompletionDecision | None:
    """Handle follow-ups like '那北京呢' or '那么北京呢' based on previous region/topic/strategy."""

    regions = ("北京", "上海", "江苏", "浙江", "深圳", "广东")
    mentioned_regions = [region for region in regions if region in query]
    if not mentioned_regions:
        return None

    # Region-switch completion should only handle short, single-target follow-ups.
    # If multiple regions are mentioned, the query is more likely a compare request.
    if len(mentioned_regions) != 1:
        return None

    is_region_switch_style = (
        query.endswith("呢")
        or query.startswith(("那", "那么", "换成", "改成"))
        or "那边" in query
    )
    if not is_region_switch_style:
        return None

    target_region = mentioned_regions[0]

    if working_memory.active_strategy == "multi_doc_summary" and working_memory.active_topic:
        return ContextCompletionDecision(
            contextualized_query=f"总结一下{target_region}的{working_memory.active_topic}",
            reason="识别为地区切换式追问，延续上一轮多文档摘要主题。",
            source="rule",
            resolved_action="summarize",
            resolved_entities=(target_region,),
        )

    if working_memory.active_topic:
        return ContextCompletionDecision(
            contextualized_query=f"{target_region}{working_memory.active_topic}",
            reason="识别为地区切换式追问，延续上一轮主题。",
            source="rule",
            resolved_entities=(target_region,),
        )
    return None


def complete_group_compare_query(
    query: str,
    working_memory: WorkingMemory,
) -> ContextCompletionDecision | None:
    """
    Handle compare references like “这两个地方有什么差异”“哪个地方更好”.

    This uses recently remembered region entities instead of relying on a fresh retrieval
    round to guess comparison anchors.
    """

    comparison_keywords = (
        "差异",
        "区别",
        "对比",
        "比较",
        "哪个更好",
        "更好",
        "优劣",
        "哪个好",
        "哪边更好",
        "适合",
        "优势",
        "场景下",
        "场景",
    )
    if not any(keyword in query for keyword in comparison_keywords):
        return None

    explicit_regions = [
        region
        for region in ("上海", "北京", "江苏", "浙江", "深圳", "广东", "长三角")
        if region in query
    ]

    region_entities = [
        entity.label
        for entity in working_memory.recent_entities
        if entity.kind == "region"
    ]
    unique_regions: list[str] = []
    for item in region_entities:
        if item not in unique_regions:
            unique_regions.append(item)

    if explicit_regions:
        ordered_regions: list[str] = []
        for item in explicit_regions:
            if item not in ordered_regions:
                ordered_regions.append(item)
        candidate_regions = ordered_regions
    else:
        candidate_regions = unique_regions

    if len(candidate_regions) < 2:
        return None

    left_region, right_region = candidate_regions[-2], candidate_regions[-1]
    topic = working_memory.active_topic or "AI政策"
    return ContextCompletionDecision(
        contextualized_query=f"比较{left_region}和{right_region}的{topic}，有什么差异，哪个地方更好",
        reason="识别到“这两个地方/哪个更好”等比较指代，使用最近两个地区实体补全 compare query。",
        source="rule",
        resolved_action="compare",
        resolved_entities=(left_region, right_region),
    )


def complete_focus_dimension_query(
    query: str,
    working_memory: WorkingMemory,
) -> ContextCompletionDecision | None:
    """
    Handle follow-ups like '继续讲支持重点' or '支持重点呢'.

    The goal is to preserve the previous policy or compare context while making
    the new user intent explicit for planner / rewrite.
    """

    focus_dimension = None
    if "支持重点" in query or "支持" in query:
        focus_dimension = "支持重点"
    elif "适用对象" in query:
        focus_dimension = "适用对象"
    elif "申报条件" in query or "申报" in query or "条件" in query:
        focus_dimension = "申报条件"

    if focus_dimension is None:
        return None

    if working_memory.active_strategy == "compare" and working_memory.left_doc_title and working_memory.right_doc_title:
        return ContextCompletionDecision(
            contextualized_query=(
                f"比较{working_memory.left_doc_title}和{working_memory.right_doc_title}，重点看{focus_dimension}"
            ),
            reason="识别到维度延续追问，沿用上一轮 compare 对象。",
            source="rule",
            resolved_action="compare",
            resolved_entities=(working_memory.left_doc_title, working_memory.right_doc_title),
        )

    if working_memory.active_doc_title:
        return ContextCompletionDecision(
            contextualized_query=f"总结一下{working_memory.active_doc_title}，重点看{focus_dimension}",
            reason="识别到维度延续追问，沿用当前单篇政策对象。",
            source="rule",
            resolved_action="summarize",
            resolved_entities=(working_memory.active_doc_title,),
        )

    if working_memory.active_region and working_memory.active_topic:
        return ContextCompletionDecision(
            contextualized_query=(
                f"总结一下{working_memory.active_region}的{working_memory.active_topic}，重点看{focus_dimension}"
            ),
            reason="识别到维度延续追问，沿用当前地区与主题摘要上下文。",
            source="rule",
            resolved_action="summarize",
            resolved_entities=(working_memory.active_region,),
        )

    return None


def complete_candidate_reference_query(
    query: str,
    working_memory: WorkingMemory,
) -> ContextCompletionDecision | None:
    """
    Handle follow-ups like '第二篇呢' based on candidate titles.

    This is useful after multi-document summary or retrieve-list style answers,
    where the user refers to items by order instead of by full title.
    """

    if "第二篇" not in query and "第一篇" not in query:
        return None

    if not working_memory.candidate_titles:
        return None

    target_index = 0 if "第一篇" in query else 1
    if target_index >= len(working_memory.candidate_titles):
        return None

    target_title = working_memory.candidate_titles[target_index]
    return ContextCompletionDecision(
        contextualized_query=f"总结一下{target_title}",
        reason="识别到按序号引用候选政策，自动下钻到对应单篇政策。",
        source="rule",
        resolved_action="summarize",
        resolved_entities=(target_title,),
    )
