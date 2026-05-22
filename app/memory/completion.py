from __future__ import annotations

from app.memory.session import SessionMemory, WorkingMemory


def contextualize_query(
    user_query: str,
    *,
    session_memory: SessionMemory | None,
) -> str:
    """
    Use working memory to expand short follow-up queries into complete task queries.

    Important: this is context completion, not retrieval-oriented rewrite.
    It only fills in omitted context such as region, focus dimension, or the
    policy title referenced in the previous turn.
    """

    normalized_query = user_query.strip()
    if not normalized_query or session_memory is None:
        return normalized_query

    working_memory = session_memory.working_memory

    expanded = (
        complete_region_switch_query(normalized_query, working_memory)
        or complete_focus_dimension_query(normalized_query, working_memory)
        or complete_candidate_reference_query(normalized_query, working_memory)
    )
    return expanded or normalized_query


def complete_region_switch_query(query: str, working_memory: WorkingMemory) -> str | None:
    """Handle follow-ups like '那北京呢' based on previous region/topic/strategy."""

    if query not in {"那北京呢", "那上海呢", "北京呢", "上海呢"}:
        return None

    target_region = "北京" if "北京" in query else "上海" if "上海" in query else None
    if not target_region:
        return None

    if working_memory.active_strategy == "multi_doc_summary" and working_memory.active_topic:
        return f"总结一下{target_region}的{working_memory.active_topic}"

    if working_memory.active_strategy == "compare":
        if working_memory.left_doc_title and working_memory.right_doc_title:
            return f"比较{target_region}和{working_memory.active_topic or 'AI政策'}"

    if working_memory.active_topic:
        return f"{target_region}{working_memory.active_topic}"
    return None


def complete_focus_dimension_query(query: str, working_memory: WorkingMemory) -> str | None:
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
        return f"比较{working_memory.left_doc_title}和{working_memory.right_doc_title}，重点看{focus_dimension}"

    if working_memory.active_doc_title:
        return f"总结一下{working_memory.active_doc_title}，重点看{focus_dimension}"

    if working_memory.active_region and working_memory.active_topic:
        return f"总结一下{working_memory.active_region}的{working_memory.active_topic}，重点看{focus_dimension}"

    return None


def complete_candidate_reference_query(query: str, working_memory: WorkingMemory) -> str | None:
    """
    Handle follow-ups like '第二篇呢' based on candidate titles.

    This is useful after multi-document summary or retrieve-list style answers,
    where the user refers to items by order instead of by full title.
    """

    if "第二篇" not in query and "第一篇" not in query:
        return None

    if not working_memory.candidate_titles:
        return None

    if "第一篇" in query:
        target_index = 0
    else:
        target_index = 1

    if target_index >= len(working_memory.candidate_titles):
        return None

    return f"总结一下{working_memory.candidate_titles[target_index]}"
