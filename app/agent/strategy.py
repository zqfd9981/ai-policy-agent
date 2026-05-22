from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent.router import ROUTE_COMPARE, ROUTE_RETRIEVE, ROUTE_SUMMARIZE
from app.tools.retrieve_policy import RetrievePolicyOutput


STRATEGY_DIRECT_ANSWER = "direct_answer"
STRATEGY_SINGLE_DOC_SUMMARY = "single_doc_summary"
STRATEGY_MULTI_DOC_SUMMARY = "multi_doc_summary"
STRATEGY_COMPARE = "compare"


@dataclass(frozen=True, slots=True)
class RetrievalDocumentCandidate:
    """Document-level aggregation of retrieval hits."""

    doc_id: str
    title: str
    total_score: float
    hit_count: int
    title_path_str: str
    snippet: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """Post-retrieval execution strategy."""

    strategy: str
    route: str
    reason: str


def aggregate_retrieval_documents(
    output: RetrievePolicyOutput,
) -> tuple[RetrievalDocumentCandidate, ...]:
    """Aggregate chunk retrieval results to document candidates."""

    grouped: dict[str, dict[str, object]] = {}
    ordered_doc_ids: list[str] = []

    for item in output.results:
        if item.doc_id not in grouped:
            grouped[item.doc_id] = {
                "doc_id": item.doc_id,
                "title": item.title,
                "total_score": float(item.score),
                "hit_count": 1,
                "title_path_str": item.title_path_str,
                "snippet": _extract_snippet(item.text),
                "metadata": {key: str(value) for key, value in item.metadata.items()},
            }
            ordered_doc_ids.append(item.doc_id)
            continue

        current = grouped[item.doc_id]
        current["total_score"] = float(current["total_score"]) + float(item.score)
        current["hit_count"] = int(current["hit_count"]) + 1
        if not current["title_path_str"] and item.title_path_str:
            current["title_path_str"] = item.title_path_str
        if not current["snippet"]:
            current["snippet"] = _extract_snippet(item.text)

    candidates = [
        RetrievalDocumentCandidate(
            doc_id=str(grouped[doc_id]["doc_id"]),
            title=str(grouped[doc_id]["title"]),
            total_score=float(grouped[doc_id]["total_score"]),
            hit_count=int(grouped[doc_id]["hit_count"]),
            title_path_str=str(grouped[doc_id]["title_path_str"]),
            snippet=str(grouped[doc_id]["snippet"]),
            metadata=dict(grouped[doc_id]["metadata"]),
        )
        for doc_id in ordered_doc_ids
    ]
    candidates.sort(key=lambda item: (-item.total_score, -item.hit_count, item.doc_id))
    return tuple(candidates)


def choose_retrieval_strategy(
    *,
    intent: str | None,
    user_query: str,
    retrieval_output: RetrievePolicyOutput,
) -> StrategyDecision:
    """
    Decide how to consume retrieved evidence.
    Strategy only selects downstream flow; it does not rewrite query.
    """

    normalized_intent = (intent or ROUTE_RETRIEVE).strip().lower()
    candidates = aggregate_retrieval_documents(retrieval_output)

    if normalized_intent == ROUTE_COMPARE and len(candidates) >= 2:
        return StrategyDecision(
            strategy=STRATEGY_COMPARE,
            route=ROUTE_COMPARE,
            reason="用户问题是对比型请求，且检索结果中已存在至少两篇可比较政策。",
        )

    if normalized_intent == ROUTE_SUMMARIZE:
        if should_use_multi_doc_summary(
            user_query=user_query,
            candidates=candidates,
        ):
            return StrategyDecision(
                strategy=STRATEGY_MULTI_DOC_SUMMARY,
                route=ROUTE_SUMMARIZE,
                reason="摘要请求更像地区或主题范围汇总，优先走多文档摘要。",
            )

        top_candidate = candidates[0] if candidates else None
        if top_candidate is not None and should_use_single_doc_summary(
            user_query=user_query,
            candidates=candidates,
        ):
            return StrategyDecision(
                strategy=STRATEGY_SINGLE_DOC_SUMMARY,
                route=ROUTE_SUMMARIZE,
                reason=f"摘要请求在检索后已明显集中到单篇政策，因此转为单篇摘要：{top_candidate.title}。",
            )

        if len(candidates) >= 2:
            return StrategyDecision(
                strategy=STRATEGY_MULTI_DOC_SUMMARY,
                route=ROUTE_SUMMARIZE,
                reason="摘要请求命中了多篇高相关政策，更适合走多文档汇总。",
            )

    return StrategyDecision(
        strategy=STRATEGY_DIRECT_ANSWER,
        route=ROUTE_RETRIEVE,
        reason="当前更适合直接基于检索结果组织回答。",
    )


def should_use_single_doc_summary(
    *,
    user_query: str,
    candidates: tuple[RetrievalDocumentCandidate, ...],
) -> bool:
    """Heuristics for upgrading a summary request to single-document summary."""

    if not candidates:
        return False

    if len(candidates) == 1:
        return True

    top_candidate = candidates[0]
    second_candidate = candidates[1]

    if _normalized_contains(user_query, top_candidate.title):
        return True

    if top_candidate.hit_count >= 3:
        return True

    if top_candidate.hit_count >= 2 and top_candidate.total_score >= second_candidate.total_score * 1.25:
        return True

    return False


def should_use_multi_doc_summary(
    *,
    user_query: str,
    candidates: tuple[RetrievalDocumentCandidate, ...],
) -> bool:
    """Heuristics for region/topic/time-range summary requests."""

    if len(candidates) < 2:
        return False

    normalized_query = user_query.strip()
    if not normalized_query:
        return False

    if any(marker in normalized_query for marker in TIME_RANGE_KEYWORDS):
        return True

    mentioned_regions = [region for region in SUMMARY_REGION_KEYWORDS if region in normalized_query]
    if mentioned_regions and not any(_normalized_contains(normalized_query, item.title) for item in candidates[:2]):
        return True

    if any(marker in normalized_query for marker in RANGE_SUMMARY_KEYWORDS):
        return True

    return False


def _extract_snippet(text: str, *, max_length: int = 120) -> str:
    normalized_text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if len(normalized_text) <= max_length:
        return normalized_text
    return normalized_text[: max_length - 1].rstrip() + "..."


def _normalized_contains(base_text: str, candidate_text: str) -> bool:
    base = _normalize_for_match(base_text)
    candidate = _normalize_for_match(candidate_text)
    return bool(base and candidate and candidate in base)


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).lower()


RANGE_SUMMARY_KEYWORDS = (
    "有哪些",
    "整体",
    "总体",
    "汇总",
    "梳理",
    "政策",
)

TIME_RANGE_KEYWORDS = (
    "近两年",
    "近三年",
    "最近",
    "近年",
)

SUMMARY_REGION_KEYWORDS = (
    "上海",
    "北京",
    "江苏",
    "浙江",
    "深圳",
    "广东",
    "长三角",
)
