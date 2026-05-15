from __future__ import annotations

from collections.abc import Iterable

from app.agent.state import AgentState
from app.models.query import AgentQuery


ROUTE_RETRIEVE = "retrieve"
ROUTE_SUMMARIZE = "summarize"
ROUTE_COMPARE = "compare"
ROUTE_MATCH = "match"
ROUTE_UNSUPPORTED = "unsupported"

DEFAULT_SUPPORTED_ROUTES = frozenset({ROUTE_RETRIEVE, ROUTE_SUMMARIZE})

SUMMARIZE_KEYWORDS = (
    "总结",
    "概括",
    "摘要",
    "梳理",
    "提炼",
)

COMPARE_KEYWORDS = (
    "对比",
    "比较",
    "差异",
    "区别",
    "不同",
)

MATCH_KEYWORDS = (
    "匹配",
    "适配",
    "适合我",
    "适合我们",
    "适合关注",
    "企业画像",
    "推荐关注",
)


def detect_intent_route(query: AgentQuery | str) -> str:
    """
    识别查询在业务语义上最接近的意图路由。

    第一版采用轻量关键词规则：
    - 明显的比较类请求 -> compare
    - 明显的摘要类请求 -> summarize
    - 明显的匹配类请求 -> match
    - 其他默认都走 retrieve
    """

    query_text = _extract_query_text(query)

    if _contains_any(query_text, COMPARE_KEYWORDS):
        return ROUTE_COMPARE

    if _contains_any(query_text, SUMMARIZE_KEYWORDS):
        return ROUTE_SUMMARIZE

    if _contains_any(query_text, MATCH_KEYWORDS):
        return ROUTE_MATCH

    return ROUTE_RETRIEVE


def route_query(
    query: AgentQuery | str,
    *,
    supported_routes: Iterable[str] = DEFAULT_SUPPORTED_ROUTES,
) -> str:
    """
    根据当前已实现能力，为查询返回最终可执行路由。

    这一步把“意图识别”和“当前实现状态”分开：
    - 先识别 query 本来想做什么
    - 再判断这条路由目前是否已实现
    - 未实现时统一回退为 unsupported
    """

    intended_route = detect_intent_route(query)
    normalized_supported_routes = {route.strip().lower() for route in supported_routes}

    if intended_route in normalized_supported_routes:
        return intended_route

    return ROUTE_UNSUPPORTED


def route_state(
    state: AgentState,
    *,
    supported_routes: Iterable[str] = DEFAULT_SUPPORTED_ROUTES,
) -> AgentState:
    """把路由结果写回 AgentState。"""

    return state.with_route(
        route_query(
            state.query,
            supported_routes=supported_routes,
        )
    )


def _extract_query_text(query: AgentQuery | str) -> str:
    """统一提取标准化后的查询文本。"""

    if isinstance(query, AgentQuery):
        return query.user_query
    return query.strip()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """判断文本是否包含给定关键词中的任意一个。"""

    return any(keyword in text for keyword in keywords)
