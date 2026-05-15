from __future__ import annotations

from collections.abc import Callable

from app.agent.router import ROUTE_RETRIEVE, ROUTE_SUMMARIZE, ROUTE_UNSUPPORTED
from app.agent.state import AgentState
from app.models.response import AgentResponse
from app.tools.retrieve_policy import RetrievePolicyOutput, RetrievePolicyTool
from app.tools.summarize_policy import (
    PolicySummaryOutput,
    SummarizePolicyTool,
    render_policy_summary,
)


AgentNode = Callable[[AgentState], AgentState]


def retrieve_node(
    state: AgentState,
    *,
    tool: RetrievePolicyTool | None = None,
) -> AgentState:
    """
    执行检索节点。

    第一版职责保持克制：
    - 调用 retrieve_policy 工具
    - 把原始工具输出挂回 state
    - 生成一个最小可展示的 AgentResponse
    """

    active_tool = tool or RetrievePolicyTool()
    route = state.route or ROUTE_RETRIEVE

    try:
        tool_output = active_tool.run(
            state.query.user_query,
            top_k=state.query.top_k,
        )
    except Exception as error:
        error_message = f"执行检索节点失败: {error}"
        return state.with_error(error_message).with_final_response(
            AgentResponse(
                success=False,
                route=route,
                message="检索执行失败。",
                error_message=error_message,
            )
        )

    citations = tuple(result.to_dict() for result in tool_output.results)
    response = AgentResponse(
        success=True,
        route=route,
        message=_build_retrieve_message(tool_output),
        citations=citations,
    )
    return state.with_tool_output(tool_output).with_final_response(response)


def summarize_node(
    state: AgentState,
    *,
    tool: SummarizePolicyTool | None = None,
) -> AgentState:
    """执行摘要节点。"""

    active_tool = tool or SummarizePolicyTool()
    route = state.route or ROUTE_SUMMARIZE

    try:
        tool_output = active_tool.run(
            state.query.user_query,
            top_k=state.query.top_k,
        )
    except Exception as error:
        error_message = f"执行摘要节点失败: {error}"
        return state.with_error(error_message).with_final_response(
            AgentResponse(
                success=False,
                route=route,
                message="政策摘要执行失败。",
                error_message=error_message,
            )
        )

    citations = tuple(item.to_dict() for item in tool_output.all_citations)
    response = AgentResponse(
        success=True,
        route=route,
        message=render_policy_summary(tool_output),
        citations=citations,
    )
    return state.with_tool_output(tool_output).with_final_response(response)


def unsupported_node(state: AgentState) -> AgentState:
    """处理当前尚未实现的路由。"""

    route = state.route or ROUTE_UNSUPPORTED
    error_message = f"当前尚未实现 route={route} 的处理流程。"

    return state.with_error(error_message).with_final_response(
        AgentResponse(
            success=False,
            route=route,
            message="当前请求类型暂未支持。",
            error_message=error_message,
        )
    )


def select_node(route: str | None) -> AgentNode:
    """根据路由选择对应节点函数。"""

    normalized_route = route.strip().lower() if route else ROUTE_UNSUPPORTED

    if normalized_route == ROUTE_RETRIEVE:
        return retrieve_node
    if normalized_route == ROUTE_SUMMARIZE:
        return summarize_node

    return unsupported_node


def _build_retrieve_message(tool_output: RetrievePolicyOutput) -> str:
    """根据检索结果生成最小可展示消息。"""

    if tool_output.result_count == 0:
        return "未检索到相关政策片段。"

    return f"已检索到 {tool_output.result_count} 条相关政策片段。"
