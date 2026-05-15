from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from app.agent.nodes import retrieve_node, select_node, summarize_node
from app.agent.router import DEFAULT_SUPPORTED_ROUTES, route_state
from app.agent.state import AgentState
from app.models.query import DEFAULT_QUERY_TOP_K, AgentQuery
from app.models.response import AgentResponse
from app.tools.retrieve_policy import RetrievePolicyTool
from app.tools.summarize_policy import SummarizePolicyTool


@dataclass(slots=True)
class PolicyAgentGraph:
    """
    第一版单 Agent 工作流编排器。

    当前目标只做最小闭环：
    - 构建初始状态
    - 执行 route
    - 选择 node
    - 返回最终状态 / 最终响应
    """

    retrieve_tool: RetrievePolicyTool | None = None
    summarize_tool: SummarizePolicyTool | None = None
    supported_routes: frozenset[str] = DEFAULT_SUPPORTED_ROUTES

    def run(
        self,
        query: AgentQuery | str,
        *,
        top_k: int = DEFAULT_QUERY_TOP_K,
    ) -> AgentState:
        """执行完整 Agent 工作流，并返回最终状态。"""

        state = build_initial_state(query, top_k=top_k)
        routed_state = route_state(
            state,
            supported_routes=self.supported_routes,
        )
        return self.execute_node(routed_state)

    def run_and_get_response(
        self,
        query: AgentQuery | str,
        *,
        top_k: int = DEFAULT_QUERY_TOP_K,
    ) -> AgentResponse:
        """执行完整 Agent 工作流，并直接返回最终响应。"""

        final_state = self.run(query, top_k=top_k)
        if final_state.final_response is not None:
            return final_state.final_response

        route = final_state.route or "unknown"
        error_message = final_state.error_message or "Agent 未生成最终响应。"
        return AgentResponse(
            success=False,
            route=route,
            message="Agent 执行未完成。",
            error_message=error_message,
        )

    def execute_node(self, state: AgentState) -> AgentState:
        """根据当前路由执行对应节点。"""

        node = select_node(state.route)
        if node is retrieve_node:
            return retrieve_node(
                state,
                tool=self.retrieve_tool,
            )
        if node is summarize_node:
            return summarize_node(
                state,
                tool=self.summarize_tool,
            )

        return node(state)


def build_initial_state(
    query: AgentQuery | str,
    *,
    top_k: int = DEFAULT_QUERY_TOP_K,
) -> AgentState:
    """构建工作流的初始状态。"""

    normalized_query = query if isinstance(query, AgentQuery) else AgentQuery(query, top_k=top_k)
    return AgentState(query=normalized_query)


def run_agent_workflow(
    query: AgentQuery | str,
    *,
    top_k: int = DEFAULT_QUERY_TOP_K,
    retrieve_tool: RetrievePolicyTool | None = None,
    summarize_tool: SummarizePolicyTool | None = None,
    supported_routes: frozenset[str] = DEFAULT_SUPPORTED_ROUTES,
) -> AgentState:
    """函数式入口：执行一次完整 Agent 工作流。"""

    graph = PolicyAgentGraph(
        retrieve_tool=retrieve_tool,
        summarize_tool=summarize_tool,
        supported_routes=cast(frozenset[str], supported_routes),
    )
    return graph.run(query, top_k=top_k)


def run_agent_query(
    query: AgentQuery | str,
    *,
    top_k: int = DEFAULT_QUERY_TOP_K,
    retrieve_tool: RetrievePolicyTool | None = None,
    summarize_tool: SummarizePolicyTool | None = None,
    supported_routes: frozenset[str] = DEFAULT_SUPPORTED_ROUTES,
) -> AgentResponse:
    """函数式入口：执行一次完整 Agent 工作流并返回最终响应。"""

    graph = PolicyAgentGraph(
        retrieve_tool=retrieve_tool,
        summarize_tool=summarize_tool,
        supported_routes=cast(frozenset[str], supported_routes),
    )
    return graph.run_and_get_response(query, top_k=top_k)
