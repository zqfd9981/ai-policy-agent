from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from app.agent.answer import PolicyAgentAnswerer
from app.agent.judge import PolicyAgentJudge
from app.agent.next_step import PolicyAgentNextStepPlanner
from app.agent.planner import PolicyAgentPlanner
from app.agent.repair import PolicyAgentRepairer
from app.agent.rewrite import PolicyAgentRewriter
from app.agent.nodes import (
    answer_node,
    judge_node,
    next_step_node,
    planner_node,
    repair_node,
    retrieve_node,
    rewrite_node,
    select_node,
    summarize_node,
)
from app.agent.router import DEFAULT_SUPPORTED_ROUTES
from app.agent.state import AgentState
from app.models.query import DEFAULT_QUERY_TOP_K, AgentQuery
from app.models.response import AgentResponse
from app.tools.retrieve_policy import RetrievePolicyTool
from app.tools.summarize_policy import SummarizePolicyTool


@dataclass(slots=True)
class PolicyAgentGraph:
    """
    第一版单 Agent 工作流编排器。

    当前已经升级成一条更完整的主线：
    - planner: 理解任务
    - rewrite: 改写检索 query
    - execute: 执行 retrieve / summarize
    - answer: 组织最终回答
    - judge: 判断回答是否达标
    - repair: 如有必要，最多再重试一次
    """

    planner: PolicyAgentPlanner | None = None
    rewriter: PolicyAgentRewriter | None = None
    answerer: PolicyAgentAnswerer | None = None
    judge: PolicyAgentJudge | None = None
    repairer: PolicyAgentRepairer | None = None
    next_step_planner: PolicyAgentNextStepPlanner | None = None
    retrieve_tool: RetrievePolicyTool | None = None
    summarize_tool: SummarizePolicyTool | None = None
    supported_routes: frozenset[str] = DEFAULT_SUPPORTED_ROUTES

    def run(
        self,
        query: AgentQuery | str,
        *,
        top_k: int = DEFAULT_QUERY_TOP_K,
        max_retries: int = 1,
    ) -> AgentState:
        """执行完整 Agent 工作流，并返回最终状态。"""

        state = build_initial_state(query, top_k=top_k, max_retries=max_retries)
        planned_state = planner_node(
            state,
            planner=self.planner,
            supported_routes=self.supported_routes,
        )
        rewritten_state = rewrite_node(
            planned_state,
            rewriter=self.rewriter,
        )
        first_pass_state = self.run_execution_cycle(rewritten_state)

        # repair 只做一层“轻量补救”：
        # - 不在 graph 里开复杂 while loop
        # - 先把“能重试一次”这条主线打通
        repaired_state = repair_node(
            first_pass_state,
            repairer=self.repairer,
        )
        current_state = (
            repaired_state
            if repaired_state.retry_count == first_pass_state.retry_count
            else self.run_execution_cycle(repaired_state)
        )

        final_state = next_step_node(
            current_state,
            planner=self.next_step_planner,
        )

        # 如果 next-step 决策器明确要求自动切摘要，
        # graph 会在这里再补跑一轮 summarize -> answer -> judge，
        # 然后再次进入 next-step，让最终输出形态稳定下来。
        if final_state.route_switch_count > current_state.route_switch_count:
            switched_state = self.run_execution_cycle(final_state)
            return next_step_node(
                switched_state,
                planner=self.next_step_planner,
            )

        return final_state

    def run_and_get_response(
        self,
        query: AgentQuery | str,
        *,
        top_k: int = DEFAULT_QUERY_TOP_K,
        max_retries: int = 1,
    ) -> AgentResponse:
        """执行完整 Agent 工作流，并直接返回最终响应。"""

        final_state = self.run(query, top_k=top_k, max_retries=max_retries)
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

    def run_execution_cycle(self, state: AgentState) -> AgentState:
        """
        执行一轮完整的 execute -> answer -> judge。

        把这一层拆成单独方法后，graph 既能跑第一次主流程，
        也能在 repair 之后复用完全同一套逻辑再跑一遍。
        """

        executed_state = self.execute_node(state)
        answered_state = answer_node(
            executed_state,
            answerer=self.answerer,
        )
        return judge_node(
            answered_state,
            judge=self.judge,
        )


def build_initial_state(
    query: AgentQuery | str,
    *,
    top_k: int = DEFAULT_QUERY_TOP_K,
    max_retries: int = 1,
) -> AgentState:
    """构建工作流的初始状态。"""

    normalized_query = query if isinstance(query, AgentQuery) else AgentQuery(query, top_k=top_k)
    return AgentState(
        query=normalized_query,
        max_retries=max_retries,
    )


def run_agent_workflow(
    query: AgentQuery | str,
    *,
    top_k: int = DEFAULT_QUERY_TOP_K,
    max_retries: int = 1,
    planner: PolicyAgentPlanner | None = None,
    rewriter: PolicyAgentRewriter | None = None,
    answerer: PolicyAgentAnswerer | None = None,
    judge: PolicyAgentJudge | None = None,
    repairer: PolicyAgentRepairer | None = None,
    next_step_planner: PolicyAgentNextStepPlanner | None = None,
    retrieve_tool: RetrievePolicyTool | None = None,
    summarize_tool: SummarizePolicyTool | None = None,
    supported_routes: frozenset[str] = DEFAULT_SUPPORTED_ROUTES,
) -> AgentState:
    """函数式入口：执行一次完整 Agent 工作流。"""

    graph = PolicyAgentGraph(
        planner=planner,
        rewriter=rewriter,
        answerer=answerer,
        judge=judge,
        repairer=repairer,
        next_step_planner=next_step_planner,
        retrieve_tool=retrieve_tool,
        summarize_tool=summarize_tool,
        supported_routes=cast(frozenset[str], supported_routes),
    )
    return graph.run(query, top_k=top_k, max_retries=max_retries)


def run_agent_query(
    query: AgentQuery | str,
    *,
    top_k: int = DEFAULT_QUERY_TOP_K,
    max_retries: int = 1,
    planner: PolicyAgentPlanner | None = None,
    rewriter: PolicyAgentRewriter | None = None,
    answerer: PolicyAgentAnswerer | None = None,
    judge: PolicyAgentJudge | None = None,
    repairer: PolicyAgentRepairer | None = None,
    next_step_planner: PolicyAgentNextStepPlanner | None = None,
    retrieve_tool: RetrievePolicyTool | None = None,
    summarize_tool: SummarizePolicyTool | None = None,
    supported_routes: frozenset[str] = DEFAULT_SUPPORTED_ROUTES,
) -> AgentResponse:
    """函数式入口：执行一次完整 Agent 工作流并返回最终响应。"""

    graph = PolicyAgentGraph(
        planner=planner,
        rewriter=rewriter,
        answerer=answerer,
        judge=judge,
        repairer=repairer,
        next_step_planner=next_step_planner,
        retrieve_tool=retrieve_tool,
        summarize_tool=summarize_tool,
        supported_routes=cast(frozenset[str], supported_routes),
    )
    return graph.run_and_get_response(query, top_k=top_k, max_retries=max_retries)
