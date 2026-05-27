from __future__ import annotations

from collections.abc import Callable

from app.agent.answer import PolicyAgentAnswerer, fallback_answer
from app.agent.judge import PolicyAgentJudge, fallback_judge
from app.agent.next_step import (
    PolicyAgentNextStepPlanner,
    append_next_step_guidance,
    fallback_next_step,
)
from app.agent.planner import PolicyAgentPlanner
from app.agent.repair import PolicyAgentRepairer, fallback_repair
from app.agent.rewrite import PolicyAgentRewriter
from app.agent.router import ROUTE_COMPARE, ROUTE_RETRIEVE, ROUTE_SUMMARIZE, ROUTE_UNSUPPORTED
from app.agent.router import detect_intent_route
from app.agent.strategy import choose_retrieval_strategy
from app.agent.state import AgentState
from app.models.response import AgentResponse
from app.tools.compare_policy import ComparePolicyTool
from app.tools.retrieve_policy import RetrievePolicyTool, RetrievePolicyOutput
from app.tools.summarize_policies import SummarizePoliciesTool
from app.tools.summarize_policy import (
    SummarizePolicyTool,
)


AgentNode = Callable[[AgentState], AgentState]


def planner_node(
    state: AgentState,
    *,
    planner: PolicyAgentPlanner | None = None,
    supported_routes: frozenset[str],
) -> AgentState:
    """
    标准 planner node。

    这里统一承接“任务理解”这一步：
    - 优先尝试 LLM planner
    - 失败时回退到规则路由
    - 无论来源如何，都把统一形状的规划结果写回 state
    """

    if state.resolved_action in {ROUTE_RETRIEVE, ROUTE_SUMMARIZE, ROUTE_COMPARE, "match", "chat"}:
        return resolve_first_class_planning_from_context(state)

    active_planner = planner or PolicyAgentPlanner()
    if not active_planner.is_available:
        return fallback_planner_node(
            state,
            supported_routes=supported_routes,
        )

    try:
        decision = active_planner.decide(state.query.user_query)
    except Exception:
        return fallback_planner_node(
            state,
            supported_routes=supported_routes,
        )

    route = initial_route_for_intent(
        decision.intent,
        needs_rag=decision.needs_rag,
    )
    return state.with_planner_result(
        intent=decision.intent,
        route=route,
        resolved_action=state.resolved_action or decision.intent,
        needs_rag=decision.needs_rag,
        needs_rewrite=decision.needs_rewrite,
        answer_style=decision.answer_style,
        response_mode=state.response_mode,
        retrieval_goal=state.retrieval_goal,
        focus=state.focus,
        answer_plan=state.answer_plan,
        resolved_entities=state.resolved_entities,
        planner_reason=decision.reason,
        planner_source="llm",
    )


def fallback_planner_node(
    state: AgentState,
    *,
    supported_routes: frozenset[str],
) -> AgentState:
    """
    规则版 planner node。

    即使当前没启用 LLM，我们也尽量把 state 填成和 LLM planner
    一样的字段形状，避免后面 rewrite / answer 节点区分两套输入。
    """

    detected_intent = detect_intent_route(state.query)
    routed_state = state.with_route(
        initial_route_for_intent(
            detected_intent,
            needs_rag=detected_intent != "chat",
        )
    )

    answer_style = "direct"
    if detected_intent == ROUTE_SUMMARIZE:
        answer_style = "structured"
    if detected_intent == ROUTE_COMPARE:
        answer_style = "comparative"

    return routed_state.with_planner_result(
        intent=detected_intent,
        route=routed_state.route or ROUTE_UNSUPPORTED,
        resolved_action=routed_state.resolved_action or detected_intent,
        needs_rag=detected_intent != "chat",
        needs_rewrite=detected_intent in {ROUTE_RETRIEVE, ROUTE_SUMMARIZE, ROUTE_COMPARE},
        answer_style=answer_style,
        response_mode=routed_state.response_mode,
        retrieval_goal=routed_state.retrieval_goal,
        focus=routed_state.focus,
        answer_plan=routed_state.answer_plan,
        resolved_entities=routed_state.resolved_entities,
        planner_reason="当前未启用 LLM planner，使用规则路由结果作为兜底规划。",
        planner_source="rule",
    )


def rewrite_node(
    state: AgentState,
    *,
    rewriter: PolicyAgentRewriter | None = None,
) -> AgentState:
    """
    标准 rewrite node。

    当前策略尽量保守：
    - planner 判断不需要 rewrite 时，直接把原始 query 透传下去
    - 如果启用了 LLM rewriter，就尝试生成更适合检索的查询
    - 失败时退回到轻量规则改写
    """

    if state.needs_rewrite is False:
        return state.with_rewrite_result(
            rewritten_query=state.query.user_query,
            rewrite_reason="planner 判断当前请求不需要额外改写，直接使用原始 query。",
            rewrite_source="skip",
        )

    active_rewriter = rewriter or PolicyAgentRewriter()
    if not active_rewriter.is_available:
        return fallback_rewrite_node(state)

    try:
        decision = active_rewriter.rewrite(
            state.query.user_query,
            intent=state.intent,
        )
    except Exception:
        return fallback_rewrite_node(state)

    return state.with_rewrite_result(
        rewritten_query=decision.primary_query,
        alternative_queries=decision.alternative_queries,
        rewrite_keywords=decision.keywords,
        rewrite_reason=decision.rewrite_reason,
        rewrite_source="llm",
    )


def fallback_rewrite_node(state: AgentState) -> AgentState:
    """规则版 rewrite node。"""

    normalized_query = state.query.user_query.strip()
    return state.with_rewrite_result(
        rewritten_query=normalized_query,
        rewrite_reason="当前未启用 LLM rewriter，使用原始 query 作为兜底检索查询。",
        rewrite_source="rule",
    )


def initial_route_for_intent(intent: str | None, *, needs_rag: bool) -> str:
    """Map user intent to the first executable route in the graph."""

    normalized_intent = intent.strip().lower() if intent else ""
    if needs_rag and normalized_intent in {ROUTE_RETRIEVE, ROUTE_SUMMARIZE, ROUTE_COMPARE}:
        return ROUTE_RETRIEVE
    return ROUTE_UNSUPPORTED


def resolve_first_class_planning_from_context(state: AgentState) -> AgentState:
    """
    Prefer resolver output as the primary understanding result.

    Once context resolver has already produced a structured action and answer mode,
    planner only needs to fill execution defaults instead of re-understanding the query.
    """

    resolved_action = state.resolved_action or ROUTE_RETRIEVE
    needs_rag = resolved_action != "chat"
    needs_rewrite = needs_rag
    answer_style = (
        "structured"
        if resolved_action == ROUTE_SUMMARIZE
        else "comparative"
        if resolved_action == ROUTE_COMPARE
        else "direct"
    )
    route = initial_route_for_intent(resolved_action, needs_rag=needs_rag)

    return state.with_planner_result(
        intent=resolved_action,
        route=route,
        resolved_action=resolved_action,
        needs_rag=needs_rag,
        needs_rewrite=needs_rewrite,
        answer_style=answer_style,
        response_mode=state.response_mode,
        retrieval_goal=state.retrieval_goal,
        focus=state.focus,
        answer_plan=state.answer_plan,
        resolved_entities=state.resolved_entities,
        planner_reason="优先采用 context resolver 的结构化理解结果，planner 仅补执行默认值。",
        planner_source="resolver",
    )


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
        # 检索节点只认当前“生效中的查询”。
        # 这样第一次走 rewrite，第二次走 repair 时，工具层都不用再分支判断。
        query_for_retrieval = state.effective_query
        tool_output = active_tool.run(
            query_for_retrieval,
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

    return state.with_retrieval_output(tool_output)


def strategy_node(state: AgentState) -> AgentState:
    """
    Select the post-retrieval execution strategy.
    The agent now retrieves first, then decides whether to answer directly,
    switch to single-document summary, or continue into compare.
    """

    if not isinstance(state.tool_output, RetrievePolicyOutput):
        return state

    decision = choose_retrieval_strategy(
        intent=state.intent,
        user_query=state.query.user_query,
        retrieval_output=state.tool_output,
        retrieval_goal=state.retrieval_goal,
        resolved_entities=state.resolved_entities,
    )
    return state.with_strategy_result(
        strategy=decision.strategy,
        route=decision.route,
        strategy_reason=decision.reason,
    )


def summarize_node(
    state: AgentState,
    *,
    tool: SummarizePolicyTool | None = None,
) -> AgentState:
    """执行摘要节点。"""

    active_tool = tool or SummarizePolicyTool()
    route = state.route or ROUTE_SUMMARIZE

    try:
        # summarize 分支和 retrieve 一样，统一消费 effective_query。
        query_for_summary = state.effective_query
        tool_output = active_tool.run(
            query_for_summary,
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

    return state.with_tool_output(tool_output)


def summarize_policies_node(
    state: AgentState,
    *,
    tool: SummarizePoliciesTool | None = None,
) -> AgentState:
    """Execute multi-document summary after retrieval aggregation."""

    active_tool = tool or SummarizePoliciesTool()
    route = state.route or ROUTE_SUMMARIZE

    try:
        tool_output = active_tool.run(
            state.effective_query,
            retrieval_output=state.retrieval_output if isinstance(state.retrieval_output, RetrievePolicyOutput) else None,
            top_k=max(3, state.query.top_k),
        )
    except Exception as error:
        error_message = f"执行多文档摘要节点失败: {error}"
        return state.with_error(error_message).with_final_response(
            AgentResponse(
                success=False,
                route=route,
                message="政策汇总执行失败。",
                error_message=error_message,
            )
        )

    return state.with_tool_output(tool_output)


def compare_node(
    state: AgentState,
    *,
    tool: ComparePolicyTool | None = None,
) -> AgentState:
    """
    执行对比节点。

    compare 的第一版刻意复用 summarize 的结构：
    - 先锁定两篇政策
    - 再对两篇政策分别做固定分区摘要
    - 最后把同名分区并排组织成对比输出
    """

    active_tool = tool or ComparePolicyTool()
    route = state.route or ROUTE_COMPARE

    try:
        query_for_compare = state.effective_query
        tool_output = active_tool.run(
            query_for_compare,
            top_k=max(2, state.query.top_k),
        )
    except Exception as error:
        error_message = f"执行对比节点失败: {error}"
        return state.with_error(error_message).with_final_response(
            AgentResponse(
                success=False,
                route=route,
                message="政策对比执行失败。",
                error_message=error_message,
            )
        )

    return state.with_tool_output(tool_output)


def answer_node(
    state: AgentState,
    *,
    answerer: PolicyAgentAnswerer | None = None,
) -> AgentState:
    """
    标准 answer node。

    这层负责把工具输出从“中间证据”变成“最终回答”：
    - 如果启用了 LLM answerer，就让模型组织自然语言答案
    - 如果没有，就回退到规则版最终回答
    """

    if state.tool_output is None:
        error_message = "answer node 缺少可用的 tool_output。"
        return state.with_error(error_message).with_final_response(
            AgentResponse(
                success=False,
                route=state.route or ROUTE_UNSUPPORTED,
                message="当前没有可用于生成回答的证据。",
                error_message=error_message,
            )
        )

    active_answerer = answerer or PolicyAgentAnswerer()
    if not active_answerer.is_available:
        draft = fallback_answer(tool_output=state.tool_output)
        return state.with_answer_source(draft.source).with_final_response(
            AgentResponse(
                success=True,
                route=state.route or ROUTE_UNSUPPORTED,
                message=draft.message,
                citations=draft.citations,
            )
        )

    try:
        draft = active_answerer.answer(
            user_query=state.query.user_query,
            intent=state.intent,
            answer_style=state.answer_style,
            response_mode=state.response_mode,
            focus=state.focus,
            answer_plan=state.answer_plan,
            tool_output=state.tool_output,
        )
    except Exception:
        draft = fallback_answer(tool_output=state.tool_output)

    return state.with_answer_source(draft.source).with_final_response(
        AgentResponse(
            success=True,
            route=state.route or ROUTE_UNSUPPORTED,
            message=draft.message,
            citations=draft.citations,
        )
    )


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


def repair_node(
    state: AgentState,
    *,
    repairer: PolicyAgentRepairer | None = None,
) -> AgentState:
    """
    标准 repair node。

    它不直接执行工具，而是只做一件事：
    - 判断当前 judge 结果是否值得重试
    - 如果值得，就把新的 repair_query 写回 state
    - 如果不值得，就保持原状态不变
    """

    active_repairer = repairer or PolicyAgentRepairer()
    if not active_repairer.is_available:
        decision = fallback_repair(state)
        if not decision.should_retry:
            return state

        return state.with_repair_result(
            repair_query=decision.repaired_query,
            repair_strategy=decision.repair_strategy,
            repair_reason=decision.repair_reason,
            repair_source="rule",
        )

    try:
        decision = active_repairer.repair(state)
        repair_source = "llm"
    except Exception:
        decision = fallback_repair(state)
        repair_source = "rule"

    if not decision.should_retry:
        return state

    return state.with_repair_result(
        repair_query=decision.repaired_query,
        repair_strategy=decision.repair_strategy,
        repair_reason=decision.repair_reason,
        repair_source=repair_source,
    )


def judge_node(
    state: AgentState,
    *,
    judge: PolicyAgentJudge | None = None,
) -> AgentState:
    """
    标准 judge node。

    这层负责在最终回答生成后补一轮“回答后自检”：
    - 如果启用了 LLM judge，就做结构化质量评估
    - 如果没有，就回退到规则版质量评估
    - 当前只记录评估结果，不直接触发 retry
    """

    if state.final_response is None:
        error_message = "judge node 缺少可评估的 final_response。"
        return state.with_error(error_message)

    active_judge = judge or PolicyAgentJudge()
    if not active_judge.is_available:
        decision = fallback_judge(
            tool_output=state.tool_output,
            final_response=state.final_response,
        )
        return state.with_judge_result(
            judge_verdict=decision.verdict,
            judge_score=decision.score,
            judge_reason=decision.reason,
            judge_followup=decision.followup,
            judge_source="rule",
        )

    try:
        decision = active_judge.judge(
            user_query=state.query.user_query,
            intent=state.intent,
            tool_output=state.tool_output,
            final_response=state.final_response,
        )
        judge_source = "llm"
    except Exception:
        decision = fallback_judge(
            tool_output=state.tool_output,
            final_response=state.final_response,
        )
        judge_source = "rule"

    return state.with_judge_result(
        judge_verdict=decision.verdict,
        judge_score=decision.score,
        judge_reason=decision.reason,
        judge_followup=decision.followup,
        judge_source=judge_source,
    )


def next_step_node(
    state: AgentState,
    *,
    planner: PolicyAgentNextStepPlanner | None = None,
) -> AgentState:
    """
    标准 next-step node。

    这层负责在整轮执行尾部做真正的“后处理分流”：
    - 当前回答已经足够好 -> 结束
    - 当前更适合自动切摘要 -> 记录 route switch，交给 graph 继续跑
    - 当前不适合自动继续 -> 把 follow-up / compare 建议追加到最终回答
    """

    if state.final_response is None:
        error_message = "next_step node 缺少可处理的 final_response。"
        return state.with_error(error_message)

    active_planner = planner or PolicyAgentNextStepPlanner()
    if not active_planner.is_available:
        decision = fallback_next_step(state)
        planner_source = "rule"
    else:
        try:
            decision = active_planner.decide(state)
            planner_source = "llm"
        except Exception:
            decision = fallback_next_step(state)
            planner_source = "rule"

    state_with_decision = state.with_next_step_result(
        next_step_action=decision.action,
        next_step_route=decision.target_route,
        next_step_query=decision.next_query,
        next_step_reason=decision.reason,
        next_step_followups=decision.followups,
        next_step_source=planner_source,
    )

    if decision.action == "auto_switch_route" and decision.target_route == ROUTE_SUMMARIZE:
        return state_with_decision.with_route_switch(
            route=ROUTE_SUMMARIZE,
            route_switch_query=decision.next_query,
            route_switch_reason=decision.reason,
            route_switch_source=planner_source,
            intent=ROUTE_SUMMARIZE,
            answer_style="structured",
        )

    if decision.action in {"ask_followup", "suggest_route"}:
        enriched_response = append_next_step_guidance(
            state_with_decision.final_response,
            decision,
        )
        return state_with_decision.with_final_response(enriched_response)

    return state_with_decision


def select_node(route: str | None) -> AgentNode:
    """根据路由选择对应节点函数。"""

    normalized_route = route.strip().lower() if route else ROUTE_UNSUPPORTED

    if normalized_route == ROUTE_RETRIEVE:
        return retrieve_node
    if normalized_route == ROUTE_SUMMARIZE:
        return summarize_node
    if normalized_route == ROUTE_COMPARE:
        return compare_node

    return unsupported_node
