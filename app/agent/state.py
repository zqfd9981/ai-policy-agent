from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from app.models.query import AgentQuery
from app.models.response import AgentResponse


@dataclass(frozen=True, slots=True)
class AgentState:
    """
    表示一次 Agent 工作流在当前阶段的共享状态。

    第一版先保持最小：
    - query: 标准化后的用户输入
    - route: 路由结果
    - intent: planner 或 router 判断的意图
    - needs_rag / needs_rewrite / answer_style: 后续 LLM Agent 链路的规划信息
    - rewritten_query / alternative_queries / rewrite_keywords: query rewrite 结果
    - retry_count / max_retries: 当前允许的 repair 重试信息
    - repair_query / repair_strategy / repair_reason: repair 阶段生成的新查询和原因
    - route_switch_count / route_switch_query: 自动切换执行路线时的状态
    - tool_output: 工具层返回的原始结果
    - answer_source: 最终回答来自 LLM 还是规则回退
    - judge_verdict / judge_score / judge_reason / judge_followup: 回答评估结果
    - next_step_action / next_step_route / next_step_followups: 最终阶段的下一步动作建议
    - final_response: 最终给用户的响应
    - error_message: 工作流中记录的错误信息
    """

    query: AgentQuery
    route: str | None = None
    intent: str | None = None
    resolved_action: str | None = None
    needs_rag: bool | None = None
    needs_rewrite: bool | None = None
    answer_style: str | None = None
    response_mode: str | None = None
    retrieval_goal: str | None = None
    focus: str | None = None
    answer_plan: dict[str, Any] | None = None
    planner_reason: str | None = None
    planner_source: str | None = None
    rewritten_query: str | None = None
    alternative_queries: tuple[str, ...] = ()
    rewrite_keywords: tuple[str, ...] = ()
    rewrite_reason: str | None = None
    rewrite_source: str | None = None
    strategy: str | None = None
    strategy_reason: str | None = None
    retrieval_output: Any | None = None
    retry_count: int = 0
    max_retries: int = 1
    repair_query: str | None = None
    repair_strategy: str | None = None
    repair_reason: str | None = None
    repair_source: str | None = None
    route_switch_count: int = 0
    max_route_switches: int = 1
    route_switch_query: str | None = None
    route_switch_reason: str | None = None
    route_switch_source: str | None = None
    tool_output: Any | None = None
    answer_source: str | None = None
    judge_verdict: str | None = None
    judge_score: int | None = None
    judge_reason: str | None = None
    judge_followup: str | None = None
    judge_source: str | None = None
    next_step_action: str | None = None
    next_step_route: str | None = None
    next_step_query: str | None = None
    next_step_reason: str | None = None
    next_step_followups: tuple[str, ...] = ()
    next_step_source: str | None = None
    final_response: AgentResponse | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        """统一可选字符串字段的格式。"""

        normalized_route = self.route.strip().lower() if self.route else None
        normalized_intent = self.intent.strip().lower() if self.intent else None
        normalized_resolved_action = self.resolved_action.strip().lower() if self.resolved_action else None
        normalized_answer_style = self.answer_style.strip().lower() if self.answer_style else None
        normalized_response_mode = self.response_mode.strip().lower() if self.response_mode else None
        normalized_retrieval_goal = self.retrieval_goal.strip().lower() if self.retrieval_goal else None
        normalized_focus = self.focus.strip().lower() if self.focus else None
        normalized_answer_plan = dict(self.answer_plan) if self.answer_plan is not None else None
        normalized_planner_reason = self.planner_reason.strip() if self.planner_reason else None
        normalized_planner_source = self.planner_source.strip().lower() if self.planner_source else None
        normalized_rewritten_query = self.rewritten_query.strip() if self.rewritten_query else None
        normalized_alternative_queries = tuple(
            item.strip() for item in self.alternative_queries if item.strip()
        )
        normalized_rewrite_keywords = tuple(
            item.strip() for item in self.rewrite_keywords if item.strip()
        )
        normalized_rewrite_reason = self.rewrite_reason.strip() if self.rewrite_reason else None
        normalized_rewrite_source = self.rewrite_source.strip().lower() if self.rewrite_source else None
        normalized_strategy = self.strategy.strip().lower() if self.strategy else None
        normalized_strategy_reason = self.strategy_reason.strip() if self.strategy_reason else None
        normalized_retry_count = max(0, int(self.retry_count))
        normalized_max_retries = max(0, int(self.max_retries))
        normalized_repair_query = self.repair_query.strip() if self.repair_query else None
        normalized_repair_strategy = self.repair_strategy.strip().lower() if self.repair_strategy else None
        normalized_repair_reason = self.repair_reason.strip() if self.repair_reason else None
        normalized_repair_source = self.repair_source.strip().lower() if self.repair_source else None
        normalized_route_switch_count = max(0, int(self.route_switch_count))
        normalized_max_route_switches = max(0, int(self.max_route_switches))
        normalized_route_switch_query = (
            self.route_switch_query.strip() if self.route_switch_query else None
        )
        normalized_route_switch_reason = (
            self.route_switch_reason.strip() if self.route_switch_reason else None
        )
        normalized_route_switch_source = (
            self.route_switch_source.strip().lower() if self.route_switch_source else None
        )
        normalized_answer_source = self.answer_source.strip().lower() if self.answer_source else None
        normalized_judge_verdict = self.judge_verdict.strip().lower() if self.judge_verdict else None
        normalized_judge_reason = self.judge_reason.strip() if self.judge_reason else None
        normalized_judge_followup = self.judge_followup.strip() if self.judge_followup else None
        normalized_judge_source = self.judge_source.strip().lower() if self.judge_source else None
        normalized_judge_score = (
            max(0, min(100, int(self.judge_score))) if self.judge_score is not None else None
        )
        normalized_next_step_action = (
            self.next_step_action.strip().lower() if self.next_step_action else None
        )
        normalized_next_step_route = self.next_step_route.strip().lower() if self.next_step_route else None
        normalized_next_step_query = self.next_step_query.strip() if self.next_step_query else None
        normalized_next_step_reason = self.next_step_reason.strip() if self.next_step_reason else None
        normalized_next_step_followups = tuple(
            item.strip() for item in self.next_step_followups if item.strip()
        )
        normalized_next_step_source = (
            self.next_step_source.strip().lower() if self.next_step_source else None
        )
        normalized_error_message = self.error_message.strip() if self.error_message else None

        object.__setattr__(self, "route", normalized_route)
        object.__setattr__(self, "intent", normalized_intent)
        object.__setattr__(self, "resolved_action", normalized_resolved_action)
        object.__setattr__(self, "answer_style", normalized_answer_style)
        object.__setattr__(self, "response_mode", normalized_response_mode)
        object.__setattr__(self, "retrieval_goal", normalized_retrieval_goal)
        object.__setattr__(self, "focus", normalized_focus)
        object.__setattr__(self, "answer_plan", normalized_answer_plan)
        object.__setattr__(self, "planner_reason", normalized_planner_reason)
        object.__setattr__(self, "planner_source", normalized_planner_source)
        object.__setattr__(self, "rewritten_query", normalized_rewritten_query)
        object.__setattr__(self, "alternative_queries", normalized_alternative_queries)
        object.__setattr__(self, "rewrite_keywords", normalized_rewrite_keywords)
        object.__setattr__(self, "rewrite_reason", normalized_rewrite_reason)
        object.__setattr__(self, "rewrite_source", normalized_rewrite_source)
        object.__setattr__(self, "strategy", normalized_strategy)
        object.__setattr__(self, "strategy_reason", normalized_strategy_reason)
        object.__setattr__(self, "retrieval_output", self.retrieval_output)
        object.__setattr__(self, "retry_count", normalized_retry_count)
        object.__setattr__(self, "max_retries", normalized_max_retries)
        object.__setattr__(self, "repair_query", normalized_repair_query)
        object.__setattr__(self, "repair_strategy", normalized_repair_strategy)
        object.__setattr__(self, "repair_reason", normalized_repair_reason)
        object.__setattr__(self, "repair_source", normalized_repair_source)
        object.__setattr__(self, "route_switch_count", normalized_route_switch_count)
        object.__setattr__(self, "max_route_switches", normalized_max_route_switches)
        object.__setattr__(self, "route_switch_query", normalized_route_switch_query)
        object.__setattr__(self, "route_switch_reason", normalized_route_switch_reason)
        object.__setattr__(self, "route_switch_source", normalized_route_switch_source)
        object.__setattr__(self, "answer_source", normalized_answer_source)
        object.__setattr__(self, "judge_verdict", normalized_judge_verdict)
        object.__setattr__(self, "judge_score", normalized_judge_score)
        object.__setattr__(self, "judge_reason", normalized_judge_reason)
        object.__setattr__(self, "judge_followup", normalized_judge_followup)
        object.__setattr__(self, "judge_source", normalized_judge_source)
        object.__setattr__(self, "next_step_action", normalized_next_step_action)
        object.__setattr__(self, "next_step_route", normalized_next_step_route)
        object.__setattr__(self, "next_step_query", normalized_next_step_query)
        object.__setattr__(self, "next_step_reason", normalized_next_step_reason)
        object.__setattr__(self, "next_step_followups", normalized_next_step_followups)
        object.__setattr__(self, "next_step_source", normalized_next_step_source)
        object.__setattr__(self, "error_message", normalized_error_message)

    @property
    def is_completed(self) -> bool:
        """判断当前工作流是否已经形成最终响应。"""

        return self.final_response is not None

    @property
    def has_error(self) -> bool:
        """判断当前状态是否已经记录错误。"""

        return bool(self.error_message)

    @property
    def can_retry(self) -> bool:
        """判断当前状态是否还允许再做一次 repair。"""

        return self.retry_count < self.max_retries

    @property
    def can_switch_route(self) -> bool:
        """判断当前状态是否还允许再做一次自动 route switch。"""

        return self.route_switch_count < self.max_route_switches

    @property
    def effective_query(self) -> str:
        """
        返回当前真正应被工具层执行的查询。

        优先级保持明确：
        1. route switch 之后生成的新 query
        2. repair 之后生成的新 query
        3. rewrite 阶段生成的 query
        4. 原始用户问题
        """

        return (
            self.route_switch_query
            or self.repair_query
            or self.rewritten_query
            or self.query.user_query
        )

    def with_route(self, route: str) -> "AgentState":
        """返回带有路由结果的新状态。"""

        return replace(self, route=route)

    def with_planner_result(
        self,
        *,
        intent: str,
        route: str,
        resolved_action: str | None,
        needs_rag: bool,
        needs_rewrite: bool,
        answer_style: str,
        response_mode: str | None,
        retrieval_goal: str | None,
        focus: str | None,
        answer_plan: dict[str, Any] | None,
        planner_reason: str,
        planner_source: str,
    ) -> "AgentState":
        """返回带有 planner 结果的新状态。"""

        return replace(
            self,
            intent=intent,
            route=route,
            resolved_action=resolved_action or self.resolved_action,
            needs_rag=needs_rag,
            needs_rewrite=needs_rewrite,
            answer_style=answer_style,
            response_mode=response_mode or self.response_mode,
            retrieval_goal=retrieval_goal or self.retrieval_goal,
            focus=focus or self.focus,
            answer_plan=answer_plan or self.answer_plan,
            planner_reason=planner_reason,
            planner_source=planner_source,
        )

    def with_tool_output(self, tool_output: Any) -> "AgentState":
        """返回带有工具输出的新状态。"""

        return replace(self, tool_output=tool_output)

    def with_retrieval_output(self, retrieval_output: Any) -> "AgentState":
        """Preserve the raw retrieval output for post-retrieval strategy branches."""

        return replace(self, retrieval_output=retrieval_output, tool_output=retrieval_output)

    def with_strategy_result(
        self,
        *,
        strategy: str,
        route: str,
        strategy_reason: str,
    ) -> "AgentState":
        """?????????????????"""

        return replace(
            self,
            strategy=strategy,
            route=route,
            strategy_reason=strategy_reason,
        )

    def with_rewrite_result(
        self,
        *,
        rewritten_query: str,
        alternative_queries: tuple[str, ...] = (),
        rewrite_keywords: tuple[str, ...] = (),
        rewrite_reason: str,
        rewrite_source: str,
    ) -> "AgentState":
        """返回带有 rewrite 结果的新状态。"""

        return replace(
            self,
            rewritten_query=rewritten_query,
            alternative_queries=alternative_queries,
            rewrite_keywords=rewrite_keywords,
            rewrite_reason=rewrite_reason,
            rewrite_source=rewrite_source,
        )

    def with_repair_result(
        self,
        *,
        repair_query: str,
        repair_strategy: str,
        repair_reason: str,
        repair_source: str,
    ) -> "AgentState":
        """
        返回带有 repair 结果的新状态。

        这里会顺手清空上一轮执行产生的 tool_output / answer / judge 结果，
        因为接下来我们要基于新的 query 再跑一轮执行链。
        """

        return replace(
            self,
            retry_count=self.retry_count + 1,
            repair_query=repair_query,
            repair_strategy=repair_strategy,
            repair_reason=repair_reason,
            repair_source=repair_source,
            tool_output=None,
            answer_source=None,
            judge_verdict=None,
            judge_score=None,
            judge_reason=None,
            judge_followup=None,
            judge_source=None,
            final_response=None,
            error_message=None,
        )

    def with_route_switch(
        self,
        *,
        route: str,
        route_switch_query: str,
        route_switch_reason: str,
        route_switch_source: str,
        intent: str | None = None,
        answer_style: str | None = None,
    ) -> "AgentState":
        """
        返回带有自动 route switch 结果的新状态。

        这一步和 repair 的区别在于：
        - repair 仍留在原 route 里补救
        - route switch 是明确切到另一条执行主线
        """

        return replace(
            self,
            route=route,
            intent=intent or self.intent,
            answer_style=answer_style or self.answer_style,
            route_switch_count=self.route_switch_count + 1,
            route_switch_query=route_switch_query,
            route_switch_reason=route_switch_reason,
            route_switch_source=route_switch_source,
            tool_output=None,
            answer_source=None,
            judge_verdict=None,
            judge_score=None,
            judge_reason=None,
            judge_followup=None,
            judge_source=None,
            next_step_action=None,
            next_step_route=None,
            next_step_query=None,
            next_step_reason=None,
            next_step_followups=(),
            next_step_source=None,
            final_response=None,
            error_message=None,
        )

    def with_final_response(self, final_response: AgentResponse) -> "AgentState":
        """返回带有最终响应的新状态。"""

        return replace(self, final_response=final_response)

    def with_answer_source(self, answer_source: str) -> "AgentState":
        """返回带有回答来源标记的新状态。"""

        return replace(self, answer_source=answer_source)

    def with_judge_result(
        self,
        *,
        judge_verdict: str,
        judge_score: int,
        judge_reason: str,
        judge_followup: str,
        judge_source: str,
    ) -> "AgentState":
        """返回带有回答评估结果的新状态。"""

        return replace(
            self,
            judge_verdict=judge_verdict,
            judge_score=judge_score,
            judge_reason=judge_reason,
            judge_followup=judge_followup,
            judge_source=judge_source,
        )

    def with_next_step_result(
        self,
        *,
        next_step_action: str,
        next_step_route: str,
        next_step_query: str,
        next_step_reason: str,
        next_step_followups: tuple[str, ...],
        next_step_source: str,
    ) -> "AgentState":
        """返回带有下一步动作建议的新状态。"""

        return replace(
            self,
            next_step_action=next_step_action,
            next_step_route=next_step_route,
            next_step_query=next_step_query,
            next_step_reason=next_step_reason,
            next_step_followups=next_step_followups,
            next_step_source=next_step_source,
        )

    def with_error(self, error_message: str) -> "AgentState":
        """返回带有错误信息的新状态。"""

        return replace(self, error_message=error_message)

    def to_dict(self) -> dict[str, Any]:
        """把当前状态转换成适合调试查看的字典。"""

        serialized_tool_output = self.tool_output
        if hasattr(serialized_tool_output, "to_dict"):
            serialized_tool_output = serialized_tool_output.to_dict()

        return {
            "query": self.query.to_dict(),
            "route": self.route,
            "intent": self.intent,
            "resolved_action": self.resolved_action,
            "needs_rag": self.needs_rag,
            "needs_rewrite": self.needs_rewrite,
            "answer_style": self.answer_style,
            "response_mode": self.response_mode,
            "retrieval_goal": self.retrieval_goal,
            "focus": self.focus,
            "answer_plan": self.answer_plan,
            "planner_reason": self.planner_reason,
            "planner_source": self.planner_source,
            "rewritten_query": self.rewritten_query,
            "alternative_queries": list(self.alternative_queries),
            "rewrite_keywords": list(self.rewrite_keywords),
            "rewrite_reason": self.rewrite_reason,
            "rewrite_source": self.rewrite_source,
            "strategy": self.strategy,
            "strategy_reason": self.strategy_reason,
            "retrieval_output": (
                self.retrieval_output.to_dict()
                if hasattr(self.retrieval_output, "to_dict")
                else self.retrieval_output
            ),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "effective_query": self.effective_query,
            "repair_query": self.repair_query,
            "repair_strategy": self.repair_strategy,
            "repair_reason": self.repair_reason,
            "repair_source": self.repair_source,
            "route_switch_count": self.route_switch_count,
            "max_route_switches": self.max_route_switches,
            "route_switch_query": self.route_switch_query,
            "route_switch_reason": self.route_switch_reason,
            "route_switch_source": self.route_switch_source,
            "tool_output": serialized_tool_output,
            "answer_source": self.answer_source,
            "judge_verdict": self.judge_verdict,
            "judge_score": self.judge_score,
            "judge_reason": self.judge_reason,
            "judge_followup": self.judge_followup,
            "judge_source": self.judge_source,
            "next_step_action": self.next_step_action,
            "next_step_route": self.next_step_route,
            "next_step_query": self.next_step_query,
            "next_step_reason": self.next_step_reason,
            "next_step_followups": list(self.next_step_followups),
            "next_step_source": self.next_step_source,
            "final_response": (
                self.final_response.to_dict() if self.final_response is not None else None
            ),
            "error_message": self.error_message,
            "is_completed": self.is_completed,
            "has_error": self.has_error,
        }
