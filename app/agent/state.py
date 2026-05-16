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
    - tool_output: 工具层返回的原始结果
    - answer_source: 最终回答来自 LLM 还是规则回退
    - judge_verdict / judge_score / judge_reason / judge_followup: 回答评估结果
    - final_response: 最终给用户的响应
    - error_message: 工作流中记录的错误信息
    """

    query: AgentQuery
    route: str | None = None
    intent: str | None = None
    needs_rag: bool | None = None
    needs_rewrite: bool | None = None
    answer_style: str | None = None
    planner_reason: str | None = None
    planner_source: str | None = None
    rewritten_query: str | None = None
    alternative_queries: tuple[str, ...] = ()
    rewrite_keywords: tuple[str, ...] = ()
    rewrite_reason: str | None = None
    rewrite_source: str | None = None
    tool_output: Any | None = None
    answer_source: str | None = None
    judge_verdict: str | None = None
    judge_score: int | None = None
    judge_reason: str | None = None
    judge_followup: str | None = None
    judge_source: str | None = None
    final_response: AgentResponse | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        """统一可选字符串字段的格式。"""

        normalized_route = self.route.strip().lower() if self.route else None
        normalized_intent = self.intent.strip().lower() if self.intent else None
        normalized_answer_style = self.answer_style.strip().lower() if self.answer_style else None
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
        normalized_answer_source = self.answer_source.strip().lower() if self.answer_source else None
        normalized_judge_verdict = self.judge_verdict.strip().lower() if self.judge_verdict else None
        normalized_judge_reason = self.judge_reason.strip() if self.judge_reason else None
        normalized_judge_followup = self.judge_followup.strip() if self.judge_followup else None
        normalized_judge_source = self.judge_source.strip().lower() if self.judge_source else None
        normalized_judge_score = (
            max(0, min(100, int(self.judge_score))) if self.judge_score is not None else None
        )
        normalized_error_message = self.error_message.strip() if self.error_message else None

        object.__setattr__(self, "route", normalized_route)
        object.__setattr__(self, "intent", normalized_intent)
        object.__setattr__(self, "answer_style", normalized_answer_style)
        object.__setattr__(self, "planner_reason", normalized_planner_reason)
        object.__setattr__(self, "planner_source", normalized_planner_source)
        object.__setattr__(self, "rewritten_query", normalized_rewritten_query)
        object.__setattr__(self, "alternative_queries", normalized_alternative_queries)
        object.__setattr__(self, "rewrite_keywords", normalized_rewrite_keywords)
        object.__setattr__(self, "rewrite_reason", normalized_rewrite_reason)
        object.__setattr__(self, "rewrite_source", normalized_rewrite_source)
        object.__setattr__(self, "answer_source", normalized_answer_source)
        object.__setattr__(self, "judge_verdict", normalized_judge_verdict)
        object.__setattr__(self, "judge_score", normalized_judge_score)
        object.__setattr__(self, "judge_reason", normalized_judge_reason)
        object.__setattr__(self, "judge_followup", normalized_judge_followup)
        object.__setattr__(self, "judge_source", normalized_judge_source)
        object.__setattr__(self, "error_message", normalized_error_message)

    @property
    def is_completed(self) -> bool:
        """判断当前工作流是否已经形成最终响应。"""

        return self.final_response is not None

    @property
    def has_error(self) -> bool:
        """判断当前状态是否已经记录错误。"""

        return bool(self.error_message)

    def with_route(self, route: str) -> "AgentState":
        """返回带有路由结果的新状态。"""

        return replace(self, route=route)

    def with_planner_result(
        self,
        *,
        intent: str,
        route: str,
        needs_rag: bool,
        needs_rewrite: bool,
        answer_style: str,
        planner_reason: str,
        planner_source: str,
    ) -> "AgentState":
        """返回带有 planner 结果的新状态。"""

        return replace(
            self,
            intent=intent,
            route=route,
            needs_rag=needs_rag,
            needs_rewrite=needs_rewrite,
            answer_style=answer_style,
            planner_reason=planner_reason,
            planner_source=planner_source,
        )

    def with_tool_output(self, tool_output: Any) -> "AgentState":
        """返回带有工具输出的新状态。"""

        return replace(self, tool_output=tool_output)

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
            "needs_rag": self.needs_rag,
            "needs_rewrite": self.needs_rewrite,
            "answer_style": self.answer_style,
            "planner_reason": self.planner_reason,
            "planner_source": self.planner_source,
            "rewritten_query": self.rewritten_query,
            "alternative_queries": list(self.alternative_queries),
            "rewrite_keywords": list(self.rewrite_keywords),
            "rewrite_reason": self.rewrite_reason,
            "rewrite_source": self.rewrite_source,
            "tool_output": serialized_tool_output,
            "answer_source": self.answer_source,
            "judge_verdict": self.judge_verdict,
            "judge_score": self.judge_score,
            "judge_reason": self.judge_reason,
            "judge_followup": self.judge_followup,
            "judge_source": self.judge_source,
            "final_response": (
                self.final_response.to_dict() if self.final_response is not None else None
            ),
            "error_message": self.error_message,
            "is_completed": self.is_completed,
            "has_error": self.has_error,
        }
