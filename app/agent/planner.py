from __future__ import annotations

from dataclasses import dataclass
import os

from pydantic import BaseModel, Field

from app.llm.client import OpenAILLMClient


PLANNER_SYSTEM_PROMPT = """
你是 Policy Agent 的任务规划器。

你的职责只有规划，不要回答用户问题。
请根据用户问题输出结构化决策，字段含义如下：
- intent: retrieve / summarize / compare / match / chat
- needs_rag: 是否需要调用政策知识库或工具
- needs_rewrite: 是否需要先改写 query 再检索
- answer_style: direct / structured / comparative / advisory
- reason: 用一句中文说明判断依据

规划原则：
1. 只要用户问题依赖具体政策事实、政策条款、政策比较、政策摘要，就倾向于 needs_rag=true。
2. “总结/摘要/概括”类请求，intent 设为 summarize。
3. “对比/比较/差异”类请求，intent 设为 compare。
4. “适合我/匹配/推荐关注”类请求，intent 设为 match。
5. 一般的政策事实查询，intent 设为 retrieve。
6. 只有明显不依赖政策库的泛闲聊问题，intent 才设为 chat 且 needs_rag=false。
""".strip()


class PlannerDecisionModel(BaseModel):
    """LLM planner 的结构化输出模式。"""

    intent: str = Field(description="retrieve / summarize / compare / match / chat")
    needs_rag: bool
    needs_rewrite: bool
    answer_style: str = Field(description="direct / structured / comparative / advisory")
    reason: str


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    """供工作流直接消费的 planner 决策结果。"""

    intent: str
    needs_rag: bool
    needs_rewrite: bool
    answer_style: str
    reason: str


class PolicyAgentPlanner:
    """
    基于 LLM 的第一版 planner。

    这层只负责“任务判断”，不直接调用任何工具。
    """

    def __init__(self, *, client: OpenAILLMClient | None = None) -> None:
        self.client = client or OpenAILLMClient()

    @property
    def is_available(self) -> bool:
        """判断当前 planner 是否具备可用 LLM。"""

        return self.client.is_available

    def decide(self, user_query: str) -> PlannerDecision:
        """
        根据用户问题生成结构化规划结果。

        这里故意只返回“规划信息”，不直接改 state，
        这样 planner 既可以被 node 复用，也可以单独做测试。
        """

        normalized_query = user_query.strip()
        if not normalized_query:
            raise ValueError("planner 输入不能为空。")

        parsed = self.client.parse_structured_response(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=f"用户问题：{normalized_query}",
            response_model=PlannerDecisionModel,
            model=os.getenv("PLANNER_MODEL"),
        )
        return PlannerDecision(
            intent=parsed.intent.strip().lower(),
            needs_rag=bool(parsed.needs_rag),
            needs_rewrite=bool(parsed.needs_rewrite),
            answer_style=parsed.answer_style.strip().lower(),
            reason=parsed.reason.strip(),
        )


def plan_query(
    user_query: str,
    *,
    planner: PolicyAgentPlanner | None = None,
) -> PlannerDecision:
    """函数式入口：生成一次 planner 决策。"""

    active_planner = planner or PolicyAgentPlanner()
    return active_planner.decide(user_query)
