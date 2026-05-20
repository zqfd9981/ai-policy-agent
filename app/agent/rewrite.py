from __future__ import annotations

from dataclasses import dataclass
import os

from pydantic import BaseModel, Field

from app.llm.client import OpenAILLMClient


REWRITE_SYSTEM_PROMPT = """
你是 Policy Agent 的查询改写器。

你的任务不是回答问题，而是把用户问题改写成更适合政策检索的表达。

输出要求：
1. primary_query: 最核心的一条检索查询
2. alternative_queries: 0 到 3 条备选查询
3. keywords: 提炼出的核心关键词
4. rewrite_reason: 用一句中文说明为什么这样改写

改写原则：
1. 保留地区、主题、政策对象、任务目标等关键信息。
2. 删除口语化和多余修饰，让 query 更适合向量检索。
3. 如果用户是总结/摘要类请求，可以把 query 改写成“政策标题/主题 + 支持重点/适用对象/申报条件”等更检索友好的形式。
4. 不要凭空编造不存在的政策名称或 doc_id。
""".strip()


class RewriteDecisionModel(BaseModel):
    """LLM rewrite 的结构化输出模式。"""

    primary_query: str = Field(description="最核心的一条检索查询")
    alternative_queries: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    rewrite_reason: str


@dataclass(frozen=True, slots=True)
class RewriteDecision:
    """供工作流直接消费的改写结果。"""

    primary_query: str
    alternative_queries: tuple[str, ...]
    keywords: tuple[str, ...]
    rewrite_reason: str


class PolicyAgentRewriter:
    """
    基于 LLM 的第一版 query rewriter。

    当前先聚焦最小能力：
    - 产出一条主查询
    - 最多补几条备选查询
    - 提供关键词线索
    """

    def __init__(self, *, client: OpenAILLMClient | None = None) -> None:
        self.client = client or OpenAILLMClient()

    @property
    def is_available(self) -> bool:
        """判断当前环境是否具备可用 LLM。"""

        return self.client.is_available

    def rewrite(self, user_query: str, *, intent: str | None = None) -> RewriteDecision:
        """
        对用户问题做结构化改写。

        这里保留 intent 作为附加上下文，让模型知道当前是在做
        检索、摘要还是其他任务，从而生成更贴合场景的检索 query。
        """

        normalized_query = user_query.strip()
        if not normalized_query:
            raise ValueError("rewrite 输入不能为空。")

        prompt_parts = [f"用户问题：{normalized_query}"]
        if intent:
            prompt_parts.append(f"当前任务类型：{intent}")

        parsed = self.client.parse_structured_response(
            system_prompt=REWRITE_SYSTEM_PROMPT,
            user_prompt="\n".join(prompt_parts),
            response_model=RewriteDecisionModel,
            model=os.getenv("REWRITE_MODEL"),
        )

        alternative_queries = tuple(
            item.strip()
            for item in parsed.alternative_queries
            if item.strip() and item.strip() != parsed.primary_query.strip()
        )
        keywords = tuple(item.strip() for item in parsed.keywords if item.strip())

        return RewriteDecision(
            primary_query=parsed.primary_query.strip(),
            alternative_queries=alternative_queries,
            keywords=keywords,
            rewrite_reason=parsed.rewrite_reason.strip(),
        )


def rewrite_query(
    user_query: str,
    *,
    intent: str | None = None,
    rewriter: PolicyAgentRewriter | None = None,
) -> RewriteDecision:
    """函数式入口：生成一次 query rewrite 结果。"""

    active_rewriter = rewriter or PolicyAgentRewriter()
    return active_rewriter.rewrite(user_query, intent=intent)
