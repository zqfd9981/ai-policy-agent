from __future__ import annotations

import re
from dataclasses import dataclass
import os

from pydantic import BaseModel, Field

from app.agent.state import AgentState
from app.agent.router import ROUTE_RETRIEVE, ROUTE_SUMMARIZE
from app.llm.client import OpenAILLMClient
from app.tools.retrieve_policy import RetrievePolicyOutput
from app.tools.summarize_policies import MultiPolicySummaryOutput
from app.tools.summarize_policy import PolicySummaryOutput


REPAIR_SYSTEM_PROMPT = """
你是 Policy Agent 的修复规划器。

你的职责不是直接回答用户，而是在回答质量不够好时，
判断当前是否值得重试，以及下一次应使用什么检索/摘要 query。

请输出结构化结果，字段含义如下：
- should_retry: 是否值得再执行一次
- repair_strategy: none / use_alternative_query / refocus_retrieve / broaden_retrieve / expand_summary
- repaired_query: 如果需要重试，给出新的 query；否则返回空字符串
- repair_reason: 用一句中文说明为什么这样修复

修复原则：
1. 只有在当前回答是 weak / fail，且仍有改进空间时才 should_retry=true。
2. repaired_query 必须基于原问题、已有证据、已有政策标题或摘要信息，不要编造政策名称。
3. 如果当前结果太泛，可以聚焦到命中的高相关政策标题或标题路径。
4. 如果当前摘要不够完整，可以把 query 改写成“政策标题 + 政策概览/支持重点/适用对象/申报条件”。
5. 如果当前 route 本身未实现，或再次重试也没有明显价值，应返回 should_retry=false。
""".strip()


class RepairDecisionModel(BaseModel):
    """LLM repair 的结构化输出模式。"""

    should_retry: bool
    repair_strategy: str = Field(
        description="none / use_alternative_query / refocus_retrieve / broaden_retrieve / expand_summary"
    )
    repaired_query: str = ""
    repair_reason: str


@dataclass(frozen=True, slots=True)
class RepairDecision:
    """供工作流直接消费的 repair 决策结果。"""

    should_retry: bool
    repair_strategy: str
    repaired_query: str
    repair_reason: str


class PolicyAgentRepairer:
    """
    基于 LLM 的第一版 repairer。

    它只负责一件事：
    - 在 judge 认为回答偏 weak/fail 时，决定要不要再跑一次
    """

    def __init__(self, *, client: OpenAILLMClient | None = None) -> None:
        self.client = client or OpenAILLMClient()

    @property
    def is_available(self) -> bool:
        """判断当前环境是否具备可用 LLM。"""

        return self.client.is_available

    def repair(self, state: AgentState) -> RepairDecision:
        """基于当前状态生成一次 repair 决策。"""

        context_text = build_repair_context(state)
        parsed = self.client.parse_structured_response(
            system_prompt=REPAIR_SYSTEM_PROMPT,
            user_prompt=context_text,
            response_model=RepairDecisionModel,
            model=os.getenv("REPAIR_MODEL"),
        )
        return RepairDecision(
            should_retry=bool(parsed.should_retry),
            repair_strategy=parsed.repair_strategy.strip().lower(),
            repaired_query=parsed.repaired_query.strip(),
            repair_reason=parsed.repair_reason.strip(),
        )


def build_repair_context(state: AgentState) -> str:
    """把 repair 所需上下文压缩成一段结构化文本。"""

    lines = [
        f"用户问题：{state.query.user_query}",
        f"任务类型：{state.intent or 'unknown'}",
        f"当前路由：{state.route or 'unknown'}",
        f"当前有效查询：{state.effective_query}",
        f"当前重试次数：{state.retry_count}/{state.max_retries}",
        f"judge_verdict：{state.judge_verdict or 'unknown'}",
        f"judge_reason：{state.judge_reason or ''}",
        f"judge_followup：{state.judge_followup or ''}",
    ]

    if state.alternative_queries:
        lines.append(f"可用备选查询：{' | '.join(state.alternative_queries)}")

    lines.extend(
        [
            "",
            "证据概览：",
            describe_repair_evidence(state),
        ]
    )
    return "\n".join(lines)


def describe_repair_evidence(state: AgentState) -> str:
    """把当前证据压缩成 repair 足够用的摘要。"""

    tool_output = state.tool_output

    if isinstance(tool_output, RetrievePolicyOutput):
        if tool_output.result_count == 0:
            return "retrieve: 当前未命中任何政策结果。"

        lines = [f"retrieve: 当前命中 {tool_output.result_count} 条结果。"]
        for item in tool_output.results[:3]:
            lines.append(
                f"- {item.doc_id} | {item.title} | score={item.score:.3f} | "
                f"path={item.title_path_str or 'N/A'}"
            )
        return "\n".join(lines)

    if isinstance(tool_output, PolicySummaryOutput):
        return (
            f"summarize: 当前已定位政策 {tool_output.title} ({tool_output.doc_id})，"
            f"摘要引用数为 {tool_output.citation_count}。"
        )

    if isinstance(tool_output, MultiPolicySummaryOutput):
        return (
            f"multi_summary: 当前已汇总 {len(tool_output.policy_summaries)} 篇政策，"
            f"总引用数为 {tool_output.citation_count}。"
        )

    return "unknown: 当前没有可供修复利用的结构化证据。"


def fallback_repair(state: AgentState) -> RepairDecision:
    """
    在未启用 LLM 时，使用规则做一次轻量 repair 决策。

    这里刻意保持克制：
    - 只允许最多一次 retry
    - 只在 retrieve / summarize 两条已实现主线里尝试修复
    - 如果没有明显更优的 query，就宁可不重试
    """

    if not should_attempt_repair(state):
        return RepairDecision(
            should_retry=False,
            repair_strategy="none",
            repaired_query="",
            repair_reason="当前回答已达标，或当前场景不适合继续重试。",
        )

    # 优先复用 rewrite 阶段已经准备好的 alternative query。
    # 这样 repair 不是另起炉灶，而是在已有 agent 决策上继续前进。
    alternative_query = choose_alternative_query(state)
    if alternative_query is not None:
        return RepairDecision(
            should_retry=True,
            repair_strategy="use_alternative_query",
            repaired_query=alternative_query,
            repair_reason="当前回答仍可改进，优先尝试 rewrite 阶段生成的备选查询。",
        )

    if state.route == ROUTE_SUMMARIZE:
        repaired_query = build_summary_repair_query(state)
        if repaired_query and repaired_query != state.effective_query:
            return RepairDecision(
                should_retry=True,
                repair_strategy="expand_summary",
                repaired_query=repaired_query,
                repair_reason="当前摘要不够稳定，改用更明确的摘要分区 query 再执行一次。",
            )

    if state.route == ROUTE_RETRIEVE:
        repaired_query, repair_strategy = build_retrieval_repair_query(state)
        if repaired_query and repaired_query != state.effective_query:
            return RepairDecision(
                should_retry=True,
                repair_strategy=repair_strategy,
                repaired_query=repaired_query,
                repair_reason="当前检索结果仍偏泛或证据可读性较弱，尝试聚焦或放宽 query 再检索一次。",
            )

    return RepairDecision(
        should_retry=False,
        repair_strategy="none",
        repaired_query="",
        repair_reason="当前虽然可继续重试，但规则层没有找到更明确的修复方向。",
    )


def should_attempt_repair(state: AgentState) -> bool:
    """判断当前状态是否值得进入 repair 流程。"""

    if state.route not in {ROUTE_RETRIEVE, ROUTE_SUMMARIZE}:
        return False

    if not state.can_retry:
        return False

    if state.final_response is None or not state.final_response.success:
        return False

    return state.judge_verdict in {"weak", "fail"}


def choose_alternative_query(state: AgentState) -> str | None:
    """从 rewrite 生成的 alternative queries 里挑一条还没真正用过的。"""

    for item in state.alternative_queries:
        normalized_item = item.strip()
        if normalized_item and normalized_item != state.effective_query:
            return normalized_item
    return None


def build_summary_repair_query(state: AgentState) -> str:
    """为 summarize 分支构造一个更明确的摘要型 query。"""

    tool_output = state.tool_output
    if isinstance(tool_output, MultiPolicySummaryOutput):
        return state.query.user_query.strip()
    if isinstance(tool_output, PolicySummaryOutput):
        base_title = tool_output.title.strip()
    else:
        base_title = state.query.user_query.strip()

    parts = [
        base_title,
        "政策概览",
        "支持重点",
        "适用对象",
        "申报条件",
    ]
    return " ".join(deduplicate_query_parts(parts))


def build_retrieval_repair_query(state: AgentState) -> tuple[str, str]:
    """为 retrieve 分支构造更聚焦或更宽松的一次重试 query。"""

    tool_output = state.tool_output

    # 如果已经命中了结果，但 judge 仍给 weak/fail，
    # 一般意味着“查到了，但太泛了”。这时优先把 query 聚焦到 top result。
    if isinstance(tool_output, RetrievePolicyOutput) and tool_output.results:
        top_result = tool_output.results[0]
        parts = [
            top_result.title,
            top_result.title_path_str.replace(">", " ") if top_result.title_path_str else "",
            *extract_query_keywords(state.query.user_query),
        ]
        focused_query = " ".join(deduplicate_query_parts(parts))

        # 如果聚焦后的 query 和当前 query 基本一样，说明用户本来就已经很聚焦了。
        # 这时补几个“政策维度词”，让第二次检索尽量避开纯标题命中的首段噪声。
        if focused_query == state.effective_query:
            focused_query = " ".join(
                deduplicate_query_parts(
                    [
                        state.effective_query,
                        "支持重点",
                        "适用对象",
                    ]
                )
            )

        return focused_query, "refocus_retrieve"

    # 如果没有稳定证据可利用，就退回到更宽一点的主题检索表达。
    parts = [
        *extract_query_keywords(state.query.user_query),
        "政策",
        "支持措施",
    ]
    return " ".join(deduplicate_query_parts(parts)), "broaden_retrieve"


def extract_query_keywords(text: str, *, limit: int = 5) -> tuple[str, ...]:
    """从原始问题里提炼一批轻量关键词，供规则 repair 拼 query。"""

    cleaned_text = text.strip()
    if not cleaned_text:
        return ()

    # 这里不做复杂分词，只提取中文词块和英文数字串，
    # 目标是让 repair 在没有额外依赖时也能稳定工作。
    candidates = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9+\-]{2,}", cleaned_text)
    stopwords = {
        "请问",
        "请帮我",
        "帮我",
        "一下",
        "相关政策",
        "政策",
        "情况",
        "内容",
        "介绍",
        "总结",
        "概括",
        "摘要",
    }

    keywords: list[str] = []
    for item in candidates:
        normalized_item = item.strip()
        if not normalized_item or normalized_item in stopwords:
            continue
        if normalized_item not in keywords:
            keywords.append(normalized_item)
        if len(keywords) >= limit:
            break

    if keywords:
        return tuple(keywords)

    return (cleaned_text,)


def deduplicate_query_parts(parts: list[str]) -> tuple[str, ...]:
    """去重并清洗 query 片段，避免 repair 后的 query 太重复。"""

    cleaned_parts: list[str] = []
    seen: set[str] = set()

    for item in parts:
        normalized_item = " ".join(item.strip().split())
        if not normalized_item or normalized_item in seen:
            continue
        seen.add(normalized_item)
        cleaned_parts.append(normalized_item)

    return tuple(cleaned_parts)
