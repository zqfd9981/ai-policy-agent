from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.agent.answer import extract_readable_snippet
from app.llm.client import OpenAILLMClient
from app.models.response import AgentResponse
from app.tools.compare_policy import PolicyCompareOutput
from app.tools.retrieve_policy import RetrievePolicyOutput
from app.tools.summarize_policies import MultiPolicySummaryOutput
from app.tools.summarize_policy import PolicySummaryOutput


JUDGE_SYSTEM_PROMPT = """
你是 Policy Agent 的回答评估器。

你的职责不是重新回答用户，而是判断当前回答质量是否达标。

请基于用户问题、工具证据和最终回答，输出结构化评估结果，字段含义如下：
- verdict: pass / weak / fail
- score: 0 到 100 的整数
- grounded: 回答是否有明确证据支撑
- reason: 用一句中文说明判断依据
- followup: 如果回答还不够好，给一句下一步改进建议；如果已达标，可返回空字符串

评估原则：
1. 如果回答明显脱离证据、编造事实，verdict=fail。
2. 如果回答有证据，但不够完整、过于简略、可读性一般，verdict=weak。
3. 如果回答和用户问题基本对齐，且能被当前证据支撑，verdict=pass。
4. 不要因为缺少完美答案就苛刻打低分，要基于当前工具能力做务实判断。
""".strip()


class JudgeDecisionModel(BaseModel):
    """LLM judge 的结构化输出模式。"""

    verdict: str = Field(description="pass / weak / fail")
    score: int = Field(ge=0, le=100)
    grounded: bool
    reason: str
    followup: str = ""


@dataclass(frozen=True, slots=True)
class JudgeDecision:
    """供工作流直接消费的回答评估结果。"""

    verdict: str
    score: int
    grounded: bool
    reason: str
    followup: str


class PolicyAgentJudge:
    """
    基于 LLM 的第一版回答评估器。

    当前只负责做“回答后自检”，不直接控制重试逻辑。
    """

    def __init__(self, *, client: OpenAILLMClient | None = None) -> None:
        self.client = client or OpenAILLMClient()

    @property
    def is_available(self) -> bool:
        """判断当前环境是否具备可用 LLM。"""

        return self.client.is_available

    def judge(
        self,
        *,
        user_query: str,
        intent: str | None,
        tool_output: Any,
        final_response: AgentResponse,
    ) -> JudgeDecision:
        """基于当前证据和最终回答生成一次结构化评估。"""

        if not final_response.message.strip():
            raise ValueError("judge 输入缺少最终回答内容。")

        context_text = build_judge_context(
            user_query=user_query,
            intent=intent,
            tool_output=tool_output,
            final_response=final_response,
        )
        parsed = self.client.parse_structured_response(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=context_text,
            response_model=JudgeDecisionModel,
        )
        return JudgeDecision(
            verdict=parsed.verdict.strip().lower(),
            score=max(0, min(100, int(parsed.score))),
            grounded=bool(parsed.grounded),
            reason=parsed.reason.strip(),
            followup=parsed.followup.strip(),
        )


def build_judge_context(
    *,
    user_query: str,
    intent: str | None,
    tool_output: Any,
    final_response: AgentResponse,
) -> str:
    """把评估所需上下文压缩成一段结构化文本。"""

    lines = [
        f"用户问题：{user_query}",
        f"任务类型：{intent or 'unknown'}",
        f"回答路由：{final_response.route}",
        f"回答是否成功：{final_response.success}",
        f"回答引用数：{final_response.citation_count}",
        "",
        "最终回答：",
        final_response.message,
        "",
        "证据摘要：",
        describe_tool_output(tool_output),
    ]
    return "\n".join(lines)


def describe_tool_output(tool_output: Any) -> str:
    """把工具输出压缩成适合 judge 消费的证据概览。"""

    if isinstance(tool_output, RetrievePolicyOutput):
        if tool_output.result_count == 0:
            return "retrieve: 未命中任何政策片段。"

        lines = [f"retrieve: 命中 {tool_output.result_count} 条结果。"]
        for item in tool_output.results[:3]:
            lines.append(
                f"- {item.doc_id} | {item.title} | score={item.score:.3f} | "
                f"path={item.title_path_str or 'N/A'}"
            )
        return "\n".join(lines)

    if isinstance(tool_output, PolicySummaryOutput):
        return (
            f"summarize: 已定位政策 {tool_output.title} ({tool_output.doc_id})，"
            f"共抽取 {tool_output.citation_count} 条摘要证据。"
        )

    if isinstance(tool_output, MultiPolicySummaryOutput):
        return (
            f"multi_summary: 宸叉眹鎬?{len(tool_output.policy_summaries)} 绡囨斂绛栵紝"
            f"鎬诲紩鐢ㄦ暟涓?{tool_output.citation_count}銆?"
        )

    if isinstance(tool_output, PolicyCompareOutput):
        return (
            f"compare: 正在对比 {tool_output.left_summary.title} "
            f"({tool_output.left_summary.doc_id}) 与 {tool_output.right_summary.title} "
            f"({tool_output.right_summary.doc_id})，总引用数为 {tool_output.citation_count}。"
        )

    return "unknown: 当前没有可识别的工具证据。"


def fallback_judge(
    *,
    tool_output: Any,
    final_response: AgentResponse,
) -> JudgeDecision:
    """在未启用 LLM 时，使用轻量规则做回答质量兜底评估。"""

    if not final_response.success:
        return JudgeDecision(
            verdict="fail",
            score=0,
            grounded=False,
            reason="最终回答本身执行失败，当前结果不可用。",
            followup="先修复当前节点错误，再重新执行完整链路。",
        )

    if isinstance(tool_output, RetrievePolicyOutput):
        return judge_retrieval_answer(tool_output, final_response)

    if isinstance(tool_output, PolicySummaryOutput):
        return judge_summary_answer(tool_output, final_response)

    if isinstance(tool_output, MultiPolicySummaryOutput):
        return judge_multi_policy_summary_answer(tool_output, final_response)

    if isinstance(tool_output, PolicyCompareOutput):
        return judge_compare_answer(tool_output, final_response)

    grounded = final_response.citation_count > 0
    return JudgeDecision(
        verdict="weak" if grounded else "fail",
        score=60 if grounded else 20,
        grounded=grounded,
        reason="当前已生成回答，但缺少更稳定的工具证据类型判断。",
        followup="建议补充更明确的工具输出，再评估回答质量。",
    )


def judge_retrieval_answer(
    tool_output: RetrievePolicyOutput,
    final_response: AgentResponse,
) -> JudgeDecision:
    """规则评估 retrieve 分支回答质量。"""

    if tool_output.result_count == 0:
        return JudgeDecision(
            verdict="weak",
            score=45,
            grounded=True,
            reason="回答如实反映了未命中证据的情况，但暂时无法直接满足查询目标。",
            followup="建议补充地区、对象、政策主题等关键词后再次检索。",
        )

    has_enough_text = len(final_response.message.strip()) >= 40
    has_citations = final_response.citation_count > 0
    has_readable_support = retrieval_has_readable_support(tool_output)
    grounded = has_citations

    # 当检索结果里能拿到可读片段或明确标题路径时，我们才更有底气给 pass。
    # 否则即使“查到了东西”，也更像一次待修复的中间结果。
    if has_enough_text and has_citations and has_readable_support:
        return JudgeDecision(
            verdict="pass",
            score=86,
            grounded=True,
            reason="回答已经结合检索证据组织出可读结果，并保留了引用支撑。",
            followup="如果用户需要更深入信息，可继续对命中政策做摘要或条件提取。",
        )

    return JudgeDecision(
        verdict="weak",
        score=58 if not has_readable_support else 65,
        grounded=grounded,
        reason=(
            "回答和检索结果基本一致，但当前命中的证据片段可读性偏弱，适合再做一次聚焦检索。"
            if not has_readable_support
            else "回答与检索结果基本一致，但仍偏简略，信息组织还可以更充分。"
        ),
        followup="建议补充政策要点提炼，或进一步聚焦单篇政策继续分析。",
    )


def judge_summary_answer(
    tool_output: PolicySummaryOutput,
    final_response: AgentResponse,
) -> JudgeDecision:
    """规则评估 summarize 分支回答质量。"""

    has_citations = final_response.citation_count > 0
    has_structure = all(
        section_label in final_response.message
        for section_label in ("政策概览", "支持重点", "适用对象", "申报条件")
    )
    has_readable_support = summary_has_readable_support(tool_output)

    if has_citations and has_structure and has_readable_support:
        return JudgeDecision(
            verdict="pass",
            score=90,
            grounded=True,
            reason="回答已经形成结构化摘要，且和当前政策证据基本对齐。",
            followup="如果用户继续追问，可围绕某一分区再做更细的定向展开。",
        )

    return JudgeDecision(
        verdict="weak",
        score=55 if not has_readable_support else 68 if has_citations else 50,
        grounded=has_citations,
        reason=(
            "回答已经切到摘要模式，但当前摘要证据可读性偏弱，更适合继续追问或指定更明确的政策文本。"
            if not has_readable_support
            else "回答已基本完成摘要任务，但结构完整性或证据支撑还不够稳定。"
            if has_citations
            else "回答已尝试摘要，但当前证据支撑偏弱。"
        ),
        followup="建议补充更明确的目标政策或增加摘要证据后再生成回答。",
    )


def retrieval_has_readable_support(tool_output: RetrievePolicyOutput) -> bool:
    """
    判断 retrieve 结果是否包含足够“能看”的证据。

    repair 是否值得触发，核心不只是“有没有命中”，
    还要看命中的文本是不是已经能支撑一个像样回答。
    """

    for item in tool_output.results[:3]:
        if item.title_path_str.strip():
            return True
        if extract_readable_snippet(item.text):
            return True
    return False


def summary_has_readable_support(tool_output: PolicySummaryOutput) -> bool:
    """判断摘要证据里是否至少有一部分正文是可读的。"""

    for item in tool_output.all_citations[:6]:
        if extract_readable_snippet(item.text):
            return True
    return False


def judge_multi_policy_summary_answer(
    tool_output: MultiPolicySummaryOutput,
    final_response: AgentResponse,
) -> JudgeDecision:
    """Rule-based evaluation for multi-policy summary answers."""

    has_citations = final_response.citation_count > 0
    has_multiple_policies = len(tool_output.policy_summaries) >= 2
    has_structure = any(section.label in final_response.message for section in tool_output.sections[:2])

    if has_citations and has_multiple_policies and has_structure:
        return JudgeDecision(
            verdict="pass",
            score=88,
            grounded=True,
            reason="回答已经形成多政策汇总结构，并保留了跨文档证据支撑。",
            followup="如果用户继续追问，可进一步下钻到单篇政策摘要或做地区间比较。",
        )

    return JudgeDecision(
        verdict="weak",
        score=64 if has_citations else 50,
        grounded=has_citations,
        reason="回答已进入多文档汇总模式，但覆盖完整性或结构化程度仍可继续加强。",
        followup="建议补充更明确的地区、主题或时间范围后继续汇总。",
    )


def judge_compare_answer(
    tool_output: PolicyCompareOutput,
    final_response: AgentResponse,
) -> JudgeDecision:
    """规则评估 compare 分支回答质量。"""

    has_citations = final_response.citation_count > 0
    has_titles = (
        tool_output.left_summary.title in final_response.message
        and tool_output.right_summary.title in final_response.message
    )
    has_structure = all(section.label in final_response.message for section in tool_output.sections[:2])

    if has_citations and has_titles and has_structure:
        return JudgeDecision(
            verdict="pass",
            score=88,
            grounded=True,
            reason="回答已经形成双政策对比结构，并保留了可追溯的证据支撑。",
            followup="如果用户继续追问，可进一步聚焦某个比较维度，例如支持重点或适用对象。",
        )

    return JudgeDecision(
        verdict="weak",
        score=62 if has_citations else 48,
        grounded=has_citations,
        reason="回答已经进入对比模式，但当前对比结构或证据完整性仍可继续加强。",
        followup="建议补充更明确的比较对象或比较维度后继续执行 compare。",
    )
