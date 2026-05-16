from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.agent.router import ROUTE_COMPARE, ROUTE_RETRIEVE, ROUTE_SUMMARIZE
from app.agent.state import AgentState
from app.llm.client import OpenAILLMClient
from app.models.response import AgentResponse
from app.tools.compare_policy import PolicyCompareOutput
from app.tools.retrieve_policy import RetrievePolicyOutput
from app.tools.summarize_policy import PolicySummaryOutput


NEXT_STEP_SYSTEM_PROMPT = """
你是 Policy Agent 的后续动作决策器。

你的职责不是重新回答用户，而是在当前回答结束后，
判断下一步最合适的动作是什么。

请输出结构化结果，字段含义如下：
- action: none / ask_followup / auto_switch_route / suggest_route
- target_route: retrieve / summarize / compare / none
- next_query: 如果要自动切路由，给出新的执行 query；否则返回空字符串
- reason: 用一句中文说明判断依据
- followups: 如果需要追问或建议下一步，给出 1 到 3 条中文建议

决策原则：
1. 如果当前回答已经 pass，通常 action=none。
2. 如果 retrieve 已经锁定到单篇政策，但回答仍 weak/fail，可以自动切到 summarize。
3. 如果当前结果里天然包含多篇政策，且用户可能更关心差异，可以建议 compare。
4. 如果当前信息仍不足以继续执行，就给出具体追问建议，而不是空泛建议。
5. 不要编造未出现过的政策名称或结论。
""".strip()


class NextStepDecisionModel(BaseModel):
    """LLM next-step 的结构化输出模式。"""

    action: str = Field(description="none / ask_followup / auto_switch_route / suggest_route")
    target_route: str = Field(description="retrieve / summarize / compare / none")
    next_query: str = ""
    reason: str
    followups: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class NextStepDecision:
    """供工作流消费的后续动作决策结果。"""

    action: str
    target_route: str
    next_query: str
    reason: str
    followups: tuple[str, ...]


class PolicyAgentNextStepPlanner:
    """
    基于 LLM 的后续动作规划器。

    这层只负责判断“接下来怎么办”，不直接执行工具。
    """

    def __init__(self, *, client: OpenAILLMClient | None = None) -> None:
        self.client = client or OpenAILLMClient()

    @property
    def is_available(self) -> bool:
        """判断当前环境是否具备可用 LLM。"""

        return self.client.is_available

    def decide(self, state: AgentState) -> NextStepDecision:
        """基于最终状态判断下一步动作。"""

        parsed = self.client.parse_structured_response(
            system_prompt=NEXT_STEP_SYSTEM_PROMPT,
            user_prompt=build_next_step_context(state),
            response_model=NextStepDecisionModel,
        )
        return NextStepDecision(
            action=parsed.action.strip().lower(),
            target_route=parsed.target_route.strip().lower(),
            next_query=parsed.next_query.strip(),
            reason=parsed.reason.strip(),
            followups=tuple(item.strip() for item in parsed.followups if item.strip()),
        )


def build_next_step_context(state: AgentState) -> str:
    """把 next-step 决策所需上下文压缩成一段文本。"""

    lines = [
        f"用户问题：{state.query.user_query}",
        f"当前路由：{state.route or 'unknown'}",
        f"当前意图：{state.intent or 'unknown'}",
        f"当前有效查询：{state.effective_query}",
        f"judge_verdict：{state.judge_verdict or 'unknown'}",
        f"judge_score：{state.judge_score if state.judge_score is not None else 'unknown'}",
        f"judge_reason：{state.judge_reason or ''}",
        f"repair_count：{state.retry_count}/{state.max_retries}",
        f"route_switch_count：{state.route_switch_count}/{state.max_route_switches}",
        "",
        "证据概览：",
        describe_next_step_evidence(state),
    ]
    return "\n".join(lines)


def describe_next_step_evidence(state: AgentState) -> str:
    """把当前证据压缩成后续动作判断足够用的概览。"""

    tool_output = state.tool_output

    if isinstance(tool_output, RetrievePolicyOutput):
        if tool_output.result_count == 0:
            return "retrieve: 当前没有命中政策结果。"

        lines = [f"retrieve: 当前命中 {tool_output.result_count} 条结果。"]
        for item in tool_output.results[:3]:
            lines.append(f"- {item.doc_id} | {item.title} | score={item.score:.3f}")
        return "\n".join(lines)

    if isinstance(tool_output, PolicySummaryOutput):
        return (
            f"summarize: 当前已定位政策 {tool_output.title} ({tool_output.doc_id})，"
            f"摘要证据数为 {tool_output.citation_count}。"
        )

    if isinstance(tool_output, PolicyCompareOutput):
        return (
            f"compare: 当前正在对比 {tool_output.left_summary.doc_id} 与 "
            f"{tool_output.right_summary.doc_id}，总引用数为 {tool_output.citation_count}。"
        )

    return "unknown: 当前没有稳定的结构化证据。"


def fallback_next_step(state: AgentState) -> NextStepDecision:
    """
    在未启用 LLM 时，使用规则判断下一步动作。

    这里的目标不是“完美自动化”，而是让系统在 weak/fail 后
    至少能做出更像 Agent 的后续动作分流：
    - 能自动切 summarize 时就自动切
    - 否则给出追问建议
    - 多文档场景下额外给 compare 建议
    """

    if state.judge_verdict == "pass":
        return NextStepDecision(
            action="none",
            target_route="none",
            next_query="",
            reason="当前回答已经达标，不需要额外后续动作。",
            followups=(),
        )

    if should_auto_switch_to_summary(state):
        summary_query = build_summary_switch_query(state)
        return NextStepDecision(
            action="auto_switch_route",
            target_route=ROUTE_SUMMARIZE,
            next_query=summary_query,
            reason="当前检索已基本锁定到单篇政策，但直接检索回答仍偏弱，适合自动切到摘要模式。",
            followups=(),
        )

    if should_suggest_compare(state):
        return NextStepDecision(
            action="suggest_route",
            target_route=ROUTE_COMPARE,
            next_query="",
            reason="当前结果中涉及多篇政策，更适合继续做差异对比而不是停留在单点检索。",
            followups=build_compare_followups(state),
        )

    return NextStepDecision(
        action="ask_followup",
        target_route=state.route or "none",
        next_query="",
        reason="当前回答仍不够稳，下一步更适合通过追问聚焦需求后再继续。",
        followups=build_followup_questions(state),
    )


def should_auto_switch_to_summary(state: AgentState) -> bool:
    """判断当前状态是否值得自动切到 summarize。"""

    if state.route != ROUTE_RETRIEVE:
        return False

    if state.judge_verdict not in {"weak", "fail"}:
        return False

    if not state.can_switch_route:
        return False

    if not isinstance(state.tool_output, RetrievePolicyOutput):
        return False

    return dominant_retrieval_doc(state.tool_output) is not None


def dominant_retrieval_doc(output: RetrievePolicyOutput) -> str | None:
    """判断检索结果里是否存在可稳定切到摘要的一篇主导政策。"""

    doc_counts: dict[str, int] = {}
    for item in output.results:
        doc_counts[item.doc_id] = doc_counts.get(item.doc_id, 0) + 1

    if not doc_counts:
        return None

    best_doc_id, best_count = max(doc_counts.items(), key=lambda pair: pair[1])

    # 一旦 top 结果主导性足够明显，就允许切摘要。
    if best_count >= 2 or len(doc_counts) == 1:
        return best_doc_id

    top_doc_id = output.results[0].doc_id
    return top_doc_id if best_doc_id == top_doc_id else None


def build_summary_switch_query(state: AgentState) -> str:
    """为自动切 summarize 构造一个更适合摘要执行的新 query。"""

    if isinstance(state.tool_output, RetrievePolicyOutput) and state.tool_output.results:
        base_title = state.tool_output.results[0].title
    else:
        base_title = state.query.user_query

    return " ".join(
        [
            base_title.strip(),
            "政策概览",
            "支持重点",
            "适用对象",
            "申报条件",
        ]
    ).strip()


def should_suggest_compare(state: AgentState) -> bool:
    """判断当前状态是否更适合建议 compare。"""

    if state.route == ROUTE_COMPARE:
        return False

    if state.route != ROUTE_RETRIEVE:
        return False

    if not isinstance(state.tool_output, RetrievePolicyOutput):
        return False

    distinct_doc_ids = {item.doc_id for item in state.tool_output.results}
    return len(distinct_doc_ids) >= 2 and state.judge_verdict in {"weak", "fail"}


def build_compare_followups(state: AgentState) -> tuple[str, ...]:
    """为 compare 建议构造更像下一步动作的话术。"""

    if isinstance(state.tool_output, RetrievePolicyOutput):
        titles: list[str] = []
        for item in state.tool_output.results:
            if item.title not in titles:
                titles.append(item.title)
            if len(titles) >= 2:
                break

        if len(titles) >= 2:
            return (
                f"如果你想看差异，我可以继续对比《{titles[0]}》和《{titles[1]}》。",
                "你也可以先指定最关心的比较维度，例如支持重点、适用对象或申报条件。",
            )

    return (
        "如果你想看多篇政策的差异，我下一步可以继续进入 compare 流程。",
        "你可以顺手补一句最关心的比较维度，例如地区差异、支持力度或适用对象。",
    )


def build_followup_questions(state: AgentState) -> tuple[str, ...]:
    """根据当前路由与结果形态，生成更具体的追问建议。"""

    if state.route == ROUTE_COMPARE:
        return (
            "你可以直接指定两篇想比较的政策名称，我会继续做定向对比。",
            "你也可以补充比较维度，例如支持重点、适用对象、申报条件或地区差异。",
        )

    if state.route == ROUTE_SUMMARIZE:
        return (
            "你可以指定最关心的摘要分区，例如支持重点、适用对象或申报条件。",
            "如果你已经有目标政策标题，也可以直接告诉我，我会围绕那一篇继续展开。",
        )

    if isinstance(state.tool_output, RetrievePolicyOutput) and state.tool_output.result_count == 0:
        return (
            "你可以补充地区信息，例如北京、上海、苏州等。",
            "你也可以补充政策对象或业务场景，例如医院、制造业、算力平台、大模型应用。",
        )

    return (
        "你可以告诉我更关心哪一篇政策，我可以继续做定向摘要。",
        "你也可以补充想看的维度，例如支持重点、适用对象、申报条件或时间范围。",
    )


def render_next_step_guidance(decision: NextStepDecision) -> str:
    """把 next-step 决策渲染成适合追加到最终回答里的提示文本。"""

    if decision.action == "none":
        return ""

    lines = ["", "下一步建议："]

    if decision.action == "suggest_route" and decision.target_route == ROUTE_COMPARE:
        lines.append("当前结果更适合继续做政策对比。")
    elif decision.action == "ask_followup":
        lines.append("为了继续缩小范围，可以补充以下信息：")
    else:
        lines.append("你可以继续这样推进：")

    for index, item in enumerate(decision.followups, start=1):
        lines.append(f"{index}. {item}")

    return "\n".join(lines).strip()


def append_next_step_guidance(
    response: AgentResponse,
    decision: NextStepDecision,
) -> AgentResponse:
    """
    把后续动作建议追加到最终回答里。

    只在“不自动继续执行”的情况下追加，
    这样用户在 CLI 或 API 里都能直接看到下一步建议。
    """

    guidance_text = render_next_step_guidance(decision)
    if not guidance_text:
        return response

    return AgentResponse(
        success=response.success,
        route=response.route,
        message=f"{response.message}\n\n{guidance_text}",
        citations=response.citations,
        error_message=response.error_message,
    )
