from __future__ import annotations

from app.agent.state import AgentState
from app.memory.session import SessionMemory, SessionTurn
from app.tools.compare_policy import PolicyCompareOutput
from app.tools.summarize_policies import MultiPolicySummaryOutput
from app.tools.summarize_policy import PolicySummaryOutput


def update_session_memory(
    session_memory: SessionMemory,
    *,
    user_query: str,
    state: AgentState,
) -> SessionMemory:
    """
    Append the latest dialogue turn and refresh working memory.

    Raw turns are stored for traceability, while WorkingMemory only keeps the
    compact state that is helpful for the next-round reasoning.
    """

    session_memory.turns.append(
        SessionTurn(
            role="user",
            content=user_query,
            metadata={"intent": state.intent or "", "strategy": state.strategy or ""},
        )
    )

    if state.final_response is not None:
        session_memory.turns.append(
            SessionTurn(
                role="assistant",
                content=state.final_response.message,
                metadata={
                    "route": state.final_response.route,
                    "judge_verdict": state.judge_verdict or "",
                },
            )
        )

    working_memory = session_memory.working_memory
    working_memory.active_intent = state.intent
    working_memory.active_strategy = state.strategy

    if state.tool_output is not None:
        tool_output = state.tool_output

        if isinstance(tool_output, PolicySummaryOutput):
            # Single-document summary: remember the active policy.
            working_memory.active_doc_id = tool_output.doc_id
            working_memory.active_doc_title = tool_output.title
            working_memory.active_region = str(tool_output.metadata.get("region", "")) or None
            working_memory.summary_scope = "single_doc"

        if isinstance(tool_output, MultiPolicySummaryOutput):
            # Multi-document summary: remember the candidate policy set.
            working_memory.candidate_doc_ids = tool_output.doc_ids
            working_memory.candidate_titles = tool_output.policy_titles
            working_memory.summary_scope = "multi_doc"
            if tool_output.policy_summaries:
                working_memory.active_region = str(
                    tool_output.policy_summaries[0].metadata.get("region", "")
                ) or None

        if isinstance(tool_output, PolicyCompareOutput):
            # Compare flow: remember left/right comparison anchors.
            working_memory.left_doc_id = tool_output.left_summary.doc_id
            working_memory.left_doc_title = tool_output.left_summary.title
            working_memory.right_doc_id = tool_output.right_summary.doc_id
            working_memory.right_doc_title = tool_output.right_summary.title
            working_memory.summary_scope = "compare"

    # Lightweight rule extraction for the current round focus.
    working_memory.active_topic = infer_topic(user_query)
    working_memory.focus_dimension = infer_focus_dimension(user_query)
    working_memory.active_region = working_memory.active_region or infer_region(user_query)

    return session_memory


def infer_topic(query: str) -> str | None:
    """Infer a coarse topic label from the current user query."""

    if "大模型" in query:
        return "大模型政策"
    if "AI" in query or "人工智能" in query:
        return "AI政策"
    return None


def infer_focus_dimension(query: str) -> str | None:
    """Infer which dimension the user is currently focusing on."""

    if "支持" in query:
        return "支持重点"
    if "适用对象" in query:
        return "适用对象"
    if "申报" in query or "条件" in query:
        return "申报条件"
    if "对比" in query or "比较" in query:
        return "对比"
    return None


def infer_region(query: str) -> str | None:
    """Infer the active region from the current user query."""

    for region in ("上海", "北京", "江苏", "浙江", "深圳", "广东", "长三角"):
        if region in query:
            return region
    return None
