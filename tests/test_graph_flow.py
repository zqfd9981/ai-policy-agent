from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.agent.graph import PolicyAgentGraph, build_initial_state
from app.agent.nodes import planner_node
from app.models.response import AgentResponse
from app.retrieval.retriever import RetrievalResult
from app.tools.compare_policy import PolicyCompareOutput, PolicyComparisonSection
from app.tools.retrieve_policy import RetrievePolicyOutput, RetrievedPolicyChunk
from app.tools.summarize_policies import (
    MultiPolicySectionSummary,
    MultiPolicySummaryOutput,
)
from app.tools.summarize_policy import PolicySummaryOutput, SummaryEvidence


def build_retrieval_result(
    *,
    rank: int,
    score: float,
    chunk_id: str,
    doc_id: str,
    title: str,
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=score,
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=title,
        title_path=("一、总体要求",),
        title_path_str="一、总体要求",
        text="支持人工智能应用和产业发展。",
        metadata={"region": "上海", "policy_type": "实施方案", "publish_date": "2025/01/01"},
    )


def build_retrieval_output(results: list[RetrievalResult], query: str) -> RetrievePolicyOutput:
    chunks = tuple(
        RetrievedPolicyChunk(
            rank=item.rank,
            score=item.score,
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            title=item.title,
            title_path=item.title_path,
            title_path_str=item.title_path_str,
            text=item.text,
            metadata=item.metadata,
        )
        for item in results
    )
    return RetrievePolicyOutput(query=query, top_k=5, results=chunks)


def build_summary_output(*, query: str, doc_id: str, title: str) -> PolicySummaryOutput:
    evidence = SummaryEvidence(
        section="support_points",
        text="支持重点行业开放人工智能应用场景。",
        chunk_id=f"{doc_id}_0001",
        doc_id=doc_id,
        title_path_str="二、重点任务",
        metadata={"region": "上海", "policy_type": "实施方案"},
    )
    return PolicySummaryOutput(
        query=query,
        doc_id=doc_id,
        title=title,
        metadata={"region": "上海", "publish_date": "2025/01/01", "policy_type": "实施方案"},
        selection_reason=f"根据标题定位到 {doc_id}",
        overview=(),
        support_points=(evidence,),
        target_audiences=(),
        application_conditions=(),
    )


def build_multi_summary_output(query: str) -> MultiPolicySummaryOutput:
    left = build_summary_output(query=query, doc_id="SH001", title="模塑申城")
    right = build_summary_output(query=query, doc_id="SH002", title="扩大应用若干措施")
    section = MultiPolicySectionSummary(
        key="support_points",
        label="支持重点",
        highlights=("模塑申城：支持重点行业开放人工智能应用场景。", "扩大应用若干措施：支持重点行业开放人工智能应用场景。"),
        evidence=left.support_points + right.support_points,
    )
    return MultiPolicySummaryOutput(
        query=query,
        doc_ids=("SH001", "SH002"),
        policy_titles=("模塑申城", "扩大应用若干措施"),
        selection_reason="根据检索结果聚合出 2 篇高相关政策。",
        policy_summaries=(left, right),
        sections=(section,),
    )


def build_compare_output(query: str) -> PolicyCompareOutput:
    left = build_summary_output(query=query, doc_id="BJ001", title="北京大模型措施")
    right = build_summary_output(query=query, doc_id="SH001", title="上海模塑申城")
    section = PolicyComparisonSection(
        key="support_points",
        label="支持重点",
        left_points=("支持重点行业开放人工智能应用场景。",),
        right_points=("支持重点行业开放人工智能应用场景。",),
        comparison_note="两篇政策在该维度都有明确表述。",
    )
    return PolicyCompareOutput(
        query=query,
        selection_reason="根据检索结果确定比较对象。",
        left_summary=left,
        right_summary=right,
        sections=(section,),
    )


class StubPlanner:
    def __init__(self, *, intent: str, needs_rag: bool = True, needs_rewrite: bool = False, answer_style: str = "direct") -> None:
        self.intent = intent
        self.needs_rag = needs_rag
        self.needs_rewrite = needs_rewrite
        self.answer_style = answer_style

    @property
    def is_available(self) -> bool:
        return True

    def decide(self, user_query: str):
        from app.agent.planner import PlannerDecision

        return PlannerDecision(
            intent=self.intent,
            needs_rag=self.needs_rag,
            needs_rewrite=self.needs_rewrite,
            answer_style=self.answer_style,
            reason="stub planner",
        )


@dataclass
class FailingPlanner:
    @property
    def is_available(self) -> bool:
        return True

    def decide(self, user_query: str):
        raise AssertionError("planner should not be called on main path")


class StubRetrieveTool:
    def __init__(self, output: RetrievePolicyOutput) -> None:
        self.output = output
        self.calls: list[tuple[str, int]] = []

    def run(self, query: str, top_k: int | None = None) -> RetrievePolicyOutput:
        self.calls.append((query, top_k or self.output.top_k))
        return self.output


class StubSummarizeTool:
    def __init__(self, output: PolicySummaryOutput) -> None:
        self.output = output
        self.calls: list[tuple[str, int | None]] = []

    def run(self, query: str, *, top_k: int | None = None, **_: object) -> PolicySummaryOutput:
        self.calls.append((query, top_k))
        return self.output


class StubSummarizePoliciesTool:
    def __init__(self, output: MultiPolicySummaryOutput) -> None:
        self.output = output
        self.calls: list[str] = []

    def run(self, query: str, **_: object) -> MultiPolicySummaryOutput:
        self.calls.append(query)
        return self.output


class StubCompareTool:
    def __init__(self, output: PolicyCompareOutput) -> None:
        self.output = output
        self.calls: list[str] = []

    def run(self, query: str, **_: object) -> PolicyCompareOutput:
        self.calls.append(query)
        return self.output


class StubJudge:
    def __init__(self, verdict: str = "pass", score: int = 90) -> None:
        self.verdict = verdict
        self.score = score

    @property
    def is_available(self) -> bool:
        return True

    def judge(self, **_: object):
        from app.agent.judge import JudgeDecision

        return JudgeDecision(
            verdict=self.verdict,
            score=self.score,
            grounded=True,
            reason="stub judge",
            followup="",
        )


class StubNextStepPlanner:
    @property
    def is_available(self) -> bool:
        return False


class GraphFlowTests(unittest.TestCase):
    def test_graph_main_path_no_longer_depends_on_planner(self) -> None:
        retrieval_output = build_retrieval_output(
            [build_retrieval_result(rank=1, score=0.93, chunk_id="SH001_0001", doc_id="SH001", title="模塑申城")],
            query="上海有哪些AI政策",
        )
        graph = PolicyAgentGraph(
            rewriter=None,
            answerer=None,
            judge=StubJudge(),
            next_step_planner=StubNextStepPlanner(),
            retrieve_tool=StubRetrieveTool(retrieval_output),
        )

        state = graph.run(
            build_initial_state(
                "上海有哪些AI政策",
                resolved_action="retrieve",
                needs_rag=True,
                needs_rewrite=False,
                answer_style="direct",
                response_mode="direct_answer",
                retrieval_goal="multi_policy_topic",
            ).query
        )

        self.assertEqual(state.resolved_action, "retrieve")
        self.assertIsInstance(state.final_response, AgentResponse)

    def test_planner_prefers_resolver_structured_result(self) -> None:
        state = build_initial_state(
            "这两个地方哪个更好",
            resolved_action="compare",
            response_mode="scenario_advice_compare",
            retrieval_goal="compare_regions",
            focus="location_choice",
        )

        next_state = planner_node(
            state,
            planner=StubPlanner(intent="retrieve", needs_rag=False, needs_rewrite=False),
            supported_routes=frozenset({"retrieve", "summarize", "compare"}),
        )

        self.assertEqual(next_state.intent, "compare")
        self.assertEqual(next_state.resolved_action, "compare")
        self.assertEqual(next_state.route, "retrieve")
        self.assertEqual(next_state.response_mode, "scenario_advice_compare")
        self.assertEqual(next_state.retrieval_goal, "compare_regions")
        self.assertEqual(next_state.planner_source, "resolver")

    def test_direct_answer_flow_keeps_retrieval_output(self) -> None:
        retrieval_output = build_retrieval_output(
            [build_retrieval_result(rank=1, score=0.93, chunk_id="SH001_0001", doc_id="SH001", title="模塑申城")],
            query="上海有哪些AI政策",
        )
        graph = PolicyAgentGraph(
            retrieve_tool=StubRetrieveTool(retrieval_output),
            judge=StubJudge(),
            next_step_planner=StubNextStepPlanner(),
        )

        state = graph.run("上海有哪些AI政策")

        self.assertEqual(state.strategy, "direct_answer")
        self.assertEqual(state.route, "retrieve")
        self.assertIsInstance(state.retrieval_output, RetrievePolicyOutput)
        self.assertIsInstance(state.final_response, AgentResponse)

    def test_single_doc_summary_flow_switches_after_retrieval(self) -> None:
        retrieval_output = build_retrieval_output(
            [
                build_retrieval_result(rank=1, score=0.95, chunk_id="SH002_0001", doc_id="SH002", title="扩大应用若干措施"),
                build_retrieval_result(rank=2, score=0.90, chunk_id="SH002_0002", doc_id="SH002", title="扩大应用若干措施"),
            ],
            query="总结扩大应用若干措施",
        )
        summary_output = build_summary_output(query="总结扩大应用若干措施", doc_id="SH002", title="扩大应用若干措施")
        summarize_tool = StubSummarizeTool(summary_output)
        graph = PolicyAgentGraph(
            retrieve_tool=StubRetrieveTool(retrieval_output),
            summarize_tool=summarize_tool,
            judge=StubJudge(),
            next_step_planner=StubNextStepPlanner(),
        )

        state = graph.run("总结扩大应用若干措施")

        self.assertEqual(state.strategy, "single_doc_summary")
        self.assertEqual(state.route, "summarize")
        self.assertIsInstance(state.tool_output, PolicySummaryOutput)
        self.assertEqual(len(summarize_tool.calls), 1)

    def test_multi_doc_summary_flow_uses_summarize_policies_tool(self) -> None:
        retrieval_output = build_retrieval_output(
            [
                build_retrieval_result(rank=1, score=0.91, chunk_id="SH001_0001", doc_id="SH001", title="模塑申城"),
                build_retrieval_result(rank=2, score=0.89, chunk_id="SH002_0001", doc_id="SH002", title="扩大应用若干措施"),
            ],
            query="总结一下上海的AI政策",
        )
        multi_output = build_multi_summary_output("总结一下上海的AI政策")
        summarize_policies_tool = StubSummarizePoliciesTool(multi_output)
        graph = PolicyAgentGraph(
            retrieve_tool=StubRetrieveTool(retrieval_output),
            summarize_policies_tool=summarize_policies_tool,
            judge=StubJudge(),
            next_step_planner=StubNextStepPlanner(),
        )

        state = graph.run("总结一下上海的AI政策")

        self.assertEqual(state.strategy, "multi_doc_summary")
        self.assertEqual(state.route, "summarize")
        self.assertIsInstance(state.tool_output, MultiPolicySummaryOutput)
        self.assertEqual(len(summarize_policies_tool.calls), 1)

    def test_compare_flow_switches_after_retrieval(self) -> None:
        retrieval_output = build_retrieval_output(
            [
                build_retrieval_result(rank=1, score=0.92, chunk_id="BJ001_0001", doc_id="BJ001", title="北京大模型措施"),
                build_retrieval_result(rank=2, score=0.90, chunk_id="SH001_0001", doc_id="SH001", title="上海模塑申城"),
            ],
            query="比较北京和上海的大模型政策",
        )
        compare_output = build_compare_output("比较北京和上海的大模型政策")
        compare_tool = StubCompareTool(compare_output)
        graph = PolicyAgentGraph(
            retrieve_tool=StubRetrieveTool(retrieval_output),
            compare_tool=compare_tool,
            judge=StubJudge(),
            next_step_planner=StubNextStepPlanner(),
        )

        state = graph.run("比较北京和上海的大模型政策")

        self.assertEqual(state.strategy, "compare")
        self.assertEqual(state.route, "compare")
        self.assertIsInstance(state.tool_output, PolicyCompareOutput)
        self.assertEqual(len(compare_tool.calls), 1)


if __name__ == "__main__":
    unittest.main()
