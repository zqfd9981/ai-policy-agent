from __future__ import annotations

import unittest

from app.agent.judge import JudgeDecision
from app.agent.next_step import NextStepDecision
from app.agent.nodes import answer_node, judge_node, next_step_node
from app.agent.state import AgentState
from app.models.query import AgentQuery
from app.models.response import AgentResponse
from app.tools.retrieve_policy import RetrievePolicyOutput


class StubAnswerer:
    @property
    def is_available(self) -> bool:
        return True

    def answer(self, **_: object):
        from app.agent.answer import AnswerDraft

        return AnswerDraft(
            message="LLM answer",
            citations=(),
            source="llm",
        )


class StubJudge:
    @property
    def is_available(self) -> bool:
        return True

    def judge(self, **_: object) -> JudgeDecision:
        return JudgeDecision(
            verdict="pass",
            score=95,
            grounded=True,
            reason="LLM judge",
            followup="",
        )


class StubNextStepPlanner:
    @property
    def is_available(self) -> bool:
        return True

    def decide(self, state: AgentState) -> NextStepDecision:
        return NextStepDecision(
            action="none",
            target_route="none",
            next_query="",
            reason="LLM next step",
            followups=(),
        )


class LLMNodeTests(unittest.TestCase):
    def test_answer_node_prefers_llm_answerer(self) -> None:
        state = AgentState(
            query=AgentQuery("上海有哪些AI政策"),
            route="retrieve",
            tool_output=RetrievePolicyOutput(query="上海有哪些AI政策", top_k=3, results=()),
        )

        next_state = answer_node(state, answerer=StubAnswerer())

        self.assertEqual(next_state.answer_source, "llm")
        self.assertEqual(next_state.final_response.message, "LLM answer")

    def test_judge_node_prefers_llm_judge(self) -> None:
        state = AgentState(
            query=AgentQuery("上海有哪些AI政策"),
            route="retrieve",
            tool_output=RetrievePolicyOutput(query="上海有哪些AI政策", top_k=3, results=()),
            final_response=AgentResponse(success=True, route="retrieve", message="ok"),
        )

        next_state = judge_node(state, judge=StubJudge())

        self.assertEqual(next_state.judge_source, "llm")
        self.assertEqual(next_state.judge_verdict, "pass")

    def test_next_step_node_prefers_llm_planner(self) -> None:
        state = AgentState(
            query=AgentQuery("上海有哪些AI政策"),
            route="retrieve",
            final_response=AgentResponse(success=True, route="retrieve", message="ok"),
            judge_verdict="pass",
            judge_score=95,
            judge_reason="ok",
        )

        next_state = next_step_node(state, planner=StubNextStepPlanner())

        self.assertEqual(next_state.next_step_source, "llm")
        self.assertEqual(next_state.next_step_action, "none")


if __name__ == "__main__":
    unittest.main()
