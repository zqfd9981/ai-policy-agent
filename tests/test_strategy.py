from __future__ import annotations

import unittest

from app.agent.router import ROUTE_COMPARE, ROUTE_SUMMARIZE
from app.agent.strategy import (
    STRATEGY_COMPARE,
    STRATEGY_MULTI_DOC_SUMMARY,
    STRATEGY_SINGLE_DOC_SUMMARY,
    aggregate_retrieval_documents,
    choose_retrieval_strategy,
)
from app.agent.state import AgentState
from app.models.query import AgentQuery
from app.models.metadata import PolicyMetadata
from app.retrieval.retriever import RetrievalResult
from app.tools.retrieve_policy import RetrievePolicyOutput, RetrievedPolicyChunk
from app.tools.compare_policy import score_metadata_candidate


def build_result(
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
        metadata={"region": "上海", "policy_type": "实施方案"},
    )


def build_output(results: list[RetrievalResult]) -> RetrievePolicyOutput:
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
    return RetrievePolicyOutput(query="上海AI政策", top_k=5, results=chunks)


class StrategyTests(unittest.TestCase):
    def test_region_compare_prefers_broader_core_policy_over_narrow_topic_policy(self) -> None:
        broad = PolicyMetadata(
            doc_id="BJ001",
            title="北京市推动“人工智能+”行动计划（2024-2025年）",
            region="北京",
            level="市级",
            issuer="北京市发展和改革委员会等",
            publish_date="2024/7/26",
            policy_type="行动计划",
            theme="人工智能+应用",
            tier="core",
            status="official_text",
            source_format="txt",
            notes="北京AI应用总纲",
        )
        narrow = PolicyMetadata(
            doc_id="BJ003",
            title="北京市加快人工智能赋能科学研究高质量发展行动计划（2025—2027年）",
            region="北京",
            level="市级",
            issuer="北京市相关主管部门",
            publish_date="2025/7/22",
            policy_type="行动计划",
            theme="AI+Science",
            tier="supplement",
            status="official_text",
            source_format="txt",
            notes="偏AI+Science专题",
        )

        broad_score = score_metadata_candidate("比较北京和上海的AI政策", broad)
        narrow_score = score_metadata_candidate("比较北京和上海的AI政策", narrow)

        self.assertGreater(broad_score, narrow_score)

    def test_aggregate_retrieval_documents_groups_by_doc(self) -> None:
        output = build_output(
            [
                build_result(rank=1, score=0.91, chunk_id="SH001_0001", doc_id="SH001", title="模塑申城"),
                build_result(rank=2, score=0.83, chunk_id="SH001_0002", doc_id="SH001", title="模塑申城"),
                build_result(rank=3, score=0.79, chunk_id="SH002_0001", doc_id="SH002", title="扩大应用若干措施"),
            ]
        )

        candidates = aggregate_retrieval_documents(output)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].doc_id, "SH001")
        self.assertEqual(candidates[0].hit_count, 2)
        self.assertGreater(candidates[0].total_score, candidates[1].total_score)

    def test_summary_intent_can_switch_to_single_doc_summary(self) -> None:
        output = build_output(
            [
                build_result(rank=1, score=0.95, chunk_id="SH002_0001", doc_id="SH002", title="扩大应用若干措施"),
                build_result(rank=2, score=0.88, chunk_id="SH002_0002", doc_id="SH002", title="扩大应用若干措施"),
                build_result(rank=3, score=0.70, chunk_id="SH003_0001", doc_id="SH003", title="AI+制造实施方案"),
            ]
        )

        decision = choose_retrieval_strategy(
            intent=ROUTE_SUMMARIZE,
            user_query="请总结扩大应用若干措施",
            retrieval_output=output,
        )

        self.assertEqual(decision.strategy, STRATEGY_SINGLE_DOC_SUMMARY)
        self.assertEqual(decision.route, ROUTE_SUMMARIZE)

    def test_summary_intent_without_clear_single_doc_uses_multi_doc_summary(self) -> None:
        output = build_output(
            [
                build_result(rank=1, score=0.89, chunk_id="SH001_0001", doc_id="SH001", title="模塑申城"),
                build_result(rank=2, score=0.86, chunk_id="SH002_0001", doc_id="SH002", title="扩大应用若干措施"),
                build_result(rank=3, score=0.84, chunk_id="SH003_0001", doc_id="SH003", title="AI+制造实施方案"),
            ]
        )

        decision = choose_retrieval_strategy(
            intent=ROUTE_SUMMARIZE,
            user_query="总结一下上海的AI政策",
            retrieval_output=output,
        )

        self.assertEqual(decision.strategy, STRATEGY_MULTI_DOC_SUMMARY)
        self.assertEqual(decision.route, ROUTE_SUMMARIZE)

    def test_compare_intent_with_two_docs_switches_to_compare(self) -> None:
        output = build_output(
            [
                build_result(rank=1, score=0.92, chunk_id="BJ001_0001", doc_id="BJ001", title="北京大模型措施"),
                build_result(rank=2, score=0.90, chunk_id="SH001_0001", doc_id="SH001", title="上海模塑申城"),
            ]
        )

        decision = choose_retrieval_strategy(
            intent=ROUTE_COMPARE,
            user_query="比较北京和上海的大模型政策",
            retrieval_output=output,
        )

        self.assertEqual(decision.strategy, STRATEGY_COMPARE)
        self.assertEqual(decision.route, ROUTE_COMPARE)

    def test_retrieval_goal_can_drive_compare_without_relying_on_query_keywords(self) -> None:
        output = build_output(
            [
                build_result(rank=1, score=0.92, chunk_id="BJ001_0001", doc_id="BJ001", title="北京大模型措施"),
                build_result(rank=2, score=0.90, chunk_id="SH001_0001", doc_id="SH001", title="上海模塑申城"),
            ]
        )

        decision = choose_retrieval_strategy(
            intent="retrieve",
            user_query="北京和上海哪个更适合",
            retrieval_output=output,
            retrieval_goal="compare_regions",
            resolved_entities=("北京", "上海"),
        )

        self.assertEqual(decision.strategy, STRATEGY_COMPARE)
        self.assertEqual(decision.route, ROUTE_COMPARE)

    def test_region_compare_prefers_resolved_entities_even_if_retrieval_is_single_sided(self) -> None:
        output = build_output(
            [
                build_result(rank=1, score=0.92, chunk_id="SH002_0001", doc_id="SH002", title="上海应用支持"),
                build_result(rank=2, score=0.90, chunk_id="SH002_0002", doc_id="SH002", title="上海应用支持"),
            ]
        )

        decision = choose_retrieval_strategy(
            intent="compare",
            user_query="北京和上海哪个更适合",
            retrieval_output=output,
            retrieval_goal="compare_regions",
            resolved_entities=("上海", "北京"),
        )

        self.assertEqual(decision.strategy, STRATEGY_COMPARE)
        self.assertEqual(decision.route, ROUTE_COMPARE)

    def test_retrieval_goal_can_drive_multi_doc_summary_without_keywords(self) -> None:
        output = build_output(
            [
                build_result(rank=1, score=0.89, chunk_id="SH001_0001", doc_id="SH001", title="模塑申城"),
                build_result(rank=2, score=0.86, chunk_id="SH002_0001", doc_id="SH002", title="扩大应用若干措施"),
                build_result(rank=3, score=0.84, chunk_id="SH003_0001", doc_id="SH003", title="AI+制造实施方案"),
            ]
        )

        decision = choose_retrieval_strategy(
            intent="retrieve",
            user_query="上海AI政策",
            retrieval_output=output,
            retrieval_goal="multi_policy_region",
        )

        self.assertEqual(decision.strategy, STRATEGY_MULTI_DOC_SUMMARY)
        self.assertEqual(decision.route, ROUTE_SUMMARIZE)

    def test_strategy_result_does_not_override_rewritten_query(self) -> None:
        state = AgentState(
            query=AgentQuery("总结一下上海的AI政策"),
            rewritten_query="上海近两年大模型应用支持政策汇总",
        )

        updated = state.with_strategy_result(
            strategy=STRATEGY_MULTI_DOC_SUMMARY,
            route=ROUTE_SUMMARIZE,
            strategy_reason="范围摘要",
        )

        self.assertEqual(updated.rewritten_query, "上海近两年大模型应用支持政策汇总")


if __name__ == "__main__":
    unittest.main()
