from __future__ import annotations

from app.agent.router import ROUTE_COMPARE, ROUTE_RETRIEVE, ROUTE_SUMMARIZE
from app.agent.strategy import (
    STRATEGY_COMPARE,
    STRATEGY_DIRECT_ANSWER,
    STRATEGY_MULTI_DOC_SUMMARY,
    STRATEGY_SINGLE_DOC_SUMMARY,
    aggregate_retrieval_documents,
    choose_retrieval_strategy,
)
from app.retrieval.retriever import RetrievalResult
from app.tools.retrieve_policy import RetrievePolicyOutput


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
    from app.tools.retrieve_policy import RetrievedPolicyChunk

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
    return RetrievePolicyOutput(query="上海 AI 政策", top_k=5, results=chunks)


def test_aggregate_retrieval_documents_groups_by_doc() -> None:
    output = build_output(
        [
            build_result(rank=1, score=0.91, chunk_id="SH001_0001", doc_id="SH001", title="模塑申城"),
            build_result(rank=2, score=0.83, chunk_id="SH001_0002", doc_id="SH001", title="模塑申城"),
            build_result(rank=3, score=0.79, chunk_id="SH002_0001", doc_id="SH002", title="扩大应用若干措施"),
        ]
    )

    candidates = aggregate_retrieval_documents(output)

    assert len(candidates) == 2
    assert candidates[0].doc_id == "SH001"
    assert candidates[0].hit_count == 2
    assert candidates[0].total_score > candidates[1].total_score


def test_summary_intent_can_switch_to_single_doc_summary() -> None:
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

    assert decision.strategy == STRATEGY_SINGLE_DOC_SUMMARY
    assert decision.route == ROUTE_SUMMARIZE
    assert "支持重点" in decision.query


def test_summary_intent_without_clear_single_doc_stays_direct_answer() -> None:
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

    assert decision.strategy == STRATEGY_MULTI_DOC_SUMMARY
    assert decision.route == ROUTE_SUMMARIZE


def test_compare_intent_with_two_docs_switches_to_compare() -> None:
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

    assert decision.strategy == STRATEGY_COMPARE
    assert decision.route == ROUTE_COMPARE
