from __future__ import annotations

import pytest

from app.retrieval.retriever import RetrievalResult
from app.tools.retrieve_policy import RetrievePolicyTool, retrieve_policy


class StubRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def build_result(*, rank: int, score: float, chunk_id: str, doc_id: str) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=score,
        chunk_id=chunk_id,
        doc_id=doc_id,
        title="示例政策",
        title_path=("一、总体要求",),
        title_path_str="一、总体要求",
        text="支持打造医疗大模型应用场景。",
        metadata={"region": "上海", "policy_type": "实施方案"},
    )


def test_retrieve_policy_wraps_retrieval_results() -> None:
    stub = StubRetriever(
        [
            build_result(rank=1, score=0.92, chunk_id="SH001_0001", doc_id="SH001"),
            build_result(rank=2, score=0.88, chunk_id="SH001_0002", doc_id="SH001"),
        ]
    )

    output = retrieve_policy("  医疗大模型  ", top_k=2, retriever=stub)

    assert stub.calls == [("医疗大模型", 2)]
    assert output.query == "医疗大模型"
    assert output.top_k == 2
    assert output.result_count == 2
    assert output.results[0].chunk_id == "SH001_0001"
    assert output.results[0].title_path == ("一、总体要求",)
    assert output.to_dict()["results"][0]["metadata"]["region"] == "上海"


def test_retrieve_policy_tool_uses_default_top_k() -> None:
    stub = StubRetriever(
        [
            build_result(rank=1, score=0.95, chunk_id="BJ001_0001", doc_id="BJ001"),
            build_result(rank=2, score=0.84, chunk_id="BJ001_0002", doc_id="BJ001"),
        ]
    )
    tool = RetrievePolicyTool(retriever=stub, default_top_k=1)

    output = tool.run("算力券")

    assert stub.calls == [("算力券", 1)]
    assert output.result_count == 1
    assert output.results[0].doc_id == "BJ001"


def test_retrieve_policy_rejects_blank_query() -> None:
    stub = StubRetriever([])

    with pytest.raises(ValueError, match="query 不能为空"):
        retrieve_policy("   ", retriever=stub)
