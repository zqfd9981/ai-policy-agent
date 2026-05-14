from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.retrieval.retriever import (
    PolicyRetriever,
    RetrievalResult,
    load_or_build_default_retriever,
)


DEFAULT_RETRIEVE_TOP_K = 5


@dataclass(frozen=True, slots=True)
class RetrievedPolicyChunk:
    """表示工具层返回的一条可直接引用的政策证据。"""

    rank: int
    score: float
    chunk_id: str
    doc_id: str
    title: str
    title_path: tuple[str, ...]
    title_path_str: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """把证据对象转换成适合上层消费的字典。"""

        return {
            "rank": self.rank,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "title_path": list(self.title_path),
            "title_path_str": self.title_path_str,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RetrievePolicyOutput:
    """表示一次政策检索工具调用的统一输出。"""

    query: str
    top_k: int
    results: tuple[RetrievedPolicyChunk, ...]

    @property
    def result_count(self) -> int:
        """返回当前实际命中的结果数。"""

        return len(self.results)

    def to_dict(self) -> dict[str, Any]:
        """把检索结果转换成适合 JSON 序列化的字典。"""

        return {
            "query": self.query,
            "top_k": self.top_k,
            "result_count": self.result_count,
            "results": [result.to_dict() for result in self.results],
        }


@lru_cache(maxsize=1)
def get_default_retriever() -> PolicyRetriever:
    """
    获取默认检索器。

    这里做一层进程内缓存，避免工具层每次调用都重复加载
    embedding 模型和 FAISS 索引。
    """

    return load_or_build_default_retriever()


def retrieve_policy(
    query: str,
    top_k: int = DEFAULT_RETRIEVE_TOP_K,
    *,
    retriever: PolicyRetriever | None = None,
) -> RetrievePolicyOutput:
    """
    执行一次政策检索，并返回统一的工具层输出。

    这是给 CLI、Agent 和其他工具复用的最小入口：
    - 不重复实现底层索引恢复逻辑
    - 不负责生成自然语言答案
    - 只返回可引用、可继续加工的证据结果
    """

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空。")

    normalized_top_k = max(1, int(top_k))
    active_retriever = retriever or get_default_retriever()
    retrieval_results = active_retriever.search(
        normalized_query,
        top_k=normalized_top_k,
    )

    return RetrievePolicyOutput(
        query=normalized_query,
        top_k=normalized_top_k,
        results=tuple(_build_chunk(result) for result in retrieval_results),
    )


def _build_chunk(result: RetrievalResult) -> RetrievedPolicyChunk:
    """把底层 RetrievalResult 包装成工具层证据对象。"""

    return RetrievedPolicyChunk(
        rank=result.rank,
        score=result.score,
        chunk_id=result.chunk_id,
        doc_id=result.doc_id,
        title=result.title,
        title_path=result.title_path,
        title_path_str=result.title_path_str,
        text=result.text,
        metadata=dict(result.metadata),
    )


class RetrievePolicyTool:
    """给 Agent 或上层流程调用的政策检索工具。"""

    name = "retrieve_policy"
    description = "根据用户 query 检索相关政策 chunk，并返回可引用证据。"

    def __init__(
        self,
        *,
        retriever: PolicyRetriever | None = None,
        default_top_k: int = DEFAULT_RETRIEVE_TOP_K,
    ) -> None:
        self.retriever = retriever
        self.default_top_k = max(1, int(default_top_k))

    def run(self, query: str, top_k: int | None = None) -> RetrievePolicyOutput:
        """执行一次检索。"""

        effective_top_k = self.default_top_k if top_k is None else top_k
        return retrieve_policy(
            query,
            top_k=effective_top_k,
            retriever=self.retriever,
        )

    def __call__(self, query: str, top_k: int | None = None) -> RetrievePolicyOutput:
        """允许把工具对象直接当作可调用对象使用。"""

        return self.run(query, top_k=top_k)


__all__ = [
    "DEFAULT_RETRIEVE_TOP_K",
    "RetrievePolicyOutput",
    "RetrievePolicyTool",
    "RetrievedPolicyChunk",
    "get_default_retriever",
    "retrieve_policy",
]
