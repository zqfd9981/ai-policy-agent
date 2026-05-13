from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 兼容直接运行 `python app\retrieval\retriever.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.chunk.chunk_builder import DEFAULT_CHUNK_OUTPUT_PATH, build_and_export_chunks
from app.models.chunk import Chunk
from app.retrieval.embedder import SimpleTfidfEmbedder
from app.retrieval.vector_store import InMemoryVectorStore, VectorSearchResult


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """表示最终暴露给上层工具或 Agent 的检索结果。"""

    rank: int
    score: float
    chunk_id: str
    doc_id: str
    title: str
    title_path: tuple[str, ...]
    title_path_str: str
    text: str
    metadata: dict[str, Any]


def load_chunk_payloads_from_jsonl(chunk_jsonl_path: Path | str) -> list[dict[str, Any]]:
    """从 jsonl 文件中读取 chunk 载荷。"""

    normalized_path = Path(chunk_jsonl_path)
    if not normalized_path.exists():
        raise FileNotFoundError(f"chunk jsonl 不存在: {normalized_path}")

    payloads: list[dict[str, Any]] = []
    with normalized_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            payloads.append(json.loads(stripped_line))

    return payloads


def serialize_chunks(chunks: list[Chunk]) -> list[dict[str, Any]]:
    """把 Chunk 对象列表转成可检索的字典载荷。"""

    return [chunk.to_dict() for chunk in chunks]


def build_retrieval_text(payload: dict[str, Any]) -> str:
    """
    构造参与检索的文本。

    这里把文档标题、标题路径和正文拼起来，
    让查询既能命中正文，也能命中结构标题。
    """

    title = str(payload.get("title", "")).strip()
    title_path_str = str(payload.get("title_path_str", "")).strip()
    text = str(payload.get("text", "")).strip()

    parts = [part for part in (title, title_path_str, text) if part]
    return "\n".join(parts)


class PolicyRetriever:
    """第一版政策检索器。"""

    def __init__(
        self,
        *,
        embedder: SimpleTfidfEmbedder,
        vector_store: InMemoryVectorStore,
        payloads: list[dict[str, Any]],
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.payloads = payloads

    @classmethod
    def from_chunk_payloads(
        cls,
        payloads: list[dict[str, Any]],
        *,
        max_features: int = 8000,
        min_df: int = 1,
    ) -> "PolicyRetriever":
        """基于 chunk 载荷列表构建检索器。"""

        retrieval_texts = [build_retrieval_text(payload) for payload in payloads]
        embedder = SimpleTfidfEmbedder(max_features=max_features, min_df=min_df)
        embeddings = embedder.fit_transform(retrieval_texts)

        vector_store = InMemoryVectorStore()
        vector_store.add(payloads, embeddings)
        return cls(embedder=embedder, vector_store=vector_store, payloads=payloads)

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        *,
        max_features: int = 8000,
        min_df: int = 1,
    ) -> "PolicyRetriever":
        """直接基于 Chunk 对象列表构建检索器。"""

        return cls.from_chunk_payloads(
            serialize_chunks(chunks),
            max_features=max_features,
            min_df=min_df,
        )

    @classmethod
    def from_jsonl(
        cls,
        chunk_jsonl_path: Path | str = DEFAULT_CHUNK_OUTPUT_PATH,
        *,
        max_features: int = 8000,
        min_df: int = 1,
    ) -> "PolicyRetriever":
        """从 chunk jsonl 文件构建检索器。"""

        payloads = load_chunk_payloads_from_jsonl(chunk_jsonl_path)
        return cls.from_chunk_payloads(
            payloads,
            max_features=max_features,
            min_df=min_df,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """对单条查询返回 top-k 检索结果。"""

        normalized_query = query.strip()
        if not normalized_query:
            return []

        query_vector = self.embedder.encode_query(normalized_query)
        vector_results = self.vector_store.search(query_vector, top_k=top_k)
        return [self._build_result(result) for result in vector_results]

    def _build_result(self, result: VectorSearchResult) -> RetrievalResult:
        """把底层向量检索结果包装成更上层可用的对象。"""

        item = result.item
        return RetrievalResult(
            rank=result.rank,
            score=result.score,
            chunk_id=str(item.get("chunk_id", "")),
            doc_id=str(item.get("doc_id", "")),
            title=str(item.get("title", "")),
            title_path=tuple(item.get("title_path", [])),
            title_path_str=str(item.get("title_path_str", "")),
            text=str(item.get("text", "")),
            metadata=dict(item.get("metadata", {})),
        )


def build_default_retriever(
    *,
    chunk_jsonl_path: Path | str = DEFAULT_CHUNK_OUTPUT_PATH,
    ensure_chunk_file: bool = True,
    max_features: int = 8000,
    min_df: int = 1,
) -> PolicyRetriever:
    """
    构建默认检索器。

    如果 chunk jsonl 还不存在，就先自动构建一份全量 chunk 文件。
    """

    normalized_path = Path(chunk_jsonl_path)
    if ensure_chunk_file and not normalized_path.exists():
        build_and_export_chunks(output_path=normalized_path)

    return PolicyRetriever.from_jsonl(
        normalized_path,
        max_features=max_features,
        min_df=min_df,
    )


def main() -> None:
    """对 retrieval 第一版做一个简单联调测试。"""

    print("=" * 60)
    print("开始测试 retrieval 第一版...")
    print("=" * 60)

    retriever = build_default_retriever()
    sample_queries = [
        "医疗服务 人工智能 大模型",
        "高质量数据集 数据要素",
        "制造业 AI 场景",
    ]

    for query in sample_queries:
        print(f"\n查询: {query}")
        results = retriever.search(query, top_k=3)

        for result in results:
            print(
                f"[{result.rank}] score={result.score:.4f} | "
                f"{result.doc_id} | {result.title_path or ('<前置信息>',)}"
            )
            print(result.text[:100])
            print("-" * 40)

    print("[OK] retrieval 第一版测试通过")


if __name__ == "__main__":
    main()
