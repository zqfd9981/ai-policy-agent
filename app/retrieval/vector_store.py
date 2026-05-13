from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """表示一次向量检索返回的单条结果。"""

    rank: int
    score: float
    item: dict[str, Any]


class InMemoryVectorStore:
    """
    一个最轻量的内存向量库。

    当前职责很单纯：
    - 保存条目和对应向量
    - 对查询向量做相似度搜索
    - 返回 top-k 结果
    """

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None

    @property
    def size(self) -> int:
        """返回当前已存条目数量。"""

        return len(self._items)

    @property
    def dimension(self) -> int:
        """返回当前向量维度。"""

        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[1])

    def add(self, items: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        批量写入条目与向量。

        这里假设 embeddings 已经按行归一化，
        后续可以直接用点积作为余弦相似度。
        """

        if len(items) != len(embeddings):
            raise ValueError(
                "items 与 embeddings 数量不一致："
                f"{len(items)} != {len(embeddings)}"
            )

        if embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维矩阵。")

        self._items = list(items)
        self._embeddings = embeddings.astype(np.float32, copy=False)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[VectorSearchResult]:
        """对单条查询向量做 top-k 检索。"""

        if self._embeddings is None or not self._items:
            return []

        if query_vector.ndim != 1:
            raise ValueError("query_vector 必须是一维向量。")

        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                "query_vector 维度与向量库不一致："
                f"{query_vector.shape[0]} != {self.dimension}"
            )

        normalized_top_k = max(1, min(top_k, self.size))
        scores = self._embeddings @ query_vector

        sorted_indices = np.argsort(-scores)[:normalized_top_k]
        results: list[VectorSearchResult] = []

        for rank, index in enumerate(sorted_indices, start=1):
            results.append(
                VectorSearchResult(
                    rank=rank,
                    score=float(scores[index]),
                    item=self._items[int(index)],
                )
            )

        return results
