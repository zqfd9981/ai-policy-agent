from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """表示一次向量检索返回的单条结果。"""

    rank: int
    score: float
    item: dict[str, Any]


class FaissVectorStore:
    """
    基于 FAISS 的正式版向量库。

    当前使用 `IndexFlatIP`：
    - 要求输入向量已做 L2 归一化
    - 用内积来近似余弦相似度
    - 实现简单、稳定，适合作为项目当前阶段的正式检索底座
    """

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._index: faiss.IndexFlatIP | None = None
        self._dimension = 0

    @property
    def size(self) -> int:
        """返回当前已入库条目数量。"""

        return len(self._items)

    @property
    def dimension(self) -> int:
        """返回向量维度。"""

        return self._dimension

    def add(self, items: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        """批量写入条目和向量。"""

        if len(items) != len(embeddings):
            raise ValueError(
                "items 与 embeddings 数量不一致："
                f"{len(items)} != {len(embeddings)}"
            )

        if embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维矩阵。")

        # 转成 float32 的连续内存数组，便于 FAISS 直接使用。
        normalized_embeddings = np.ascontiguousarray(
            embeddings.astype(np.float32, copy=False)
        )

        self._dimension = int(normalized_embeddings.shape[1])
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(normalized_embeddings)
        self._items = list(items)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[VectorSearchResult]:
        """对单条查询向量做 top-k 搜索。"""

        if self._index is None or not self._items:
            return []

        if query_vector.ndim != 1:
            raise ValueError("query_vector 必须是一维向量。")

        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                "query_vector 维度与向量库不一致："
                f"{query_vector.shape[0]} != {self.dimension}"
            )

        normalized_top_k = max(1, min(top_k, self.size))
        # FAISS 的 search 接口需要二维批量输入，因此这里把单条 query reshape 成 (1, dim)。
        normalized_query = np.ascontiguousarray(
            query_vector.astype(np.float32, copy=False).reshape(1, -1)
        )

        scores, indices = self._index.search(normalized_query, normalized_top_k)
        results: list[VectorSearchResult] = []

        for rank, (score, index) in enumerate(
            zip(scores[0].tolist(), indices[0].tolist()),
            start=1,
        ):
            if index < 0:
                continue

            results.append(
                VectorSearchResult(
                    rank=rank,
                    score=float(score),
                    item=self._items[index],
                )
            )

        return results

    def save_index(self, output_path: Path | str) -> Path:
        """把当前 FAISS 索引保存到本地文件。"""

        if self._index is None:
            raise RuntimeError("当前没有可保存的 FAISS 索引。")

        normalized_output_path = Path(output_path)
        normalized_output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(normalized_output_path))
        return normalized_output_path

    @classmethod
    def from_index_file(
        cls,
        index_path: Path | str,
        *,
        items: list[dict[str, Any]],
    ) -> "FaissVectorStore":
        """从已保存的 FAISS 索引文件恢复向量库。"""

        normalized_index_path = Path(index_path)
        if not normalized_index_path.exists():
            raise FileNotFoundError(f"FAISS 索引文件不存在: {normalized_index_path}")

        instance = cls()
        instance._index = faiss.read_index(str(normalized_index_path))
        instance._dimension = int(instance._index.d)
        instance._items = list(items)

        if instance._index.ntotal != len(instance._items):
            raise ValueError(
                "FAISS 索引向量数量与 payload 数量不一致："
                f"{instance._index.ntotal} != {len(instance._items)}"
            )

        return instance
