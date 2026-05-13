from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


# 抽取英文单词、数字词和常见技术符号组合，例如 AI、AIGC、GPT-4、RAG。
ALNUM_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+\-./]*")
# 抽取中文字符，用于后续构造中文双字切分。
CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def tokenize_text(text: str) -> list[str]:
    """
    把文本切成轻量 token 列表。

    这版不依赖第三方中文分词库，而是采用一个更稳的折中方案：
    1. 英文、数字、技术缩写按词提取
    2. 中文连续字符转成双字切分

    这样虽然不如专业分词细，但足够支撑第一版检索底座。
    """

    normalized_text = text.strip().lower()
    if not normalized_text:
        return []

    tokens: list[str] = []

    # 先保留英文词、缩写和数字类 token。
    tokens.extend(ALNUM_TOKEN_PATTERN.findall(normalized_text))

    # 对中文采用双字切分，兼顾简单性和一定的语义稳定性。
    chinese_chars = CHINESE_CHAR_PATTERN.findall(normalized_text)
    if len(chinese_chars) == 1:
        tokens.append(chinese_chars[0])
    elif len(chinese_chars) >= 2:
        tokens.extend(
            chinese_chars[index] + chinese_chars[index + 1]
            for index in range(len(chinese_chars) - 1)
        )

    return tokens


class SimpleTfidfEmbedder:
    """
    一个纯本地、零额外依赖的轻量 TF-IDF 向量器。

    设计目标不是追求最强效果，而是先把 retrieval 主链路跑通：
    - 能 fit 语料
    - 能把 chunk 转成向量
    - 能把 query 转成同维度向量
    - 后面如果想替换成真正 embedding 模型，只需要替换这一层
    """

    def __init__(self, *, max_features: int = 8000, min_df: int = 1) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None
        self.is_fitted = False

    @property
    def dimension(self) -> int:
        """返回当前向量维度。"""

        return len(self.vocabulary_)

    def fit(self, texts: list[str]) -> "SimpleTfidfEmbedder":
        """基于一批文本建立词表并计算 IDF。"""

        document_frequency: Counter[str] = Counter()

        for text in texts:
            unique_tokens = set(tokenize_text(text))
            for token in unique_tokens:
                document_frequency[token] += 1

        filtered_tokens = [
            token
            for token, doc_freq in document_frequency.items()
            if doc_freq >= self.min_df
        ]

        # 第一版优先保留更常见、覆盖面更广的 token，便于稳定检索。
        filtered_tokens.sort(key=lambda token: (-document_frequency[token], token))

        if self.max_features > 0:
            filtered_tokens = filtered_tokens[: self.max_features]

        self.vocabulary_ = {
            token: index
            for index, token in enumerate(filtered_tokens)
        }

        total_documents = max(len(texts), 1)
        idf_values = np.ones(len(self.vocabulary_), dtype=np.float32)

        for token, index in self.vocabulary_.items():
            doc_freq = document_frequency[token]
            idf_values[index] = math.log((1 + total_documents) / (1 + doc_freq)) + 1.0

        self.idf_ = idf_values
        self.is_fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """把文本列表转成归一化后的 TF-IDF 向量矩阵。"""

        self._ensure_fitted()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for row_index, text in enumerate(texts):
            token_counts = Counter(tokenize_text(text))
            if not token_counts:
                continue

            total_tokens = sum(token_counts.values())
            for token, count in token_counts.items():
                token_index = self.vocabulary_.get(token)
                if token_index is None:
                    continue

                # 采用最基础的词频归一化，保持实现简单透明。
                term_frequency = count / total_tokens
                matrix[row_index, token_index] = term_frequency

        matrix *= self.idf_
        return self._l2_normalize(matrix)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """先 fit 再 transform。"""

        self.fit(texts)
        return self.transform(texts)

    def encode_query(self, query: str) -> np.ndarray:
        """把单条查询转成 1 维向量。"""

        transformed = self.transform([query])
        if transformed.size == 0:
            return np.zeros(self.dimension, dtype=np.float32)
        return transformed[0]

    def _ensure_fitted(self) -> None:
        """在 transform 前确认 embedder 已完成 fit。"""

        if not self.is_fitted or self.idf_ is None:
            raise RuntimeError("SimpleTfidfEmbedder 尚未 fit，不能直接 transform。")

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """按行做 L2 归一化，便于后续直接用点积近似余弦相似度。"""

        if matrix.size == 0:
            return matrix

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms
