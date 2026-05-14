from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


# 正式版默认采用中文检索效果更稳的 BGE v1.5 模型。
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
# BGE 系列做中文检索时，通常会给 query 加一条检索任务提示词。
BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class SentenceTransformerEmbedder:
    """
    基于 sentence-transformers 的正式版向量器。

    这一层只保留项目当前真正使用的正式方案：
    - 加载预训练 embedding 模型
    - 把 chunk / query 转成 dense vector
    - 在查询时自动附加检索指令
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        query_instruction: str | None = BGE_ZH_QUERY_INSTRUCTION,
        device: str | None = None,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """返回向量维度；如模型未加载则先自动加载。"""

        self._ensure_model_loaded()
        if self._dimension is None:
            raise RuntimeError("模型维度尚未初始化。")
        return self._dimension

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """把一批文档文本编码成向量矩阵。"""

        self._ensure_model_loaded()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        normalized_texts = [text.strip() for text in texts]
        embeddings = self._model.encode(
            normalized_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32, copy=False)

    def encode_query(self, query: str) -> np.ndarray:
        """把单条查询编码成向量。"""

        self._ensure_model_loaded()
        normalized_query = query.strip()
        if not normalized_query:
            return np.zeros(self.dimension, dtype=np.float32)

        if self.query_instruction:
            normalized_query = f"{self.query_instruction}{normalized_query}"

        embedding = self._model.encode(
            normalized_query,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype=np.float32)

    def _ensure_model_loaded(self) -> None:
        """按需加载底层 SentenceTransformer 模型。"""

        if self._model is not None:
            return

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=False,
        )
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = int(self._model.get_embedding_dimension())
        else:
            self._dimension = int(self._model.get_sentence_embedding_dimension())
