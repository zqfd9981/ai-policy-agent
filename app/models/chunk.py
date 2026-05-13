from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.document import CleanDocument
from app.models.metadata import PolicyMetadata


@dataclass(frozen=True, slots=True)
class Chunk:
    """表示一段已经可用于检索和引用的政策文本切片。"""

    # 该切片来源的清洗后文档对象。
    # 保留这个引用，后续做回溯、调试、导出时会很方便。
    clean_document: CleanDocument

    # 切片唯一编号。
    # 建议格式类似：JS002_0001
    chunk_id: str

    # 切片在当前文档中的顺序编号，从 1 开始。
    chunk_index: int

    # 标题路径，表示这个切片位于哪一层结构下。
    # 例如：["二、释放数据要素价值", "（一）加大多元数据资源供给。"]
    title_path: tuple[str, ...]

    # 当前切片的正文内容。
    text: str

    def __post_init__(self) -> None:
        """统一字段格式，并做基础合法性校验。"""

        normalized_chunk_id = self.chunk_id.strip()
        normalized_text = self.text.strip()
        normalized_title_path = tuple(title.strip() for title in self.title_path if title.strip())

        object.__setattr__(self, "chunk_id", normalized_chunk_id)
        object.__setattr__(self, "text", normalized_text)
        object.__setattr__(self, "title_path", normalized_title_path)

        if not normalized_chunk_id:
            raise ValueError("Chunk.chunk_id 不能为空。")

        if self.chunk_index <= 0:
            raise ValueError(f"Chunk.chunk_index 必须大于 0，当前值为: {self.chunk_index}")

        if not normalized_text:
            raise ValueError(f"Chunk.text 不能为空: {normalized_chunk_id}")

    @property
    def metadata(self) -> PolicyMetadata:
        """直接暴露 metadata，方便 retrieval 层使用。"""

        return self.clean_document.metadata

    @property
    def doc_id(self) -> str:
        """直接暴露 doc_id，减少后续频繁写 clean_document.doc_id。"""

        return self.clean_document.doc_id

    @property
    def title(self) -> str:
        """直接暴露源文档标题。"""

        return self.clean_document.title

    @property
    def region(self) -> str:
        """直接暴露地区。"""

        return self.clean_document.region

    @property
    def source_format(self) -> str:
        """直接暴露源文档格式。"""

        return self.clean_document.source_format

    @property
    def source_path(self):
        """直接暴露原始文件路径。"""

        return self.clean_document.source_path

    @property
    def text_length(self) -> int:
        """返回切片正文长度。"""

        return len(self.text)

    @property
    def title_path_str(self) -> str:
        """把标题路径拼成一个更适合展示和导出的字符串。"""

        return " > ".join(self.title_path)

    def to_dict(self) -> dict[str, Any]:
        """
        把 Chunk 转成可序列化字典。

        这一步主要给后续的 jsonl 导出、检索入库和调试查看使用。
        """

        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
            "title": self.title,
            "region": self.region,
            "source_format": self.source_format,
            "source_path": str(self.source_path),
            "title_path": list(self.title_path),
            "title_path_str": self.title_path_str,
            "text": self.text,
            "text_length": self.text_length,
            "metadata": {
                "doc_id": self.metadata.doc_id,
                "title": self.metadata.title,
                "region": self.metadata.region,
                "level": self.metadata.level,
                "issuer": self.metadata.issuer,
                "publish_date": self.metadata.publish_date,
                "policy_type": self.metadata.policy_type,
                "theme": self.metadata.theme,
                "tier": self.metadata.tier,
                "status": self.metadata.status,
                "source_format": self.metadata.source_format,
                "doc_no": self.metadata.doc_no,
                "source_url": self.metadata.source_url,
                "notes": self.metadata.notes,
            },
        }
