from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.models.metadata import PolicyMetadata


@dataclass(frozen=True, slots=True)
class Document:
    """表示一篇已经完成原始读取、但尚未清洗的政策文档。"""

    # 该文档对应的元数据对象。
    # 这里不把 metadata 打散成很多重复字段，是为了保证“文档事实”只有一份来源。
    metadata: PolicyMetadata

    # 原始文件在本地磁盘上的真实路径。
    # 后续如果清洗、切片、报错定位，都可以直接追溯到这个源文件。
    source_path: Path

    # 原始文件格式，例如 pdf、txt。
    # 这个值理论上应与 metadata.source_format 一致，后续 loader 可以顺手校验。
    source_format: str
    
    # 从原始文件中读取出来的全文文本，保持“原始读取结果”语义。
    # 这里先不做复杂清洗，方便后续 clean 模块单独处理。
    raw_text: str

    def __post_init__(self) -> None:
        """统一字段格式，并做最基础的数据一致性校验。"""

        normalized_path = Path(self.source_path)
        normalized_format = self.source_format.strip().lower()

        object.__setattr__(self, "source_path", normalized_path)
        object.__setattr__(self, "source_format", normalized_format)

        if not self.raw_text:
            raise ValueError(f"Document.raw_text 不能为空: {normalized_path}")

        actual_suffix = normalized_path.suffix.lstrip(".").lower()
        if actual_suffix and actual_suffix != normalized_format:
            raise ValueError(
                "Document 的 source_format 与 source_path 扩展名不一致："
                f"path={normalized_path}, source_format={normalized_format}"
            )

    @property
    def doc_id(self) -> str:
        """直接暴露 doc_id，减少后续频繁写 metadata.doc_id。"""

        return self.metadata.doc_id

    @property
    def title(self) -> str:
        """直接暴露标题，方便日志、调试和展示。"""

        return self.metadata.title

    @property
    def region(self) -> str:
        """直接暴露地区，方便后续清洗、切片和分析阶段使用。"""

        return self.metadata.region

    @property
    def text_length(self) -> int:
        """返回原始正文长度，方便做调试和质量检查。"""

        return len(self.raw_text)
