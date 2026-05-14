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
from app.retrieval.embedder import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    SentenceTransformerEmbedder,
)
from app.retrieval.vector_store import (
    FaissVectorStore,
    VectorSearchResult,
)


# 默认把正式版检索索引放在 outputs/retrieval 目录下。
DEFAULT_RETRIEVAL_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "outputs" / "retrieval"
)
DEFAULT_FAISS_INDEX_PATH = DEFAULT_RETRIEVAL_OUTPUT_DIR / "policy_chunks.faiss"
DEFAULT_RETRIEVAL_PAYLOAD_PATH = DEFAULT_RETRIEVAL_OUTPUT_DIR / "policy_payloads.jsonl"
DEFAULT_RETRIEVAL_MANIFEST_PATH = DEFAULT_RETRIEVAL_OUTPUT_DIR / "policy_retriever_manifest.json"


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


def export_payloads_to_jsonl(
    payloads: list[dict[str, Any]],
    output_path: Path | str = DEFAULT_RETRIEVAL_PAYLOAD_PATH,
) -> Path:
    """把检索 payload 列表导出到 jsonl。"""

    normalized_output_path = Path(output_path)
    normalized_output_path.parent.mkdir(parents=True, exist_ok=True)

    with normalized_output_path.open("w", encoding="utf-8") as file:
        for payload in payloads:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return normalized_output_path


def save_retriever_manifest(
    manifest: dict[str, Any],
    output_path: Path | str = DEFAULT_RETRIEVAL_MANIFEST_PATH,
) -> Path:
    """保存检索器构建信息，便于后续直接恢复。"""

    normalized_output_path = Path(output_path)
    normalized_output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return normalized_output_path


def load_retriever_manifest(manifest_path: Path | str) -> dict[str, Any]:
    """读取已保存的检索器 manifest。"""

    normalized_manifest_path = Path(manifest_path)
    if not normalized_manifest_path.exists():
        raise FileNotFoundError(f"retriever manifest 不存在: {normalized_manifest_path}")

    return json.loads(normalized_manifest_path.read_text(encoding="utf-8"))


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
    """政策检索器。"""

    def __init__(
        self,
        *,
        embedder: SentenceTransformerEmbedder,
        vector_store: FaissVectorStore,
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
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
    ) -> "PolicyRetriever":
        """基于 chunk 载荷列表构建正式版检索器。"""

        retrieval_texts = [build_retrieval_text(payload) for payload in payloads]
        embedder = SentenceTransformerEmbedder(
            model_name=embedding_model_name,
            device=device,
            batch_size=batch_size,
        )
        embeddings = embedder.encode_texts(retrieval_texts)

        vector_store = FaissVectorStore()
        vector_store.add(payloads, embeddings)
        return cls(embedder=embedder, vector_store=vector_store, payloads=payloads)

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        *,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
    ) -> "PolicyRetriever":
        """直接基于 Chunk 对象列表构建检索器。"""

        return cls.from_chunk_payloads(
            serialize_chunks(chunks),
            embedding_model_name=embedding_model_name,
            device=device,
            batch_size=batch_size,
        )

    @classmethod
    def from_jsonl(
        cls,
        chunk_jsonl_path: Path | str = DEFAULT_CHUNK_OUTPUT_PATH,
        *,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
    ) -> "PolicyRetriever":
        """从 chunk jsonl 文件构建检索器。"""

        payloads = load_chunk_payloads_from_jsonl(chunk_jsonl_path)
        return cls.from_chunk_payloads(
            payloads,
            embedding_model_name=embedding_model_name,
            device=device,
            batch_size=batch_size,
        )

    @classmethod
    def from_saved_artifacts(
        cls,
        *,
        index_path: Path | str = DEFAULT_FAISS_INDEX_PATH,
        payload_path: Path | str = DEFAULT_RETRIEVAL_PAYLOAD_PATH,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
    ) -> "PolicyRetriever":
        """
        从已保存的 payload + FAISS 索引直接恢复检索器。

        这条路径不会重新编码全部 chunk，因此启动更快。
        """

        payloads = load_chunk_payloads_from_jsonl(payload_path)
        vector_store = FaissVectorStore.from_index_file(index_path, items=payloads)
        embedder = SentenceTransformerEmbedder(
            model_name=embedding_model_name,
            device=device,
            batch_size=batch_size,
        )
        return cls(embedder=embedder, vector_store=vector_store, payloads=payloads)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path | str = DEFAULT_RETRIEVAL_MANIFEST_PATH,
        *,
        device: str | None = None,
        batch_size: int = 16,
    ) -> "PolicyRetriever":
        """根据 manifest 恢复检索器。"""

        manifest = load_retriever_manifest(manifest_path)
        return cls.from_saved_artifacts(
            index_path=manifest["index_path"],
            payload_path=manifest["payload_path"],
            embedding_model_name=manifest.get(
                "embedding_model_name",
                DEFAULT_EMBEDDING_MODEL_NAME,
            ),
            device=device,
            batch_size=batch_size,
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

    def save_artifacts(
        self,
        *,
        index_path: Path | str = DEFAULT_FAISS_INDEX_PATH,
        payload_path: Path | str = DEFAULT_RETRIEVAL_PAYLOAD_PATH,
        manifest_path: Path | str = DEFAULT_RETRIEVAL_MANIFEST_PATH,
        source_chunk_path: Path | str = DEFAULT_CHUNK_OUTPUT_PATH,
    ) -> dict[str, Path]:
        """
        保存当前检索器的持久化产物。

        当前只支持保存正式版 FAISS 检索器；
        baseline 版没有必要额外持久化。
        """

        if not isinstance(self.vector_store, FaissVectorStore):
            raise TypeError("只有正式版 FAISS 检索器支持 save_artifacts。")

        saved_index_path = self.vector_store.save_index(index_path)
        saved_payload_path = export_payloads_to_jsonl(self.payloads, payload_path)

        manifest = {
            "embedding_model_name": self.embedder.model_name,
            "index_path": str(saved_index_path),
            "payload_path": str(saved_payload_path),
            "source_chunk_path": str(source_chunk_path),
            "payload_count": len(self.payloads),
            "vector_dimension": self.vector_store.dimension,
        }
        saved_manifest_path = save_retriever_manifest(manifest, manifest_path)

        return {
            "index_path": saved_index_path,
            "payload_path": saved_payload_path,
            "manifest_path": saved_manifest_path,
        }


def build_and_save_default_retriever(
    *,
    chunk_jsonl_path: Path | str = DEFAULT_CHUNK_OUTPUT_PATH,
    ensure_chunk_file: bool = True,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    device: str | None = None,
    batch_size: int = 16,
    index_path: Path | str = DEFAULT_FAISS_INDEX_PATH,
    payload_path: Path | str = DEFAULT_RETRIEVAL_PAYLOAD_PATH,
    manifest_path: Path | str = DEFAULT_RETRIEVAL_MANIFEST_PATH,
) -> tuple[PolicyRetriever, dict[str, Path]]:
    """构建正式版检索器并把索引相关产物一起保存。"""

    retriever = build_default_retriever(
        chunk_jsonl_path=chunk_jsonl_path,
        ensure_chunk_file=ensure_chunk_file,
        embedding_model_name=embedding_model_name,
        device=device,
        batch_size=batch_size,
    )
    saved_paths = retriever.save_artifacts(
        index_path=index_path,
        payload_path=payload_path,
        manifest_path=manifest_path,
        source_chunk_path=chunk_jsonl_path,
    )
    return retriever, saved_paths


def build_default_retriever(
    *,
    chunk_jsonl_path: Path | str = DEFAULT_CHUNK_OUTPUT_PATH,
    ensure_chunk_file: bool = True,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    device: str | None = None,
    batch_size: int = 16,
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
        embedding_model_name=embedding_model_name,
        device=device,
        batch_size=batch_size,
    )


def load_or_build_default_retriever(
    *,
    chunk_jsonl_path: Path | str = DEFAULT_CHUNK_OUTPUT_PATH,
    ensure_chunk_file: bool = True,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    device: str | None = None,
    batch_size: int = 16,
    manifest_path: Path | str = DEFAULT_RETRIEVAL_MANIFEST_PATH,
) -> PolicyRetriever:
    """
    优先从已保存产物恢复检索器；不存在时再重新构建。

    这是后面 tools / agent 最适合直接调用的入口。
    """

    normalized_manifest_path = Path(manifest_path)
    if normalized_manifest_path.exists():
        return PolicyRetriever.from_manifest(
            normalized_manifest_path,
            device=device,
            batch_size=batch_size,
        )

    retriever, _ = build_and_save_default_retriever(
        chunk_jsonl_path=chunk_jsonl_path,
        ensure_chunk_file=ensure_chunk_file,
        embedding_model_name=embedding_model_name,
        device=device,
        batch_size=batch_size,
        manifest_path=normalized_manifest_path,
    )
    return retriever


def main() -> None:
    """对 retrieval 第一版做一个简单联调测试。"""

    print("=" * 60)
    print("开始测试 retrieval 第一版...")
    print("=" * 60)

    retriever, saved_paths = build_and_save_default_retriever()
    print(f"[OK] 已保存 FAISS 索引: {saved_paths['index_path']}")
    print(f"[OK] 已保存 payload 文件: {saved_paths['payload_path']}")
    print(f"[OK] 已保存 manifest: {saved_paths['manifest_path']}")
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

    restored_retriever = PolicyRetriever.from_manifest()
    restored_results = restored_retriever.search("医疗服务 人工智能 大模型", top_k=1)
    if restored_results:
        print(f"\n[OK] 恢复索引测试通过: {restored_results[0].chunk_id}")

    print("[OK] retrieval 第一版测试通过")


if __name__ == "__main__":
    main()
