from __future__ import annotations

from dataclasses import dataclass


DEFAULT_QUERY_TOP_K = 5


@dataclass(frozen=True, slots=True)
class AgentQuery:
    """表示一次进入 Agent 的标准化用户查询。"""

    user_query: str
    top_k: int = DEFAULT_QUERY_TOP_K

    def __post_init__(self) -> None:
        """统一字段格式，并做最基础的输入校验。"""

        normalized_user_query = self.user_query.strip()
        normalized_top_k = max(1, int(self.top_k))

        object.__setattr__(self, "user_query", normalized_user_query)
        object.__setattr__(self, "top_k", normalized_top_k)

        if not normalized_user_query:
            raise ValueError("AgentQuery.user_query 不能为空。")

    def to_dict(self) -> dict[str, object]:
        """把查询对象转成便于日志或序列化的字典。"""

        return {
            "user_query": self.user_query,
            "top_k": self.top_k,
        }
