from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """表示 Agent 对一次请求的统一输出。"""

    success: bool
    route: str
    message: str
    citations: tuple[dict[str, Any], ...] = ()
    error_message: str | None = None

    def __post_init__(self) -> None:
        """统一输出字段格式。"""

        normalized_route = self.route.strip().lower()
        normalized_message = self.message.strip()
        normalized_error_message = self.error_message.strip() if self.error_message else None
        normalized_citations = tuple(dict(citation) for citation in self.citations)

        object.__setattr__(self, "route", normalized_route)
        object.__setattr__(self, "message", normalized_message)
        object.__setattr__(self, "error_message", normalized_error_message)
        object.__setattr__(self, "citations", normalized_citations)

        if not normalized_route:
            raise ValueError("AgentResponse.route 不能为空。")

        if not normalized_message:
            raise ValueError("AgentResponse.message 不能为空。")

    @property
    def citation_count(self) -> int:
        """返回当前响应中携带的引用条数。"""

        return len(self.citations)

    def to_dict(self) -> dict[str, Any]:
        """把响应对象转换成适合 JSON 序列化的字典。"""

        return {
            "success": self.success,
            "route": self.route,
            "message": self.message,
            "citation_count": self.citation_count,
            "citations": [dict(citation) for citation in self.citations],
            "error_message": self.error_message,
        }
