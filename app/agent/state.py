from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from app.models.query import AgentQuery
from app.models.response import AgentResponse


@dataclass(frozen=True, slots=True)
class AgentState:
    """
    表示一次 Agent 工作流在当前阶段的共享状态。

    第一版先保持最小：
    - query: 标准化后的用户输入
    - route: 路由结果
    - tool_output: 工具层返回的原始结果
    - final_response: 最终给用户的响应
    - error_message: 工作流中记录的错误信息
    """

    query: AgentQuery
    route: str | None = None
    tool_output: Any | None = None
    final_response: AgentResponse | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        """统一可选字符串字段的格式。"""

        normalized_route = self.route.strip().lower() if self.route else None
        normalized_error_message = self.error_message.strip() if self.error_message else None

        object.__setattr__(self, "route", normalized_route)
        object.__setattr__(self, "error_message", normalized_error_message)

    @property
    def is_completed(self) -> bool:
        """判断当前工作流是否已经形成最终响应。"""

        return self.final_response is not None

    @property
    def has_error(self) -> bool:
        """判断当前状态是否已经记录错误。"""

        return bool(self.error_message)

    def with_route(self, route: str) -> "AgentState":
        """返回带有路由结果的新状态。"""

        return replace(self, route=route)

    def with_tool_output(self, tool_output: Any) -> "AgentState":
        """返回带有工具输出的新状态。"""

        return replace(self, tool_output=tool_output)

    def with_final_response(self, final_response: AgentResponse) -> "AgentState":
        """返回带有最终响应的新状态。"""

        return replace(self, final_response=final_response)

    def with_error(self, error_message: str) -> "AgentState":
        """返回带有错误信息的新状态。"""

        return replace(self, error_message=error_message)

    def to_dict(self) -> dict[str, Any]:
        """把当前状态转换成适合调试查看的字典。"""

        serialized_tool_output = self.tool_output
        if hasattr(serialized_tool_output, "to_dict"):
            serialized_tool_output = serialized_tool_output.to_dict()

        return {
            "query": self.query.to_dict(),
            "route": self.route,
            "tool_output": serialized_tool_output,
            "final_response": (
                self.final_response.to_dict() if self.final_response is not None else None
            ),
            "error_message": self.error_message,
            "is_completed": self.is_completed,
            "has_error": self.has_error,
        }
