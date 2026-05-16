from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel


DEFAULT_LLM_MODEL = "gpt-4.1-mini"

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class LLMClientUnavailableError(RuntimeError):
    """当前环境无法使用 LLM client 时抛出的异常。"""


@dataclass(slots=True)
class OpenAILLMClient:
    """
    对 OpenAI 调用做一层很薄的统一封装。

    当前先只支持 planner 需要的“结构化解析”能力，
    后面再把 answer / judge 等节点都收敛到这一层。
    """

    model: str = DEFAULT_LLM_MODEL
    api_key: str | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")

        env_model = os.getenv("OPENAI_PLANNER_MODEL") or os.getenv("OPENAI_MODEL")
        if env_model:
            self.model = env_model.strip()

        self._client: OpenAI | None = None

    @property
    def is_available(self) -> bool:
        """判断当前环境是否具备可用的 API Key。"""

        return bool(self.api_key)

    def parse_structured_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        temperature: float = 0.1,
    ) -> ResponseModelT:
        """
        请求模型并直接解析为结构化对象。

        这里优先使用 SDK 的 parse 能力，避免我们手写 JSON 提取逻辑。
        """

        if not self.is_available:
            raise LLMClientUnavailableError("未配置 OPENAI_API_KEY，无法使用 LLM planner。")

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        completion = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_model,
            temperature=temperature,
        )

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise LLMClientUnavailableError("LLM 未返回可解析的结构化结果。")

        return parsed

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_completion_tokens: int = 1200,
    ) -> str:
        """请求模型生成普通文本输出。"""

        if not self.is_available:
            raise LLMClientUnavailableError("未配置 OPENAI_API_KEY，无法使用 LLM answer。")

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        message = completion.choices[0].message.content or ""
        normalized_message = message.strip()
        if not normalized_message:
            raise LLMClientUnavailableError("LLM 未返回有效文本结果。")
        return normalized_message
