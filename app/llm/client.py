from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv


DEFAULT_LLM_MODEL = "gpt-4o"

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class LLMClientUnavailableError(RuntimeError):
    """Raised when the configured LLM client is unavailable."""


@dataclass(slots=True)
class OpenAILLMClient:
    """
    OpenAI-compatible client wrapper.
    Designed to work with both official OpenAI endpoints and compatible providers
    such as YUNWU.
    """

    model: str = DEFAULT_LLM_MODEL
    api_key: str | None = None
    base_url: str | None = None
    _client: OpenAI | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        load_dotenv()

        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("YUNWU_API_KEY")

        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("YUNWU_BASE_URL")
            if not self.base_url and os.getenv("YUNWU_API_KEY"):
                self.base_url = "https://yunwu.ai/v1"

        env_model = (
            os.getenv("OPENAI_MODEL")
            or os.getenv("LLM_MODEL_NAME")
            or os.getenv("OPENAI_PLANNER_MODEL")
        )
        if env_model:
            self.model = env_model.strip()

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def parse_structured_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        temperature: float = 0.1,
        model: str | None = None,
    ) -> ResponseModelT:
        """
        Ask a model to return JSON text, then validate it with Pydantic.
        This is more compatible with third-party OpenAI-compatible providers
        than relying on SDK-specific parse helpers.
        """

        content = self.generate_text(
            system_prompt=system_prompt,
            user_prompt=(
                f"{user_prompt}\n\n"
                "请只返回一个合法 JSON 对象，不要输出 Markdown，不要输出解释。"
            ),
            temperature=temperature,
            max_completion_tokens=1200,
            model=model,
        )

        json_text = extract_json_object(content)
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as error:
            raise LLMClientUnavailableError(f"LLM 未返回可解析 JSON：{error}") from error

        try:
            return response_model.model_validate(payload)
        except ValidationError as error:
            raise LLMClientUnavailableError(f"LLM JSON 结构不符合预期：{error}") from error

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_completion_tokens: int = 1200,
        model: str | None = None,
    ) -> str:
        if not self.is_available:
            raise LLMClientUnavailableError("未配置可用 API Key。")

        if self._client is None:
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)

        completion = self._client.chat.completions.create(
            model=(model or self.model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            timeout=30.0,
        )
        message = completion.choices[0].message.content or ""
        normalized_message = message.strip()
        if not normalized_message:
            raise LLMClientUnavailableError("LLM 未返回有效文本结果。")
        return normalized_message


def extract_json_object(text: str) -> str:
    """Extract the outermost JSON object from model output."""

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start_index = stripped.find("{")
    end_index = stripped.rfind("}")
    if start_index < 0 or end_index < 0 or end_index <= start_index:
        raise LLMClientUnavailableError("未在模型输出中找到 JSON 对象。")

    return stripped[start_index : end_index + 1]
