from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.llm.client import OpenAILLMClient


class LLMClientTests(unittest.TestCase):
    def test_client_reads_yunwu_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "YUNWU_API_KEY": "test-key",
                "YUNWU_BASE_URL": "https://yunwu.ai/v1",
                "PLANNER_MODEL": "gpt-5.4",
            },
            clear=False,
        ):
            client = OpenAILLMClient()
            self.assertTrue(client.is_available)
            self.assertEqual(client.base_url, "https://yunwu.ai/v1")

    def test_client_falls_back_to_default_yunwu_base_url(self) -> None:
        with patch.dict(
            os.environ,
            {
                "YUNWU_API_KEY": "test-key",
            },
            clear=False,
        ):
            os.environ.pop("YUNWU_BASE_URL", None)
            os.environ.pop("OPENAI_BASE_URL", None)
            client = OpenAILLMClient()
            self.assertEqual(client.base_url, "https://yunwu.ai/v1")


if __name__ == "__main__":
    unittest.main()
