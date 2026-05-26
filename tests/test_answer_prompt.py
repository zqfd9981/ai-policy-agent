from __future__ import annotations

import unittest

from app.agent.answer import build_answer_prompt, is_scenario_compare_query


class AnswerPromptTests(unittest.TestCase):
    def test_is_scenario_compare_query_detects_scene_based_compare(self) -> None:
        self.assertTrue(
            is_scenario_compare_query("举例子说明一下，在什么具体场景下，北京好或者是上海好")
        )
        self.assertTrue(
            is_scenario_compare_query("这两个地方哪个更适合做大模型落地？")
        )
        self.assertFalse(
            is_scenario_compare_query("比较北京和上海的大模型政策")
        )

    def test_build_answer_prompt_adds_scenario_compare_instructions(self) -> None:
        prompt = build_answer_prompt(
            user_query="举例子说明一下，在什么具体场景下，北京好或者是上海好",
            intent="compare",
            answer_style="comparative",
            context_text="mock context",
        )

        self.assertIn("场景化比较建议", prompt)
        self.assertIn("至少给出 2 到 4 个具体场景例子", prompt)
        self.assertIn("不要再按“政策概览 / 支持重点 / 适用对象 / 申报条件”四段模板展开", prompt)


if __name__ == "__main__":
    unittest.main()
