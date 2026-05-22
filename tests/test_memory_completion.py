from __future__ import annotations

import unittest

from app.memory.completion import contextualize_query
from app.memory.session import SessionMemory, WorkingMemory


class MemoryCompletionTests(unittest.TestCase):
    def test_region_switch_query_uses_working_memory_topic(self) -> None:
        session_memory = SessionMemory(
            session_id="demo-1",
            working_memory=WorkingMemory(
                active_region="上海",
                active_topic="AI政策",
                active_strategy="multi_doc_summary",
            ),
        )

        result = contextualize_query("那北京呢", session_memory=session_memory)

        self.assertEqual(result, "总结一下北京的AI政策")

    def test_focus_dimension_query_uses_compare_context(self) -> None:
        session_memory = SessionMemory(
            session_id="demo-2",
            working_memory=WorkingMemory(
                active_strategy="compare",
                left_doc_title="北京市推动“人工智能+”行动计划（2024-2025年）",
                right_doc_title="关于人工智能“模塑申城”的实施方案",
            ),
        )

        result = contextualize_query("继续讲支持重点", session_memory=session_memory)

        self.assertEqual(
            result,
            "比较北京市推动“人工智能+”行动计划（2024-2025年）和关于人工智能“模塑申城”的实施方案，重点看支持重点",
        )

    def test_candidate_reference_query_uses_second_title(self) -> None:
        session_memory = SessionMemory(
            session_id="demo-3",
            working_memory=WorkingMemory(
                active_strategy="multi_doc_summary",
                candidate_titles=(
                    "上海市加快推动“AI+制造”发展的实施方案",
                    "上海市进一步扩大人工智能应用的若干措施",
                ),
            ),
        )

        result = contextualize_query("第二篇呢", session_memory=session_memory)

        self.assertEqual(result, "总结一下上海市进一步扩大人工智能应用的若干措施")


if __name__ == "__main__":
    unittest.main()
