from __future__ import annotations

import unittest

from app.memory.completion import (
    ContextCompletionDecision,
    contextualize_query,
    resolve_context_query,
)
from app.memory.session import MemoryEntity, SessionMemory, WorkingMemory


class MemoryCompletionTests(unittest.TestCase):
    def test_resolve_context_query_prefers_llm_completion_when_available(self) -> None:
        class StubCompleter:
            is_available = True

            def complete(self, *, user_query: str, session_memory: SessionMemory) -> ContextCompletionDecision:
                return ContextCompletionDecision(
                    contextualized_query="比较上海和北京的AI政策，有什么差异，哪个地方更好",
                    reason="stub llm",
                    source="llm",
                    resolved_action="compare",
                    resolved_entities=("上海", "北京"),
                )

        session_memory = SessionMemory(
            session_id="demo-0",
            working_memory=WorkingMemory(
                active_topic="AI政策",
                recent_entities=(
                    MemoryEntity(kind="region", key="上海", label="上海"),
                    MemoryEntity(kind="region", key="北京", label="北京"),
                ),
            ),
        )

        decision = resolve_context_query(
            "这两个地方哪个更好",
            session_memory=session_memory,
            completer=StubCompleter(),
        )

        self.assertEqual(decision.source, "llm")
        self.assertEqual(decision.resolved_action, "compare")
        self.assertEqual(decision.resolved_entities, ("上海", "北京"))

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

    def test_region_switch_query_accepts_natural_variant(self) -> None:
        session_memory = SessionMemory(
            session_id="demo-1b",
            working_memory=WorkingMemory(
                active_region="上海",
                active_topic="AI政策",
                active_strategy="multi_doc_summary",
            ),
        )

        result = contextualize_query("那么北京呢", session_memory=session_memory)

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

    def test_group_compare_query_uses_recent_region_entities(self) -> None:
        session_memory = SessionMemory(
            session_id="demo-4",
            working_memory=WorkingMemory(
                active_topic="AI政策",
                recent_entities=(
                    MemoryEntity(kind="region", key="上海", label="上海"),
                    MemoryEntity(kind="region", key="北京", label="北京"),
                ),
            ),
        )

        result = contextualize_query(
            "这两个地方的政策有什么差异？哪个地方更好？",
            session_memory=session_memory,
        )

        self.assertEqual(
            result,
            "比较上海和北京的AI政策，有什么差异，哪个地方更好",
        )

    def test_resolve_context_query_returns_structured_decision(self) -> None:
        session_memory = SessionMemory(
            session_id="demo-5",
            working_memory=WorkingMemory(
                active_topic="AI政策",
                recent_entities=(
                    MemoryEntity(kind="region", key="上海", label="上海"),
                    MemoryEntity(kind="region", key="北京", label="北京"),
                ),
            ),
        )

        decision = resolve_context_query(
            "这两个地方的政策有什么差异？哪个地方更好？",
            session_memory=session_memory,
        )

        self.assertEqual(decision.source, "rule")
        self.assertEqual(decision.resolved_action, "compare")
        self.assertEqual(decision.resolved_entities, ("上海", "北京"))
        self.assertEqual(
            decision.contextualized_query,
            "比较上海和北京的AI政策，有什么差异，哪个地方更好",
        )

    def test_region_switch_rule_does_not_hijack_compare_query(self) -> None:
        session_memory = SessionMemory(
            session_id="demo-6",
            working_memory=WorkingMemory(
                active_topic="AI政策",
                recent_entities=(
                    MemoryEntity(kind="region", key="上海", label="上海"),
                    MemoryEntity(kind="region", key="北京", label="北京"),
                ),
            ),
        )

        decision = resolve_context_query(
            "举例子说明一下，在什么具体场景下，北京好或者是上海好",
            session_memory=session_memory,
        )

        self.assertEqual(decision.source, "rule")
        self.assertEqual(decision.resolved_action, "compare")
        self.assertEqual(decision.resolved_entities, ("上海", "北京"))
        self.assertEqual(
            decision.contextualized_query,
            "比较上海和北京的AI政策，有什么差异，哪个地方更好",
        )

    def test_group_compare_query_can_expand_existing_comparison_group(self) -> None:
        from app.memory.session import ComparisonMemory

        session_memory = SessionMemory(
            session_id="demo-7",
            working_memory=WorkingMemory(
                active_topic="AI政策",
                recent_entities=(
                    MemoryEntity(kind="region", key="上海", label="上海"),
                    MemoryEntity(kind="region", key="北京", label="北京"),
                    MemoryEntity(kind="region", key="安徽", label="安徽"),
                ),
                active_comparison=ComparisonMemory(
                    kind="region",
                    members=("上海", "北京"),
                    topic="AI政策",
                ),
            ),
        )

        decision = resolve_context_query(
            "那安徽和它们比又怎么样",
            session_memory=session_memory,
            completer=None,
        )

        self.assertEqual(decision.source, "rule")
        self.assertEqual(decision.resolved_action, "compare")
        self.assertEqual(decision.retrieval_goal, "compare_regions_multi")
        self.assertEqual(decision.resolved_entities, ("上海", "北京", "安徽"))
        self.assertEqual(
            decision.contextualized_query,
            "比较上海、北京、安徽的AI政策，它们分别适合什么场景，各有什么差异",
        )


if __name__ == "__main__":
    unittest.main()
