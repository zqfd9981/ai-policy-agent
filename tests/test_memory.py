from __future__ import annotations

import unittest

from app.agent.state import AgentState
from app.memory.store import SessionStore
from app.memory.session import MemoryEntity
from app.memory.updater import update_session_memory
from app.models.query import AgentQuery
from app.models.response import AgentResponse


class MemoryTests(unittest.TestCase):
    def test_session_store_reuses_same_session(self) -> None:
        store = SessionStore()

        left = store.get_or_create("demo-001")
        right = store.get_or_create("demo-001")

        self.assertIs(left, right)

    def test_update_session_memory_writes_turns_and_working_memory(self) -> None:
        store = SessionStore()
        session_memory = store.get_or_create("demo-002")
        state = AgentState(
            query=AgentQuery("总结一下上海的AI政策"),
            intent="summarize",
            strategy="multi_doc_summary",
            final_response=AgentResponse(
                success=True,
                route="summarize",
                message="ok",
            ),
        )

        update_session_memory(
            session_memory,
            user_query="总结一下上海的AI政策",
            state=state,
        )

        self.assertEqual(len(session_memory.turns), 2)
        self.assertEqual(session_memory.working_memory.active_intent, "summarize")
        self.assertEqual(session_memory.working_memory.active_strategy, "multi_doc_summary")
        self.assertEqual(session_memory.working_memory.active_region, "上海")
        self.assertEqual(session_memory.working_memory.active_topic, "AI政策")

    def test_update_session_memory_accumulates_recent_region_entities(self) -> None:
        store = SessionStore()
        session_memory = store.get_or_create("demo-003")
        session_memory.working_memory.recent_entities = (
            MemoryEntity(kind="region", key="上海", label="上海"),
        )

        state = AgentState(
            query=AgentQuery("那么北京呢"),
            intent="summarize",
            strategy="multi_doc_summary",
            final_response=AgentResponse(
                success=True,
                route="summarize",
                message="ok",
            ),
        )

        update_session_memory(
            session_memory,
            user_query="那么北京呢",
            state=state,
        )

        region_labels = [
            item.label
            for item in session_memory.working_memory.recent_entities
            if item.kind == "region"
        ]
        self.assertIn("上海", region_labels)
        self.assertIn("北京", region_labels)


if __name__ == "__main__":
    unittest.main()
