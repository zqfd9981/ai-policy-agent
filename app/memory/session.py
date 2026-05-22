from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SessionTurn:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkingMemory:
    active_region: str | None = None
    active_topic: str | None = None
    active_intent: str | None = None
    active_strategy: str | None = None
    focus_dimension: str | None = None

    active_doc_id: str | None = None
    active_doc_title: str | None = None

    candidate_doc_ids: tuple[str, ...] = ()
    candidate_titles: tuple[str, ...] = ()

    left_doc_id: str | None = None
    left_doc_title: str | None = None
    right_doc_id: str | None = None
    right_doc_title: str | None = None

    summary_scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_region": self.active_region,
            "active_topic": self.active_topic,
            "active_intent": self.active_intent,
            "active_strategy": self.active_strategy,
            "focus_dimension": self.focus_dimension,
            "active_doc_id": self.active_doc_id,
            "active_doc_title": self.active_doc_title,
            "candidate_doc_ids": list(self.candidate_doc_ids),
            "candidate_titles": list(self.candidate_titles),
            "left_doc_id": self.left_doc_id,
            "left_doc_title": self.left_doc_title,
            "right_doc_id": self.right_doc_id,
            "right_doc_title": self.right_doc_title,
            "summary_scope": self.summary_scope,
        }


@dataclass(slots=True)
class SessionMemory:
    session_id: str
    turns: list[SessionTurn] = field(default_factory=list)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [
                {
                    "role": item.role,
                    "content": item.content,
                    "metadata": dict(item.metadata),
                }
                for item in self.turns
            ],
            "working_memory": self.working_memory.to_dict(),
        }
