from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MemoryEntity:
    """A lightweight entity remembered across turns for reference resolution."""

    kind: str
    key: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "key": self.key,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class ComparisonMemory:
    """Tracks the currently active comparison group, if one exists."""

    kind: str
    members: tuple[str, ...]
    topic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "members": list(self.members),
            "topic": self.topic,
        }


@dataclass(slots=True)
class SessionTurn:
    """One raw conversation turn kept for session history display and tracing."""

    role: str
    content: str
    # Lightweight metadata for debugging, such as route / strategy / verdict.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkingMemory:
    """
    Compact, structured context distilled from previous turns.

    Unlike SessionTurn, this is not a full transcript. It only keeps the pieces
    that are most useful for the next-round reasoning and query completion.
    """

    # Current conversation focus.
    active_region: str | None = None
    active_topic: str | None = None
    active_intent: str | None = None
    active_strategy: str | None = None
    focus_dimension: str | None = None

    # Used when the user is currently drilling into one specific policy.
    active_doc_id: str | None = None
    active_doc_title: str | None = None

    # Used when the previous turn returned a list / summary over multiple policies.
    candidate_doc_ids: tuple[str, ...] = ()
    candidate_titles: tuple[str, ...] = ()

    # Used when the previous turn is a compare flow.
    left_doc_id: str | None = None
    left_doc_title: str | None = None
    right_doc_id: str | None = None
    right_doc_title: str | None = None

    # single_doc / multi_doc / compare
    summary_scope: str | None = None

    # Recent dialogue objects used for resolving phrases like “这两个地方”“最后一个”.
    recent_entities: tuple[MemoryEntity, ...] = ()
    active_comparison: ComparisonMemory | None = None

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
            "recent_entities": [item.to_dict() for item in self.recent_entities],
            "active_comparison": (
                self.active_comparison.to_dict() if self.active_comparison is not None else None
            ),
        }


@dataclass(slots=True)
class SessionMemory:
    """
    Session-level memory container.

    - turns: raw dialogue history
    - working_memory: compact current task state distilled from history
    """

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
