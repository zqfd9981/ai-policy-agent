from __future__ import annotations

from dataclasses import dataclass, field

from app.memory.session import SessionMemory


@dataclass(slots=True)
class SessionStore:
    sessions: dict[str, SessionMemory] = field(default_factory=dict)

    def get_or_create(self, session_id: str) -> SessionMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionMemory(session_id=session_id)
        return self.sessions[session_id]


_SESSION_STORE = SessionStore()


def get_session_store() -> SessionStore:
    return _SESSION_STORE
