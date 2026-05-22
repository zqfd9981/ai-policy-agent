from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent.graph import run_agent_workflow
from app.memory.store import get_session_store
from app.memory.updater import update_session_memory


APP_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_DIR / "api" / "static"

app = FastAPI(title="Policy Agent Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str = Field(default="default")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    session_store = get_session_store()
    session_memory = session_store.get_or_create(request.session_id)
    state = run_agent_workflow(request.query, top_k=request.top_k)
    update_session_memory(
        session_memory,
        user_query=request.query,
        state=state,
    )

    payload = state.to_dict()
    payload["session_id"] = request.session_id
    payload["session_memory"] = session_memory.to_dict()
    return payload
