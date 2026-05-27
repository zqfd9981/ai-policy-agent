from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent.graph import run_agent_workflow
from app.memory.completion import resolve_context_query
from app.memory.store import get_session_store
from app.memory.updater import update_session_memory


APP_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_DIR / "api" / "static"

app = FastAPI(title="Policy Agent Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    # Requests with the same session_id share one SessionMemory.
    session_id: str = Field(default="default")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    session_store = get_session_store()
    session_memory = session_store.get_or_create(request.session_id)

    # Step 1: use working memory to complete short follow-up queries.
    context_resolution = resolve_context_query(
        request.query,
        session_memory=session_memory,
    )
    contextualized_query = context_resolution.contextualized_query

    # Step 2: run the main agent workflow on the completed query.
    state = run_agent_workflow(
        contextualized_query,
        top_k=request.top_k,
        resolved_action=context_resolution.resolved_action,
        response_mode=context_resolution.response_mode,
        retrieval_goal=context_resolution.retrieval_goal,
        focus=context_resolution.focus,
        answer_plan=context_resolution.answer_plan,
    )

    # Step 3: persist the latest turn and distilled working memory.
    update_session_memory(
        session_memory,
        user_query=request.query,
        state=state,
    )

    payload = state.to_dict()
    payload["session_id"] = request.session_id
    payload["contextualized_query"] = contextualized_query
    payload["context_resolution"] = {
        "contextualized_query": context_resolution.contextualized_query,
        "reason": context_resolution.reason,
        "source": context_resolution.source,
        "resolved_action": context_resolution.resolved_action,
        "response_mode": context_resolution.response_mode,
        "retrieval_goal": context_resolution.retrieval_goal,
        "focus": context_resolution.focus,
        "answer_plan": context_resolution.answer_plan,
        "resolved_entities": list(context_resolution.resolved_entities),
    }
    payload["session_memory"] = session_memory.to_dict()
    return payload
