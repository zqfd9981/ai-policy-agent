from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent.graph import run_agent_workflow


APP_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_DIR / "api" / "static"

app = FastAPI(title="Policy Agent Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    state = run_agent_workflow(request.query, top_k=request.top_k)
    return state.to_dict()
