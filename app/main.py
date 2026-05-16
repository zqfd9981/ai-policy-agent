from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 兼容直接运行 `python app\main.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.graph import run_agent_workflow
from app.models.query import DEFAULT_QUERY_TOP_K
from app.agent.state import AgentState


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="Policy Agent 最小 CLI demo",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="要查询的政策问题；不传时会进入交互式输入。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_QUERY_TOP_K,
        help=f"检索返回的最大结果数，默认 {DEFAULT_QUERY_TOP_K}。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 形式输出最终响应。",
    )
    return parser


def resolve_query(query_parts: list[str]) -> str:
    """统一解析最终 query 文本。"""

    if query_parts:
        return " ".join(part.strip() for part in query_parts if part.strip()).strip()

    return input("请输入政策问题: ").strip()


def render_state(state: AgentState) -> str:
    """把最终 AgentState 渲染成适合终端展示的文本。"""

    response = state.final_response
    if response is None:
        return "Agent 未生成最终响应。"

    lines = [
        f"success: {response.success}",
        f"route: {response.route}",
        f"intent: {state.intent or 'unknown'}",
        f"planner_source: {state.planner_source or 'unknown'}",
    ]
    if state.needs_rag is not None:
        lines.append(f"needs_rag: {state.needs_rag}")
    if state.needs_rewrite is not None:
        lines.append(f"needs_rewrite: {state.needs_rewrite}")
    if state.answer_style:
        lines.append(f"answer_style: {state.answer_style}")
    if state.planner_reason:
        lines.extend(render_labeled_block("planner_reason", state.planner_reason))
    if state.rewritten_query:
        lines.append(f"rewritten_query: {state.rewritten_query}")
    if state.rewrite_source:
        lines.append(f"rewrite_source: {state.rewrite_source}")
    if state.rewrite_reason:
        lines.extend(render_labeled_block("rewrite_reason", state.rewrite_reason))
    if state.rewrite_keywords:
        lines.append(f"rewrite_keywords: {', '.join(state.rewrite_keywords)}")
    if state.alternative_queries:
        lines.extend(
            render_labeled_block(
                "alternative_queries",
                "\n".join(state.alternative_queries),
            )
        )
    if state.answer_source:
        lines.append(f"answer_source: {state.answer_source}")
    if state.judge_verdict:
        lines.append(f"judge_verdict: {state.judge_verdict}")
    if state.judge_score is not None:
        lines.append(f"judge_score: {state.judge_score}")
    if state.judge_source:
        lines.append(f"judge_source: {state.judge_source}")
    if state.judge_reason:
        lines.extend(render_labeled_block("judge_reason", state.judge_reason))
    if state.judge_followup:
        lines.extend(render_labeled_block("judge_followup", state.judge_followup))

    lines.extend(render_labeled_block("message", response.message))

    if response.error_message:
        lines.extend(render_labeled_block("error", response.error_message))

    if response.citations:
        lines.append("")
        lines.append(f"citations ({response.citation_count}):")
        for citation in response.citations:
            lines.extend(render_citation(citation))

    return "\n".join(lines)


def render_labeled_block(label: str, text: str) -> list[str]:
    """把单行或多行文本渲染成统一的 label block。"""

    if "\n" not in text:
        return [f"{label}: {text}"]

    lines = [f"{label}:"]
    lines.extend(text.splitlines())
    return lines


def render_citation(citation: dict[str, object]) -> list[str]:
    """把单条 citation 渲染成多行文本。"""

    rank = citation.get("rank", "")
    score = citation.get("score", "")
    doc_id = citation.get("doc_id", "")
    title = citation.get("title", "")
    title_path_str = citation.get("title_path_str", "")
    text = str(citation.get("text", "")).replace("\n", " ").strip()

    path_part = f" | {title_path_str}" if title_path_str else ""
    preview = text[:120]
    prefix_parts: list[str] = []
    if rank != "":
        prefix_parts.append(f"[{rank}]")
    if score != "":
        prefix_parts.append(f"score={score}")
    if doc_id:
        prefix_parts.append(str(doc_id))
    if title:
        prefix_parts.append(str(title))
    if path_part:
        prefix_parts.append(path_part[3:] if path_part.startswith(" | ") else path_part)

    return [
        f"- {' | '.join(prefix_parts)}",
        f"  {preview}",
    ]


def main() -> None:
    """运行最小 Policy Agent CLI demo。"""

    parser = build_argument_parser()
    args = parser.parse_args()

    query = resolve_query(args.query)
    if not query:
        parser.error("query 不能为空。")

    state = run_agent_workflow(
        query,
        top_k=args.top_k,
    )
    response = state.final_response
    if response is None:
        parser.error("Agent 未生成最终响应。")

    if args.json:
        print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2))
        return

    print(render_state(state))


if __name__ == "__main__":
    main()
