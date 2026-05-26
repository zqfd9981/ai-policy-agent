from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from app.llm.client import OpenAILLMClient
from app.tools.compare_policy import PolicyCompareOutput, render_policy_comparison
from app.tools.retrieve_policy import RetrievePolicyOutput
from app.tools.summarize_policies import MultiPolicySummaryOutput, render_multi_policy_summary
from app.tools.summarize_policy import PolicySummaryOutput, render_policy_summary


ANSWER_SYSTEM_PROMPT = """
你是 Policy Agent 的最终回答生成器。

你的职责是基于给定证据组织最终回答，而不是凭空补充事实。

要求：
1. 只根据提供的 evidence/context 回答。
2. 回答要直接、清晰、中文自然。
3. 如果任务类型是 summarize，优先输出结构化总结。
4. 如果任务类型是 compare，且用户要求“举例 / 场景 / 哪个更好 / 更适合”，优先输出场景化建议，而不是八股式政策综述。
5. 如果证据不足，要明确说明，而不是编造。
6. 不要输出“根据以上内容”等空泛套话，直接给结果。
""".strip()


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    """表示 answer node 产出的最终回答草稿。"""

    message: str
    citations: tuple[dict[str, Any], ...]
    source: str


class PolicyAgentAnswerer:
    """
    基于 LLM 的第一版 answerer。

    当前只负责一件事：
    - 把工具层证据组织成更自然的最终回答
    """

    def __init__(self, *, client: OpenAILLMClient | None = None) -> None:
        self.client = client or OpenAILLMClient()

    @property
    def is_available(self) -> bool:
        """判断当前环境是否具备可用 LLM。"""

        return self.client.is_available

    def answer(
        self,
        *,
        user_query: str,
        intent: str | None,
        answer_style: str | None,
        tool_output: Any,
    ) -> AnswerDraft:
        """基于当前工具结果生成最终回答。"""

        context_text, citations = build_answer_context(
            user_query=user_query,
            intent=intent,
            tool_output=tool_output,
        )
        user_prompt = build_answer_prompt(
            user_query=user_query,
            intent=intent,
            answer_style=answer_style,
            context_text=context_text,
        )
        message = self.client.generate_text(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=os.getenv("ANSWER_MODEL"),
        )
        return AnswerDraft(
            message=message,
            citations=citations,
            source="llm",
        )


def build_answer_context(
    *,
    user_query: str,
    intent: str | None,
    tool_output: Any,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """根据不同工具输出构建 answer node 的证据上下文。"""

    if isinstance(tool_output, RetrievePolicyOutput):
        citations = tuple(result.to_dict() for result in tool_output.results)
        lines = [
            f"用户问题：{user_query}",
            f"任务类型：{intent or 'retrieve'}",
            f"命中结果数：{tool_output.result_count}",
            "",
            "检索证据：",
        ]
        for item in build_retrieval_evidence_lines(citations):
            lines.append(
                f"- doc_id={item['doc_id']} | title={item['title']} | "
                f"region={item['region']} | publish_date={item['publish_date']} | "
                f"type={item['policy_type']} | score={item['score']}"
            )
            if item["title_path_str"]:
                lines.append(f"命中位置：{item['title_path_str']}")
            if item["snippet"]:
                lines.append(f"相关片段：{item['snippet']}")
        return "\n".join(lines), citations

    if isinstance(tool_output, PolicySummaryOutput):
        citations = tuple(item.to_dict() for item in tool_output.all_citations)
        lines = [
            f"用户问题：{user_query}",
            f"任务类型：{intent or 'summarize'}",
            "",
            f"目标政策：{tool_output.title} ({tool_output.doc_id})",
            f"定位依据：{tool_output.selection_reason}",
            "",
            "结构化摘要草稿：",
            render_policy_summary(tool_output),
        ]
        return "\n".join(lines), citations

    if isinstance(tool_output, MultiPolicySummaryOutput):
        citations = tuple(item.to_dict() for item in tool_output.all_citations)
        lines = [
            f"用户问题：{user_query}",
            f"任务类型：{intent or 'summarize'}",
            "",
            f"纳入政策数：{len(tool_output.policy_summaries)}",
            f"选择依据：{tool_output.selection_reason}",
            "",
            "多文档汇总草稿：",
            render_multi_policy_summary(tool_output),
        ]
        return "\n".join(lines), citations

    if isinstance(tool_output, PolicyCompareOutput):
        citations = tuple(dict(item) for item in tool_output.all_citations)
        lines = [
            f"用户问题：{user_query}",
            f"任务类型：{intent or 'compare'}",
            "",
            f"对比对象A：{tool_output.left_summary.title} ({tool_output.left_summary.doc_id})",
            f"对比对象B：{tool_output.right_summary.title} ({tool_output.right_summary.doc_id})",
            f"定位依据：{tool_output.selection_reason}",
            "",
            "结构化对比草稿：",
            render_policy_comparison(tool_output),
        ]
        return "\n".join(lines), citations

    return f"用户问题：{user_query}\n任务类型：{intent or 'unknown'}", ()


def build_answer_prompt(
    *,
    user_query: str,
    intent: str | None,
    answer_style: str | None,
    context_text: str,
) -> str:
    """构建给 LLM final answer 的用户提示词。"""

    prompt_lines = [
        f"用户问题：{user_query}",
        f"任务类型：{intent or 'unknown'}",
        f"回答风格：{answer_style or 'direct'}",
    ]

    if intent == "compare" and is_scenario_compare_query(user_query):
        prompt_lines.extend(
            [
                "",
                "额外回答要求：",
                "- 这轮不是普通政策综述，而是场景化比较建议。",
                "- 先直接回答：什么场景更适合北京，什么场景更适合上海/另一方。",
                "- 至少给出 2 到 4 个具体场景例子，例如科研平台、制造业落地、企业补贴申请、AI for Science、人才引进等。",
                "- 不要再按“政策概览 / 支持重点 / 适用对象 / 申报条件”四段模板展开。",
                "- 输出风格更像顾问建议：先结论，再场景例子，再给出简短原因。",
            ]
        )

    prompt_lines.extend(
        [
            "",
            "请基于以下证据生成最终回答：",
            context_text,
        ]
    )

    return "\n".join(prompt_lines)


def is_scenario_compare_query(query: str) -> bool:
    """Detect compare questions that ask for concrete scenes / recommendations."""

    scenario_keywords = (
        "举例",
        "场景",
        "哪个更好",
        "哪个好",
        "更适合",
        "适合什么",
        "什么情况下",
        "哪种情况",
        "在什么情况下",
        "具体场景",
    )
    return any(keyword in query for keyword in scenario_keywords)


def fallback_answer(
    *,
    tool_output: Any,
) -> AnswerDraft:
    """当未启用 LLM 时，退回到规则版最终回答。"""

    if isinstance(tool_output, PolicySummaryOutput):
        return AnswerDraft(
            message=render_policy_summary(tool_output),
            citations=tuple(item.to_dict() for item in tool_output.all_citations),
            source="rule",
        )

    if isinstance(tool_output, MultiPolicySummaryOutput):
        return AnswerDraft(
            message=render_multi_policy_summary(tool_output),
            citations=tuple(item.to_dict() for item in tool_output.all_citations),
            source="rule",
        )

    if isinstance(tool_output, PolicyCompareOutput):
        return AnswerDraft(
            message=render_policy_comparison(tool_output),
            citations=tuple(dict(item) for item in tool_output.all_citations),
            source="rule",
        )

    if isinstance(tool_output, RetrievePolicyOutput):
        return AnswerDraft(
            message=render_retrieval_answer(tool_output),
            citations=tuple(item.to_dict() for item in tool_output.results),
            source="rule",
        )

    return AnswerDraft(
        message="当前没有可用于生成回答的证据。",
        citations=(),
        source="rule",
    )


def build_retrieval_evidence_lines(
    citations: tuple[dict[str, Any], ...],
) -> tuple[dict[str, str], ...]:
    """把检索结果压缩成更适合 answer prompt 的证据行。"""

    evidence_lines: list[dict[str, str]] = []
    for item in citations:
        metadata = item.get("metadata", {})
        evidence_lines.append(
            {
                "doc_id": str(item.get("doc_id", "")),
                "title": str(item.get("title", "")),
                "region": str(metadata.get("region", "")),
                "publish_date": str(metadata.get("publish_date", "")),
                "policy_type": str(metadata.get("policy_type", "")),
                "score": f"{float(item.get('score', 0.0)):.3f}",
                "title_path_str": str(item.get("title_path_str", "")),
                "snippet": extract_readable_snippet(str(item.get("text", ""))),
            }
        )

    return tuple(evidence_lines)


def render_retrieval_answer(output: RetrievePolicyOutput) -> str:
    """把纯检索结果渲染成更适合直接展示的最终回答。"""

    if output.result_count == 0:
        return (
            "暂未检索到直接相关的政策证据。"
            "可以继续补充地区、政策主题、适用对象或业务场景后再检索。"
        )

    grouped_docs = group_retrieval_results(output)
    lines = [
        f"围绕“{output.query}”，我先检索到 {output.result_count} 条相关政策证据。",
        f"优先可关注以下 {len(grouped_docs)} 篇政策：",
        "",
    ]

    for index, item in enumerate(grouped_docs, start=1):
        meta_parts = []
        if item["region"]:
            meta_parts.append(f"地区：{item['region']}")
        if item["publish_date"]:
            meta_parts.append(f"发布日期：{item['publish_date']}")
        if item["policy_type"]:
            meta_parts.append(f"类型：{item['policy_type']}")
        if item["issuer"]:
            meta_parts.append(f"发文单位：{item['issuer']}")

        lines.append(f"{index}. {item['title']} ({item['doc_id']})")
        if meta_parts:
            lines.append("   " + " | ".join(meta_parts))
        if item["title_path_str"]:
            lines.append(f"   命中位置：{item['title_path_str']}")
        if item["snippet"]:
            lines.append(f"   相关片段：{item['snippet']}")

    lines.extend(
        [
            "",
            "如果你愿意，下一步可以继续让我对其中某一篇政策做摘要，或者进一步提取申报条件、支持重点、适用对象。",
        ]
    )
    return "\n".join(lines)


def group_retrieval_results(output: RetrievePolicyOutput) -> tuple[dict[str, str], ...]:
    """按政策文档聚合检索结果，避免最终回答重复罗列同一篇政策。"""

    grouped: dict[str, dict[str, str]] = {}
    ordered_doc_ids: list[str] = []

    for result in output.results:
        if result.doc_id not in grouped:
            metadata = result.metadata
            grouped[result.doc_id] = {
                "doc_id": result.doc_id,
                "title": result.title,
                "region": str(metadata.get("region", "")),
                "publish_date": str(metadata.get("publish_date", "")),
                "policy_type": str(metadata.get("policy_type", "")),
                "issuer": str(metadata.get("issuer", "")),
                "title_path_str": result.title_path_str,
                "snippet": extract_readable_snippet(result.text),
            }
            ordered_doc_ids.append(result.doc_id)
            continue

        current = grouped[result.doc_id]
        if not current["title_path_str"] and result.title_path_str:
            current["title_path_str"] = result.title_path_str
        if not current["snippet"]:
            current["snippet"] = extract_readable_snippet(result.text)

    return tuple(grouped[doc_id] for doc_id in ordered_doc_ids)


def extract_readable_snippet(text: str, *, max_length: int = 120) -> str:
    """尽量从 chunk 中抽取一段可读摘要，过滤明显的 PDF/OCR 噪声。"""

    normalized_text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if not normalized_text or looks_like_garbled_text(normalized_text):
        return ""

    sentences = re.split(r"(?<=[。！？；])", normalized_text)
    for sentence in sentences:
        cleaned_sentence = sentence.strip()
        if len(cleaned_sentence) < 12:
            continue
        if looks_like_garbled_text(cleaned_sentence):
            continue
        return clip_text(cleaned_sentence, max_length=max_length)

    return clip_text(normalized_text, max_length=max_length)


def looks_like_garbled_text(text: str) -> bool:
    """判断文本是否更像抽取损坏后的 OCR/PDF 噪声。"""

    if "/G" in text:
        return True

    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
    alnum_count = sum(char.isalnum() for char in text)
    effective_length = max(1, cjk_count + alnum_count)

    # 如果可见内容里中文占比极低，通常不是可直接展示的正文片段。
    return cjk_count / effective_length < 0.2


def clip_text(text: str, *, max_length: int) -> str:
    """把文本截断到适合终端展示的长度。"""

    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"
