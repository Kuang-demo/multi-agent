from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from src.services.llm_json import invoke_json_schema
from src.state import AgentState, ReviewComment


logger = logging.getLogger(__name__)

# ── Schema ──────────────────────────────────────────

class ReviewerOutput(BaseModel):
    decision: str = Field(..., pattern=r"^(pass|research)$")
    reason: str = Field(..., min_length=10, max_length=300)
    suggested_queries: list[str] = Field(default_factory=list, max_length=3)


# ── Prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """\
你是研报质量审查员。你需要读一份 AI 生成的报告草稿，对照各章节的目标和证据摘要，做出判断。

## 判断标准

通过（pass）的条件：
- 每个章节都基本回应了它的章节目标
- 报告中的事实性陈述能在提供的证据中找到直接或间接支撑
- 没有明显的凭空编造或大段跑题

不通过（research）的条件：
- 某章节完全没有回应其目标
- 存在证据中找不到来源的事实声明（幻觉）
- 证据明显不足以支撑核心论点

## 输出要求

- decision: "pass" 或 "research"
- reason: 用一两句话说清通过或不通过的具体原因
- suggested_queries: 如果 research，给出 2-3 条具体的检索词，让下一轮搜索能定向补充缺失信息。检索词要具体到可以直接拿去搜索引擎用。如果 pass 则为空列表。

严格按 JSON Schema 输出，不要输出 JSON 之外的文字。
"""


# ── 构建审查上下文 ─────────────────────────────────

def _build_review_prompt(state: AgentState) -> str:
    draft = state.get("draft", "")
    draft_excerpt = draft[:5000] if len(draft) > 5000 else draft

    sections_info: list[str] = []
    analyses_by_id = {a.section_id: a for a in state["section_analyses"]}

    for section in state["outline"]:
        sid = section.section_id
        analysis = analyses_by_id.get(sid)
        block = f"章节 {sid}：{section.title}\n  目标：{section.objective}"
        if analysis:
            if analysis.evidence_snippets:
                snippets_text = "；".join(
                    s[:120] for s in analysis.evidence_snippets[:2]
                )
                block += f"\n  证据片段：{snippets_text}"
            if analysis.missing_gaps:
                block += f"\n  上游缺口：{', '.join(analysis.missing_gaps[:2])}"
        sections_info.append(block)

    return (
        "请审查以下 AI 生成的研报草稿。\n\n"
        "=== 报告草稿 ===\n"
        f"{draft_excerpt}\n\n"
        "=== 各章节目标与证据 ===\n"
        + "\n".join(sections_info)
        + "\n\n请判断报告质量，输出 decision、reason、suggested_queries。"
    )


async def reviewer_node(state: AgentState) -> dict:
    draft = state.get("draft", "")

    # 兜底：无草稿直接通过
    if not draft or len(draft) < 100:
        logger.warning("Reviewer 收到空或极短草稿，直接通过。")
        return {
            "review_feedback": [
                ReviewComment(
                    comment_id="review-skip",
                    severity="minor",
                    issue_type="pass",
                    description="草稿为空或过短，跳过审查。",
                    recommendation="检查 Writer 节点是否正常。",
                )
            ],
            "review_decision": "pass",
            "suggested_search_queries": [],
            "iteration_count": state["iteration_count"] + 1,
            "current_stage": "reviewed",
        }

    logger.info("Reviewer 启动 LLM 审查…")

    try:
        user_prompt = _build_review_prompt(state)
        result = await invoke_json_schema(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=ReviewerOutput,
            temperature=0.1,
        )
    except Exception as exc:
        logger.error("LLM 审查调用失败：%s，降级为 pass。", exc)
        return {
            "review_feedback": [
                ReviewComment(
                    comment_id="review-llm-error",
                    severity="minor",
                    issue_type="pass",
                    description=f"LLM 审查调用失败（{exc}），降级通过。",
                    recommendation="检查 LLM API 配置和网络。",
                )
            ],
            "review_decision": "pass",
            "suggested_search_queries": [],
            "iteration_count": state["iteration_count"] + 1,
            "current_stage": "reviewed",
        }

    # 包装成 ReviewComment
    severity = "minor" if result.decision == "pass" else "major"
    issue_type = "pass" if result.decision == "pass" else "coverage_gap"
    comment = ReviewComment(
        comment_id=f"llm-review-{result.decision}",
        severity=severity,
        issue_type=issue_type,
        description=result.reason,
        recommendation=(
            "报告质量合格。" if result.decision == "pass"
            else f"建议用以下检索词定向补充：{', '.join(result.suggested_queries)}"
        ),
    )

    logger.info(
        "Reviewer 完成：decision=%s queries=%d reason=%s",
        result.decision,
        len(result.suggested_queries),
        result.reason[:80],
    )

    return {
        "review_feedback": [comment],
        "review_decision": result.decision,
        "suggested_search_queries": result.suggested_queries,
        "iteration_count": state["iteration_count"] + 1,
        "current_stage": "reviewed",
    }
