from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from src.services.llm_json import invoke_json_schema
from src.state import AgentState, IterationRecord, ReviewComment


logger = logging.getLogger(__name__)
HARD_GAP_CONFIDENCE_THRESHOLD = 0.45


class ReviewerOutput(BaseModel):
    decision: str = Field(..., pattern=r"^(pass|research)$")
    reason: str = Field(..., min_length=10, max_length=300)
    target_section_ids: list[int] = Field(default_factory=list, max_length=3)
    suggested_queries: list[str] = Field(default_factory=list, max_length=3)


SYSTEM_PROMPT = """\
你是研究报告质量审查节点。你需要阅读一份 AI 生成的报告草稿，对照各章节目标和证据摘要，做出判断。

通过（pass）的条件：
- 每个章节都基本回应了章节目标。
- 报告中的事实性陈述能在提供的证据中找到直接或间接支撑。
- 没有明显的凭空编造或大段跑题。

不通过（research）的条件：
- 某些章节没有回应章节目标。
- 存在证据中找不到来源的事实声明。
- 证据明显不足以支撑核心论点。

输出要求：
- decision: "pass" 或 "research"。
- reason: 用一两句话说清通过或不通过的具体原因。
- target_section_ids: 如果 research，列出需要补充检索的章节编号；如果 pass 则为空列表。
- suggested_queries: 如果 research，给出 2-3 条具体检索词，让下一轮搜索能定向补充缺失信息；如果 pass 则为空列表。

严格按 JSON Schema 输出，不要输出 JSON 之外的文字。
"""


def _build_review_prompt(state: AgentState) -> str:
    draft = state.get("draft", "")
    draft_excerpt = draft[:5000] if len(draft) > 5000 else draft

    sections_info: list[str] = []
    analyses_by_id = {analysis.section_id: analysis for analysis in state["section_analyses"]}

    for section in state["outline"]:
        analysis = analyses_by_id.get(section.section_id)
        block = f"章节 {section.section_id}: {section.title}\n  目标: {section.objective}"
        if analysis:
            if analysis.evidence_snippets:
                snippets_text = "；".join(
                    snippet[:120] for snippet in analysis.evidence_snippets[:2]
                )
                block += f"\n  证据片段: {snippets_text}"
            if analysis.missing_gaps:
                block += f"\n  上游缺口: {', '.join(analysis.missing_gaps[:2])}"
        sections_info.append(block)

    return (
        "请审查以下 AI 生成的研究报告草稿。\n\n"
        "=== 报告草稿 ===\n"
        f"{draft_excerpt}\n\n"
        "=== 各章节目标与证据 ===\n"
        + "\n".join(sections_info)
        + "\n\n请判断报告质量，输出 decision、reason、target_section_ids、suggested_queries。"
    )


def _all_section_ids(state: AgentState) -> list[int]:
    return [section.section_id for section in state["outline"]]


def _normalize_queries(queries: list[str], limit: int = 3) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = " ".join((query or "").split())[:80].strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) >= limit:
            break
    return normalized


def _fallback_queries(state: AgentState) -> list[str]:
    return _normalize_queries(state.get("search_queries", []))


def _section_queries(state: AgentState, section_ids: list[int]) -> list[str]:
    selected = set(section_ids)
    queries: list[str] = []
    for section in state["outline"]:
        if section.section_id in selected:
            queries.extend(section.search_queries)
    return _normalize_queries(queries) or _fallback_queries(state)


def _new_iteration_record(
    iteration: int,
    decision: str,
    reason: str,
    target_section_ids: list[int],
    suggested_queries: list[str],
    draft_version: int,
) -> list[IterationRecord]:
    return [
        IterationRecord(
            iteration=iteration,
            decision=decision,
            reason=reason,
            target_section_ids=target_section_ids,
            suggested_search_queries=suggested_queries,
            draft_version=draft_version,
        )
    ]


def _rule_based_review(state: AgentState) -> tuple[str, list[int]] | None:
    analyses_by_id = {analysis.section_id: analysis for analysis in state["section_analyses"]}
    missing_analysis_ids: list[int] = []
    no_evidence_ids: list[int] = []
    hard_gap_ids: list[int] = []

    for section in state["outline"]:
        analysis = analyses_by_id.get(section.section_id)
        if analysis is None:
            missing_analysis_ids.append(section.section_id)
            continue
        if not analysis.evidence_doc_ids or analysis.confidence <= 0:
            no_evidence_ids.append(section.section_id)
        elif analysis.missing_gaps and analysis.confidence < HARD_GAP_CONFIDENCE_THRESHOLD:
            hard_gap_ids.append(section.section_id)

    if missing_analysis_ids:
        return "部分章节缺少 Analyzer 结构化分析结果，需要重新检索或分析。", missing_analysis_ids
    if no_evidence_ids:
        return "部分章节缺少可支撑事实性结论的有效证据，需要补充检索。", no_evidence_ids
    if hard_gap_ids:
        return "部分章节证据置信度较低且存在信息缺口，需要定向补充证据。", hard_gap_ids
    return None


async def reviewer_node(state: AgentState) -> dict:
    draft = state.get("draft", "")
    next_iteration = state["iteration_count"] + 1

    if not draft or len(draft) < 100:
        reason = "草稿为空或过短，无法完成质量审查。"
        target_section_ids = _all_section_ids(state)
        suggested_queries = _fallback_queries(state)
        logger.warning("Reviewer received an empty or too short draft; requesting research.")
        return {
            "review_feedback": [
                ReviewComment(
                    comment_id=f"review-empty-draft-{next_iteration}",
                    severity="critical",
                    issue_type="coverage_gap",
                    description=reason,
                    recommendation="检查 Writer 节点输出，并重新执行检索和生成。",
                )
            ],
            "review_decision": "research",
            "target_section_ids": target_section_ids,
            "suggested_search_queries": suggested_queries,
            "iteration_history": _new_iteration_record(
                next_iteration,
                "research",
                reason,
                target_section_ids,
                suggested_queries,
                state.get("draft_version", 0),
            ),
            "iteration_count": next_iteration,
            "current_stage": "reviewed",
        }

    logger.info("Reviewer starting LLM review.")

    rule_result = _rule_based_review(state)
    if rule_result:
        reason, target_section_ids = rule_result
        suggested_queries = _section_queries(state, target_section_ids)
        logger.info(
            "Reviewer rule-based check requested research: sections=%s reason=%s",
            target_section_ids,
            reason,
        )
        return {
            "review_feedback": [
                ReviewComment(
                    comment_id=f"rule-review-research-{next_iteration}",
                    severity="major",
                    issue_type="evidence_gap",
                    description=reason,
                    recommendation=f"建议补充检索：{', '.join(suggested_queries)}",
                )
            ],
            "review_decision": "research",
            "target_section_ids": target_section_ids,
            "suggested_search_queries": suggested_queries,
            "iteration_history": _new_iteration_record(
                next_iteration,
                "research",
                reason,
                target_section_ids,
                suggested_queries,
                state.get("draft_version", 0),
            ),
            "iteration_count": next_iteration,
            "current_stage": "reviewed",
        }

    try:
        result = await invoke_json_schema(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_build_review_prompt(state),
            schema=ReviewerOutput,
            temperature=0.1,
        )
    except Exception as exc:
        reason = f"Reviewer 调用失败，无法确认报告质量：{exc}"
        target_section_ids = _all_section_ids(state)
        suggested_queries = _fallback_queries(state)
        logger.error("Reviewer LLM call failed; requesting research: %s", exc)
        return {
            "review_feedback": [
                ReviewComment(
                    comment_id=f"review-llm-error-{next_iteration}",
                    severity="critical",
                    issue_type="coverage_gap",
                    description=reason,
                    recommendation="检查 LLM API 配置、网络和 JSON 输出格式后重新审查。",
                )
            ],
            "review_decision": "research",
            "target_section_ids": target_section_ids,
            "suggested_search_queries": suggested_queries,
            "iteration_history": _new_iteration_record(
                next_iteration,
                "research",
                reason,
                target_section_ids,
                suggested_queries,
                state.get("draft_version", 0),
            ),
            "iteration_count": next_iteration,
            "current_stage": "reviewed",
        }

    valid_section_ids = set(_all_section_ids(state))
    target_section_ids = [
        section_id for section_id in result.target_section_ids if section_id in valid_section_ids
    ]
    if result.decision == "research" and not target_section_ids:
        target_section_ids = _all_section_ids(state)
    suggested_queries = _normalize_queries(result.suggested_queries)
    if result.decision == "pass":
        target_section_ids = []
        suggested_queries = []

    severity = "minor" if result.decision == "pass" else "major"
    issue_type = "pass" if result.decision == "pass" else "coverage_gap"
    comment = ReviewComment(
        comment_id=f"llm-review-{result.decision}-{next_iteration}",
        severity=severity,
        issue_type=issue_type,
        description=result.reason,
        recommendation=(
            "报告质量合格。"
            if result.decision == "pass"
            else f"建议补充检索：{', '.join(suggested_queries)}"
        ),
    )

    logger.info(
        "Reviewer complete: decision=%s sections=%s queries=%d reason=%s",
        result.decision,
        target_section_ids,
        len(suggested_queries),
        result.reason[:80],
    )

    return {
        "review_feedback": [comment],
        "review_decision": result.decision,
        "target_section_ids": target_section_ids,
        "suggested_search_queries": suggested_queries,
        "iteration_history": _new_iteration_record(
            next_iteration,
            result.decision,
            result.reason,
            target_section_ids,
            suggested_queries,
            state.get("draft_version", 0),
        ),
        "iteration_count": next_iteration,
        "current_stage": "reviewed",
    }
