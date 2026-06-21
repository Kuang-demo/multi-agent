import logging
import re

from pydantic import BaseModel, Field

from src.services.llm_json import invoke_json_schema
from src.services.report_exporter import export_markdown_report
from src.state import AgentState, SectionAnalysis


logger = logging.getLogger(__name__)
EXECUTIVE_SUMMARY_PLACEHOLDER = "__EXECUTIVE_SUMMARY__"
MAX_DISPLAY_SNIPPETS = 2
MAX_DISPLAY_SNIPPET_CHARS = 240
MAX_EXECUTIVE_SUMMARY_CHARS = 700


class WriterOutputSchema(BaseModel):
    executive_summary: str = Field(..., min_length=40, max_length=320)
    section_body: str = Field(..., min_length=120, max_length=1500)


WRITER_SYSTEM_PROMPT = """
你是研究工作流中的写作节点。
你要基于上游分析结果和证据片段生成中文报告正文。
每个重要的事实性句子都应该尽量带上 [C1] [C2] [C3] 这样的引用标记。
不要编造证据之外的信息。
不要删除缺口说明。
"""


def _format_sources(sources: list[str]) -> str:
    if not sources:
        return "- 暂无来源"
    return "\n".join(f"- {source}" for source in sources)


def _format_snippets(snippets: list[str]) -> str:
    if not snippets:
        return "- 暂无证据摘录"
    display_snippets = []
    for snippet in snippets[:MAX_DISPLAY_SNIPPETS]:
        compact = " ".join(snippet.split())
        if len(compact) > MAX_DISPLAY_SNIPPET_CHARS:
            compact = compact[:MAX_DISPLAY_SNIPPET_CHARS].rstrip() + "..."
        display_snippets.append(compact)
    return "\n".join(f"> {snippet}" for snippet in display_snippets)


def _format_prompt_items(items: list[str], empty_text: str) -> str:
    if not items:
        return f"- {empty_text}"
    return "\n".join(f"- {item}" for item in items)


def _compact_text(text: str) -> str:
    return " ".join((text or "").split())


def _strip_citation_marks(text: str) -> str:
    return re.sub(r"\[C\d+\]", "", text)


def _first_sentence(text: str, max_chars: int = 150) -> str:
    compact = _strip_citation_marks(_compact_text(text))
    if not compact:
        return ""

    best_pos = 0
    for boundary_char in "。！？.!?":
        pos = compact.find(boundary_char)
        if pos != -1 and (best_pos == 0 or pos < best_pos):
            best_pos = pos

    sentence = compact[: best_pos + 1] if best_pos else compact
    if len(sentence) <= max_chars:
        return sentence
    return sentence[:max_chars].rstrip("，,；;。 ") + "。"


def _build_executive_summary(topic: str, analyses: list[SectionAnalysis]) -> str:
    valid_analyses = [
        item
        for item in sorted(analyses, key=lambda analysis: analysis.section_id)
        if item.evidence_doc_ids and item.confidence > 0
    ]
    if not valid_analyses:
        return f"本报告围绕“{topic}”展开，但当前检索证据不足，暂不形成事实性总括结论。"

    findings: list[str] = []
    seen: set[str] = set()
    for analysis in valid_analyses:
        source_text = analysis.key_points[0] if analysis.key_points else analysis.summary
        sentence = _first_sentence(source_text)
        dedupe_key = sentence[:40]
        if sentence and dedupe_key not in seen:
            findings.append(sentence)
            seen.add(dedupe_key)
        if len(findings) >= 4:
            break

    gap_items: list[str] = []
    for analysis in valid_analyses:
        for gap in analysis.missing_gaps:
            gap_text = _first_sentence(gap, max_chars=80)
            if gap_text and gap_text not in gap_items:
                gap_items.append(gap_text)
            if len(gap_items) >= 2:
                break
        if len(gap_items) >= 2:
            break

    parts = [f"本报告围绕“{topic}”梳理风险能力、评估框架和治理缺口。"]
    if findings:
        parts.append("综合证据看，" + "；".join(item.rstrip("。") for item in findings) + "。")
    if gap_items:
        parts.append("主要信息缺口包括：" + "；".join(item.rstrip("。") for item in gap_items) + "。")

    summary = _compact_text(" ".join(parts))
    if len(summary) <= MAX_EXECUTIVE_SUMMARY_CHARS:
        return summary
    return summary[:MAX_EXECUTIVE_SUMMARY_CHARS].rstrip("，,；;。 ") + "。"


def _ensure_citations(text: str, citations: list[str]) -> str:
    if not citations:
        return text
    if re.search(r"\[C\d+\]", text):
        return text
    return f"{text.rstrip()}\n\n本节依据上述证据综合整理，主要参考 [C1]。"


def _build_no_evidence_output(analysis: SectionAnalysis) -> tuple[str, str]:
    gaps = analysis.missing_gaps or ["未检索到可支撑本章节的有效证据"]
    executive_summary = f"{analysis.section_title}：当前证据不足，暂不生成事实性结论。"
    section_body = (
        "当前章节没有检索到足够的有效证据，因此不生成事实性结论。"
        "后续需要补充检索后再进行分析和写作。\n\n"
        "主要缺口：\n"
        + "\n".join(f"- {gap}" for gap in gaps)
    )
    return executive_summary, section_body


def _build_missing_analysis_output(section_title: str) -> tuple[str, str]:
    executive_summary = f"{section_title}：缺少上游分析结果，暂不生成章节结论。"
    section_body = (
        "本章节缺少 Analyzer 节点输出的结构化分析结果，"
        "因此 Writer 不生成事实性正文。请检查上游检索和分析流程。"
    )
    return executive_summary, section_body


def _append_section_report(
    report_lines: list[str],
    section_id: int,
    section_title: str,
    section_objective: str,
    section_body: str,
    citations: list[str],
    evidence_snippets: list[str],
    evidence_doc_ids: list[str],
    missing_gaps: list[str],
) -> None:
    report_lines.append("")
    report_lines.append(f"## {section_id}. {section_title}")
    report_lines.append(f"章节目标：{section_objective}")
    report_lines.append("")
    report_lines.append("### 章节正文")
    report_lines.append(section_body)
    report_lines.append("")
    report_lines.append("### 引用索引")
    report_lines.append(_format_sources(citations))
    report_lines.append("")
    report_lines.append("### 证据摘录")
    report_lines.append(_format_snippets(evidence_snippets))
    report_lines.append("")
    report_lines.append("### 证据ID")
    report_lines.append(", ".join(evidence_doc_ids) if evidence_doc_ids else "暂无")
    report_lines.append("")
    if missing_gaps:
        report_lines.append("### 缺口")
        report_lines.extend(f"- {gap}" for gap in missing_gaps)
        report_lines.append("")


async def _llm_write_section(topic: str, analysis: SectionAnalysis) -> WriterOutputSchema:
    user_prompt = (
        f"研究主题：{topic}\n"
        f"章节标题：{analysis.section_title}\n"
        f"章节摘要：{analysis.summary}\n"
        "关键要点：\n"
        f"{_format_prompt_items(analysis.key_points, '暂无关键要点')}\n"
        "引用索引：\n"
        f"{_format_prompt_items(analysis.citations, '暂无引用索引')}\n"
        "证据摘录：\n"
        f"{_format_prompt_items(analysis.evidence_snippets, '暂无证据摘录')}\n"
        "缺口：\n"
        f"{_format_prompt_items(analysis.missing_gaps, '暂无明确缺口')}\n\n"
        "请输出一句执行摘要，以及一段带引用标记的章节正文。"
    )
    return await invoke_json_schema(
        system_prompt=WRITER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=WriterOutputSchema,
        temperature=0.2,
    )


async def writer_node(state: AgentState) -> dict:
    analyses_by_section = {item.section_id: item for item in state["section_analyses"]}
    completed_analyses: list[SectionAnalysis] = []

    report_lines: list[str] = [
        f"# {state['topic']}",
        "",
        "## 报告说明",
        "- 生成方式：离线多智能体工作流",
        f"- 草稿版本：v{state['draft_version'] + 1}",
        "- 节点流程：Planner -> Searcher -> Analyzer -> Writer -> Reviewer",
        "",
        "## 执行摘要",
        EXECUTIVE_SUMMARY_PLACEHOLDER,
    ]

    for section in state["outline"]:
        analysis = analyses_by_section.get(section.section_id)
        if analysis is None:
            _, section_body = _build_missing_analysis_output(section.title)
            _append_section_report(
                report_lines=report_lines,
                section_id=section.section_id,
                section_title=section.title,
                section_objective=section.objective,
                section_body=section_body,
                citations=[],
                evidence_snippets=[],
                evidence_doc_ids=[],
                missing_gaps=["缺少上游结构化分析结果"],
            )
            logger.warning(
                "Writer skipped LLM for section %s because analysis is missing.",
                section.section_id,
            )
            continue

        if not analysis.evidence_doc_ids or analysis.confidence <= 0:
            _, section_body = _build_no_evidence_output(analysis)
            logger.warning(
                "Writer skipped LLM for section %s because no evidence was available.",
                section.section_id,
            )
        else:
            llm_result = await _llm_write_section(state["topic"], analysis)
            section_body = _ensure_citations(llm_result.section_body, analysis.citations)
            logger.info("Writer 使用 LLM 完成了 section %s。", section.section_id)

        completed_analyses.append(analysis)
        _append_section_report(
            report_lines=report_lines,
            section_id=section.section_id,
            section_title=section.title,
            section_objective=section.objective,
            section_body=section_body,
            citations=analysis.citations,
            evidence_snippets=analysis.evidence_snippets,
            evidence_doc_ids=analysis.evidence_doc_ids,
            missing_gaps=analysis.missing_gaps,
        )

    executive_summary_text = _build_executive_summary(state["topic"], completed_analyses)
    report_lines = [
        executive_summary_text if line == EXECUTIVE_SUMMARY_PLACEHOLDER else line
        for line in report_lines
    ]
    draft = "\n".join(report_lines).strip() + "\n"
    draft_version = state["draft_version"] + 1
    report_path = export_markdown_report(state["topic"], draft, draft_version)
    return {
        "draft": draft,
        "draft_version": draft_version,
        "report_path": report_path,
        "current_stage": "written",
    }
