import logging
import re

from pydantic import BaseModel, Field

from src.services.llm_json import invoke_json_schema
from src.services.report_exporter import export_markdown_report
from src.state import AgentState, SectionAnalysis


logger = logging.getLogger(__name__)


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
    return "\n".join(f"> {snippet}" for snippet in snippets)


def _ensure_citations(text: str, citations: list[str]) -> str:
    if not citations:
        return text
    if re.search(r"\[C\d+\]", text):
        return text
    return f"{text.rstrip()} [C1]"


async def _llm_write_section(topic: str, analysis: SectionAnalysis) -> WriterOutputSchema:
    user_prompt = (
        f"研究主题：{topic}\n"
        f"章节标题：{analysis.section_title}\n"
        f"章节摘要：{analysis.summary}\n"
        f"关键要点：{analysis.key_points}\n"
        f"引用索引：{analysis.citations}\n"
        f"证据摘录：{analysis.evidence_snippets}\n"
        f"缺口：{analysis.missing_gaps}\n\n"
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
    executive_summaries: list[str] = []

    report_lines: list[str] = [
        f"# {state['topic']}",
        "",
        "## 报告说明",
        "- 生成方式：离线多智能体工作流",
        f"- 草稿版本：v{state['draft_version'] + 1}",
        "- 节点流程：Planner -> Searcher -> Analyzer -> Writer -> Reviewer",
        "",
        "## 执行摘要",
    ]

    for section in state["outline"]:
        analysis = analyses_by_section.get(section.section_id)
        if analysis is None:
            continue

        llm_result = await _llm_write_section(state["topic"], analysis)
        executive_summaries.append(llm_result.executive_summary)

        section_body = _ensure_citations(llm_result.section_body, analysis.citations)
        logger.info("Writer 使用 LLM 完成了 section %s。", section.section_id)

        report_lines.append("")
        report_lines.append(f"## {section.section_id}. {section.title}")
        report_lines.append(f"章节目标：{section.objective}")
        report_lines.append("")
        report_lines.append("### 章节正文")
        report_lines.append(section_body)
        report_lines.append("")
        report_lines.append("### 引用索引")
        report_lines.append(_format_sources(analysis.citations))
        report_lines.append("")
        report_lines.append("### 证据摘录")
        report_lines.append(_format_snippets(analysis.evidence_snippets))
        report_lines.append("")
        report_lines.append("### 证据ID")
        report_lines.append(
            ", ".join(analysis.evidence_doc_ids) if analysis.evidence_doc_ids else "暂无"
        )
        report_lines.append("")
        if analysis.missing_gaps:
            report_lines.append("### 缺口")
            report_lines.extend(f"- {gap}" for gap in analysis.missing_gaps)
            report_lines.append("")

    report_lines.insert(8, " ".join(executive_summaries))
    draft = "\n".join(report_lines).strip() + "\n"
    draft_version = state["draft_version"] + 1
    report_path = export_markdown_report(state["topic"], draft, draft_version)
    return {
        "draft": draft,
        "draft_version": draft_version,
        "report_path": report_path,
        "current_stage": "written",
    }
