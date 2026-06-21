import logging
from collections import defaultdict

from pydantic import BaseModel, Field

from src.services.llm_json import invoke_json_schema
from src.state import AgentState, Insight, RawDocument, SectionAnalysis


logger = logging.getLogger(__name__)


class AnalyzerOutputSchema(BaseModel):
    summary: str = Field(..., min_length=20, max_length=600)
    key_points: list[str] = Field(..., min_length=2, max_length=4)
    missing_gaps: list[str] = Field(default_factory=list, max_length=3)
    confidence: float = Field(..., ge=0.0, le=1.0)


ANALYZER_SYSTEM_PROMPT = """
你是研究工作流中的分析节点。
你会收到一个章节目标和若干条检索到的证据片段。
你的任务是基于这些证据做章节级分析。
不要编造证据中没有的信息。
如果证据不足，要明确指出缺口。
输出尽量简洁、专业、中文化。
"""


def _truncate_text(text: str, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact

    best_pos = 0
    for boundary_char in "。！？.!?\n":
        pos = compact.rfind(boundary_char, max_chars - 120, max_chars)
        if pos > best_pos:
            best_pos = pos
    cut = best_pos + 1 if best_pos > 0 else max_chars
    return compact[:cut].rstrip() + "..."


def _select_top_documents(docs: list[RawDocument], limit: int = 4) -> list[RawDocument]:
    return sorted(
        docs,
        key=lambda doc: (doc.relevance_score, len(doc.content or doc.summary)),
        reverse=True,
    )[:limit]


def _build_evidence_snippets(docs: list[RawDocument]) -> list[str]:
    return [_truncate_text(doc.content or doc.summary, 500) for doc in docs]


def _build_empty_analysis(section_title: str, section_id: int) -> SectionAnalysis:
    return SectionAnalysis(
        section_id=section_id,
        section_title=section_title,
        summary="当前章节没有检索到可支撑事实性分析的有效证据，暂不生成结论。",
        key_points=[],
        evidence_doc_ids=[],
        evidence_snippets=[],
        citations=[],
        missing_gaps=["未检索到可支撑本章节的有效证据"],
        confidence=0.0,
    )


def _build_insight_claim(summary: str, key_points: list[str], missing_gaps: list[str]) -> str:
    if key_points:
        return key_points[0]
    if summary:
        return summary
    if missing_gaps:
        return f"当前章节仍存在信息缺口：{missing_gaps[0]}"
    return "当前章节缺少足够证据，暂不形成事实性洞察。"


#   函数会把前 4 条证据加工成引用索引字符串。
def _build_citations(docs: list[RawDocument]) -> list[str]:
    citations: list[str] = []
    for index, doc in enumerate(docs[:4], start=1):
        citations.append(
            f"[C{index}] {doc.title} | {doc.retrieval_method} | score={doc.relevance_score:.3f} | {doc.url or doc.source}"
        )
    return citations


async def _llm_analyze_section(
    section_title: str,
    objective: str,
    docs: list[RawDocument],
) -> AnalyzerOutputSchema:
    evidence_blocks = []
    for index, doc in enumerate(docs[:4], start=1):
        evidence_blocks.append(
            f"[C{index}]\n"
            f"标题：{doc.title}\n"  ## 证据标题，例如文件标题或网页标题。
            f"检索方式：{doc.retrieval_method}\n"
            f"分数：{doc.relevance_score:.3f}\n"
            f"内容：{_truncate_text(doc.content or doc.summary, 900)}"
        )
    user_prompt = (
        f"章节标题：{section_title}\n"
        f"章节目标：{objective}\n\n"
        "证据片段：\n"
        + "\n\n".join(evidence_blocks)
        + "\n\n请让关键事实尽量和 [C1] [C2] [C3] [C4]这些证据标签保持对应。"
    )
    return await invoke_json_schema(
        system_prompt=ANALYZER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=AnalyzerOutputSchema,
        temperature=0.1,
    )


async def analyzer_node(state: AgentState) -> dict:
    grouped_docs = defaultdict(list)
    for doc in state["raw_documents"]:
        grouped_docs[doc.section_id].append(doc)
    # "cat" 不存在 → 自动生成空列表 []，再执行 append
    # grouped_docs["cat"].append("小猫文档1")
    # grouped_docs["cat"].append("小猫文档2")
    # grouped_docs["dog"].append("小狗文档1")

    # print(dict(grouped_docs))
    # # {'cat': ['小猫文档1', '小猫文档2'], 'dog': ['小狗文档1']}

    insights: list[Insight] = []
    analyses: list[SectionAnalysis] = []

    for section in state["outline"]:
        docs = grouped_docs.get(section.section_id, [])
        top_docs = _select_top_documents(docs)
        evidence_doc_ids = [doc.doc_id for doc in top_docs]
        evidence_snippets = _build_evidence_snippets(top_docs)

        if not top_docs:
            analysis = _build_empty_analysis(section.title, section.section_id)
            analyses.append(analysis)
            insights.append(
                Insight(
                    insight_id=f"insight-{section.section_id}",
                    section_id=section.section_id,
                    claim=_build_insight_claim(
                        analysis.summary,
                        analysis.key_points,
                        analysis.missing_gaps,
                    ),
                    evidence_doc_ids=[],
                    confidence=0.0,
                )
            )
            logger.warning(
                "Analyzer skipped section %s because no evidence was available.",
                section.section_id,
            )
            continue

        llm_result = await _llm_analyze_section(section.title, section.objective, top_docs)
        summary = llm_result.summary[:600]
        key_points = llm_result.key_points
        missing_gaps = llm_result.missing_gaps
        confidence = llm_result.confidence
        logger.info("Analyzer 使用 LLM 完成了 section %s。", section.section_id)

        analyses.append(
            SectionAnalysis(
                section_id=section.section_id,
                section_title=section.title,
                summary=summary,
                key_points=key_points,
                evidence_doc_ids=evidence_doc_ids,
                evidence_snippets=evidence_snippets,
                citations=_build_citations(top_docs),
                missing_gaps=missing_gaps,
                confidence=confidence,
            )
        )

        insights.append(
            Insight(
                insight_id=f"insight-{section.section_id}",
                section_id=section.section_id,
                claim=_build_insight_claim(summary, key_points, missing_gaps),
                evidence_doc_ids=evidence_doc_ids,
                confidence=confidence,
            )
        )

    return {
        "key_insights": insights,
        "section_analyses": analyses,
        "current_stage": "analyzed",
    }
