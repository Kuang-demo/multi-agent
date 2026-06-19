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

#   函数会把前 3 条证据加工成引用索引字符串。
def _build_citations(docs: list[RawDocument]) -> list[str]:
    citations: list[str] = []
    for index, doc in enumerate(docs[:3], start=1):
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
    for index, doc in enumerate(docs[:3], start=1):
        evidence_blocks.append(
            f"[C{index}]\n"
            f"标题：{doc.title}\n"  ## 证据标题，例如文件标题或网页标题。
            f"检索方式：{doc.retrieval_method}\n"
            f"分数：{doc.relevance_score:.3f}\n"
            f"内容：{doc.content or doc.summary}"
        )
    user_prompt = (
        f"章节标题：{section_title}\n"
        f"章节目标：{objective}\n\n"
        "证据片段：\n"
        + "\n\n".join(evidence_blocks)
        + "\n\n请让关键事实尽量和 [C1] [C2] [C3] 这些证据标签保持对应。"
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
        docs = sorted(
            grouped_docs.get(section.section_id, []),
            key=lambda item: item.relevance_score,  #按 relevance_score 降序排序
            reverse=True,
        )
        top_docs = docs[:3]
        evidence_doc_ids = [doc.doc_id for doc in top_docs]
        # 用 500 字符，优先在句子边界处截断，避免切碎单词
        _raw_snippets: list[str] = []
        for doc in top_docs:
            text = doc.content or doc.summary
            if len(text) <= 500:
                _raw_snippets.append(text)
            else:
                # 在 [400, 500] 区间内找最靠后的句子边界
                best_pos = 0
                for boundary_char in "。！？\n":
                    pos = text.rfind(boundary_char, 400, 500)
                    if pos > best_pos:
                        best_pos = pos
                cut = best_pos + 1 if best_pos > 0 else 500
                _raw_snippets.append(text[:cut])
        evidence_snippets = _raw_snippets

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
                claim=f"{section.title} 可以基于当前证据形成一版可解释的章节草稿。",
                evidence_doc_ids=evidence_doc_ids,
                confidence=confidence,
            )
        )

    return {
        "key_insights": insights,
        "section_analyses": analyses,
        "current_stage": "analyzed",
    }
