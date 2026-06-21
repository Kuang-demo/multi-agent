from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


ReviewDecision = Literal["", "pass", "research"]
IterationDecision = Literal["pass", "research"]
WorkflowStage = Literal[
    "created",
    "planned",
    "searched",
    "analyzed",
    "written",
    "reviewed",
]


# 一章报告的规划结果。
# Planner 会先把用户的大主题拆成多个 Section，后面的 Searcher/Analyzer/Writer
# 都是围绕这些章节来工作。
class Section(BaseModel):
    # 章节编号，用来在整个工作流里唯一标识这一章。
    section_id: int = Field(..., ge=1)
    # 章节标题，例如“业务价值与使用场景”。
    title: str
    # 这一章要回答什么问题，相当于章节写作目标。
    objective: str
    # 为这一章准备的检索词列表，Searcher 会拿这些词去做检索。
    search_queries: list[str] = Field(default_factory=list)


# 一条原始证据文档。
# 它可能来自本地知识库、向量检索、网页搜索。
class RawDocument(BaseModel):
    # 这条证据在当前系统中的唯一 ID。"vector-sec1-chunk89757"
    doc_id: str
    # 如果证据来自本地 chunk，这里记录 chunk 的 ID；网页结果通常为空。
    chunk_id: str = ""
    # 证据标题，例如文件标题或网页标题。
    title: str
    # 证据来源名称，例如 local-knowledge-base / tavily 。
    source: str
    # 原始链接或文件路径。
    url: str = ""
    # 给后续节点快速预览用的摘要。
    summary: str
    # 更完整的正文内容，Analyzer/Writer 往往会用它来生成分析和报告。
    content: str = ""
    # 这条证据属于哪一个章节。
    section_id: int = Field(..., ge=1)
    # 当前系统给这条证据打的相关度分数，范围 0~1。
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # 证据是通过什么方式召回的，例如 keyword / vector / web。
    retrieval_method: str = ""
    # 额外元数据，通常放 file_path、chunk_id、query 等补充信息。
    metadata: dict[str, str] = Field(default_factory=dict)


# 一条“提炼后的观点”。
# 它比 RawDocument 更抽象，代表系统基于若干证据总结出的一个 claim。
class Insight(BaseModel):
    # 洞察 ID，用来在状态里唯一标识一条洞察。
    insight_id: str
    # 这条洞察属于哪一个章节。
    section_id: int = Field(..., ge=1)
    # 洞察本身的文字描述。
    claim: str
    # 支撑这条洞察的证据 ID 列表。
    evidence_doc_ids: list[str] = Field(default_factory=list)
    # 系统对这条洞察的置信度估计，范围 0~1。
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# 某一章节的结构化分析结果。
# Analyzer 的核心输出就是 SectionAnalysis，Writer 会根据它来写报告正文。
class SectionAnalysis(BaseModel):
    # 对应哪一章。
    section_id: int = Field(..., ge=1)
    # 章节标题，便于后续节点直接展示。
    section_title: str
    # 这一章的整体结论或摘要。
    summary: str
    # 这一章提炼出的关键要点列表。
    key_points: list[str] = Field(default_factory=list)
    # 支撑本章分析的证据 ID。
    evidence_doc_ids: list[str] = Field(default_factory=list)
    # 证据正文片段摘录，方便 Writer 和 Reviewer 使用。
    evidence_snippets: list[str] = Field(default_factory=list)
    # 引用标签或引用说明，通常会在正文中配合 [C1]/[C2] 使用。
    citations: list[str] = Field(default_factory=list)
    # 当前章节还缺什么信息，帮助 Reviewer 判断是否需要回退重做。
    missing_gaps: list[str] = Field(default_factory=list)
    # 对整个章节分析结果的置信度估计。
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# 审查节点给出的反馈意见。
# Reviewer 会根据证据覆盖、引用完整性等规则生成这些评论。
class ReviewComment(BaseModel):
    # 评论 ID，用于去重和合并。
    comment_id: str
    # 问题严重程度。
    severity: Literal["critical", "major", "minor"]
    # 问题类型，例如证据缺失、引用缺失、覆盖不足、或通过。
    issue_type: Literal["evidence_gap", "citation_gap", "coverage_gap", "pass"]
    # 对问题本身的描述。
    description: str
    # 给下一步改进或回退的建议。
    recommendation: str


class IterationRecord(BaseModel):
    iteration: int = Field(..., ge=1)
    decision: IterationDecision
    reason: str = ""
    target_section_ids: list[int] = Field(default_factory=list)
    suggested_search_queries: list[str] = Field(default_factory=list)
    draft_version: int = 0


# LangGraph 在多次节点更新同一个字段时，需要知道"怎么合并"。
# 下面这些 merge_* 函数就是在定义各类结构化数据的合并策略。
def _document_quality_key(doc: RawDocument) -> tuple[float, int, int]:
    return (
        doc.relevance_score,
        len(doc.content or ""),
        len(doc.summary or ""),
    )


def _document_identity_key(doc: RawDocument) -> tuple[int, str, str]:
    if doc.chunk_id:
        return (doc.section_id, "chunk", doc.chunk_id)
    if doc.url:
        return (doc.section_id, "url", doc.url)
    return (doc.section_id, "doc", doc.doc_id)


def merge_documents(existing: list[RawDocument], new: list[RawDocument]) -> list[RawDocument]:
    merged = {_document_identity_key(doc): doc for doc in existing}
    for doc in new:
        key = _document_identity_key(doc)
        current = merged.get(key)
        if current is None or _document_quality_key(doc) >= _document_quality_key(current):
            merged[key] = doc
    return sorted(
        merged.values(),
        key=lambda doc: (doc.section_id, -doc.relevance_score, doc.doc_id),
    )


def merge_insights(existing: list[Insight], new: list[Insight]) -> list[Insight]:
    merged = {item.insight_id: item for item in existing}
    for item in new:
        merged[item.insight_id] = item
    return sorted(merged.values(), key=lambda item: (item.section_id, item.insight_id))


def merge_section_analyses(
    existing: list[SectionAnalysis], new: list[SectionAnalysis]
) -> list[SectionAnalysis]:
    merged = {item.section_id: item for item in existing}
    for item in new:
        merged[item.section_id] = item
    return sorted(merged.values(), key=lambda item: item.section_id)


def merge_iteration_history(
    existing: list[IterationRecord], new: list[IterationRecord]
) -> list[IterationRecord]:
    merged = {item.iteration: item for item in existing}
    for item in new:
        merged[item.iteration] = item
    return sorted(merged.values(), key=lambda item: item.iteration)


# 整个多智能体工作流共享的“全局状态”。
# 你可以把它理解成：所有节点围绕同一个大字典读写数据。
class AgentState(TypedDict):
    # 用户原始输入的问题或研究主题。
    query: str
    # Planner 最终确认的主题标题，通常比 query 更适合写入报告标题。
    topic: str
    # 报告大纲，每个元素都是一章 Section。
    outline: list[Section]
    # 所有章节的检索词汇总，便于整体查看或调试。
    search_queries: list[str]
    # Searcher 找回来的原始证据池。
    # Annotated + merge_documents 表示多个节点更新它时要按自定义规则合并。
    raw_documents: Annotated[list[RawDocument], merge_documents]
    # 从证据中提炼出的关键洞察。
    key_insights: Annotated[list[Insight], merge_insights]
    # 每一章的结构化分析结果。
    section_analyses: Annotated[list[SectionAnalysis], merge_section_analyses]
    # Writer 生成的整份报告草稿文本。
    draft: str
    # 报告草稿版本号，每次重写通常会递增。
    draft_version: int
    # 导出的报告文件路径。
    report_path: str
    # Reviewer 产生的审查反馈列表。
    review_feedback: list[ReviewComment]
    # Reviewer 的最终判断，例如 pass / research。
    review_decision: ReviewDecision
    # Reviewer 建议的定向补充检索词，当 decision=research 时 Searcher 会合并使用。
    suggested_search_queries: list[str]
    # Reviewer 判断需要补充检索的章节 ID；为空时表示没有定向补充目标。
    target_section_ids: list[int]
    # 每轮审查和回流的摘要记录，便于 API/CLI 展示运行过程。
    iteration_history: Annotated[list[IterationRecord], merge_iteration_history]
    # 当前工作流执行到了哪个阶段，例如 planned / searched / analyzed。
    current_stage: WorkflowStage
    # 当前已经循环了多少轮，用来避免无限回环。
    iteration_count: int

# 创建工作流的初始状态。
# 所有字段先放一个合理的默认值，后续由各节点逐步填充。
def create_initial_state(query: str) -> AgentState:
    return {
        "query": query,
        "topic": "",
        "outline": [],
        "search_queries": [],
        "raw_documents": [],
        "key_insights": [],
        "section_analyses": [],
        "draft": "",
        "draft_version": 0,
        "report_path": "",
        "review_feedback": [],
        "review_decision": "",
        "suggested_search_queries": [],
        "target_section_ids": [],
        "iteration_history": [],
        "current_stage": "created",
        "iteration_count": 0,
    }
