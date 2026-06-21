import logging

from pydantic import BaseModel, Field

from src.config import settings
from src.services.llm_json import invoke_json_schema
from src.state import AgentState, Section


logger = logging.getLogger(__name__)


class PlannerSectionSchema(BaseModel):
    title: str = Field(..., min_length=4, max_length=30)
    objective: str = Field(..., min_length=10, max_length=80)
    search_queries: list[str] = Field(..., min_length=3, max_length=4)


class PlannerOutputSchema(BaseModel):
    topic: str = Field(..., min_length=4, max_length=80)
    sections: list[PlannerSectionSchema] = Field(
        ...,
        min_length=settings.min_sections,
        max_length=settings.max_sections,
    )


PLANNER_SYSTEM_PROMPT = """
你是一个研究工作流中的规划节点。
你的任务是把用户给出的研究主题拆成可执行的报告结构和检索查询。
你最优先考虑的是后续检索质量，而不是文风。
章节标题要简洁、明确、可落地。
检索词要尽量贴近中文知识库和网页搜索的实际表达。
"""


def _clean_text(value: str, max_length: int) -> str:
    return " ".join(value.split())[:max_length].strip()


def _normalize_search_queries(
    topic: str,
    section_title: str,
    objective: str,
    queries: list[str],
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for query in queries:
        cleaned = _clean_text(query, 80)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) >= 4:
            break

    fallback_candidates = [
        f"{section_title} {topic}",
        f"{section_title} {objective}",
        f"{topic} {section_title}",
    ]
    for candidate in fallback_candidates:
        cleaned = _clean_text(candidate, 80)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            normalized.append(cleaned)
        if len(normalized) >= 3:
            break

    return normalized[:4]


async def _llm_plan(query: str) -> tuple[str, list[Section]]:
    user_prompt = (
        f"请为下面这个研究主题生成一个 {settings.min_sections} 到 {settings.max_sections} 章的报告规划。\n"
        "要求：\n"
        "1. 章节标题用中文。\n"
        "2. 每章都要有明确目标。\n"
        "3. 每章提供 3 到 4 个检索查询。\n"
        "4. 检索查询要尽量利于中文知识库命中。\n\n"
        f"研究主题：\n{query}"
    )
    result = await invoke_json_schema(
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=PlannerOutputSchema,
        temperature=0.1,
    )
    outline = [
        Section(
            section_id=index,
            title=_clean_text(section.title, 30),
            objective=_clean_text(section.objective, 80),
            search_queries=_normalize_search_queries(
                topic=result.topic,
                section_title=section.title,
                objective=section.objective,
                queries=section.search_queries,
            ),
        )
        for index, section in enumerate(result.sections, start=1)
    ]
    return _clean_text(result.topic, 80), outline


async def planner_node(state: AgentState) -> dict:
    topic, outline = await _llm_plan(state["query"])
    logger.info("Planner 使用 LLM 生成了动态大纲。")

    all_queries = list(
        dict.fromkeys(query_text for section in outline for query_text in section.search_queries)
    )
    return {
        "topic": topic,
        "outline": outline,
        "search_queries": all_queries,
        "current_stage": "planned",
    }
