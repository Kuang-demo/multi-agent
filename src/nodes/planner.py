import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.services.llm_json import invoke_json_schema
from src.state import AgentState, Section


logger = logging.getLogger(__name__)


class PlannerSectionSchema(BaseModel):
    title: str = Field(..., min_length=4, max_length=30)
    objective: str = Field(..., min_length=10, max_length=80)
    search_queries: list[str] = Field(..., min_length=3, max_length=4)


class PlannerOutputSchema(BaseModel):
    topic: str = Field(..., min_length=4, max_length=80)
    sections: list[PlannerSectionSchema] = Field(..., min_length=5, max_length=6)


PLANNER_SYSTEM_PROMPT = """
你是一个研究工作流中的规划节点。
你的任务是把用户给出的研究主题拆成可执行的报告结构和检索查询。
你最优先考虑的是后续检索质量，而不是文风。
章节标题要简洁、明确、可落地。
检索词要尽量贴近中文知识库和网页搜索的实际表达。
"""


async def _llm_plan(query: str) -> tuple[str, list[Section]]:
    user_prompt = (
        "请为下面这个研究主题生成一个 5 到 6 章的报告规划。\n"
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
            title=section.title,
            objective=section.objective,
            search_queries=section.search_queries,
        )
        for index, section in enumerate(result.sections, start=1)
    ]
    return result.topic, outline


async def planner_node(state: AgentState) -> dict:
    topic, outline = await _llm_plan(state["query"])
    logger.info("Planner 使用 LLM 生成了动态大纲。")

    all_queries = [query_text for section in outline for query_text in section.search_queries]
    return {
        "topic": topic,
        "outline": outline,
        "search_queries": all_queries,
        "current_stage": "planned",
        "messages": [
            SystemMessage(content="你是研究工作流中的规划节点。"),
            HumanMessage(content=state["query"]),
        ],
    }
