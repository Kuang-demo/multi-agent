import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.nodes.analyzer import analyzer_node
from src.nodes.planner import planner_node
from src.nodes.reviewer import reviewer_node
from src.nodes.searcher import searcher_node
from src.nodes.writer import writer_node
from src.state import AgentState, create_initial_state


logger = logging.getLogger(__name__)


def route_after_reviewer(state: AgentState) -> Literal["Searcher", "__end__"]:
    if state["review_decision"] == "pass":
        return "__end__"
    if state["iteration_count"] >= settings.max_iterations:
        return "__end__"
    return "Searcher"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("Planner", planner_node)
    builder.add_node("Searcher", searcher_node)
    builder.add_node("Analyzer", analyzer_node)
    builder.add_node("Writer", writer_node)
    builder.add_node("Reviewer", reviewer_node)

    builder.add_edge(START, "Planner")
    builder.add_edge("Planner", "Searcher")
    builder.add_edge("Searcher", "Analyzer")
    builder.add_edge("Analyzer", "Writer")
    builder.add_edge("Writer", "Reviewer")
    builder.add_conditional_edges(
        "Reviewer",
        route_after_reviewer,
        {"Searcher": "Searcher", "__end__": END},
    )
    return builder.compile(checkpointer=MemorySaver())


async def run_report(query: str, thread_id: str = "default") -> AgentState:
    graph = build_graph()
    initial_state = create_initial_state(query)
    config = {"configurable": {"thread_id": thread_id}}
    final_state = await graph.ainvoke(initial_state, config=config)
    logger.info(
        "Run complete: topic=%s decision=%s draft_version=%s report_path=%s",
        final_state["topic"],
        final_state["review_decision"],
        final_state["draft_version"],
        final_state["report_path"],
    )
    return final_state
