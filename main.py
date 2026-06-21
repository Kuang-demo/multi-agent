import argparse
import asyncio
import json
import logging
import sys

from src.graph import run_report


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="DeepResearch-Agent CLI runner")
    parser.add_argument("--query", "-q", type=str, help="Research topic.")
    parser.add_argument(
        "--thread-id",
        type=str,
        default="cli",
        help="LangGraph thread id for checkpoint isolation.",
    )
    args = parser.parse_args()

    query = args.query or input("请输入研究主题: ").strip()
    if not query:
        print("研究主题不能为空。")
        return 1

    final_state = await run_report(query=query, thread_id=args.thread_id)
    summary = {
        "topic": final_state["topic"],
        "stage": final_state["current_stage"],
        "decision": final_state["review_decision"],
        "iterations": final_state["iteration_count"],
        "target_section_ids": final_state["target_section_ids"],
        "draft_version": final_state["draft_version"],
        "documents": len(final_state["raw_documents"]),
        "analyses": len(final_state["section_analyses"]),
        "report_path": final_state["report_path"],
        "iteration_history": [
            item.model_dump() for item in final_state["iteration_history"]
        ],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(final_state["draft"])
    return 0


def main() -> None:
    exit_code = asyncio.run(async_main())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
