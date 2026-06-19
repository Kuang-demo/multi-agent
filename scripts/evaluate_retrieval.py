import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.knowledge_base import (  # noqa: E402
    build_knowledge_base,
    search_hybrid_documents,
    search_keyword_documents,
    search_vector_documents,
)


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def hit_at_k(results: list[str], expected_titles: list[str]) -> int:
    expected = {title.lower() for title in expected_titles}
    return 1 if any(title.lower() in expected for title in results) else 0


def reciprocal_rank(results: list[str], expected_titles: list[str]) -> float:
    expected = {title.lower() for title in expected_titles}
    for index, title in enumerate(results, start=1):
        if title.lower() in expected:
            return 1.0 / index
    return 0.0


def evaluate_case(case: dict) -> dict:
    query = case["query"]
    section_id = case.get("section_id", 1)
    expected_titles = case["expected_titles"]

    keyword_docs = search_keyword_documents(query=query, section_id=section_id, top_k=4)
    vector_docs = search_vector_documents(query=query, section_id=section_id, top_k=4)
    hybrid_docs = search_hybrid_documents(query=query, section_id=section_id)

    keyword_titles = [doc.title for doc in keyword_docs]
    vector_titles = [doc.title for doc in vector_docs]
    hybrid_titles = [doc.title for doc in hybrid_docs]

    return {
        "query": query,
        "expected_titles": expected_titles,
        "keyword_titles": keyword_titles,
        "vector_titles": vector_titles,
        "hybrid_titles": hybrid_titles,
        "keyword_hit": hit_at_k(keyword_titles, expected_titles),
        "vector_hit": hit_at_k(vector_titles, expected_titles),
        "hybrid_hit": hit_at_k(hybrid_titles, expected_titles),
        "keyword_mrr": reciprocal_rank(keyword_titles, expected_titles),
        "vector_mrr": reciprocal_rank(vector_titles, expected_titles),
        "hybrid_mrr": reciprocal_rank(hybrid_titles, expected_titles),
    }


def summarize(results: list[dict]) -> dict:
    total = max(len(results), 1)
    return {
        "cases": len(results),
        "keyword_hit_rate": round(sum(item["keyword_hit"] for item in results) / total, 4),
        "vector_hit_rate": round(sum(item["vector_hit"] for item in results) / total, 4),
        "hybrid_hit_rate": round(sum(item["hybrid_hit"] for item in results) / total, 4),
        "keyword_mrr": round(sum(item["keyword_mrr"] for item in results) / total, 4),
        "vector_mrr": round(sum(item["vector_mrr"] for item in results) / total, 4),
        "hybrid_mrr": round(sum(item["hybrid_mrr"] for item in results) / total, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate keyword/vector/hybrid retrieval.")
    parser.add_argument(
        "--cases",
        type=str,
        default="data/eval/retrieval_cases.json",
        help="Path to retrieval evaluation cases.",
    )
    args = parser.parse_args()

    build_knowledge_base(force_rebuild=False)
    cases = load_cases(Path(args.cases))
    results = [evaluate_case(case) for case in cases]
    summary = summarize(results)

    print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
