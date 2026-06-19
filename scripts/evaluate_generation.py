import argparse
import json
import re
from pathlib import Path


SECTION_PATTERN = re.compile(r"^##\s+\d+\.\s+", re.MULTILINE)
CITATION_PATTERN = re.compile(r"\[C\d+\]")


def evaluate_report(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")

    section_count = len(SECTION_PATTERN.findall(text))
    citation_count = len(CITATION_PATTERN.findall(text))
    citation_index_count = text.count("### 引用索引")
    evidence_snippet_count = text.count("### 证据摘录")
    evidence_id_count = text.count("### 证据ID")
    gap_count = text.count("### 缺口")

    has_exec_summary = "## 执行摘要" in text
    has_report_info = "## 报告说明" in text

    score = 0
    score += 20 if has_report_info else 0
    score += 20 if has_exec_summary else 0
    score += min(section_count * 8, 24)
    score += min(citation_index_count * 4, 16)
    score += min(evidence_snippet_count * 4, 12)
    score += min(evidence_id_count * 4, 12)
    score += min(citation_count, 16)

    return {
        "report_path": str(path),
        "section_count": section_count,
        "citation_count": citation_count,
        "citation_index_count": citation_index_count,
        "evidence_snippet_count": evidence_snippet_count,
        "evidence_id_count": evidence_id_count,
        "gap_count": gap_count,
        "has_report_info": has_report_info,
        "has_exec_summary": has_exec_summary,
        "structure_score": min(score, 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated report structure.")
    parser.add_argument("--report", type=str, required=True, help="Path to markdown report.")
    args = parser.parse_args()

    result = evaluate_report(Path(args.report))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
