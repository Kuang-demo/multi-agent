from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[\\/:*?\"<>|]+", "_", value.strip())
    normalized = re.sub(r"\s+", "_", normalized)
    return normalized[:60] or "report"


def export_markdown_report(topic: str, content: str, draft_version: int) -> str:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}_v{draft_version}_{_safe_name(topic)}.md"
    path = output_dir / file_name
    path.write_text(content, encoding="utf-8")
    return str(path)
