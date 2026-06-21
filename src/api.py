from pathlib import Path
import re
import shutil
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import settings
from src.graph import run_report
from src.services.knowledge_base import build_knowledge_base, get_knowledge_base_stats


app = FastAPI(title=settings.api_title, version=settings.api_version)


class ResearchRequest(BaseModel):
    query: str
    thread_id: str = "api"


ALLOWED_UPLOAD_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}


def _safe_upload_name(filename: str, suffix: str) -> str:
    raw_name = Path(filename or f"upload{suffix}").name
    stem = Path(raw_name).stem or "upload"
    safe_stem = re.sub(r"[^\w.-]+", "_", stem).strip("._") or "upload"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{safe_stem[:60]}{suffix}"


def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / _safe_upload_name(file.filename or "", suffix)

    with target.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return str(target)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": settings.api_title, "version": settings.api_version}


@app.get("/knowledge-base/stats")
async def knowledge_base_stats() -> dict:
    return get_knowledge_base_stats()


@app.post("/knowledge-base/rebuild")
async def rebuild_knowledge_base() -> dict:
    return build_knowledge_base(force_rebuild=True)


@app.post("/knowledge-base/upload")
async def upload_knowledge_file(file: UploadFile = File(...)) -> dict:
    saved_path = _save_upload(file)
    stats = build_knowledge_base(force_rebuild=False)
    return {
        "filename": file.filename,
        "saved_path": saved_path,
        "knowledge_base": stats,
    }


@app.post("/research")
async def create_research(request: ResearchRequest) -> dict:
    final_state = await run_report(query=request.query, thread_id=request.thread_id)
    return {
        "topic": final_state["topic"],
        "stage": final_state["current_stage"],
        "decision": final_state["review_decision"],
        "target_section_ids": final_state["target_section_ids"],
        "draft_version": final_state["draft_version"],
        "report_path": final_state["report_path"],
        "outline": [section.model_dump() for section in final_state["outline"]],
        "documents": [doc.model_dump() for doc in final_state["raw_documents"]],
        "analyses": [item.model_dump() for item in final_state["section_analyses"]],
        "insights": [item.model_dump() for item in final_state["key_insights"]],
        "review_feedback": [item.model_dump() for item in final_state["review_feedback"]],
        "iteration_history": [
            item.model_dump() for item in final_state["iteration_history"]
        ],
        "draft": final_state["draft"],
    }
