import asyncio
import hashlib
import logging

from tavily import TavilyClient

from src.config import settings
from src.services.knowledge_base import build_knowledge_base, search_hybrid_documents
from src.state import AgentState, RawDocument


logger = logging.getLogger(__name__)

NOISY_WEB_LINE_PATTERNS = (
    "联系我们",
    "邮箱登录",
    "ENGLISH",
    "中国科学院",
    "版权所有",
    "ICP备",
    "Copyright",
    "登录",
    "注册",
    "分享",
    "微信",
    "微博",
)


def _normalize_text(text: str) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= settings.search_max_content_chars:
        return compact
    return compact[: settings.search_max_content_chars].rstrip() + "..."


def _is_noisy_web_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if len(stripped) <= 8 and any(char in stripped for char in "|[]{}"):
        return True
    if any(pattern in stripped for pattern in NOISY_WEB_LINE_PATTERNS):
        return True

    link_count = stripped.count("http://") + stripped.count("https://") + stripped.count("](")
    if link_count >= 3:
        return True
    if stripped.startswith(("[![", "![", "* [", "- [")) and link_count >= 1:
        return True
    return False


def _clean_web_content(text: str, fallback: str) -> str:
    raw = text or fallback
    lines = []
    seen: set[str] = set()
    for line in raw.replace("\r", "\n").split("\n"):
        compact = " ".join(line.split())
        if _is_noisy_web_line(compact):
            continue
        key = compact[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(compact)

    cleaned = " ".join(lines).strip()
    if not cleaned:
        cleaned = fallback
    return _normalize_text(cleaned or fallback)


def _normalize_queries(queries: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = " ".join((query or "").split())[:80].strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _document_identity_key(doc: RawDocument) -> tuple[int, str, str]:
    if doc.chunk_id:
        return (doc.section_id, "chunk", doc.chunk_id)
    if doc.url:
        return (doc.section_id, "url", doc.url.strip().lower())
    summary_key = doc.summary[:160].strip().lower()
    return (doc.section_id, "summary", summary_key or doc.doc_id)


def _document_quality_key(doc: RawDocument) -> tuple[float, int, int]:
    return (
        doc.relevance_score,
        len(doc.content or ""),
        len(doc.summary or ""),
    )


def _dedupe_documents(documents: list[RawDocument]) -> list[RawDocument]:
    deduped: dict[tuple[int, str, str], RawDocument] = {}
    for doc in documents:
        key = _document_identity_key(doc)
        current = deduped.get(key)
        if current is None or _document_quality_key(doc) >= _document_quality_key(current):
            deduped[key] = doc
    return sorted(
        deduped.values(),
        key=lambda doc: (doc.section_id, -doc.relevance_score, doc.doc_id),
    )


def _search_query(client: TavilyClient, query: str) -> list[dict]:
    response = client.search(
        query=query,
        max_results=settings.search_results_per_query,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=True,
    )
    return response.get("results", [])


def _web_doc_id(section_id: int, query: str, url: str, index: int) -> str:
    raw = f"{section_id}:{query}:{url}:{index}".encode("utf-8")
    return f"web-{hashlib.sha1(raw).hexdigest()[:12]}"


async def _search_web_documents(section_id: int, topic: str, queries: list[str]) -> list[RawDocument]:
    if not settings.tavily_api_key:
        return []

    client = TavilyClient(api_key=settings.tavily_api_key)
    documents: list[RawDocument] = []

    for query in queries:
        try:
            results = await asyncio.to_thread(_search_query, client, query)
        except Exception as exc:
            logger.warning("Tavily search failed for query '%s': %s", query, exc)
            continue

        for index, item in enumerate(results, start=1):
            title = item.get("title") or f"{topic} search result {index}"
            url = item.get("url", "")
            content = item.get("raw_content") or item.get("content") or item.get("snippet") or ""
            cleaned_content = _clean_web_content(content, title)
            documents.append(
                RawDocument(
                    doc_id=_web_doc_id(section_id, query, url, index),
                    title=title,
                    source=item.get("source") or "tavily",
                    url=url,
                    summary=cleaned_content,
                    content=cleaned_content,
                    section_id=section_id,
                    relevance_score=float(item.get("score") or 0.7),
                    retrieval_method="web",
                    metadata={"query": query},
                )
            )

    return _dedupe_documents(documents)


async def _search_section(section_id: int, topic: str, queries: list[str]) -> list[RawDocument]:
    local_documents: list[RawDocument] = []
    for query in queries:
        local_documents.extend(search_hybrid_documents(query=query, section_id=section_id))

    local_documents = _dedupe_documents(local_documents)
    if local_documents:
        logger.info(
            "Hybrid local retrieval returned %s documents for section %s.",
            len(local_documents),
            section_id,
        )

    web_documents = await _search_web_documents(section_id=section_id, topic=topic, queries=queries)
    if web_documents:
        logger.info(
            "Web retrieval returned %s documents for section %s.",
            len(web_documents),
            section_id,
        )

    # 分别取 top-k：避免跨源分数不可比。两边各占一半，一方空则由另一方补足。
    total = settings.hybrid_retrieval_top_k
    half = max(total // 2, 1)
    local_sorted = sorted(local_documents, key=lambda d: d.relevance_score, reverse=True)
    web_sorted = sorted(web_documents, key=lambda d: d.relevance_score, reverse=True)
    merged = _dedupe_documents(local_sorted[:half] + web_sorted[:half])
    if not merged:
        merged = _dedupe_documents(local_sorted[:total] + web_sorted[:total])
    if not merged:
        raise RuntimeError(
            f"Section {section_id} failed to retrieve any evidence from local or web sources."
        )

    return merged[:total]


async def searcher_node(state: AgentState) -> dict:
    # 整轮搜索只构建一次知识库，避免每章重复跑语义切分；
    # 用 to_thread 避免同步 I/O 阻塞异步事件循环。
    await asyncio.to_thread(build_knowledge_base)

    # 合并 Reviewer 建议的定向检索词（如果有的话）。
    # 如果 Reviewer 指定了章节，只对这些章节追加补充 query，避免全局重搜污染其它章节结果。
    extra_queries = _normalize_queries(state.get("suggested_search_queries", []))
    target_section_ids = set(state.get("target_section_ids", []))
    if extra_queries:
        logger.info(
            "Searcher received %s reviewer queries for target sections: %s.",
            len(extra_queries),
            sorted(target_section_ids) if target_section_ids else "all",
        )

    documents: list[RawDocument] = []
    for section in state["outline"]:
        if extra_queries and target_section_ids and section.section_id not in target_section_ids:
            continue

        should_apply_extra = extra_queries and (
            not target_section_ids or section.section_id in target_section_ids
        )
        combined = _normalize_queries(section.search_queries)
        if should_apply_extra:
            combined = _normalize_queries(combined + extra_queries)
        section_documents = await _search_section(
            section_id=section.section_id,
            topic=state["topic"],
            queries=combined,
        )
        documents.extend(section_documents)

    return {
        "raw_documents": documents,
        "current_stage": "searched",
    }
