import asyncio
import hashlib
import logging

from tavily import TavilyClient

from src.config import settings
from src.services.knowledge_base import build_knowledge_base, search_hybrid_documents
from src.state import AgentState, RawDocument


logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= settings.search_max_content_chars:
        return compact
    return compact[: settings.search_max_content_chars].rstrip() + "..."


def _dedupe_documents(documents: list[RawDocument]) -> list[RawDocument]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[RawDocument] = []
    for doc in documents:
        key = (
            doc.chunk_id.strip(),
            doc.url.strip(),
            doc.summary[:160].strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


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
            documents.append(
                RawDocument(
                    doc_id=_web_doc_id(section_id, query, url, index),
                    title=title,
                    source=item.get("source") or "tavily",
                    url=url,
                    summary=_normalize_text(content or title),
                    content=_normalize_text(content or title),
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

    # 合并 Reviewer 建议的定向检索词（如果有的话）
    extra_queries = state.get("suggested_search_queries", [])
    if extra_queries:
        logger.info("Searcher 合并了 %s 条 Reviewer 建议的检索词。", len(extra_queries))

    documents: list[RawDocument] = []
    for section in state["outline"]:
        combined = list(section.search_queries) + extra_queries
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
