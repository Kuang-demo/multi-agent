from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

if importlib.util.find_spec("langchain_community"):
    from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
else:
    TextLoader = PyPDFLoader = Docx2txtLoader = None  # type: ignore

from src.config import has_embedding_credentials, settings
from src.services.embeddings import DashScopeEmbeddings
from src.services.storage import ensure_database, get_connection
from src.state import RawDocument


logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}
LATIN_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+|\n+")


@dataclass(frozen=True)
class LocalChunk:
    chunk_id: str
    title: str
    path: Path
    text: str
    content_hash: str


def _normalize_title(path: Path) -> str:
    return path.stem.replace("_", " ")


def _hash_content(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _chunk_id_for_path(path: Path, chunk_index: int) -> str:
    try:
        relative_path = path.relative_to(Path(settings.local_data_dir))
    except ValueError:
        relative_path = path
    path_hash = hashlib.sha1(str(relative_path).encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}-{path_hash}-{chunk_index}"


def _load_documents_from_file(path: Path) -> list[dict[str, str]]:
    """
    PDF / DOCX 解析，直接用 LangChain 的 loader
    """
    if TextLoader is None:
        raise RuntimeError(
            "langchain_community is required to load documents. "
            "Install it with: pip install langchain-community"
        )

    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        return []

    docs = loader.load()
    normalized_docs: list[dict[str, str]] = []
    for index, doc in enumerate(docs, start=1):
        content = _clean_text((doc.page_content or "").replace("\r\n", "\n"))
        if not content:
            continue
        normalized_docs.append(
            {
                "title": _normalize_title(path),
                "path": str(path),
                "content": content,
            }
        )
    return normalized_docs


def _split_into_sentences(text: str) -> list[str]:
    """把长文本拆成句子序列，中文优先兼顾英文句号。

    这是结构感知分块的第三层兜底：当段落超过 chunk_size 时，
    退到句子级累积，保证不会把任何句子腰斩。"""
    normalized = re.sub(r"\n{2,}", "\n", text.replace("\r\n", "\n")).strip()
    if not normalized:
        return []

    pieces = SENTENCE_SPLIT_PATTERN.split(normalized)
    sentences = []
    for piece in pieces:
        sentence = piece.strip()
        if sentence:
            sentences.append(sentence)
    return sentences


# ── 结构化分块 ─────────────────────────────────────
# 三层降级：md 标题拆节 → 段落拆段 → 句子兜底
# 无外部 API 依赖，纯 Python 操作

_MD_HEADER_RE = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
_PAGE_NUM_RE = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE)


def _clean_text(text: str) -> str:
    """文档清洗：去页眉页码、多余换行、零宽字符。"""
    # 纯数字页码行
    text = _PAGE_NUM_RE.sub('', text)
    # 连续 3 个以上换行 → 2 个
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 零宽字符（常见于 PDF 提取）
    text = text.replace('​', '').replace('‌', '').replace('‍', '').replace('﻿', '')
    # 行首行尾空白统一
    return text.strip()


def _split_md_headers(text: str) -> list[str]:
    """按 Markdown 标题拆成节。无标题时返回整篇。"""
    matches = list(_MD_HEADER_RE.finditer(text))
    if not matches:
        return [text] if text.strip() else []

    sections: list[str] = []
    for i, match in enumerate(matches):
        section_start = match.start()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[section_start:section_end].strip()
        if section:
            sections.append(section)

    if matches and matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.insert(0, preamble)

    return sections


def _split_paragraphs(text: str) -> list[str]:
    """按双换行拆段，过滤空行。"""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def _accumulate_units(
    units: list[str], chunk_size: int, overlap_chars: int
) -> list[str]:
    """逐段累积到 chunk_size，超出时在段落边界断开。

    重叠窗口是字符级的：下一个 chunk 会从上一个 chunk 末尾往回包含约
    overlap_chars 个字符的段落，避免语义被切在边界上。"""
    if not units:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(units):
        current: list[str] = []
        current_len = 0
        i = start
        while i < len(units):
            if current and current_len + len(units[i]) > chunk_size:
                break
            current.append(units[i])
            current_len += len(units[i])
            i += 1

        if not current:
            long_unit = units[start]
            sentences = _split_into_sentences(long_unit)
            sub_chunks = _accumulate_units(
                sentences, chunk_size, overlap_chars=overlap_chars
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append(' '.join(current))

        if i >= len(units):
            break

        # 字符级重叠：从上一个 chunk 末尾往前数 overlap_chars 字
        overlap_count = 0
        new_start = i
        for j in range(i - 1, start - 1, -1):
            overlap_count += len(units[j])
            new_start = j
            if overlap_count >= overlap_chars:
                break
        start = max(start + 1, new_start)
        if start >= len(units):
            break

    return chunks


def _chunk_text(text: str, file_ext: str) -> list[str]:
    """结构化分块编排：根据格式选择分层策略。"""
    # 第 1 层：.md 按标题拆节，其他格式整篇当一节
    if file_ext == ".md":
        sections = _split_md_headers(text)
    else:
        sections = [text] if text.strip() else []

    all_chunks: list[str] = []
    for section in sections:
        # 第 2 层：节内按段落拆段
        paragraphs = _split_paragraphs(section)
        # 第 3 层：段落累积成 chunk，超长段落退到句子级（_accumulate_units 内部处理）
        section_chunks = _accumulate_units(
            paragraphs,
            settings.local_chunk_size,
            overlap_chars=settings.local_chunk_overlap,
        )
        all_chunks.extend(section_chunks)

    return all_chunks


# ── 文件加载与知识库构建 ─────────────────────────

def _load_chunks_from_files() -> list[LocalChunk]:
    base_dir = Path(settings.local_data_dir)
    if not base_dir.exists():
        return []

    chunks: list[LocalChunk] = []

    for path in sorted(base_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        documents = _load_documents_from_file(path)
        title = _normalize_title(path)
        chunk_index = 1

        for document in documents:
            chunk_texts = _chunk_text(document["content"], path.suffix.lower())
            for chunk_text in chunk_texts:
                chunk_id = _chunk_id_for_path(path, chunk_index)
                chunk_index += 1
                chunks.append(
                    LocalChunk(
                        chunk_id=chunk_id,
                        title=title,
                        path=path,
                        text=chunk_text,
                        content_hash=_hash_content(chunk_text),
                    )
                )
    return chunks


def _tokenize(text: str) -> set[str]:
    latin_tokens = {token.lower() for token in LATIN_TOKEN_PATTERN.findall(text)}
    chinese_bigrams: set[str] = set()
    chinese_chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    for index in range(len(chinese_chars)):
        chinese_bigrams.add(chinese_chars[index])
        if index + 1 < len(chinese_chars):
            chinese_bigrams.add(chinese_chars[index] + chinese_chars[index + 1])
    return latin_tokens | chinese_bigrams


def _keyword_score(query: str, title: str, text: str) -> float:
    query_terms = _tokenize(query)
    content_terms = _tokenize(f"{title} {text}")
    if not query_terms or not content_terms:
        return 0.0

    overlap = len(query_terms & content_terms) / len(query_terms)
    exact_bonus = 0.2 if query.lower() in text.lower() else 0.0
    return min(1.0, overlap + exact_bonus)


def _get_vector_store() -> Any | None:
    if not has_embedding_credentials():
        return None
    if not importlib.util.find_spec("langchain_chroma"):
        return None

    from langchain_chroma import Chroma

    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=DashScopeEmbeddings(),
    )


def _upsert_chunk_rows(chunks: list[LocalChunk]) -> None:
    ensure_database()
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO knowledge_chunks (chunk_id, file_path, title, content, content_hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                file_path=excluded.file_path,
                title=excluded.title,
                content=excluded.content,
                content_hash=excluded.content_hash
            """,
            [
                (
                    chunk.chunk_id,
                    str(chunk.path),
                    chunk.title,
                    chunk.text,
                    chunk.content_hash,
                )
                for chunk in chunks
            ],
        )
        conn.commit()


def build_knowledge_base(force_rebuild: bool = False) -> dict[str, int]:
    chunks = _load_chunks_from_files()
    _upsert_chunk_rows(chunks)

    vector_store = _get_vector_store()
    if vector_store:
        existing = vector_store.get(include=["metadatas"])
        existing_ids = set(existing["ids"])
        existing_hashes = {
            chunk_id: metadata.get("content_hash", "")
            for chunk_id, metadata in zip(existing["ids"], existing["metadatas"])
            if metadata
        }
        if force_rebuild and existing_ids:
            vector_store.delete(ids=list(existing_ids))
            existing_ids = set()
            existing_hashes = {}

        stale_ids = [
            chunk.chunk_id
            for chunk in chunks
            if chunk.chunk_id in existing_ids
            and existing_hashes.get(chunk.chunk_id) != chunk.content_hash
        ]
        if stale_ids:
            vector_store.delete(ids=stale_ids)
            existing_ids.difference_update(stale_ids)

        chunks_to_upsert = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
        if chunks_to_upsert:
            vector_store.add_texts(
                texts=[chunk.text for chunk in chunks_to_upsert],
                ids=[chunk.chunk_id for chunk in chunks_to_upsert],
                metadatas=[
                    {
                        "title": chunk.title,
                        "file_path": str(chunk.path),
                        "chunk_id": chunk.chunk_id,
                        "content_hash": chunk.content_hash,
                    }
                    for chunk in chunks_to_upsert
                ],
            )

    load_local_chunks.cache_clear()
    return {
        "chunk_count": len(chunks),
        "vector_enabled": 1 if vector_store else 0,
    }


def get_knowledge_base_stats() -> dict[str, int]:
    chunks = load_local_chunks()
    return {
        "chunk_count": len(chunks),
        "file_count": len({str(chunk.path) for chunk in chunks}),
        "vector_enabled": 1 if _get_vector_store() else 0,
    }


@lru_cache(maxsize=1)
def load_local_chunks() -> tuple[LocalChunk, ...]:
    ensure_database()
    chunks = _load_chunks_from_files()
    if not chunks:
        return tuple()
    _upsert_chunk_rows(chunks)
    return tuple(chunks)


def _to_raw_document(
    chunk: LocalChunk,
    section_id: int,
    score: float,
    retrieval_method: str,
) -> RawDocument:
    normalized_score = max(0.0, min(1.0, round(score, 4)))
    return RawDocument(
        doc_id=f"{retrieval_method}-sec{section_id}-{chunk.chunk_id}",
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        source="local-knowledge-base",
        url=str(chunk.path),
        summary=chunk.text[: settings.search_max_content_chars].strip(),
        content=chunk.text,
        section_id=section_id,
        relevance_score=normalized_score,
        retrieval_method=retrieval_method,
        metadata={
            "file_path": str(chunk.path),
            "chunk_id": chunk.chunk_id,
        },
    )


def search_keyword_documents(query: str, section_id: int, top_k: int) -> list[RawDocument]:
    scored: list[tuple[float, LocalChunk]] = []
    for chunk in load_local_chunks():
        score = _keyword_score(query=query, title=chunk.title, text=chunk.text)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        _to_raw_document(chunk, section_id, score, "keyword")
        for score, chunk in scored[:top_k]
    ]


def search_vector_documents(query: str, section_id: int, top_k: int) -> list[RawDocument]:
    vector_store = _get_vector_store()
    if not vector_store:
        return []

    results = vector_store.similarity_search_with_score(query, k=top_k)
    documents: list[RawDocument] = []
    chunks_by_id = {chunk.chunk_id: chunk for chunk in load_local_chunks()}
    for doc, score in results:
        metadata = doc.metadata or {}
        chunk_id = metadata.get("chunk_id", "")
        chunk = chunks_by_id.get(chunk_id)
        if not chunk:
            continue
        normalized = 1.0 / (1.0 + max(float(score), 0.0))
        documents.append(_to_raw_document(chunk, section_id, normalized, "vector"))
    return documents


def _merge_hybrid_documents(documents: list[RawDocument], top_k: int) -> list[RawDocument]:
    merged: dict[str, RawDocument] = {}
    for doc in documents:
        existing = merged.get(doc.chunk_id or doc.doc_id)
        if not existing or doc.relevance_score > existing.relevance_score:
            merged[doc.chunk_id or doc.doc_id] = doc
    ranked = sorted(merged.values(), key=lambda item: item.relevance_score, reverse=True)
    return ranked[:top_k]


def rerank_documents(query: str, documents: list[RawDocument], top_k: int) -> list[RawDocument]:
    query_terms = _tokenize(query)
    rescored: list[RawDocument] = []
    for doc in documents:
        content_terms = _tokenize(f"{doc.title} {doc.content or doc.summary}")
        overlap = len(query_terms & content_terms) / len(query_terms) if query_terms else 0.0
        title_bonus = 0.15 if any(term in _tokenize(doc.title) for term in query_terms) else 0.0
        method_bonus = 0.1 if doc.retrieval_method == "vector" else 0.05 if doc.retrieval_method == "keyword" else 0.0
        reranked_score = min(1.0, 0.55 * doc.relevance_score + 0.35 * overlap + title_bonus + method_bonus)
        rescored.append(doc.model_copy(update={"relevance_score": round(reranked_score, 4)}))

    rescored.sort(key=lambda item: item.relevance_score, reverse=True)
    return rescored[:top_k]


def log_retrieval(query: str, section_id: int, method: str, document_ids: list[str]) -> None:
    try:
        ensure_database()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_logs (query, section_id, retrieval_method, document_ids)
                VALUES (?, ?, ?, ?)
                """,
                (query, section_id, method, json.dumps(document_ids, ensure_ascii=False)),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Failed to write retrieval log: %s", exc)


def search_hybrid_documents(query: str, section_id: int) -> list[RawDocument]:
    keyword_docs = search_keyword_documents(
        query=query,
        section_id=section_id,
        top_k=settings.local_retrieval_top_k,
    )
    vector_docs = search_vector_documents(
        query=query,
        section_id=section_id,
        top_k=settings.vector_retrieval_top_k,
    )
    merged = _merge_hybrid_documents(
        keyword_docs + vector_docs,
        top_k=settings.hybrid_retrieval_top_k * 2,
    )
    merged = rerank_documents(query=query, documents=merged, top_k=settings.hybrid_retrieval_top_k)
    method = "hybrid" if vector_docs else "keyword"
    log_retrieval(query=query, section_id=section_id, method=method, document_ids=[doc.doc_id for doc in merged])
    return merged
