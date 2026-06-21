import os
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()


def _get_int_env(name: str, default: int, min_value: int | None = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        value = default
    else:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc

    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}.")
    return value


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat")
    llm_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    llm_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    llm_request_timeout: int = _get_int_env("LLM_REQUEST_TIMEOUT", 120, min_value=1)

    embedding_model: str = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
    embedding_api_url: str = os.getenv(
        "DASHSCOPE_EMBEDDING_API_URL",
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
    ).strip()
    embedding_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    embedding_batch_size: int = _get_int_env("EMBEDDING_BATCH_SIZE", 10, min_value=1)

    max_iterations: int = _get_int_env("MAX_ITERATIONS", 2, min_value=1)
    min_sections: int = _get_int_env("MIN_SECTIONS", 5, min_value=1)
    max_sections: int = _get_int_env("MAX_SECTIONS", 6, min_value=1)

    search_results_per_query: int = _get_int_env(
        "SEARCH_MAX_RESULTS_PER_KEYWORD", 3, min_value=1
    )
    search_max_content_chars: int = _get_int_env(
        "SEARCH_MAX_CONTENT_LENGTH", 1200, min_value=100
    )

    local_data_dir: str = os.getenv("LOCAL_DATA_DIR", "data/raw")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/raw/uploads")
    
    local_chunk_size: int = _get_int_env("LOCAL_CHUNK_SIZE", 700, min_value=100)
    local_chunk_overlap: int = _get_int_env("LOCAL_CHUNK_OVERLAP", 120, min_value=0)
    local_retrieval_top_k: int = _get_int_env("LOCAL_RETRIEVAL_TOP_K", 3, min_value=1)
    vector_retrieval_top_k: int = _get_int_env("VECTOR_RETRIEVAL_TOP_K", 4, min_value=1)
    hybrid_retrieval_top_k: int = _get_int_env("HYBRID_RETRIEVAL_TOP_K", 4, min_value=1)

    sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", "data/app.db")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "research_chunks")

    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "").strip()

    api_title: str = "DeepResearch-Agent API"
    api_version: str = "0.4.0"


def _validate_settings(value: Settings) -> None:
    if value.min_sections > value.max_sections:
        raise ValueError(
            f"MIN_SECTIONS must be <= MAX_SECTIONS, got "
            f"{value.min_sections} > {value.max_sections}."
        )
    if value.local_chunk_overlap >= value.local_chunk_size:
        raise ValueError(
            "LOCAL_CHUNK_OVERLAP must be smaller than LOCAL_CHUNK_SIZE, got "
            f"{value.local_chunk_overlap} >= {value.local_chunk_size}."
        )


settings = Settings()
_validate_settings(settings)


def has_llm_credentials() -> bool:
    return bool(settings.llm_api_key)


def has_embedding_credentials() -> bool:
    return bool(settings.embedding_api_key)


def get_chat_llm(temperature: float):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        request_timeout=settings.llm_request_timeout,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
