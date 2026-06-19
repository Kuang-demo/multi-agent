import os
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat")
    llm_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    llm_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    llm_request_timeout: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

    embedding_model: str = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
    embedding_api_url: str = os.getenv(
        "DASHSCOPE_EMBEDDING_API_URL",
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
    ).strip()
    embedding_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))

    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "2"))
    min_sections: int = int(os.getenv("MIN_SECTIONS", "5"))
    max_sections: int = int(os.getenv("MAX_SECTIONS", "6"))

    search_results_per_query: int = int(os.getenv("SEARCH_MAX_RESULTS_PER_KEYWORD", "3"))
    search_max_content_chars: int = int(os.getenv("SEARCH_MAX_CONTENT_LENGTH", "1200"))

    local_data_dir: str = os.getenv("LOCAL_DATA_DIR", "data/raw")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/raw/uploads")
    
    local_chunk_size: int = int(os.getenv("LOCAL_CHUNK_SIZE", "700"))
    local_chunk_overlap: int = int(os.getenv("LOCAL_CHUNK_OVERLAP", "120"))
    local_retrieval_top_k: int = int(os.getenv("LOCAL_RETRIEVAL_TOP_K", "3"))
    vector_retrieval_top_k: int = int(os.getenv("VECTOR_RETRIEVAL_TOP_K", "4"))
    hybrid_retrieval_top_k: int = int(os.getenv("HYBRID_RETRIEVAL_TOP_K", "4"))

    sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", "data/app.db")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "research_chunks")

    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "").strip()

    api_title: str = "DeepResearch-Agent API"
    api_version: str = "0.4.0"


settings = Settings()


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
