from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    deepseek_api_key: str = "sk-placeholder"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # Embedding
    embedding_model_name: str = "BAAI/bge-base-zh-v1.5"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "prd_knowledge"

    # RAG
    max_retrieved_docs: int = 3
    default_chunk_size: int = 500
    default_chunk_overlap: int = 50

    # LLM generation
    llm_temperature: float = 0.3
    max_retries: int = 2
    request_timeout: int = 120

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def chroma_persist_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @property
    def samples_dir(self) -> Path:
        return self.project_root / "prd_samples"


settings = Settings()
