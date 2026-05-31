from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    deepseek_api_key: str = "sk-placeholder"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    available_models: list[str] = ["deepseek-chat", "deepseek-v4-pro"]

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
    llm_temperature_min: float = 0.0
    llm_temperature_max: float = 1.0
    max_retries: int = 2
    request_timeout: int = 120

    # History
    max_history: int = 5

    # Vision (Qwen-VL via DashScope, or OpenAI/Anthropic)
    vision_provider: str = "qwen"
    vision_model: str = "qwen-vl-max"
    dashscope_api_key: str = ""
    zhipu_api_key: str = ""

    # Multi-agent reflection
    reflection_max_rounds: int = 2
    reviewer_score_threshold: int = 80

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = ["*"]

    # Multi-provider text LLM (all OpenAI-compatible)
    dashscope_text_model: str = "qwen-plus"
    zhipu_text_model: str = "glm-4-flash"

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

    @property
    def uploads_dir(self) -> Path:
        return self.project_root / "data" / "uploads"


settings = Settings()
