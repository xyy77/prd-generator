"""FastAPI application for PRD Generator.

Usage:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models.responses import HealthResponse, ProviderInfo
from src.api.routes.generate import router as generate_router
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API server starting on %s:%s", settings.api_host, settings.api_port)
    yield
    logger.info("API server shutting down")


app = FastAPI(
    title="智能 PRD 自动生成平台 API",
    description="Multi-Agent PRD Generation Platform with LangGraph Supervisor Architecture",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router)


@app.get("/api/health", response_model=HealthResponse)
def health():
    """Health check with RAG status."""
    try:
        from src.rag.retriever import Retriever
        from src.rag.embedder import EmbeddingService
        from src.rag.store import ChromaStore

        store = ChromaStore()
        retriever = Retriever(store, EmbeddingService())
        doc_count = store.count()
        graph_ready = retriever.graph_retriever.is_ready
    except Exception:
        doc_count = 0
        graph_ready = False

    return HealthResponse(
        status="ok",
        version="2.0.0",
        rag_documents=doc_count,
        graph_index_ready=graph_ready,
    )


@app.get("/api/providers", response_model=list[ProviderInfo])
def list_providers():
    """List available LLM providers and their status."""
    providers = []
    # DeepSeek
    providers.append(ProviderInfo(
        name="deepseek",
        model=settings.deepseek_model,
        priority=1,
        available=bool(settings.deepseek_api_key and "placeholder" not in settings.deepseek_api_key),
    ))
    # Bailian
    providers.append(ProviderInfo(
        name="bailian",
        model=settings.dashscope_text_model,
        priority=2,
        available=bool(settings.dashscope_api_key),
    ))
    # Zhipu
    providers.append(ProviderInfo(
        name="zhipu",
        model=settings.zhipu_text_model,
        priority=3,
        available=bool(settings.zhipu_api_key),
    ))
    return providers
