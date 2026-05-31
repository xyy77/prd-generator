from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class NodeEvent(BaseModel):
    node_name: str = Field(..., description="节点名称")
    display_name: str = Field(..., description="节点显示名（中文）")
    status: str = Field(default="completed", description="completed / running / failed")
    output_summary: dict[str, Any] = Field(default_factory=dict, description="节点输出摘要")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class GenerateResponse(BaseModel):
    success: bool
    product_idea: str
    final_prd_json: dict[str, Any] | None = None
    final_prd_markdown: str | None = None
    reviewer_score: int | None = None
    reviewer_summary: str | None = None
    reflection_rounds: int = 0
    reflection_history: list[dict[str, Any]] = Field(default_factory=list)
    completed_agents: list[str] = Field(default_factory=list)
    planner_output: dict[str, Any] | None = None
    provider_history: list[str] = Field(default_factory=list, description="LLM provider fallback 历史")
    elapsed_seconds: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    rag_documents: int = 0
    graph_index_ready: bool = False


class ProviderInfo(BaseModel):
    name: str
    model: str
    priority: int
    available: bool
