"""Generate routes: /api/generate and /api/generate/stream"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from src.api.models.requests import GenerateRequest, RevisionRequest
from src.api.models.responses import GenerateResponse, NodeEvent
from src.api.sse import sse_generate
from src.rag.retriever import Retriever
from src.rag.embedder import EmbeddingService
from src.rag.store import ChromaStore
from src.workflow.multi_agent.graph import (
    NODE_DISPLAY_NAMES,
    run_multi_agent_workflow,
    run_multi_agent_revision,
)
from src.utils.input_validation import (
    validate_product_idea,
    validate_supplementary,
    check_prompt_injection,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["generate"])

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        store = ChromaStore()
        embedder = EmbeddingService()
        _retriever = Retriever(store, embedder)
        _retriever.ensure_methodology_loaded()
        if store.count() > 0:
            try:
                _retriever.build_graph_index()
            except Exception:
                pass
    return _retriever


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Synchronous PRD generation. Returns the complete result."""
    start = time.time()

    product_idea = validate_product_idea(req.product_idea)
    if not product_idea:
        raise HTTPException(status_code=400, detail="product_idea 不能为空")
    supplementary_info = validate_supplementary(req.supplementary_info)

    # Log injection warnings for monitoring
    suspicious = check_prompt_injection(product_idea)
    if suspicious:
        logger.warning("Prompt injection patterns in request: %s", suspicious)

    try:
        retriever = _get_retriever()
        retrieved_context = retriever.search_as_context(product_idea)
    except Exception as e:
        logger.warning("RAG retrieval failed: %s, continuing without context", e)
        retrieved_context = ""

    result = run_multi_agent_workflow(
        product_idea=product_idea,
        supplementary_info=supplementary_info,
        retrieved_context=retrieved_context,
        selected_model=req.selected_model,
        temperature=req.temperature,
        reflection_max_rounds=req.reflection_max_rounds,
        reviewer_score_threshold=req.reviewer_score_threshold,
    )

    elapsed = round(time.time() - start, 1)

    error = result.get("error_message") or None
    success = error is None and bool(result.get("final_prd_json"))

    return GenerateResponse(
        success=success,
        product_idea=product_idea,
        final_prd_json=result.get("final_prd_json"),
        final_prd_markdown=result.get("final_prd_markdown"),
        reviewer_score=result.get("reviewer_score"),
        reviewer_summary=result.get("reviewer_summary"),
        reflection_rounds=result.get("reflection_round", 0),
        reflection_history=result.get("reflection_history", []),
        completed_agents=result.get("completed_agents", []),
        planner_output=result.get("planner_output"),
        elapsed_seconds=elapsed,
        error=error,
    )


@router.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """SSE streaming PRD generation. Yields node_complete events in real time."""
    try:
        retriever = _get_retriever()
        retrieved_context = retriever.search_as_context(req.product_idea)
    except Exception:
        retrieved_context = ""

    # Note: sse_generate runs the workflow directly since
    # the SSE generator currently handles its own RAG.
    # In a future refactor, RAG context could be injected.
    stream = sse_generate(
        product_idea=req.product_idea,
        supplementary_info=req.supplementary_info,
        selected_model=req.selected_model,
        temperature=req.temperature,
        reflection_max_rounds=req.reflection_max_rounds,
        reviewer_score_threshold=req.reviewer_score_threshold,
    )
    return EventSourceResponse(stream)


@router.post("/generate/revision", response_model=GenerateResponse)
def revise(req: RevisionRequest) -> GenerateResponse:
    """Run revision on an existing state with user feedback."""
    start = time.time()
    try:
        result = run_multi_agent_revision(
            existing_state=req.existing_state,
            feedback=req.feedback,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Revision failed: {e}") from e

    elapsed = round(time.time() - start, 1)
    error = result.get("error_message") or None
    success = error is None and bool(result.get("final_prd_json"))

    return GenerateResponse(
        success=success,
        product_idea=result.get("product_idea", ""),
        final_prd_json=result.get("final_prd_json"),
        final_prd_markdown=result.get("final_prd_markdown"),
        reviewer_score=result.get("reviewer_score"),
        reviewer_summary=result.get("reviewer_summary"),
        reflection_rounds=result.get("reflection_round", 0),
        reflection_history=result.get("reflection_history", []),
        completed_agents=result.get("completed_agents", []),
        elapsed_seconds=elapsed,
        error=error,
    )
