"""SSE (Server-Sent Events) streaming helper for multi-agent workflow.

Wraps graph.stream() + on_node_complete callback into an async
generator that yields SSE-formatted events.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

from src.workflow.multi_agent.graph import (
    NODE_DISPLAY_NAMES,
    run_multi_agent_workflow,
    run_multi_agent_revision,
)
from src.workflow.multi_agent.state import AGENT_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_KEYS_BY_NODE: dict[str, list[str]] = {
    "node_planner": ["product_type", "complexity", "execution_plan"],
    "node_supervisor": ["agents_to_call", "decision_rationale"],
    "node_requirements_analyst": ["persona_count"],
    "node_feature_planner": ["module_count"],
    "node_ux_designer": ["page_count"],
    "node_tech_advisor": ["tech_count"],
    "node_reviewer": ["reviewer_score", "reviewer_summary"],
    "node_revision_router": ["agents_to_revise", "current_stage"],
    "node_document_synthesis": [],
    "node_image_analysis": [],
}


def _extract_summary(node_name: str, output: dict[str, Any]) -> dict[str, Any]:
    """Extract a lightweight summary from node output for SSE events."""
    summary: dict[str, Any] = {}
    keys = OUTPUT_KEYS_BY_NODE.get(node_name, [])

    for k in keys:
        # Check both snake_case and direct field names
        if k in output:
            summary[k] = output[k]

    # Special handling
    if node_name == "node_planner":
        planner = output.get("planner_output", {})
        if isinstance(planner, dict):
            summary.setdefault("product_type", planner.get("product_type"))
            summary.setdefault("complexity", planner.get("complexity"))
    elif node_name == "node_reviewer":
        summary.setdefault("reviewer_score", output.get("reviewer_score"))
        summary.setdefault("reviewer_summary", output.get("reviewer_summary"))
    elif node_name == "node_supervisor":
        sd = output.get("supervisor_decision", {})
        if isinstance(sd, dict):
            summary.setdefault("agents_to_call", output.get("agents_to_call"))
            summary.setdefault("decision_rationale", sd.get("decision_rationale"))

    return summary


async def sse_generate(
    product_idea: str,
    supplementary_info: str = "",
    selected_model: str | None = None,
    temperature: float | None = None,
    reflection_max_rounds: int | None = None,
    reviewer_score_threshold: int | None = None,
    image_paths: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Run multi-agent workflow and yield SSE events for each node completion."""
    start_time = time.time()

    async def _emit(event_type: str, data: dict[str, Any]) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event_type}\ndata: {payload}\n\n"

    # Yield start event
    yield await _emit("start", {"product_idea": product_idea[:100]})

    try:
        # We run the sync graph.stream in a blocking way but yield events
        # In production you'd use asyncio.to_thread or loop.run_in_executor
        # but for simplicity we keep it sync here and yield from the callback
        result_state: dict[str, Any] = {}
        node_count = 0

        def on_node(node_name: str, node_output: dict[str, Any]) -> None:
            nonlocal node_count
            node_count += 1
            display = NODE_DISPLAY_NAMES.get(node_name, node_name)
            summary = _extract_summary(node_name, node_output)
            # We can't yield from a callback, so we store events
            on_node._events.append({
                "node_name": node_name,
                "display_name": display,
                "status": "completed",
                "output_summary": summary,
                "index": node_count,
            })
            result_state.update(node_output)

        on_node._events = []  # type: ignore[attr-defined]

        # Run the workflow synchronously (blocks, but SSE client handles it)
        result_state = run_multi_agent_workflow(
            product_idea=product_idea,
            supplementary_info=supplementary_info,
            retrieved_context="",
            image_paths=image_paths or [],
            selected_model=selected_model,
            temperature=temperature,
            reflection_max_rounds=reflection_max_rounds,
            reviewer_score_threshold=reviewer_score_threshold,
            on_node_complete=on_node,
        )

        # Yield all collected events
        for evt in on_node._events:  # type: ignore[attr-defined]
            yield await _emit("node_complete", evt)

        elapsed = round(time.time() - start_time, 1)
        yield await _emit("complete", {
            "success": True,
            "reviewer_score": result_state.get("reviewer_score"),
            "reviewer_summary": result_state.get("reviewer_summary"),
            "reflection_rounds": result_state.get("reflection_round", 0),
            "completed_agents": result_state.get("completed_agents", []),
            "elapsed_seconds": elapsed,
        })

    except Exception as e:
        logger.error("SSE workflow failed: %s", e)
        yield await _emit("error", {"success": False, "error": str(e)})
