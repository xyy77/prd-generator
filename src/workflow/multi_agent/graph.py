from langgraph.graph import StateGraph, START, END

from src.config import settings
from src.workflow.multi_agent.state import AGENT_NAMES
from src.workflow.multi_agent.node_utils import run_parallel_agents
from src.workflow.multi_agent.agents import (
    image_analyst_node,
    reviewer_node,
    revision_router_node,
)
from src.workflow.multi_agent.agents.document_synthesis import document_synthesis_node
from src.workflow.state import WorkflowState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _route_after_revision(state: WorkflowState) -> str:
    agents_to_revise = state.get("agents_to_revise", [])
    max_rounds = state.get("reflection_max_rounds", settings.reflection_max_rounds)
    current_round = state.get("reflection_round", 0)

    if agents_to_revise and current_round < max_rounds:
        logger.info("Re-running agents %s (round %d/%d)", agents_to_revise, current_round, max_rounds)
        return "revise"
    logger.info("No agents to revise or max rounds reached, skipping to finalize")
    return "finalize"


def _route_after_review(state: WorkflowState) -> str:
    score = state.get("reviewer_score", 0)
    round_num = state.get("reflection_round", 0)
    max_rounds = state.get("reflection_max_rounds", settings.reflection_max_rounds)
    threshold = state.get("reviewer_score_threshold", settings.reviewer_score_threshold)

    if score < threshold and round_num < max_rounds:
        logger.info("Review score %d < %d, round %d/%d — routing to revision", score, threshold, round_num, max_rounds)
        return "revision"
    logger.info("Review passed (score=%d, round=%d) — routing to finalize", score, round_num)
    return "finalize"


def _parallel_agents_wrapper(state: WorkflowState) -> dict:
    """Wrapper that runs only agents flagged for revision, or all agents on first pass."""
    agents_to_revise = state.get("agents_to_revise")
    if agents_to_revise:
        agent_names = [a for a in agents_to_revise if a in AGENT_NAMES]
        logger.info("Re-running agents: %s", agent_names)
    else:
        agent_names = list(AGENT_NAMES)
        logger.info("Running all agents (initial pass)")

    reference_context = state.get("retrieved_context", "")
    return run_parallel_agents(dict(state), agent_names, reference_context)


def _image_analyst_wrapper(state: WorkflowState) -> dict:
    """Wrapper that runs image analysis. Skips gracefully if no images."""
    reference_context = state.get("retrieved_context", "")
    return image_analyst_node(dict(state), reference_context)


def _reviewer_wrapper(state: WorkflowState) -> dict:
    """Wrapper for reviewer with default max_rounds/threshold injection."""
    s = dict(state)
    if "reflection_max_rounds" not in s or s["reflection_max_rounds"] is None:
        s["reflection_max_rounds"] = settings.reflection_max_rounds
    return reviewer_node(s)


def _revision_router_wrapper(state: WorkflowState) -> dict:
    """Wrapper for revision router."""
    return revision_router_node(dict(state))


def build_multi_agent_graph() -> StateGraph:
    builder = StateGraph(WorkflowState)

    builder.add_node("node_image_analysis", _image_analyst_wrapper)
    builder.add_node("node_parallel_agents", _parallel_agents_wrapper)
    builder.add_node("node_reviewer", _reviewer_wrapper)
    builder.add_node("node_revision_router", _revision_router_wrapper)
    builder.add_node("node_document_synthesis", document_synthesis_node)

    builder.add_edge(START, "node_image_analysis")
    builder.add_edge("node_image_analysis", "node_parallel_agents")
    builder.add_edge("node_parallel_agents", "node_reviewer")

    builder.add_conditional_edges(
        "node_reviewer",
        _route_after_review,
        {
            "finalize": "node_document_synthesis",
            "revision": "node_revision_router",
        },
    )

    builder.add_conditional_edges(
        "node_revision_router",
        _route_after_revision,
        {
            "revise": "node_parallel_agents",
            "finalize": "node_document_synthesis",
        },
    )
    builder.add_edge("node_document_synthesis", END)

    compiled = builder.compile()
    logger.info("Multi-agent graph compiled successfully")
    return compiled


_multi_agent_graph: StateGraph | None = None


def get_multi_agent_graph() -> StateGraph:
    global _multi_agent_graph
    if _multi_agent_graph is None:
        _multi_agent_graph = build_multi_agent_graph()
    return _multi_agent_graph


def run_multi_agent_workflow(
    product_idea: str,
    supplementary_info: str = "",
    retrieved_context: str = "",
    image_paths: list[str] | None = None,
    selected_model: str | None = None,
    temperature: float | None = None,
    reflection_max_rounds: int | None = None,
    reviewer_score_threshold: int | None = None,
) -> WorkflowState:
    graph = get_multi_agent_graph()
    initial_state: WorkflowState = {
        "product_idea": product_idea,
        "supplementary_info": supplementary_info,
        "retrieved_context": retrieved_context,
        "image_paths": image_paths or [],
        "reflection_round": 0,
        "reflection_max_rounds": reflection_max_rounds or settings.reflection_max_rounds,
        "reviewer_score_threshold": reviewer_score_threshold or settings.reviewer_score_threshold,
        "reflection_history": [],
        "revision_history": [],
        "revision_count": 0,
    }
    if selected_model:
        initial_state["selected_model"] = selected_model
    if temperature is not None:
        initial_state["temperature"] = temperature

    logger.info("Starting multi-agent workflow for: %s...", product_idea[:60])
    result = graph.invoke(initial_state)
    logger.info(
        "Multi-agent workflow complete, score=%s, reflection_rounds=%s",
        result.get("reviewer_score", "N/A"),
        result.get("reflection_round", 0),
    )
    return result
