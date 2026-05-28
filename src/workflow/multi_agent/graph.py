from langgraph.graph import StateGraph, START, END
from typing import Callable, Any

from src.config import settings
from src.workflow.multi_agent.state import AGENT_NAMES
from src.workflow.multi_agent.agents import (
    image_analyst_node,
    reviewer_node,
    revision_router_node,
    planner_node,
)
from src.workflow.multi_agent.agents.supervisor import supervisor_node
from src.workflow.multi_agent.agents.document_synthesis import document_synthesis_node
from src.workflow.multi_agent.node_utils import run_agent_node
from src.workflow.state import WorkflowState
from src.utils.llm_client import LLMClient
from src.prompts.manager import PromptManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

AGENT_OUTPUT_KEYS = {
    "requirements_analyst": "requirement_analysis",
    "feature_planner": "feature_plan",
    "ux_designer": "ux_design",
    "tech_advisor": "tech_advice",
}

NODE_DISPLAY_NAMES: dict[str, str] = {
    "node_image_analysis": "图片分析",
    "node_planner": "产品架构规划",
    "node_supervisor": "Agent 调度",
    "node_requirements_analyst": "需求分析师",
    "node_feature_planner": "功能规划师",
    "node_ux_designer": "UX 设计师",
    "node_tech_advisor": "技术顾问",
    "node_reviewer": "评审官",
    "node_revision_router": "修订路由",
    "node_document_synthesis": "文档合成",
}


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


def _route_entry(state: WorkflowState) -> str:
    """Skip image_analysis and planner when user_feedback is present (revision mode)."""
    if state.get("user_feedback"):
        logger.info("User feedback detected, skipping to supervisor (revision mode)")
        return "node_supervisor"
    return "node_image_analysis"


def _route_from_supervisor(state: WorkflowState) -> str:
    """Pick the next agent from execution_order that hasn't completed yet."""
    execution_order = state.get("execution_order", [])
    completed = state.get("completed_agents", [])
    for agent in execution_order:
        if agent not in completed:
            return agent
    return "reviewer"


# ── Wrapper nodes ──────────────────────────────────────────────

def _image_analyst_wrapper(state: WorkflowState) -> dict:
    reference_context = state.get("retrieved_context", "")
    return image_analyst_node(dict(state), reference_context)


def _planner_wrapper(state: WorkflowState) -> dict:
    reference_context = state.get("retrieved_context", "")
    return planner_node(dict(state), reference_context)


def _supervisor_wrapper(state: WorkflowState) -> dict:
    s = dict(state)
    if "reflection_max_rounds" not in s or s["reflection_max_rounds"] is None:
        s["reflection_max_rounds"] = settings.reflection_max_rounds
    return supervisor_node(s)


def _make_agent_wrapper(agent_name: str):
    """Create a wrapper for a single agent that tracks completion."""
    output_key = AGENT_OUTPUT_KEYS[agent_name]

    def wrapper(state: WorkflowState) -> dict:
        from src.workflow.multi_agent.agents import AGENT_NODE_MAP
        agent_fn = AGENT_NODE_MAP.get(agent_name)
        if agent_fn is None:
            return {"error_message": f"Unknown agent: {agent_name}"}

        reference_context = state.get("retrieved_context", "")
        try:
            output = agent_fn(dict(state), reference_context)
        except Exception as e:
            logger.error("Agent %s failed: %s", agent_name, e)
            output = {output_key: {}, "error_message": str(e)}

        # Track completion
        completed = list(state.get("completed_agents", []))
        if agent_name not in completed:
            completed.append(agent_name)
        output["completed_agents"] = completed
        output["current_stage"] = agent_name
        return output

    return wrapper


def _reviewer_wrapper(state: WorkflowState) -> dict:
    s = dict(state)
    if "reflection_max_rounds" not in s or s["reflection_max_rounds"] is None:
        s["reflection_max_rounds"] = settings.reflection_max_rounds
    return reviewer_node(s)


def _revision_router_wrapper(state: WorkflowState) -> dict:
    return revision_router_node(dict(state))


# ── Graph construction ─────────────────────────────────────────

def build_multi_agent_graph() -> StateGraph:
    builder = StateGraph(WorkflowState)

    # Nodes
    builder.add_node("node_image_analysis", _image_analyst_wrapper)
    builder.add_node("node_planner", _planner_wrapper)
    builder.add_node("node_supervisor", _supervisor_wrapper)
    builder.add_node("node_requirements_analyst", _make_agent_wrapper("requirements_analyst"))
    builder.add_node("node_feature_planner", _make_agent_wrapper("feature_planner"))
    builder.add_node("node_ux_designer", _make_agent_wrapper("ux_designer"))
    builder.add_node("node_tech_advisor", _make_agent_wrapper("tech_advisor"))
    builder.add_node("node_reviewer", _reviewer_wrapper)
    builder.add_node("node_revision_router", _revision_router_wrapper)
    builder.add_node("node_document_synthesis", document_synthesis_node)

    # Edges
    builder.add_conditional_edges(START, _route_entry, {
        "node_supervisor": "node_supervisor",
        "node_image_analysis": "node_image_analysis",
    })
    builder.add_edge("node_image_analysis", "node_planner")
    builder.add_edge("node_planner", "node_supervisor")

    # Supervisor → conditional routing to agents → back to supervisor
    builder.add_conditional_edges("node_supervisor", _route_from_supervisor, {
        "requirements_analyst": "node_requirements_analyst",
        "feature_planner": "node_feature_planner",
        "ux_designer": "node_ux_designer",
        "tech_advisor": "node_tech_advisor",
        "reviewer": "node_reviewer",
    })
    for agent_node in ["node_requirements_analyst", "node_feature_planner",
                        "node_ux_designer", "node_tech_advisor"]:
        builder.add_edge(agent_node, "node_supervisor")

    # Reviewer → conditional: finalize or revision
    builder.add_conditional_edges("node_reviewer", _route_after_review, {
        "finalize": "node_document_synthesis",
        "revision": "node_revision_router",
    })

    # Revision router → conditional: revise or finalize
    builder.add_conditional_edges("node_revision_router", _route_after_revision, {
        "revise": "node_supervisor",
        "finalize": "node_document_synthesis",
    })
    builder.add_edge("node_document_synthesis", END)

    compiled = builder.compile()
    logger.info("Multi-agent graph compiled successfully (Supervisor mode)")
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
    on_node_complete: Callable[[str, dict[str, Any]], None] | None = None,
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
        "completed_agents": [],
        "execution_order": [],
        "agents_to_call": [],
    }
    if selected_model:
        initial_state["selected_model"] = selected_model
    if temperature is not None:
        initial_state["temperature"] = temperature

    logger.info("Starting multi-agent workflow for: %s...", product_idea[:60])

    if on_node_complete is not None:
        result: WorkflowState = {}
        for chunk in graph.stream(initial_state):
            for node_name, node_output in chunk.items():
                logger.info("Node completed: %s", node_name)
                on_node_complete(node_name, node_output)
                result.update(node_output)
    else:
        result = graph.invoke(initial_state)

    logger.info(
        "Multi-agent workflow complete, score=%s, reflection_rounds=%s",
        result.get("reviewer_score", "N/A"),
        result.get("reflection_round", 0),
    )
    return result


def run_multi_agent_revision(
    existing_state: WorkflowState,
    feedback: str,
    on_node_complete: Callable[[str, dict[str, Any]], None] | None = None,
) -> WorkflowState:
    """Run revision through the multi-agent graph with user feedback.

    Skips image_analysis and planner via _route_entry, goes straight to
    supervisor which re-runs all agents with the user feedback injected.
    """
    graph = get_multi_agent_graph()
    state = dict(existing_state)
    state["user_feedback"] = feedback
    state["agents_to_revise"] = list(AGENT_NAMES)
    state["completed_agents"] = []
    state["execution_order"] = []
    state["agents_to_call"] = []
    state["current_stage"] = "awaiting_revision"
    logger.info("Running multi-agent revision with feedback: %s...", feedback[:80])

    if on_node_complete is not None:
        result: WorkflowState = {}
        for chunk in graph.stream(state):
            for node_name, node_output in chunk.items():
                logger.info("Node completed (revision): %s", node_name)
                on_node_complete(node_name, node_output)
                result.update(node_output)
    else:
        result = graph.invoke(state)

    logger.info(
        "Multi-agent revision complete, score=%s, revision_count=%s",
        result.get("reviewer_score", "N/A"),
        result.get("revision_count", 0),
    )
    return result
