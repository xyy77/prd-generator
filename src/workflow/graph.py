from langgraph.graph import StateGraph, START, END

from src.workflow.nodes.document_finalization import document_finalization_node
from src.workflow.nodes.parallel_analysis import parallel_analysis_node
from src.workflow.nodes.prd_revision import prd_revision_node
from src.workflow.nodes.process_flow import process_flow_node
from src.workflow.state import WorkflowState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _route_entry(state: WorkflowState) -> str:
    """Route to revision node if user_feedback is set, otherwise normal pipeline."""
    if state.get("user_feedback"):
        logger.info("Routing to revision node (user_feedback present)")
        return "node_prd_revision"
    return "node_parallel_analysis"


def _route_after_revision(state: WorkflowState) -> str:
    """After revision, always go to END."""
    return END


def build_workflow_graph() -> StateGraph:
    builder = StateGraph(WorkflowState)

    # Stage 1+2 run in parallel inside a single node (ThreadPoolExecutor)
    builder.add_node("node_parallel_analysis", parallel_analysis_node)
    builder.add_node("node_process_flow", process_flow_node)
    builder.add_node("node_document_finalization", document_finalization_node)
    builder.add_node("node_prd_revision", prd_revision_node)

    # Conditional entry: revision path vs normal generation path
    builder.add_conditional_edges(START, _route_entry, {
        "node_prd_revision": "node_prd_revision",
        "node_parallel_analysis": "node_parallel_analysis",
    })

    # Normal pipeline
    builder.add_edge("node_parallel_analysis", "node_process_flow")
    builder.add_edge("node_process_flow", "node_document_finalization")
    builder.add_edge("node_document_finalization", END)

    # Revision path
    builder.add_edge("node_prd_revision", END)

    return builder.compile()


_graph: StateGraph | None = None


def get_graph() -> StateGraph:
    global _graph
    if _graph is None:
        _graph = build_workflow_graph()
        logger.info("Workflow graph compiled and cached")
    return _graph


def run_workflow(
    product_idea: str,
    supplementary_info: str = "",
    retrieved_context: str = "",
    selected_model: str | None = None,
    temperature: float | None = None,
) -> WorkflowState:
    graph = get_graph()
    initial_state: WorkflowState = {
        "product_idea": product_idea,
        "supplementary_info": supplementary_info,
        "retrieved_context": retrieved_context,
        "current_stage": "start",
    }
    if selected_model:
        initial_state["selected_model"] = selected_model
    if temperature is not None:
        initial_state["temperature"] = temperature

    logger.info("Starting workflow for: %s...", product_idea[:60])
    result = graph.invoke(initial_state)
    logger.info("Workflow complete, stages: %s",
                [k for k in result if k in ["requirement_analysis", "architecture_design", "process_flow", "final_prd_json"]])
    return result


def run_revision(
    existing_state: WorkflowState,
    feedback: str,
) -> WorkflowState:
    """Run prd_revision node with the existing workflow state."""
    graph = get_graph()
    state = dict(existing_state)
    state["user_feedback"] = feedback
    state["current_stage"] = "awaiting_revision"
    logger.info("Running PRD revision with feedback: %s...", feedback[:80])
    result = graph.invoke(state)
    logger.info("Revision complete, revision_count=%s", result.get("revision_count", 0))
    return result


def run_single_stage(
    stage_name: str,
    existing_state: WorkflowState,
) -> WorkflowState:
    """Run a single stage node standalone, then re-run downstream stages.

    Directly calls node functions to avoid re-running the entire graph pipeline.
    The updated output is merged with existing state, and subsequent stages
    are re-run to keep the full PRD consistent.
    """
    state = dict(existing_state)

    if stage_name == "requirement_analysis" or stage_name == "architecture_design":
        # Re-run parallel analysis (both sub-stages to keep them aligned)
        logger.info("Running single stage: parallel_analysis")
        node_result = parallel_analysis_node(state)
        state.update(node_result)
        # Re-run downstream stages
        logger.info("Re-running downstream: process_flow")
        pf_result = process_flow_node(state)
        state.update(pf_result)
        logger.info("Re-running downstream: document_finalization")
        df_result = document_finalization_node(state)
        state.update(df_result)
    elif stage_name == "process_flow":
        logger.info("Running single stage: process_flow")
        node_result = process_flow_node(state)
        state.update(node_result)
        # Re-run downstream
        logger.info("Re-running downstream: document_finalization")
        df_result = document_finalization_node(state)
        state.update(df_result)
    elif stage_name == "document_finalization":
        logger.info("Running single stage: document_finalization")
        node_result = document_finalization_node(state)
        state.update(node_result)
    elif stage_name == "prd_revision":
        logger.info("Running single stage: prd_revision")
        graph = get_graph()
        state["user_feedback"] = state.get("user_feedback", "")
        state["current_stage"] = "awaiting_revision"
        result = graph.invoke(state)
        state.update(result)
    else:
        raise ValueError(f"Unknown stage for single run: {stage_name}")

    logger.info("Single stage '%s' complete", stage_name)
    return state
