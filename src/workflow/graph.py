from langgraph.graph import StateGraph, START, END

from src.workflow.nodes.document_finalization import document_finalization_node
from src.workflow.nodes.parallel_analysis import parallel_analysis_node
from src.workflow.nodes.process_flow import process_flow_node
from src.workflow.state import WorkflowState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_workflow_graph() -> StateGraph:
    builder = StateGraph(WorkflowState)

    # Stage 1+2 run in parallel inside a single node (ThreadPoolExecutor)
    builder.add_node("node_parallel_analysis", parallel_analysis_node)
    builder.add_node("node_process_flow", process_flow_node)
    builder.add_node("node_document_finalization", document_finalization_node)

    builder.add_edge(START, "node_parallel_analysis")
    builder.add_edge("node_parallel_analysis", "node_process_flow")
    builder.add_edge("node_process_flow", "node_document_finalization")
    builder.add_edge("node_document_finalization", END)

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
) -> WorkflowState:
    graph = get_graph()
    initial_state: WorkflowState = {
        "product_idea": product_idea,
        "supplementary_info": supplementary_info,
        "retrieved_context": retrieved_context,
        "current_stage": "start",
    }
    logger.info("Starting workflow for: %s...", product_idea[:60])
    result = graph.invoke(initial_state)
    logger.info("Workflow complete, stages: %s",
                [k for k in result if k in ["requirement_analysis", "architecture_design", "process_flow", "final_prd_json"]])
    return result
