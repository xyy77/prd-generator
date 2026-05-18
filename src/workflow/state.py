from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class StageOutput(TypedDict, total=False):
    stage_name: str
    raw_json: dict[str, Any]
    llm_response: str


class WorkflowState(TypedDict, total=False):
    # Input
    product_idea: str
    supplementary_info: str

    # RAG
    retrieved_docs: list[dict[str, Any]]
    retrieved_context: str

    # Messages (for LangGraph compatibility)
    messages: Annotated[list, add_messages]

    # Stage outputs
    requirement_analysis: dict[str, Any]
    architecture_design: dict[str, Any]
    process_flow: dict[str, Any]
    final_prd_json: dict[str, Any]

    # Control
    current_stage: str
    error_message: str

    # Final output
    final_prd_markdown: str


STAGE_ORDER = [
    "requirement_analysis",
    "architecture_design",
    "process_flow",
    "document_finalization",
]
