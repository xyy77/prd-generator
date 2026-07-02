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

    # Stage outputs (classic pipeline)
    requirement_analysis: dict[str, Any]
    architecture_design: dict[str, Any]
    process_flow: dict[str, Any]
    final_prd_json: dict[str, Any]

    # Multi-agent outputs
    feature_plan: dict[str, Any]
    ux_design: dict[str, Any]
    tech_advice: dict[str, Any]

    # Multimodal
    image_paths: list[str]
    image_analysis: dict[str, Any]

    # Reviewer
    reviewer_score: int
    reviewer_scores: dict[str, int]
    reviewer_feedback: dict[str, str]
    reviewer_summary: str

    # Reflection control
    reflection_round: int
    reflection_max_rounds: int
    agents_to_revise: list[str]
    reflection_history: list[dict[str, Any]]

    # Control
    current_stage: str
    error_message: str
    selected_model: str
    temperature: float

    # Planner
    planner_output: dict[str, Any]

    # Supervisor
    supervisor_decision: dict[str, Any]
    agents_to_call: list[str]
    completed_agents: list[str]

    # Revision
    user_feedback: str
    revision_history: list[dict[str, Any]]
    revision_count: int
    locked_agents: list[str]  # User-locked agents — revision will skip these

    # Final output
    final_prd_markdown: str


STAGE_ORDER = [
    "requirement_analysis",
    "architecture_design",
    "process_flow",
    "document_finalization",
]
