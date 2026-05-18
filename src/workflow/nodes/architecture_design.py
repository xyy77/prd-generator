import json

from src.prompts.manager import PromptManager
from src.utils.llm_client import LLMClient
from src.workflow.node_utils import run_stage_node
from src.workflow.state import WorkflowState


def architecture_design_node(state: WorkflowState) -> dict:
    client = LLMClient()
    pm = PromptManager()

    req_json = state.get("requirement_analysis", {})
    requirement_analysis_str = json.dumps(req_json, ensure_ascii=False, indent=2)

    messages = pm.get_stage_prompt(
        stage="architecture_design",
        product_idea=state.get("product_idea", ""),
        reference_context=state.get("retrieved_context", ""),
        requirement_analysis=requirement_analysis_str,
    )

    return run_stage_node(
        client=client,
        messages=messages,
        stage_name="architecture_design",
        output_key="architecture_design",
        state=state,
    )
