import json

from src.prompts.manager import PromptManager
from src.utils.llm_client import LLMClient
from src.workflow.node_utils import run_stage_node
from src.workflow.state import WorkflowState


def process_flow_node(state: WorkflowState) -> dict:
    client = LLMClient()
    pm = PromptManager()

    req_json = state.get("requirement_analysis", {})
    arch_json = state.get("architecture_design", {})

    messages = pm.get_stage_prompt(
        stage="process_flow",
        product_idea=state.get("product_idea", ""),
        reference_context=state.get("retrieved_context", ""),
        requirement_analysis=json.dumps(req_json, ensure_ascii=False, indent=2),
        architecture_design=json.dumps(arch_json, ensure_ascii=False, indent=2),
    )

    return run_stage_node(
        client=client,
        messages=messages,
        stage_name="process_flow",
        output_key="process_flow",
        state=state,
    )
