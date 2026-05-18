import json

from src.prompts.manager import PromptManager
from src.utils.llm_client import LLMClient
from src.workflow.node_utils import run_stage_node
from src.workflow.state import WorkflowState


def document_finalization_node(state: WorkflowState) -> dict:
    client = LLMClient()
    pm = PromptManager()

    req_json = state.get("requirement_analysis", {})
    arch_json = state.get("architecture_design", {})
    flow_json = state.get("process_flow", {})

    messages = pm.get_stage_prompt(
        stage="document_finalization",
        product_idea=state.get("product_idea", ""),
        reference_context=state.get("retrieved_context", ""),
        requirement_analysis=json.dumps(req_json, ensure_ascii=False, indent=2),
        architecture_design=json.dumps(arch_json, ensure_ascii=False, indent=2),
        process_flow=json.dumps(flow_json, ensure_ascii=False, indent=2),
    )

    model = state.get("selected_model") or None
    temperature = state.get("temperature", None)

    return run_stage_node(
        client=client,
        messages=messages,
        stage_name="document_finalization",
        output_key="final_prd_json",
        state=state,
        model=model,
        temperature=temperature,
    )
