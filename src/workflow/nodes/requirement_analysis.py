from src.prompts.manager import PromptManager
from src.utils.llm_client import LLMClient
from src.workflow.node_utils import run_stage_node
from src.workflow.state import WorkflowState


def requirement_analysis_node(state: WorkflowState) -> dict:
    client = LLMClient()
    pm = PromptManager()

    messages = pm.get_stage_prompt(
        stage="requirement_analysis",
        product_idea=state.get("product_idea", ""),
        supplementary_info=state.get("supplementary_info", ""),
        reference_context=state.get("retrieved_context", ""),
    )

    return run_stage_node(
        client=client,
        messages=messages,
        stage_name="requirement_analysis",
        output_key="requirement_analysis",
        state=state,
    )
