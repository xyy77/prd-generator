import json

from src.prompts.manager import PromptManager
from src.utils.llm_client import LLMClient
from src.workflow.node_utils import run_stage_node
from src.utils.logger import get_logger

logger = get_logger(__name__)


def document_synthesis_node(state: dict) -> dict:
    """Synthesize multi-agent outputs into final PRD JSON."""
    client = LLMClient()
    pm = PromptManager()

    req_json = state.get("requirement_analysis", {})
    feature_json = state.get("feature_plan", {})
    ux_json = state.get("ux_design", {})
    tech_json = state.get("tech_advice", {})

    combined_arch = {
        "feature_plan": feature_json,
        "tech_advice": tech_json,
    }

    messages = pm.get_stage_prompt(
        stage="document_finalization",
        product_idea=state.get("product_idea", ""),
        reference_context=state.get("retrieved_context", ""),
        requirement_analysis=json.dumps(req_json, ensure_ascii=False, indent=2),
        architecture_design=json.dumps(combined_arch, ensure_ascii=False, indent=2),
        process_flow=json.dumps(ux_json, ensure_ascii=False, indent=2),
    )

    model = state.get("selected_model") or None

    result = run_stage_node(
        client=client,
        messages=messages,
        stage_name="document_synthesis",
        output_key="final_prd_json",
        state=state,
        model=model,
    )

    reviewer_detail = state.get("_reviewer_detail")
    if reviewer_detail:
        result["_reviewer_detail"] = reviewer_detail

    return result
