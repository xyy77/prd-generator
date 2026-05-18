import json

from src.prompts.manager import PromptManager
from src.utils.llm_client import LLMClient
from src.workflow.node_utils import run_stage_node
from src.workflow.state import WorkflowState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prd_revision_node(state: WorkflowState) -> dict:
    client = LLMClient()
    pm = PromptManager()

    existing_prd = state.get("final_prd_json", {})
    user_feedback = state.get("user_feedback", "")
    revision_history = list(state.get("revision_history", []))
    revision_count = state.get("revision_count", 0)

    if not user_feedback:
        logger.warning("No user feedback provided for revision")
        return {"error_message": "未提供修改意见"}

    messages = pm.get_stage_prompt(
        stage="prd_revision",
        product_idea=state.get("product_idea", ""),
        reference_context=state.get("retrieved_context", ""),
        existing_prd_json=json.dumps(existing_prd, ensure_ascii=False, indent=2),
        user_feedback=user_feedback,
    )

    # Save current version to history before revision
    revision_entry = {
        "version": revision_count + 1,
        "feedback": user_feedback,
        "prd_before": existing_prd,
    }
    revision_history.append(revision_entry)

    model = state.get("selected_model") or None
    temperature = state.get("temperature", None)

    result = run_stage_node(
        client=client,
        messages=messages,
        stage_name="prd_revision",
        output_key="final_prd_json",
        state=state,
        model=model,
        temperature=temperature,
    )

    result["revision_history"] = revision_history
    result["revision_count"] = revision_count + 1
    result["current_stage"] = "prd_revision_complete"
    logger.info("PRD revision #%d complete", revision_count + 1)
    return result
