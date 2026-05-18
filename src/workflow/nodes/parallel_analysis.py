import json
from concurrent.futures import ThreadPoolExecutor

from src.prompts.manager import PromptManager
from src.utils.llm_client import LLMClient
from src.workflow.node_utils import safe_json_extract, llm_call_with_logging
from src.workflow.state import WorkflowState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _run_requirement_analysis(state: dict, reference_context: str) -> dict:
    client = LLMClient()
    pm = PromptManager()
    messages = pm.get_stage_prompt(
        stage="requirement_analysis",
        product_idea=state.get("product_idea", ""),
        supplementary_info=state.get("supplementary_info", ""),
        reference_context=reference_context,
    )
    raw = llm_call_with_logging(client, messages, "requirement_analysis")
    parsed = safe_json_extract(raw)
    return {"requirement_analysis": parsed}


def _run_architecture_design(state: dict, reference_context: str) -> dict:
    client = LLMClient()
    pm = PromptManager()
    messages = pm.get_stage_prompt(
        stage="architecture_design",
        product_idea=state.get("product_idea", ""),
        supplementary_info=state.get("supplementary_info", ""),
        reference_context=reference_context,
    )
    raw = llm_call_with_logging(client, messages, "architecture_design")
    parsed = safe_json_extract(raw)
    return {"architecture_design": parsed}


def parallel_analysis_node(state: WorkflowState) -> dict:
    reference_context = state.get("retrieved_context", "")

    with ThreadPoolExecutor(max_workers=2) as executor:
        req_future = executor.submit(_run_requirement_analysis, state, reference_context)
        arch_future = executor.submit(_run_architecture_design, state, reference_context)

        req_result = req_future.result()
        arch_result = arch_future.result()

    merged: dict = {}
    if "error_message" in req_result:
        merged.setdefault("error_message", req_result["error_message"])
    if "error_message" in arch_result:
        if "error_message" in merged:
            merged["error_message"] += "; " + arch_result["error_message"]
        else:
            merged["error_message"] = arch_result["error_message"]

    merged.update(req_result)
    merged.update(arch_result)
    merged["current_stage"] = "parallel_analysis_complete"
    logger.info("Parallel analysis complete")
    return merged
