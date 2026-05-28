import json

from src.utils.llm_client import LLMClient
from src.prompts.manager import PromptManager
from src.workflow.multi_agent.node_utils import run_agent_with_reflexion
from src.utils.logger import get_logger

logger = get_logger(__name__)


def requirements_analyst_node(state: dict, reference_context: str = "") -> dict:
    client = LLMClient()
    prompt_mgr = PromptManager()
    model = state.get("selected_model") or None

    messages = prompt_mgr.get_agent_prompt(
        agent="requirements_analyst",
        product_idea=state.get("product_idea", ""),
        supplementary_info=state.get("supplementary_info", ""),
        reference_context=reference_context or state.get("retrieved_context", ""),
        image_analysis=json.dumps(state.get("image_analysis", {}), ensure_ascii=False),
        product_type=state.get("planner_output", {}).get("product_type", ""),
        user_feedback=state.get("user_feedback", ""),
    )

    return run_agent_with_reflexion(client, messages, "requirements_analyst", "requirement_analysis", model=model)
