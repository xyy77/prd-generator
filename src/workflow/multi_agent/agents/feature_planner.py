import json

from src.utils.llm_client import LLMClient
from src.prompts.manager import PromptManager
from src.workflow.multi_agent.node_utils import run_agent_node
from src.utils.logger import get_logger

logger = get_logger(__name__)


def feature_planner_node(state: dict, reference_context: str = "") -> dict:
    client = LLMClient()
    prompt_mgr = PromptManager()
    model = state.get("selected_model") or None

    messages = prompt_mgr.get_agent_prompt(
        agent="feature_planner",
        product_idea=state.get("product_idea", ""),
        supplementary_info=state.get("supplementary_info", ""),
        reference_context=reference_context or state.get("retrieved_context", ""),
        requirement_analysis=json.dumps(state.get("requirement_analysis", {}), ensure_ascii=False),
    )

    return run_agent_node(client, messages, "feature_planner", "feature_plan", model=model)
