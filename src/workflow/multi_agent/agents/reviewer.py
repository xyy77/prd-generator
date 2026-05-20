import json

from src.utils.llm_client import LLMClient
from src.prompts.manager import PromptManager
from src.workflow.node_utils import safe_json_extract
from src.workflow.multi_agent.node_utils import _build_correction_messages
from src.utils.logger import get_logger

logger = get_logger(__name__)


def reviewer_node(state: dict, reference_context: str = "") -> dict:
    client = LLMClient()
    prompt_mgr = PromptManager()
    model = state.get("selected_model") or None
    agent_errors = _collect_agent_errors(state)

    messages = prompt_mgr.get_agent_prompt(
        agent="reviewer",
        product_idea=state.get("product_idea", ""),
        supplementary_info=state.get("supplementary_info", ""),
        reference_context=reference_context or state.get("retrieved_context", ""),
        requirement_analysis=json.dumps(state.get("requirement_analysis", {}), ensure_ascii=False),
        feature_plan=json.dumps(state.get("feature_plan", {}), ensure_ascii=False),
        ux_design=json.dumps(state.get("ux_design", {}), ensure_ascii=False),
        tech_advice=json.dumps(state.get("tech_advice", {}), ensure_ascii=False),
        image_analysis=json.dumps(state.get("image_analysis", {}), ensure_ascii=False),
    )

    try:
        parsed = _call_reviewer_with_retry(client, messages, model)
        return _build_reviewer_result(parsed, agent_errors)
    except Exception as e:
        logger.error("Reviewer failed after retry: %s", e)
        return {
            "reviewer_score": 75,
            "reviewer_scores": {},
            "reviewer_feedback": {},
            "reviewer_summary": "评审官评估失败，已跳过评审直接生成文档",
            "current_stage": "reviewer",
            "error_message": str(e),
            "_agent_errors": agent_errors,
            "_reviewer_failed": True,
        }


def _call_reviewer_with_retry(
    client: LLMClient, messages: list[dict], model: str | None
) -> dict:
    raw = ""
    try:
        raw = client.chat_with_json_mode(messages, model=model)
        logger.info("Reviewer complete, response length: %d", len(raw))
        return safe_json_extract(raw)
    except json.JSONDecodeError as e:
        logger.warning("Reviewer JSON parse failed: %s. Retrying with correction...", e)
        try:
            correction_msgs = _build_correction_messages(messages, raw, str(e))
            raw2 = client.chat(correction_msgs, model=model)
            logger.info("Reviewer correction response length: %d", len(raw2))
            return safe_json_extract(raw2)
        except Exception as e2:
            logger.error("Reviewer correction also failed: %s", e2)
            raise
    except Exception:
        raise


def _build_reviewer_result(parsed: dict, agent_errors: list[str]) -> dict:
    result: dict = {
        "reviewer_score": parsed.get("overall_score", 0),
        "reviewer_scores": parsed.get("scores", {}),
        "reviewer_feedback": parsed.get("feedback", {}),
        "reviewer_summary": parsed.get("summary", ""),
        "current_stage": "reviewer",
    }

    feedback = parsed.get("feedback", {})
    agents_to_revise = [name for name, fb in feedback.items() if fb and str(fb).lower() != "null"]
    if agents_to_revise:
        result["agents_to_revise"] = agents_to_revise

    consistency_issues = parsed.get("consistency_issues", [])
    completeness_gaps = parsed.get("completeness_gaps", [])
    if consistency_issues or completeness_gaps:
        result["_reviewer_detail"] = {
            "consistency_issues": consistency_issues,
            "completeness_gaps": completeness_gaps,
        }

    if agent_errors:
        result.setdefault("_agent_errors", agent_errors)

    return result


def _collect_agent_errors(state: dict) -> list[str]:
    errors: list[str] = []
    for key in ("requirement_analysis", "feature_plan", "ux_design", "tech_advice"):
        val = state.get(key)
        if isinstance(val, dict) and not val:
            errors.append(f"{key} 输出为空（可能解析失败）")
    return errors
