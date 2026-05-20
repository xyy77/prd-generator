import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.llm_client import LLMClient
from src.workflow.node_utils import safe_json_extract
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_correction_messages(
    original_messages: list[dict],
    failed_raw: str,
    parse_error: str,
) -> list[dict]:
    """Build a correction prompt asking the LLM to fix its malformed JSON output."""
    correction_prompt = f"""你的上一次输出不是有效的 JSON 格式。请将以下内容修复为标准 JSON。

JSON 解析错误：{parse_error}

--- 你的上一次输出（格式有误）---
{failed_raw[:8000]}

---

请将以上内容修复为有效的 JSON。严格输出纯 JSON，不要包含 markdown 代码块标记或任何解释性文字。"""
    return [
        original_messages[0],  # system message
        {"role": "user", "content": correction_prompt},
    ]


def run_agent_node(
    client: LLMClient,
    messages: list[dict],
    agent_name: str,
    output_key: str,
    model: str | None = None,
) -> dict:
    raw = ""
    try:
        raw = client.chat_with_json_mode(messages, model=model)
        logger.info("Agent %s complete, response length: %d", agent_name, len(raw))
        parsed = safe_json_extract(raw)
        return {output_key: parsed, "current_stage": agent_name}
    except json.JSONDecodeError as e:
        logger.warning(
            "Agent %s JSON parse failed: %s. Retrying with correction prompt...",
            agent_name, e,
        )
        try:
            correction_msgs = _build_correction_messages(messages, raw, str(e))
            raw2 = client.chat(correction_msgs, model=model)
            logger.info("Agent %s correction response length: %d", agent_name, len(raw2))
            parsed = safe_json_extract(raw2)
            return {output_key: parsed, "current_stage": agent_name, "_corrected": True}
        except Exception as e2:
            logger.error("Agent %s correction also failed: %s", agent_name, e2)
            return {output_key: {}, "current_stage": agent_name, "error_message": str(e2)}
    except Exception as e:
        logger.error("Agent %s failed: %s", agent_name, e)
        return {output_key: {}, "current_stage": agent_name, "error_message": str(e)}


def run_parallel_agents(
    state: dict,
    agent_names: list[str],
    reference_context: str,
) -> dict:
    from src.workflow.multi_agent.agents import AGENT_NODE_MAP

    results: dict = {}
    errors: list[str] = []

    def run_agent(agent_name: str) -> tuple[str, dict]:
        agent_fn = AGENT_NODE_MAP.get(agent_name)
        if agent_fn is None:
            return agent_name, {"error_message": f"Unknown agent: {agent_name}"}

        try:
            output = agent_fn(state, reference_context)
            return agent_name, output
        except Exception as e:
            logger.error("Agent %s execution failed: %s", agent_name, e)
            return agent_name, {"error_message": str(e)}

    with ThreadPoolExecutor(max_workers=len(agent_names)) as executor:
        future_map = {executor.submit(run_agent, name): name for name in agent_names}
        for future in as_completed(future_map):
            agent_name, output = future.result()
            results[agent_name] = output
            if output.get("error_message"):
                errors.append(f"{agent_name}: {output['error_message']}")

    merged: dict = {}
    for agent_name, output in results.items():
        merged.update(output)

    if errors:
        merged["error_message"] = "; ".join(errors)
    merged["current_stage"] = "parallel_agents_complete"
    return merged
