import json
import re

from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def safe_json_extract(raw: str) -> dict:
    raw = raw.strip()

    # Strip markdown code fences
    fence_pattern = r"^```(?:json)?\s*\n(.*?)\n```\s*$"
    m = re.match(fence_pattern, raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()

    # Find JSON object boundaries
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fix common LLM JSON issues
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    raw = re.sub(r"(?<!\\)\\(?=[^{}\"\[\]\\])", r"\\\\", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON: %s\nRaw (first 500): %s", e, raw[:500])
        raise


def llm_call_with_logging(
    client: LLMClient,
    messages: list[dict],
    stage_name: str,
) -> str:
    logger.info("Calling LLM for stage: %s", stage_name)
    result = client.chat_with_json_mode(messages)
    logger.info("Stage %s complete, response length: %d", stage_name, len(result))
    return result


def run_stage_node(
    client: LLMClient,
    messages: list[dict],
    stage_name: str,
    output_key: str,
    state: dict,
) -> dict:
    try:
        raw = llm_call_with_logging(client, messages, stage_name)
        parsed = safe_json_extract(raw)
        return {
            output_key: parsed,
            "current_stage": stage_name,
        }
    except Exception as e:
        logger.error("Stage %s failed: %s", stage_name, e)
        return {
            output_key: {},
            "current_stage": stage_name,
            "error_message": str(e),
        }
