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

    # Multi-pass repair loop
    for _ in range(3):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raw = _repair_json(raw)

    raise json.JSONDecodeError("safe_json_extract: unable to repair JSON", raw, 0)


def _repair_json(raw: str) -> str:
    """Apply successive JSON repair heuristics. Returns repaired string."""
    # Remove trailing commas in objects/arrays
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)

    # Fix bad escape sequences (backslash before non-escape char)
    raw = re.sub(r"\\(?=[^{}\"\[\]\\/bfnrtu])", r"\\\\", raw)

    # Fix unquoted property names: {key: or , key: → {"key": or , "key":
    raw = re.sub(r'([\{,]\s*)([a-zA-Z_一-鿿][a-zA-Z0-9_一-鿿]*)\s*:', r'\1"\2":', raw)

    # Fix single-quoted strings (keys and values): 'xxx' → "xxx"
    raw = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', raw)

    # Fix missing commas between consecutive values
    # "string" followed by " (newline or space)
    raw = re.sub(r'("(?:\\.|[^"\\])*")\s*\n\s*"', r'\1,\n  "', raw)
    # } or ] followed by "
    raw = re.sub(r"([}\]])\s*\n\s*\"", r'\1,\n  "', raw)
    # number/true/false/null followed by "
    raw = re.sub(r"(\d+|true|false|null)\s*\n\s*\"", r'\1,\n  "', raw)
    # } followed by { or [
    raw = re.sub(r"([}\]])\s*\n\s*([\{[])", r"\1,\n  \2", raw)

    # Balance unclosed braces/brackets
    open_braces = raw.count("{") - raw.count("}")
    open_brackets = raw.count("[") - raw.count("]")
    raw = raw.rstrip(",\n\r\t ")
    raw += "}" * max(0, open_braces)
    raw += "]" * max(0, open_brackets)

    # Remove trailing content after the last balanced closer
    last_brace = raw.rfind("}")
    last_bracket = raw.rfind("]")
    last = max(last_brace, last_bracket)
    if last != -1 and last < len(raw) - 1:
        after = raw[last + 1:].strip()
        if after and not after.startswith((",", "}", "]")):
            raw = raw[:last + 1]

    return raw


def llm_call_with_logging(
    client: LLMClient,
    messages: list[dict],
    stage_name: str,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    logger.info("Calling LLM for stage: %s (model=%s)", stage_name, model or "default")
    result = client.chat_with_json_mode(messages, model=model)
    logger.info("Stage %s complete, response length: %d", stage_name, len(result))
    return result


def run_stage_node(
    client: LLMClient,
    messages: list[dict],
    stage_name: str,
    output_key: str,
    state: dict,
    model: str | None = None,
    temperature: float | None = None,
) -> dict:
    try:
        raw = llm_call_with_logging(client, messages, stage_name, model=model, temperature=temperature)
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
