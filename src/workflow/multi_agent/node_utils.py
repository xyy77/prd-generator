import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.llm_client import LLMClient, MultiProviderLLMClient
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
    use_json_mode: bool = True,
) -> dict:
    raw = ""
    try:
        if use_json_mode:
            raw = client.chat_with_json_mode(messages, model=model)
        else:
            raw = client.chat(messages, model=model)
        logger.info("Agent %s complete, response length: %d", agent_name, len(raw))
        parsed = safe_json_extract(raw)
        return {output_key: parsed, "current_stage": agent_name}
    except json.JSONDecodeError as e:
        logger.warning(
            "Agent %s JSON parse failed: %s. Raw preview: %.300s",
            agent_name, e, raw,
        )
        if not raw.strip():
            logger.error("Agent %s returned empty response", agent_name)
            return {output_key: {}, "current_stage": agent_name, "error_message": "Empty LLM response"}
        try:
            correction_msgs = _build_correction_messages(messages, raw, str(e))
            raw2 = client.chat(correction_msgs, model=model)
            logger.info("Agent %s correction response length: %d", agent_name, len(raw2))
            parsed = safe_json_extract(raw2)
            logger.info("Agent %s correction succeeded", agent_name)
            return {output_key: parsed, "current_stage": agent_name, "_corrected": True}
        except json.JSONDecodeError as e2:
            logger.error("Agent %s correction also failed: %s. Raw preview: %.300s",
                         agent_name, e2, raw2)
            return {output_key: {}, "current_stage": agent_name, "error_message": str(e2)}
        except Exception as e2:
            logger.error("Agent %s correction unexpected error: %s", agent_name, e2)
            return {output_key: {}, "current_stage": agent_name, "error_message": str(e2)}
    except Exception as e:
        logger.error("Agent %s failed: %s", agent_name, e)
        return {output_key: {}, "current_stage": agent_name, "error_message": str(e)}


def _build_self_eval_messages(agent_name: str, output: dict) -> list[dict]:
    """Build messages for self-evaluation."""
    output_json = json.dumps(output, ensure_ascii=False)
    eval_prompt = f"""请评估以下 {agent_name} 的输出质量（0-100分）。

输出内容：
{output_json[:4000]}

评估标准：
- 完整性：是否覆盖所有要求字段，内容是否充实
- 具体性：是否给出可执行的具体内容而非泛泛而谈
- 一致性：内容是否与产品想法一致，内部是否有矛盾

返回 JSON: {{"score": 整数, "issues": ["具体问题1", "具体问题2"], "improvement_suggestions": ["改进建议1"]}}

只输出 JSON，不要包含任何其他文字。"""
    return [
        {"role": "system", "content": "你是一位严格的质量评估专家。只输出JSON。"},
        {"role": "user", "content": eval_prompt},
    ]


def _build_reflexion_correction_messages(
    original_messages: list[dict],
    draft_output: dict,
    issues: list[str],
) -> list[dict]:
    """Build correction messages for reflexion self-improvement."""
    issues_text = "\n".join(f"- {i}" for i in issues)
    correction_prompt = f"""你之前的输出存在以下问题：

{issues_text}

请修正你的输出，确保：
1. 内容完整、具体、可执行
2. 所有字段都有实质内容（不要有空数组或占位文本）
3. 各部分之间保持一致

严格返回纯 JSON，不要包含 markdown 代码块标记或任何解释性文字。

原始输出供参考：
{json.dumps(draft_output, ensure_ascii=False)[:2000]}"""
    return [
        original_messages[0],
        {"role": "user", "content": correction_prompt},
    ]


def run_agent_with_reflexion(
    client,
    messages: list[dict],
    agent_name: str,
    output_key: str,
    model: str | None = None,
    reflexion_threshold: int = 75,
    use_json_mode: bool = True,
) -> dict:
    """Run an agent with self-reflection: generate → self-evaluate → correct if needed."""
    # 1. Initial generation
    result = run_agent_node(client, messages, agent_name, output_key, model=model, use_json_mode=use_json_mode)
    draft = result.get(output_key, {})

    if not isinstance(draft, dict) or not draft:
        logger.warning("Agent %s produced empty output, skipping reflexion", agent_name)
        return result

    # 2. Self-evaluation
    try:
        eval_messages = _build_self_eval_messages(agent_name, draft)
        eval_raw = client.chat_with_json_mode(eval_messages, model=model)
        eval_result = safe_json_extract(eval_raw)
        score = eval_result.get("score", 0)

        if isinstance(score, bool):
            score = 0
        if not isinstance(score, (int, float)):
            score = 0

        logger.info("Agent %s self-evaluation score: %s", agent_name, score)

        # 3. If below threshold, self-correct (max 1 round)
        if score < reflexion_threshold:
            issues = eval_result.get("issues", [])
            logger.info("Agent %s score %d < %d, triggering self-correction", agent_name, score, reflexion_threshold)
            correction_msgs = _build_reflexion_correction_messages(messages, draft, issues)
            raw2 = client.chat_with_json_mode(correction_msgs, model=model)
            corrected = safe_json_extract(raw2)
            if isinstance(corrected, dict) and corrected:
                result[output_key] = corrected
                result["_reflexion_applied"] = True
                result["_reflexion_score_before"] = score
            else:
                logger.warning("Agent %s reflexion correction produced invalid output, keeping draft", agent_name)
        else:
            result["_reflexion_evaluated"] = True
            result["_reflexion_score"] = score
    except Exception as e:
        logger.warning("Agent %s reflexion failed: %s, keeping original output", agent_name, e)

    return result


def run_agent_with_tools(
    client,
    messages: list[dict],
    tools: list[dict],
    tool_fns: dict,
    agent_name: str,
    output_key: str,
    model: str | None = None,
    max_tool_rounds: int = 3,
    reflexion_threshold: int = 75,
) -> dict:
    """Run an agent with Function Calling tool access (ReAct loop).

    The agent can decide autonomously when to call tools during reasoning.
    After tool use rounds complete and a final JSON output is produced,
    the standard Reflexion self-evaluation + correction is applied.

    Args:
        client: MultiProviderLLMClient (must support chat_with_tools).
        messages: Initial system + user messages.
        tools: OpenAI Function Calling tool definitions.
        tool_fns: Mapping from tool name to callable, e.g. {"search": _search_impl}.
        agent_name: Name of the agent (for logging).
        output_key: Key to store the parsed result under.
        model: Optional model override.
        max_tool_rounds: Max tool-call rounds (prevents infinite loops).
        reflexion_threshold: Score threshold for Reflexion self-correction.
    """
    conversation = list(messages)
    raw = ""
    tool_round = 0

    while tool_round < max_tool_rounds:
        tool_round += 1
        try:
            result = client.chat_with_tools(
                conversation, tools=tools, model=model, force_tool=False,
            )
        except Exception as e:
            logger.error("Agent %s chat_with_tools failed (round %d): %s", agent_name, tool_round, e)
            raw = ""
            break

        if not isinstance(result, dict):
            raw = str(result) if result else ""
            break

        # Check if LLM made a tool call or returned final answer
        is_tool_call = result.pop("_is_tool_call", False)
        tool_name = result.pop("_tool_name", None)

        if not is_tool_call or not tool_name:
            # Final answer — strip sentinel fields and break the loop
            result.pop("_is_tool_call", None)
            result.pop("_tool_name", None)
            raw = json.dumps(result, ensure_ascii=False)
            break

        tool_args = {k: v for k, v in result.items() if not k.startswith("_")}
        fn = tool_fns.get(tool_name)
        if fn is None:
            tool_output = json.dumps({"error": f"Unknown tool: {tool_name}"})
            logger.warning("Agent %s called unknown tool '%s'", agent_name, tool_name)
        else:
            try:
                tool_output = fn(**tool_args)
                logger.info("Agent %s tool '%s' OK (%d chars)", agent_name, tool_name, len(tool_output))
            except Exception as e:
                tool_output = json.dumps({"error": str(e), "tool": tool_name})
                logger.warning("Agent %s tool '%s' exec failed: %s", agent_name, tool_name, e)

        conversation.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{tool_name}_{tool_round}",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(tool_args, ensure_ascii=False)},
            }],
        })
        conversation.append({
            "role": "tool",
            "tool_call_id": f"call_{tool_name}_{tool_round}",
            "content": tool_output,
        })

    if not raw:
        try:
            raw = client.chat_with_json_mode(conversation, model=model)
        except Exception as e:
            logger.error("Agent %s final JSON call failed: %s", agent_name, e)
            return {output_key: {}, "current_stage": agent_name, "error_message": str(e)}

    logger.info("Agent %s (tools) final response length: %d, tool rounds: %d", agent_name, len(raw), tool_round)

    try:
        parsed = safe_json_extract(raw)
    except json.JSONDecodeError as e:
        logger.warning("Agent %s (tools) JSON parse failed: %s", agent_name, e)
        return {output_key: {}, "current_stage": agent_name, "error_message": str(e)}

    if not isinstance(parsed, dict) or not parsed:
        logger.warning("Agent %s (tools) produced empty output, skipping reflexion", agent_name)
        return {output_key: parsed, "current_stage": agent_name}

    # Reflexion self-evaluation
    try:
        eval_messages = _build_self_eval_messages(agent_name, parsed)
        eval_raw = client.chat_with_json_mode(eval_messages, model=model)
        eval_result = safe_json_extract(eval_raw)
        score = eval_result.get("score", 0)

        if isinstance(score, bool):
            score = 0
        if not isinstance(score, (int, float)):
            score = 0

        logger.info("Agent %s (tools) self-evaluation score: %s", agent_name, score)

        if score < reflexion_threshold:
            issues = eval_result.get("issues", [])
            logger.info("Agent %s (tools) score %d < %d, triggering self-correction",
                        agent_name, score, reflexion_threshold)
            correction_msgs = _build_reflexion_correction_messages(messages, parsed, issues)
            raw2 = client.chat_with_json_mode(correction_msgs, model=model)
            corrected = safe_json_extract(raw2)
            if isinstance(corrected, dict) and corrected:
                result = {output_key: corrected, "current_stage": agent_name}
                result["_reflexion_applied"] = True
                result["_reflexion_score_before"] = score
                result["_tool_rounds"] = tool_round
                return result
        else:
            result = {output_key: parsed, "current_stage": agent_name}
            result["_reflexion_evaluated"] = True
            result["_reflexion_score"] = score
            result["_tool_rounds"] = tool_round
            return result
    except Exception as e:
        logger.warning("Agent %s (tools) reflexion failed: %s, keeping output", agent_name, e)

    result = {output_key: parsed, "current_stage": agent_name}
    result["_tool_rounds"] = tool_round
    return result


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
