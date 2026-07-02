import json
import re

from src.utils.llm_client import LLMClient, MultiProviderLLMClient
from src.prompts.manager import PromptManager
from src.workflow.multi_agent.node_utils import run_agent_with_reflexion, run_agent_with_tools
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALIDATE_MERMAID_TOOL = {
    "type": "function",
    "function": {
        "name": "validate_mermaid",
        "description": "校验 Mermaid 图表代码的语法正确性。生成的 Mermaid 代码在输出前务必调用此工具检查。如果校验失败，请根据返回的错误提示修正代码。",
        "parameters": {
            "type": "object",
            "properties": {
                "mermaid_code": {
                    "type": "string",
                    "description": "需要校验的 Mermaid 代码（不含 ```mermaid 代码块标记）",
                },
            },
            "required": ["mermaid_code"],
        },
    },
}


def _validate_mermaid(mermaid_code: str) -> str:
    """Hard validation of Mermaid syntax — deterministic, no LLM involved.

    Checks:
    1. Required diagram type declaration (graph/flowchart/sequenceDiagram etc.)
    2. Bracket pairing ([]  ()  {}  "")
    3. Arrow syntax validity
    4. Chinese punctuation misuse
    5. Semicolon-terminated lines (Mermaid-style warnings)
    """
    errors = []
    warnings_list = []

    code = mermaid_code.strip()
    if not code:
        return json.dumps({"valid": False, "errors": ["Mermaid 代码为空"], "warnings": [], "suggestions": ["请提供有效的 Mermaid 图表代码"]})

    # 1. Diagram type declaration
    diagram_types = r'\b(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|gitGraph|mindmap|timeline)\b'
    if not re.search(diagram_types, code, re.IGNORECASE):
        errors.append("缺少图表类型声明（如 graph TD、flowchart LR、sequenceDiagram 等）")

    # 2. Bracket pairing
    brackets = {"[": "]", "(": ")", "{": "}", '"': '"'}
    for open_b, close_b in [("[", "]"), ("(", ")"), ("{", "}")]:
        count = code.count(open_b) - code.count(close_b)
        if count > 0:
            errors.append(f"存在未闭合的 {open_b}（多出 {count} 个）")
        elif count < 0:
            errors.append(f"存在多余的 {close_b}（多出 {-count} 个）")

    # 3. Arrow syntax
    arrow_count = len(re.findall(r'-->|-->\|.*?\||->|-.->|==>|\.\.->|--->|---', code))
    if arrow_count == 0 and re.search(r'\b(graph|flowchart)\b', code, re.IGNORECASE):
        warnings_list.append("graph/flowchart 类型建议包含箭头连接（如 A --> B）")

    # 4. Chinese punctuation
    for cn_char, en_char in [("，", ","), ("。", "."), ("；", ";"), ("：", ":"), ("（", "("), ("）", ")")]:
        if cn_char in code:
            warnings_list.append(f"包含中文标点 '{cn_char}'，建议替换为 '{en_char}'")

    # 5. Semicolon-terminated lines
    if re.search(r';[ \t]*$', code, re.MULTILINE):
        warnings_list.append("包含分号结尾的行，部分 Mermaid 渲染器可能报错")

    valid = len(errors) == 0
    return json.dumps({
        "valid": valid,
        "errors": errors,
        "warnings": warnings_list,
        "suggestions": ["请根据上述错误修正 Mermaid 代码"] if errors else [],
    }, ensure_ascii=False)


def ux_designer_node(state: dict, reference_context: str = "") -> dict:
    prompt_mgr = PromptManager()
    model = state.get("selected_model") or None

    messages = prompt_mgr.get_agent_prompt(
        agent="ux_designer",
        product_idea=state.get("product_idea", ""),
        supplementary_info=state.get("supplementary_info", ""),
        reference_context=reference_context or state.get("retrieved_context", ""),
        requirement_analysis=json.dumps(state.get("requirement_analysis", {}), ensure_ascii=False),
        feature_plan=json.dumps(state.get("feature_plan", {}), ensure_ascii=False),
        image_analysis=json.dumps(state.get("image_analysis", {}), ensure_ascii=False),
        product_type=state.get("planner_output", {}).get("product_type", ""),
        user_feedback=state.get("user_feedback", ""),
    )

    try:
        client = MultiProviderLLMClient()
        tools = [VALIDATE_MERMAID_TOOL]
        tool_fns = {"validate_mermaid": _validate_mermaid}

        logger.info("UX designer running with tools (validate_mermaid available)")
        return run_agent_with_tools(
            client, messages, tools, tool_fns,
            "ux_designer", "ux_design", model=model,
        )
    except Exception as e:
        logger.error("UX designer with tools failed: %s, falling back to reflexion mode", e)
        client = LLMClient()
        return run_agent_with_reflexion(client, messages, "ux_designer", "ux_design", model=model)
