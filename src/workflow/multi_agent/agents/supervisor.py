import json

from src.utils.llm_client import MultiProviderLLMClient
from src.prompts.manager import PromptManager
from src.workflow.multi_agent.state import AGENT_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)

ALL_AGENTS = list(AGENT_NAMES)

DISPATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "dispatch_agents",
        "description": "根据产品复杂度、当前进度和用户反馈，决定下一步需要调度哪些Agent及其执行顺序",
        "parameters": {
            "type": "object",
            "properties": {
                "agents_to_call": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["requirements_analyst", "feature_planner", "ux_designer", "tech_advisor"],
                    },
                    "description": "本次需要调用的Agent列表",
                },
                "execution_order": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Agent的执行顺序（有依赖关系时按序排列）",
                },
                "decision_rationale": {
                    "type": "string",
                    "description": "调度决策的理由说明",
                },
            },
            "required": ["agents_to_call", "execution_order", "decision_rationale"],
        },
    },
}


def supervisor_node(state: dict, reference_context: str = "") -> dict:
    """Decide which agents to call next based on planner output and current progress."""
    completed = state.get("completed_agents", [])
    agents_to_revise = state.get("agents_to_revise", [])
    planner_output = state.get("planner_output", {})
    execution_order = state.get("execution_order", [])

    # Pass-through: supervisor already decided, agents are running
    if execution_order and completed:
        remaining = [a for a in execution_order if a not in completed]
        if remaining:
            return {
                "agents_to_call": remaining,
                "execution_order": execution_order,
                "current_stage": "supervisor",
            }
        return {
            "agents_to_call": [],
            "execution_order": [],
            "current_stage": "supervisor",
        }

    # Revision mode: only re-run flagged agents
    if agents_to_revise:
        pending = [a for a in agents_to_revise if a in ALL_AGENTS and a not in completed]
        if pending:
            rationale = "用户修改意见触发全量修订" if state.get("user_feedback") else f"修订模式：重新执行 {', '.join(pending)}"
            logger.info("Supervisor (revision): re-running %s", pending)
            return {
                "agents_to_call": pending,
                "execution_order": pending,
                "supervisor_decision": {"decision_rationale": rationale},
                "current_stage": "supervisor",
            }
        # All revision agents done
        return {
            "agents_to_call": [],
            "execution_order": [],
            "supervisor_decision": {"decision_rationale": "修订完成"},
            "current_stage": "supervisor",
        }

    # First call after planner: ask LLM to decide
    if not completed:
        complexity = planner_output.get("complexity", "medium")
        product_type = planner_output.get("product_type", "")

        # Heuristic fallback for simple products (skip LLM call)
        if complexity == "simple":
            agents = ["requirements_analyst", "tech_advisor"]
            logger.info("Supervisor (heuristic): simple product, using %s", agents)
            return {
                "agents_to_call": agents,
                "execution_order": agents,
                "supervisor_decision": {
                    "decision_rationale": f"简单产品（{product_type}），只需需求分析+技术建议",
                    "skip_reason": {
                        "feature_planner": "简单产品无需专门功能规划",
                        "ux_designer": "简单产品无需专门UX设计",
                    },
                },
                "current_stage": "supervisor",
            }

        # For medium/complex: use LLM with function calling to decide
        try:
            client = MultiProviderLLMClient()
            prompt_mgr = PromptManager()
            model = state.get("selected_model") or None

            messages = prompt_mgr.get_agent_prompt(
                agent="supervisor",
                product_idea=state.get("product_idea", ""),
                planner_output=json.dumps(planner_output, ensure_ascii=False),
                execution_plan=json.dumps(planner_output.get("execution_plan", []), ensure_ascii=False),
                agents_to_call=json.dumps(completed, ensure_ascii=False),
            )

            sd = client.chat_with_tools(messages, tools=[DISPATCH_TOOL], model=model)
            logger.info("Supervisor function calling result: agents=%s", sd.get("agents_to_call", []))

            return {
                "agents_to_call": sd.get("agents_to_call", list(ALL_AGENTS)),
                "execution_order": sd.get("execution_order", list(ALL_AGENTS)),
                "supervisor_decision": sd,
                "current_stage": "supervisor",
            }

        except Exception as e:
            logger.error("Supervisor function calling failed: %s, falling back to all agents", e)
            return {
                "agents_to_call": list(ALL_AGENTS),
                "execution_order": list(ALL_AGENTS),
                "supervisor_decision": {"decision_rationale": "Supervisor function calling 失败，fallback 全部 Agent"},
                "current_stage": "supervisor",
            }

    # Check remaining agents from previous supervisor decision
    remaining = [a for a in ALL_AGENTS if a not in completed]
    if remaining:
        logger.info("Supervisor: remaining agents %s", remaining)
        return {
            "agents_to_call": remaining,
            "execution_order": remaining,
            "supervisor_decision": {"decision_rationale": f"继续执行剩余 Agent: {', '.join(remaining)}"},
            "current_stage": "supervisor",
        }

    # All done
    return {
        "agents_to_call": [],
        "execution_order": [],
        "supervisor_decision": {"decision_rationale": "所有 Agent 已完成"},
        "current_stage": "supervisor",
    }
