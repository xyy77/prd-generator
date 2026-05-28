from src.prompts.templates import STAGE_PROMPTS, StagePromptTemplate
from src.prompts.multi_agent_prompts import AGENT_PROMPTS, PRODUCT_TYPE_HINTS, AgentPromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_product_type_hint(product_type: str) -> str:
    """Get product type hint for injection into agent prompts."""
    if not product_type:
        return ""
    return PRODUCT_TYPE_HINTS.get(product_type, "")


class PromptManager:
    def build_messages(
        self,
        template: StagePromptTemplate | AgentPromptTemplate,
        **kwargs: str,
    ) -> list[dict]:
        system_msg = template.system_message
        user_msg = template.user_message_template.format(**kwargs)
        stage_name = getattr(template, "stage_name", None) or getattr(template, "agent_name", "unknown")
        logger.debug("Built prompt for: %s", stage_name)
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def get_stage_prompt(
        self,
        stage: str,
        product_idea: str,
        supplementary_info: str = "",
        reference_context: str = "",
        requirement_analysis: str = "",
        architecture_design: str = "",
        process_flow: str = "",
        existing_prd_json: str = "",
        user_feedback: str = "",
    ) -> list[dict]:
        from datetime import date

        template = STAGE_PROMPTS.get(stage)
        if template is None:
            raise ValueError(f"Unknown stage: {stage}")

        kwargs: dict[str, str] = dict(
            product_idea=product_idea,
            supplementary_info=supplementary_info or "无额外补充信息",
            reference_context=reference_context or "暂无参考案例",
            current_date=date.today().strftime("%Y-%m-%d"),
        )

        if stage == "prd_revision":
            kwargs["existing_prd_json"] = existing_prd_json or "{}"
            kwargs["user_feedback"] = user_feedback or "无修改意见"

        if stage in ("architecture_design", "process_flow", "document_finalization"):
            kwargs["requirement_analysis"] = requirement_analysis or "{}"
        if stage in ("process_flow", "document_finalization"):
            kwargs["architecture_design"] = architecture_design or "{}"
        if stage == "document_finalization":
            kwargs["process_flow"] = process_flow or "{}"

        return self.build_messages(template, **kwargs)

    def get_agent_prompt(
        self,
        agent: str,
        product_idea: str,
        supplementary_info: str = "",
        reference_context: str = "",
        requirement_analysis: str = "",
        feature_plan: str = "",
        ux_design: str = "",
        tech_advice: str = "",
        image_analysis: str = "",
        product_type: str = "",
        planner_output: str = "",
        execution_plan: str = "",
        agents_to_call: str = "",
        user_feedback: str = "",
    ) -> list[dict]:
        from datetime import date

        template = AGENT_PROMPTS.get(agent)
        if template is None:
            raise ValueError(f"Unknown agent: {agent}")

        kwargs: dict[str, str] = dict(
            product_idea=product_idea,
            supplementary_info=supplementary_info or "无额外补充信息",
            reference_context=reference_context or "暂无参考案例",
            image_analysis=image_analysis or "无图片输入，请基于文本描述进行分析。",
            current_date=date.today().strftime("%Y-%m-%d"),
        )

        if agent == "planner":
            return self.build_messages(template, **kwargs)

        if agent == "supervisor":
            kwargs["planner_output"] = planner_output or "{}"
            kwargs["execution_plan"] = execution_plan or "[]"
            kwargs["agents_to_call"] = agents_to_call or "[]"
            return self.build_messages(template, **kwargs)

        if agent in ("feature_planner", "ux_designer", "tech_advisor", "reviewer"):
            kwargs["requirement_analysis"] = requirement_analysis or "{}"
        if agent in ("ux_designer", "tech_advisor", "reviewer"):
            kwargs["feature_plan"] = feature_plan or "{}"
        if agent in ("tech_advisor", "reviewer"):
            kwargs["ux_design"] = ux_design or "{}"
        if agent == "reviewer":
            kwargs["tech_advice"] = tech_advice or "{}"

        messages = self.build_messages(template, **kwargs)

        # Inject user feedback for worker agents (revision mode)
        if user_feedback and agent in ("requirements_analyst", "feature_planner", "ux_designer", "tech_advisor"):
            messages[1]["content"] += (
                f"\n\n【用户修改意见】\n{user_feedback}\n\n"
                "请根据以上修改意见调整你的输出。保留原输出中合理且不冲突的部分，重点修改与意见相关的部分。"
            )

        # Inject product type hint for worker agents
        hint = _get_product_type_hint(product_type)
        if hint and agent in ("requirements_analyst", "feature_planner", "ux_designer", "tech_advisor"):
            messages[1]["content"] += f"\n\n{hint}"

        return messages
