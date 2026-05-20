from src.prompts.templates import STAGE_PROMPTS, StagePromptTemplate
from src.prompts.multi_agent_prompts import AGENT_PROMPTS, AgentPromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


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

        if agent in ("feature_planner", "ux_designer", "tech_advisor", "reviewer"):
            kwargs["requirement_analysis"] = requirement_analysis or "{}"
        if agent in ("ux_designer", "tech_advisor", "reviewer"):
            kwargs["feature_plan"] = feature_plan or "{}"
        if agent in ("tech_advisor", "reviewer"):
            kwargs["ux_design"] = ux_design or "{}"
        if agent == "reviewer":
            kwargs["tech_advice"] = tech_advice or "{}"

        return self.build_messages(template, **kwargs)
