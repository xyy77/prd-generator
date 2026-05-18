from src.prompts.templates import STAGE_PROMPTS, StagePromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptManager:
    def build_messages(
        self,
        template: StagePromptTemplate,
        **kwargs: str,
    ) -> list[dict]:
        system_msg = template.system_message
        user_msg = template.user_message_template.format(**kwargs)
        logger.debug("Built prompt for stage: %s", template.stage_name)
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
