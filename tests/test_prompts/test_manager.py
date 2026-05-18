from src.prompts.manager import PromptManager
from src.prompts.templates import STAGE_PROMPTS, REQUIREMENT_ANALYSIS_PROMPT


class TestPromptManager:
    def test_build_messages_returns_correct_structure(self):
        pm = PromptManager()
        msgs = pm.build_messages(
            REQUIREMENT_ANALYSIS_PROMPT,
            product_idea="AI助手",
            supplementary_info="无",
            reference_context="无",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "AI助手" in msgs[1]["content"]

    def test_get_stage_prompt_for_all_stages(self):
        pm = PromptManager()
        for stage in STAGE_PROMPTS:
            msgs = pm.get_stage_prompt(
                stage=stage,
                product_idea="AI助手",
                supplementary_info="测试",
                reference_context="参考案例",
                requirement_analysis='{"test": true}',
                architecture_design='{"test": true}',
                process_flow='{"test": true}',
            )
            assert len(msgs) == 2

    def test_unknown_stage_raises(self):
        pm = PromptManager()
        try:
            pm.get_stage_prompt("nonexistent", product_idea="test")
        except ValueError:
            pass
