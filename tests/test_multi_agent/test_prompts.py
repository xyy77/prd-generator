from src.prompts.manager import PromptManager
from src.prompts.multi_agent_prompts import AGENT_PROMPTS


class TestMultiAgentPrompts:
    def test_all_agent_templates_present(self):
        expected = [
            "requirements_analyst",
            "feature_planner",
            "ux_designer",
            "tech_advisor",
            "reviewer",
            "image_analyst",
        ]
        for agent in expected:
            assert agent in AGENT_PROMPTS, f"Missing agent: {agent}"

    def test_get_agent_prompt_returns_messages(self):
        pm = PromptManager()
        messages = pm.get_agent_prompt(
            agent="requirements_analyst",
            product_idea="测试产品",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "测试产品" in messages[1]["content"]

    def test_get_agent_prompt_unknown_raises(self):
        pm = PromptManager()
        try:
            pm.get_agent_prompt(agent="nonexistent", product_idea="X")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_reviewer_gets_all_agent_outputs(self):
        pm = PromptManager()
        messages = pm.get_agent_prompt(
            agent="reviewer",
            product_idea="测试产品",
            requirement_analysis='{"key":"req"}',
            feature_plan='{"key":"feat"}',
            ux_design='{"key":"ux"}',
            tech_advice='{"key":"tech"}',
        )
        content = messages[1]["content"]
        assert "requirement_analysis" in content.lower() or "需求分析师" in content
        assert "feature_plan" in content.lower() or "功能规划师" in content

    def test_image_analysis_default_value(self):
        pm = PromptManager()
        messages = pm.get_agent_prompt(
            agent="requirements_analyst",
            product_idea="测试产品",
        )
        content = messages[1]["content"]
        assert "无图片输入" in content
