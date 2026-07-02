import json
from unittest.mock import patch

from src.workflow.multi_agent.agents.requirements_analyst import requirements_analyst_node
from src.workflow.multi_agent.agents.feature_planner import feature_planner_node
from src.workflow.multi_agent.agents.ux_designer import ux_designer_node
from src.workflow.multi_agent.agents.tech_advisor import tech_advisor_node
from src.workflow.multi_agent.agents.reviewer import reviewer_node
from src.workflow.multi_agent.agents.image_analyst import image_analyst_node
from src.workflow.multi_agent.agents.revision_router import revision_router_node


REQUIREMENTS_JSON = json.dumps({
    "product_name": "测试产品",
    "product_overview": "概述",
    "target_market": "市场",
    "user_personas": [{"role": "用户", "scenario": "场景", "pain_point": "痛点", "goal": "目标", "frequency": "每天"}],
    "user_stories": [{"as_a": "用户", "i_want": "需求", "so_that": "目的", "acceptance_criteria": ["条件1"]}],
    "success_metrics": [{"metric": "活跃度", "target": "80%", "measurement_method": "DAU"}],
    "core_scenarios": [{"name": "场景1", "description": "描述", "priority": "P0"}],
})

FEATURE_JSON = json.dumps({
    "feature_list": [{"name": "功能1", "description": "描述", "moscow": "Must", "effort_estimate": "5人天", "dependencies": []}],
    "mvp_scope": {"included_features": ["功能1"], "excluded_from_mvp": [], "mvp_success_criteria": "标准"},
    "priority_matrix": {"p0_critical": ["功能1"], "p1_important": [], "p2_nice_to_have": []},
    "feature_dependencies_graph": "无依赖",
})

UX_JSON = json.dumps({
    "screens": [{"screen_name": "首页", "layout_description": "布局", "ui_elements": [], "states": {"normal": "正常", "loading": "加载", "empty": "空", "error": "错误"}}],
    "interaction_flows": [{"flow_name": "流程1", "steps": [{"user_action": "点击", "system_response": "响应", "next_state": "下一页"}]}],
    "mermaid_user_flow": "graph TD\nA-->B",
    "edge_cases": [{"case": "边界", "handling": "处理"}],
    "error_handling": [{"error": "错误", "user_message": "提示", "recovery": "恢复"}],
    "design_tokens": {"visual_style": "简洁", "color_suggestion": "蓝色", "component_patterns": []},
    "image_references": [],
})

TECH_JSON = json.dumps({
    "tech_stack": {"frontend": "React", "backend": "Python", "database": "PostgreSQL", "caching": "Redis", "message_queue": "无", "infrastructure": "Docker", "ai_ml": "DeepSeek"},
    "architecture_overview": "概述",
    "system_components": [{"name": "前端", "responsibility": "UI", "tech": "React", "interfaces": []}],
    "data_models": [{"entity": "User", "fields": [{"name": "id", "type": "int", "constraints": "PK"}], "relationships": "无"}],
    "api_endpoints": [{"method": "GET", "path": "/api/users", "description": "列表", "request_body": {}, "response": {}}],
    "non_functional": {"performance": {"target_latency": "100ms", "target_throughput": "1000qps"}, "security": {"auth_method": "JWT", "data_encryption": "TLS"}, "availability": "99.9%", "scalability": "水平扩展"},
    "tech_risks": [{"risk": "风险", "probability": "低", "impact": "高", "mitigation": "措施"}],
})


class TestRequirementsAnalyst:
    def test_node_returns_parsed_json(self):
        with patch("src.workflow.multi_agent.agents.requirements_analyst.MultiProviderLLMClient") as mock_cls:
            # Simulate: no tool call → final JSON output directly
            mock_cls.return_value.chat_with_tools.return_value = {
                "_is_tool_call": False,
                "product_name": "测试产品",
                "product_overview": "概述",
            }
            # Self-evaluation returns high score to skip correction
            mock_cls.return_value.chat_with_json_mode.return_value = '{"score": 90, "issues": []}'
            state = {"product_idea": "AI助手", "retrieved_context": "参考"}
            result = requirements_analyst_node(state)
        assert "requirement_analysis" in result
        assert result["requirement_analysis"]["product_name"] == "测试产品"


class TestFeaturePlanner:
    def test_node_returns_parsed_json(self):
        with patch("src.workflow.multi_agent.agents.feature_planner.LLMClient") as mock_cls:
            mock_cls.return_value.chat_with_json_mode.return_value = FEATURE_JSON
            state = {"product_idea": "AI助手", "requirement_analysis": {}}
            result = feature_planner_node(state)
        assert "feature_plan" in result
        assert len(result["feature_plan"]["feature_list"]) == 1


class TestUXDesigner:
    def test_node_returns_parsed_json(self):
        import json as _json
        ux_data = _json.loads(UX_JSON)
        ux_data["_is_tool_call"] = False

        with patch("src.workflow.multi_agent.agents.ux_designer.MultiProviderLLMClient") as mock_cls:
            mock_cls.return_value.chat_with_tools.return_value = ux_data
            mock_cls.return_value.chat_with_json_mode.return_value = '{"score": 90, "issues": []}'
            state = {"product_idea": "AI助手", "requirement_analysis": {}, "feature_plan": {}}
            result = ux_designer_node(state)
        assert "ux_design" in result
        assert "screens" in result["ux_design"]


class TestTechAdvisor:
    def test_node_returns_parsed_json(self):
        with patch("src.workflow.multi_agent.agents.tech_advisor.LLMClient") as mock_cls:
            # tech_advisor uses use_json_mode=False, so mock chat() not chat_with_json_mode()
            mock_cls.return_value.chat.return_value = TECH_JSON
            state = {"product_idea": "AI助手", "requirement_analysis": {}, "feature_plan": {}, "ux_design": {}}
            result = tech_advisor_node(state)
        assert "tech_advice" in result
        assert "tech_stack" in result["tech_advice"]


class TestReviewer:
    def test_node_sets_score_and_feedback(self):
        reviewer_response = json.dumps({
            "overall_score": 82,
            "scores": {"requirements_analyst": 85, "feature_planner": 80, "ux_designer": 88, "tech_advisor": 75},
            "feedback": {"requirements_analyst": None, "feature_planner": "需要改进", "ux_designer": None, "tech_advisor": None},
            "consistency_issues": [],
            "completeness_gaps": [{"missing_element": "安全", "suggested_addition": "补充"}],
            "summary": "整体OK",
        })
        with patch("src.workflow.multi_agent.agents.reviewer.LLMClient") as mock_cls:
            mock_cls.return_value.chat_with_json_mode.return_value = reviewer_response
            state = {
                "product_idea": "AI助手",
                "requirement_analysis": {},
                "feature_plan": {},
                "ux_design": {},
                "tech_advice": {},
            }
            result = reviewer_node(state)

        assert result["reviewer_score"] == 82
        assert result["reviewer_scores"]["requirements_analyst"] == 85
        assert "feature_planner" in result["agents_to_revise"]

    def test_node_no_feedback_no_revision(self):
        reviewer_response = json.dumps({
            "overall_score": 92,
            "scores": {"requirements_analyst": 92, "feature_planner": 90, "ux_designer": 93, "tech_advisor": 91},
            "feedback": {"requirements_analyst": None, "feature_planner": None, "ux_designer": None, "tech_advisor": None},
            "consistency_issues": [],
            "completeness_gaps": [],
            "summary": "优秀",
        })
        with patch("src.workflow.multi_agent.agents.reviewer.LLMClient") as mock_cls:
            mock_cls.return_value.chat_with_json_mode.return_value = reviewer_response
            state = {"product_idea": "AI助手", "requirement_analysis": {}, "feature_plan": {}, "ux_design": {}, "tech_advice": {}}
            result = reviewer_node(state)

        assert result["reviewer_score"] == 92
        assert "agents_to_revise" not in result


class TestImageAnalyst:
    def test_skips_without_images(self):
        state = {"image_paths": []}
        result = image_analyst_node(state)
        assert result["current_stage"] == "image_analysis_skipped"


class TestRevisionRouter:
    def test_router_flags_agents(self):
        state = {
            "reviewer_feedback": {"feature_planner": "需要改进", "tech_advisor": None},
            "reflection_round": 0,
            "reflection_max_rounds": 2,
            "reviewer_score": 65,
            "reflection_history": [],
        }
        result = revision_router_node(state)
        assert result["reflection_round"] == 1
        assert "feature_planner" in result["agents_to_revise"]

    def test_router_no_revision_needed(self):
        state = {
            "reviewer_feedback": {"requirements_analyst": None, "feature_planner": None},
            "reflection_round": 0,
            "reflection_max_rounds": 2,
            "reviewer_score": 90,
            "reflection_history": [],
        }
        result = revision_router_node(state)
        assert result["agents_to_revise"] == []
