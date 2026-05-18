from unittest.mock import patch

from src.workflow.nodes.requirement_analysis import requirement_analysis_node
from src.workflow.nodes.architecture_design import architecture_design_node
from src.workflow.nodes.process_flow import process_flow_node
from src.workflow.nodes.document_finalization import document_finalization_node
from src.workflow.nodes.prd_revision import prd_revision_node
from src.workflow.state import WorkflowState


REQUIREMENT_JSON = '{"product_name": "测试产品", "product_overview": "概述", "background": {"market_context": "市场", "user_pain_points": ["痛点1"], "opportunity": "机会"}, "goals": {"core_objective": "目标", "key_results": ["KR1"]}, "user_personas": [{"role": "用户", "scenario": "场景", "core_needs": ["需求"]}], "scope": {"in_scope": ["功能1"], "out_of_scope": ["不做1"]}}'

ARCHITECTURE_JSON = '{"functional_modules": [{"module_name": "模块", "description": "描述", "features": ["F1"], "priority": "P0", "dependencies": []}], "tech_stack": {"frontend": "React", "backend": "Python", "database": "PG", "ai_ml": "LLM", "infrastructure": "Cloud"}, "data_flow": {"key_entities": ["E1"], "data_relationships": "关系", "external_integrations": ["API1"]}, "system_components": [{"name": "组件", "responsibility": "职责", "interfaces": ["I1"]}]}'

FLOW_JSON = '{"user_flows": [{"flow_name": "核心流程", "description": "描述", "steps": ["S1"], "entry_point": "入口", "exit_point": "出口"}], "mermaid_diagrams": [{"diagram_name": "流程图", "diagram_type": "graph TD", "mermaid_code": "graph TD\\n    A-->B"}], "state_machine": {"states": ["S1"], "transitions": [{"from": "S1", "to": "S2", "trigger": "触发"}]}, "edge_cases": [{"case": "异常", "handling": "处理", "user_experience": "体验"}], "error_handling": [{"error_type": "网络错误", "strategy": "重试"}]}'


def make_state(**overrides) -> WorkflowState:
    base: WorkflowState = {
        "product_idea": "AI助手",
        "supplementary_info": "",
        "retrieved_context": "参考案例",
    }
    base.update(overrides)
    return base


class TestRequirementAnalysisNode:
    def test_node_returns_parsed_json(self):
        with patch("src.workflow.nodes.requirement_analysis.LLMClient") as mock_llm:
            mock_llm.return_value.chat_with_json_mode.return_value = REQUIREMENT_JSON
            result = requirement_analysis_node(make_state())
            assert "requirement_analysis" in result
            assert result["requirement_analysis"]["product_name"] == "测试产品"


class TestArchitectureDesignNode:
    def test_node_returns_parsed_json(self):
        with patch("src.workflow.nodes.architecture_design.LLMClient") as mock_llm:
            mock_llm.return_value.chat_with_json_mode.return_value = ARCHITECTURE_JSON
            state = make_state(requirement_analysis={"product_name": "测试"})
            result = architecture_design_node(state)
            assert "architecture_design" in result
            assert len(result["architecture_design"]["functional_modules"]) == 1


class TestProcessFlowNode:
    def test_node_returns_mermaid(self):
        with patch("src.workflow.nodes.process_flow.LLMClient") as mock_llm:
            mock_llm.return_value.chat_with_json_mode.return_value = FLOW_JSON
            state = make_state(
                requirement_analysis={"product_name": "测试"},
                architecture_design={"functional_modules": []},
            )
            result = process_flow_node(state)
            assert "process_flow" in result
            assert len(result["process_flow"]["mermaid_diagrams"]) == 1


class TestDocumentFinalizationNode:
    def test_node_returns_final_prd(self):
        with patch("src.workflow.nodes.document_finalization.LLMClient") as mock_llm:
            mock_llm.return_value.chat_with_json_mode.return_value = '{"version_record": {"document_version": "1.0", "product_name": "最终产品"}, "background_and_goals": {}, "user_personas": [], "functional_requirements": [], "non_functional_requirements": {}, "tech_architecture": {}, "analytics_and_iteration": {}, "risks_and_mitigation": [], "appendix": {}}'
            state = make_state(
                requirement_analysis={"product_name": "测试"},
                architecture_design={"functional_modules": []},
                process_flow={"mermaid_diagrams": []},
            )
            result = document_finalization_node(state)
            assert "final_prd_json" in result
            assert result["final_prd_json"]["version_record"]["product_name"] == "最终产品"


class TestPrdRevisionNode:
    def test_revision_node_updates_prd(self):
        revision_json = '{"version_record": {"document_version": "1.1", "product_name": "修订产品"}, "background_and_goals": {}, "user_personas": [], "functional_requirements": [], "non_functional_requirements": {}, "tech_architecture": {}, "analytics_and_iteration": {}, "risks_and_mitigation": [], "appendix": {}}'
        with patch("src.workflow.nodes.prd_revision.LLMClient") as mock_llm:
            mock_llm.return_value.chat_with_json_mode.return_value = revision_json
            state = make_state(
                final_prd_json={"version_record": {"product_name": "初始产品"}},
                user_feedback="增加社交分享功能",
            )
            result = prd_revision_node(state)
            assert "final_prd_json" in result
            assert result["final_prd_json"]["version_record"]["product_name"] == "修订产品"
            assert result.get("revision_count") == 1
            assert len(result.get("revision_history", [])) == 1

    def test_revision_node_no_feedback(self):
        state = make_state(final_prd_json={"test": True})
        result = prd_revision_node(state)
        assert "未提供修改意见" in result.get("error_message", "")
