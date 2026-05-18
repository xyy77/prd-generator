from unittest.mock import patch

from src.workflow.graph import build_workflow_graph
from src.workflow.state import WorkflowState


class TestWorkflowGraph:
    def test_graph_compiles(self):
        graph = build_workflow_graph()
        assert graph is not None

    def test_graph_invoke(self):
        import json

        graph = build_workflow_graph()

        canned_responses = iter([
            # Parallel analysis: requirement_analysis then architecture_design
            json.dumps({"product_name": "测试", "product_overview": "概述", "background": {}, "goals": {}, "user_personas": [], "scope": {}}),
            json.dumps({"functional_modules": [], "tech_stack": {}, "data_flow": {}, "system_components": []}),
            # process_flow
            json.dumps({"user_flows": [], "mermaid_diagrams": [], "state_machine": {}, "edge_cases": [], "error_handling": []}),
            # document_finalization
            json.dumps({"version_record": {"product_name": "测试"}, "background_and_goals": {}, "user_personas": [], "functional_requirements": [], "non_functional_requirements": {}, "tech_architecture": {}, "analytics_and_iteration": {}, "risks_and_mitigation": [], "appendix": {}}),
        ])

        def mock_chat(*args, **kwargs):
            return next(canned_responses)

        with patch("src.workflow.nodes.parallel_analysis.LLMClient") as mock_parallel, \
             patch("src.workflow.nodes.process_flow.LLMClient") as mock_flow, \
             patch("src.workflow.nodes.document_finalization.LLMClient") as mock_doc:
            mock_parallel.return_value.chat_with_json_mode.side_effect = mock_chat
            mock_flow.return_value.chat_with_json_mode.side_effect = mock_chat
            mock_doc.return_value.chat_with_json_mode.side_effect = mock_chat

            result = graph.invoke({
                "product_idea": "AI助手",
                "retrieved_context": "参考案例",
            })

        assert "requirement_analysis" in result
        assert "architecture_design" in result
        assert "process_flow" in result
        assert "final_prd_json" in result
