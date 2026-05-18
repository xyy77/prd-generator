from unittest.mock import patch

from src.workflow.graph import build_workflow_graph, run_workflow, run_revision, run_single_stage
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

    def test_graph_with_model_and_temperature(self):
        import json

        graph = build_workflow_graph()

        canned = json.dumps({"version_record": {"product_name": "X"}, "background_and_goals": {}, "user_personas": [], "functional_requirements": [], "non_functional_requirements": {}, "tech_architecture": {}, "analytics_and_iteration": {}, "risks_and_mitigation": [], "appendix": {}})

        with patch("src.workflow.nodes.parallel_analysis.LLMClient") as mock_p, \
             patch("src.workflow.nodes.process_flow.LLMClient") as mock_f, \
             patch("src.workflow.nodes.document_finalization.LLMClient") as mock_d:
            mock_p.return_value.chat_with_json_mode.return_value = canned
            mock_f.return_value.chat_with_json_mode.return_value = canned
            mock_d.return_value.chat_with_json_mode.return_value = canned

            result = graph.invoke({
                "product_idea": "AI助手",
                "retrieved_context": "参考案例",
                "selected_model": "deepseek-v4-pro",
                "temperature": 0.5,
            })

        assert result.get("selected_model") == "deepseek-v4-pro"
        assert result.get("temperature") == 0.5


class TestRunWorkflow:
    def test_run_workflow_passes_model_and_temperature(self):
        import json

        canned = json.dumps({"user_flows": [], "mermaid_diagrams": [], "state_machine": {}, "edge_cases": [], "error_handling": []})

        with patch("src.workflow.nodes.parallel_analysis.LLMClient") as mock_p, \
             patch("src.workflow.nodes.process_flow.LLMClient") as mock_f, \
             patch("src.workflow.nodes.document_finalization.LLMClient") as mock_d:
            mock_p.return_value.chat_with_json_mode.return_value = canned
            mock_f.return_value.chat_with_json_mode.return_value = canned
            mock_d.return_value.chat_with_json_mode.return_value = canned

            result = run_workflow(
                product_idea="测试产品",
                supplementary_info="补充说明",
                retrieved_context="上下文",
                selected_model="deepseek-v4-pro",
                temperature=0.7,
            )

        assert result["product_idea"] == "测试产品"
        assert result["selected_model"] == "deepseek-v4-pro"


class TestRunRevision:
    def test_run_revision_route(self):
        import json

        revision_response = json.dumps({"version_record": {"product_name": "修订版"}, "background_and_goals": {}, "user_personas": [], "functional_requirements": [], "non_functional_requirements": {}, "tech_architecture": {}, "analytics_and_iteration": {}, "risks_and_mitigation": [], "appendix": {}})

        with patch("src.workflow.nodes.prd_revision.LLMClient") as mock_rev:
            mock_rev.return_value.chat_with_json_mode.return_value = revision_response

            state: WorkflowState = {
                "product_idea": "AI助手",
                "final_prd_json": {"version_record": {"product_name": "初始版"}},
                "retrieved_context": "参考",
                "user_feedback": "增加社交功能",
            }
            result = run_revision(state, "增加社交功能")

        assert result.get("revision_count") == 1
        assert len(result.get("revision_history", [])) == 1


class TestRunSingleStage:
    def test_run_single_stage(self):
        import json

        canned = json.dumps({"version_record": {"product_name": "修订文档"}, "background_and_goals": {}, "user_personas": [], "functional_requirements": [], "non_functional_requirements": {}, "tech_architecture": {}, "analytics_and_iteration": {}, "risks_and_mitigation": [], "appendix": {}})

        with patch("src.workflow.nodes.document_finalization.LLMClient") as mock_doc:
            mock_doc.return_value.chat_with_json_mode.return_value = canned

            state: WorkflowState = {
                "product_idea": "AI助手",
                "requirement_analysis": {"product_name": "测试"},
                "architecture_design": {"functional_modules": []},
                "process_flow": {"mermaid_diagrams": []},
                "retrieved_context": "参考",
            }
            result = run_single_stage("document_finalization", state)

        assert "final_prd_json" in result
