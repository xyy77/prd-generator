import json
from unittest.mock import patch

from src.workflow.multi_agent.graph import build_multi_agent_graph, run_multi_agent_workflow


class TestMultiAgentGraph:
    def test_graph_compiles(self):
        graph = build_multi_agent_graph()
        assert graph is not None

    def test_graph_invoke_all_agents(self):
        graph = build_multi_agent_graph()

        canned = json.dumps({"images": [], "aggregated_insights": {}})
        agent_output = json.dumps({"test_key": "test_value"})
        reviewer_output = json.dumps({
            "overall_score": 85,
            "scores": {"requirements_analyst": 85, "feature_planner": 82, "ux_designer": 88, "tech_advisor": 84},
            "feedback": {"requirements_analyst": None, "feature_planner": None, "ux_designer": None, "tech_advisor": None},
            "consistency_issues": [],
            "completeness_gaps": [],
            "summary": "Good",
        })

        with patch("src.workflow.multi_agent.graph.image_analyst_node") as mock_img, \
             patch("src.workflow.multi_agent.graph.run_parallel_agents") as mock_agents, \
             patch("src.workflow.multi_agent.graph.reviewer_node") as mock_review, \
             patch("src.workflow.multi_agent.graph.document_synthesis_node") as mock_doc:
            mock_img.return_value = {"image_analysis": json.loads(canned), "current_stage": "image_analysis_complete"}
            mock_agents.return_value = {
                "requirement_analysis": {}, "feature_plan": {}, "ux_design": {}, "tech_advice": {},
                "current_stage": "parallel_agents_complete",
            }
            mock_review.return_value = {
                "reviewer_score": 85,
                "reviewer_scores": {"requirements_analyst": 85},
                "reviewer_feedback": {},
                "reviewer_summary": "Good",
                "current_stage": "reviewer",
            }
            mock_doc.return_value = {"final_prd_json": {"version_record": {"product_name": "Test"}}}

            result = graph.invoke({
                "product_idea": "AI助手",
                "retrieved_context": "参考",
            })

        assert "final_prd_json" in result
        assert result.get("reviewer_score") == 85
        assert mock_agents.called

    def test_reflection_loop_triggers_below_threshold(self):
        graph = build_multi_agent_graph()

        with patch("src.workflow.multi_agent.graph.image_analyst_node") as mock_img, \
             patch("src.workflow.multi_agent.graph.run_parallel_agents") as mock_agents, \
             patch("src.workflow.multi_agent.graph.reviewer_node") as mock_review, \
             patch("src.workflow.multi_agent.graph.revision_router_node") as mock_router, \
             patch("src.workflow.multi_agent.graph.document_synthesis_node") as mock_doc:
            mock_img.return_value = {"current_stage": "image_analysis_skipped"}
            mock_agents.return_value = {
                "requirement_analysis": {}, "feature_plan": {}, "ux_design": {}, "tech_advice": {},
                "current_stage": "parallel_agents_complete",
            }
            # First review: low score
            mock_review.side_effect = [
                {
                    "reviewer_score": 65,
                    "reviewer_scores": {"requirements_analyst": 65},
                    "reviewer_feedback": {"feature_planner": "Needs improvement"},
                    "reviewer_summary": "Needs work",
                    "current_stage": "reviewer",
                },
                # Second review: pass
                {
                    "reviewer_score": 88,
                    "reviewer_scores": {"requirements_analyst": 88},
                    "reviewer_feedback": {},
                    "reviewer_summary": "Passed",
                    "current_stage": "reviewer",
                },
            ]
            mock_router.return_value = {
                "reflection_round": 1,
                "reflection_history": [{"round": 1, "score": 65}],
                "agents_to_revise": ["feature_planner"],
                "current_stage": "revision_router_revise",
            }
            mock_doc.return_value = {"final_prd_json": {"version_record": {"product_name": "Test"}}}

            result = graph.invoke({
                "product_idea": "AI助手",
                "retrieved_context": "参考",
                "reflection_round": 0,
                "reflection_max_rounds": 2,
                "reviewer_score_threshold": 80,
                "reflection_history": [],
            })

        assert "final_prd_json" in result
        # Reflection should have happened
        assert mock_router.called

    def test_reflection_stops_at_max_rounds(self):
        graph = build_multi_agent_graph()

        with patch("src.workflow.multi_agent.graph.image_analyst_node") as mock_img, \
             patch("src.workflow.multi_agent.graph.run_parallel_agents") as mock_agents, \
             patch("src.workflow.multi_agent.graph.reviewer_node") as mock_review, \
             patch("src.workflow.multi_agent.graph.document_synthesis_node") as mock_doc:
            mock_img.return_value = {"current_stage": "image_analysis_skipped"}
            mock_agents.return_value = {
                "requirement_analysis": {}, "feature_plan": {}, "ux_design": {}, "tech_advice": {},
            }
            mock_review.return_value = {
                "reviewer_score": 65,
                "reviewer_scores": {},
                "reviewer_feedback": {"feature_planner": "Needs improvement"},
                "reviewer_summary": "Needs work",
                "current_stage": "reviewer",
            }
            mock_doc.return_value = {"final_prd_json": {"version_record": {"product_name": "Test"}}}

            result = graph.invoke({
                "product_idea": "AI助手",
                "reflection_round": 2,  # Already at max
                "reflection_max_rounds": 2,
                "reviewer_score_threshold": 80,
            })

        assert "final_prd_json" in result


class TestRunMultiAgentWorkflow:
    def test_run_workflow_passes_params(self):
        with patch("src.workflow.multi_agent.graph.build_multi_agent_graph") as mock_build:
            mock_graph = mock_build.return_value
            mock_graph.invoke.return_value = {
                "product_idea": "测试产品",
                "supplementary_info": "补充",
                "image_paths": ["test.png"],
                "selected_model": "deepseek-v4-pro",
                "temperature": 0.5,
                "final_prd_json": {"version_record": {"product_name": "Test"}},
                "reviewer_score": 90,
            }

            result = run_multi_agent_workflow(
                product_idea="测试产品",
                supplementary_info="补充",
                image_paths=["test.png"],
                selected_model="deepseek-v4-pro",
                temperature=0.5,
            )

        assert result["product_idea"] == "测试产品"
        assert result["selected_model"] == "deepseek-v4-pro"
