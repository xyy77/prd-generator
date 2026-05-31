import json
from unittest.mock import patch

import src.workflow.multi_agent.graph as graph_module

AGENT_KEYS = ["requirements_analyst", "feature_planner", "ux_designer", "tech_advisor"]

# The graph module caches the compiled graph; we need to rebuild inside patches.
_rebuild_graph = graph_module.build_multi_agent_graph


class TestMultiAgentGraph:
    def test_graph_compiles(self):
        graph = _rebuild_graph()
        assert graph is not None

    def test_graph_invoke_all_agents(self):
        # Build graph INSIDE patch block so mock references are captured
        with patch.object(graph_module, "_image_analyst_wrapper") as mock_img, \
             patch.object(graph_module, "_planner_wrapper") as mock_planner, \
             patch.object(graph_module, "_supervisor_wrapper") as mock_sup, \
             patch.object(graph_module, "_make_agent_wrapper") as mock_make, \
             patch.object(graph_module, "_reviewer_wrapper") as mock_review, \
             patch.object(graph_module, "_revision_router_wrapper") as mock_router, \
             patch.object(graph_module, "document_synthesis_node") as mock_doc:

            mock_img.return_value = {"image_analysis": {}, "current_stage": "image_analysis"}
            mock_planner.return_value = {"planner_output": {"complexity": "medium", "product_type": "web"}, "current_stage": "planner"}

            # Supervisor: return all agents first, then empty to proceed to reviewer
            mock_sup.side_effect = [
                {"agents_to_call": AGENT_KEYS, "execution_order": AGENT_KEYS, "current_stage": "supervisor"},
                {"agents_to_call": AGENT_KEYS[1:], "execution_order": AGENT_KEYS, "completed_agents": [AGENT_KEYS[0]], "current_stage": "supervisor"},
                {"agents_to_call": AGENT_KEYS[2:], "execution_order": AGENT_KEYS, "completed_agents": AGENT_KEYS[:2], "current_stage": "supervisor"},
                {"agents_to_call": AGENT_KEYS[3:], "execution_order": AGENT_KEYS, "completed_agents": AGENT_KEYS[:3], "current_stage": "supervisor"},
                {"agents_to_call": [], "execution_order": AGENT_KEYS, "completed_agents": AGENT_KEYS, "current_stage": "supervisor"},
            ]
            # Agent wrapper factory
            mock_make.side_effect = lambda name: (lambda state: {
                "completed_agents": list(state.get("completed_agents", [])) + [name],
                "current_stage": name,
            })
            mock_review.return_value = {
                "reviewer_score": 85,
                "reviewer_scores": dict.fromkeys(AGENT_KEYS, 85),
                "reviewer_feedback": {},
                "reviewer_summary": "Good",
                "current_stage": "reviewer",
            }
            mock_router.return_value = {
                "reflection_round": 0, "agents_to_revise": [], "current_stage": "revision_router",
            }
            mock_doc.return_value = {"final_prd_json": {"version_record": {"product_name": "Test"}}}

            graph = _rebuild_graph()
            result = graph.invoke({
                "product_idea": "AI助手",
                "retrieved_context": "参考",
            })

        assert "final_prd_json" in result
        assert result.get("reviewer_score") == 85

    def test_reflection_loop_triggers_below_threshold(self):
        with patch.object(graph_module, "_image_analyst_wrapper") as mock_img, \
             patch.object(graph_module, "_planner_wrapper") as mock_planner, \
             patch.object(graph_module, "_supervisor_wrapper") as mock_sup, \
             patch.object(graph_module, "_make_agent_wrapper") as mock_make, \
             patch.object(graph_module, "_reviewer_wrapper") as mock_review, \
             patch.object(graph_module, "_revision_router_wrapper") as mock_router, \
             patch.object(graph_module, "document_synthesis_node") as mock_doc:

            mock_img.return_value = {"image_analysis": {}, "current_stage": "image_analysis"}
            mock_planner.return_value = {"planner_output": {"complexity": "medium"}, "current_stage": "planner"}
            mock_sup.return_value = {
                "agents_to_call": [], "execution_order": AGENT_KEYS,
                "completed_agents": AGENT_KEYS, "current_stage": "supervisor",
            }
            mock_make.side_effect = lambda name: (lambda state: {
                "completed_agents": list(state.get("completed_agents", [])) + [name],
                "current_stage": name,
            })
            # First review low score → second review pass
            mock_review.side_effect = [
                {"reviewer_score": 65, "reviewer_feedback": {"feature_planner": "bad"}, "current_stage": "reviewer"},
                {"reviewer_score": 88, "reviewer_feedback": {}, "current_stage": "reviewer"},
            ]
            # First call: revise; second call: finalize
            mock_router.side_effect = [
                {"reflection_round": 1, "agents_to_revise": ["feature_planner"], "current_stage": "revision_router_revise"},
                {"reflection_round": 1, "agents_to_revise": [], "current_stage": "revision_router_finalize"},
            ]
            mock_doc.return_value = {"final_prd_json": {"version_record": {"product_name": "Test"}}}

            graph = _rebuild_graph()
            result = graph.invoke({
                "product_idea": "AI助手",
                "reflection_round": 0,
                "reflection_max_rounds": 2,
                "reviewer_score_threshold": 80,
                "reflection_history": [],
            })

        assert "final_prd_json" in result
        assert mock_router.called

    def test_reflection_stops_at_max_rounds(self):
        with patch.object(graph_module, "_image_analyst_wrapper") as mock_img, \
             patch.object(graph_module, "_planner_wrapper") as mock_planner, \
             patch.object(graph_module, "_supervisor_wrapper") as mock_sup, \
             patch.object(graph_module, "_make_agent_wrapper") as mock_make, \
             patch.object(graph_module, "_reviewer_wrapper") as mock_review, \
             patch.object(graph_module, "_revision_router_wrapper") as mock_router, \
             patch.object(graph_module, "document_synthesis_node") as mock_doc:

            mock_img.return_value = {"current_stage": "image_analysis"}
            mock_planner.return_value = {"planner_output": {"complexity": "medium"}, "current_stage": "planner"}
            mock_sup.return_value = {
                "agents_to_call": [], "execution_order": AGENT_KEYS,
                "completed_agents": AGENT_KEYS, "current_stage": "supervisor",
            }
            mock_make.side_effect = lambda name: (lambda state: {
                "completed_agents": list(state.get("completed_agents", [])) + [name],
                "current_stage": name,
            })
            mock_review.return_value = {
                "reviewer_score": 65, "reviewer_feedback": {}, "current_stage": "reviewer",
            }
            # At max rounds, router should not revise
            mock_router.return_value = {
                "reflection_round": 2, "agents_to_revise": [], "current_stage": "revision_router_finalize",
            }
            mock_doc.return_value = {"final_prd_json": {"version_record": {"product_name": "Test"}}}

            graph = _rebuild_graph()
            result = graph.invoke({
                "product_idea": "AI助手",
                "reflection_round": 2,
                "reflection_max_rounds": 2,
                "reviewer_score_threshold": 80,
                "reflection_history": [{"round": 1, "score": 65}],
            })

        assert "final_prd_json" in result


class TestRunMultiAgentWorkflow:
    def test_run_workflow_passes_params(self):
        with patch.object(graph_module, "build_multi_agent_graph") as mock_build:
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

            result = graph_module.run_multi_agent_workflow(
                product_idea="测试产品",
                supplementary_info="补充",
                image_paths=["test.png"],
                selected_model="deepseek-v4-pro",
                temperature=0.5,
            )

        assert result["product_idea"] == "测试产品"
        assert result["selected_model"] == "deepseek-v4-pro"
