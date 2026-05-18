from src.workflow.state import WorkflowState, STAGE_ORDER


class TestWorkflowState:
    def test_minimal_state(self):
        state: WorkflowState = {
            "product_idea": "AI助手",
        }
        assert state["product_idea"] == "AI助手"

    def test_full_state(self):
        state: WorkflowState = {
            "product_idea": "AI助手",
            "supplementary_info": "额外信息",
            "retrieved_context": "参考内容",
            "requirement_analysis": {"test": True},
            "architecture_design": {"test": True},
            "process_flow": {"test": True},
            "final_prd_json": {"test": True},
            "current_stage": "document_finalization",
            "error_message": "",
            "final_prd_markdown": "",
        }
        assert state["final_prd_json"] == {"test": True}


class TestStageOrder:
    def test_four_stages(self):
        assert len(STAGE_ORDER) == 4
        assert STAGE_ORDER[0] == "requirement_analysis"
        assert STAGE_ORDER[-1] == "document_finalization"
