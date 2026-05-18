"""End-to-end integration tests.

These tests require DEEPSEEK_API_KEY to be set in .env or environment.
Run with: pytest tests/test_integration/ -v -m integration
Skip with:  pytest -m "not integration"
"""

import pytest

from src.config import settings
from src.output.json_to_markdown import convert_to_prd_markdown
from src.output.validator import parse_and_validate
from src.rag.embedder import EmbeddingService
from src.rag.retriever import Retriever
from src.rag.store import ChromaStore
from src.workflow.graph import run_workflow


def requires_api_key():
    if not settings.deepseek_api_key or settings.deepseek_api_key == "sk-placeholder":
        pytest.skip("DEEPSEEK_API_KEY not configured")


class TestRagPipelineIntegration:
    @pytest.fixture(scope="class")
    def retriever(self):
        embedder = EmbeddingService()
        store = ChromaStore()
        return Retriever(store, embedder)

    def test_retrieval_on_real_kb(self, retriever):
        if retriever.store.count() == 0:
            pytest.skip("Knowledge base is empty, seed it first")
        results = retriever.search("AI助手产品", top_k=3)
        if results:
            assert results[0].score > 0


class TestWorkflowIntegration:
    def test_full_workflow_real_llm(self):
        requires_api_key()

        result = run_workflow(
            product_idea="一个帮助用户管理个人财务的AI助手，可以自动记账、分析消费、生成预算建议",
            supplementary_info="目标用户是一二线城市白领",
            retrieved_context="无参考案例",
        )

        assert "final_prd_json" in result
        final = result.get("final_prd_json", {})
        assert "version_record" in final
        assert "background_and_goals" in final

    def test_output_quality(self):
        requires_api_key()

        result = run_workflow(
            product_idea="面向中小企业的智能客服系统",
            retrieved_context="无参考案例",
        )

        final = result.get("final_prd_json", {})
        markdown = convert_to_prd_markdown(final)

        assert len(markdown) > 500
        sections = [
            "产品需求文档",
            "项目背景与目标",
            "用户角色",
            "功能需求",
            "非功能性需求",
            "技术架构",
            "迭代规划",
            "风险",
        ]
        for section in sections:
            assert section in markdown, f"Missing section: {section}"
