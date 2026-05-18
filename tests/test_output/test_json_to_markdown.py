from src.output.json_to_markdown import convert_to_prd_markdown


SAMPLE_PRD_JSON = {
    "version_record": {
        "document_version": "1.0",
        "create_date": "2026-05-19",
        "product_name": "AI写作助手",
        "product_manager": "AI产品经理",
        "target_users": "职场人士",
        "project_codename": "AI-WRITE",
    },
    "background_and_goals": {
        "background": "写作助手市场背景",
        "goals": {
            "core": "提高写作效率",
            "key_results": ["KR1: 日活1000", "KR2: 满意度4.0"],
        },
    },
    "user_personas": [
        {"role": "职场人士", "scenario": "写周报", "core_needs": "快速生成模板"},
    ],
    "functional_requirements": [
        {
            "epic": "EPIC 1：写作引擎",
            "modules": [
                {"module_name": "智能生成", "description": "AI生成内容", "priority": "P0"},
            ],
        },
    ],
    "non_functional_requirements": {
        "performance": "响应时间<3秒",
        "security": "数据加密",
    },
    "tech_architecture": {
        "overview": "基于大模型的SaaS平台",
        "tech_stack": {"frontend": "React", "backend": "Python"},
        "mermaid_diagrams": ["graph TD\n    A-->B"],
    },
    "analytics_and_iteration": {
        "key_metrics": ["DAU", "留存率"],
        "tracking_plan": "全埋点方案",
        "iteration_plan": [
            {"phase": "MVP", "timeline": "Week 1-3", "deliverables": ["核心功能"]},
        ],
    },
    "risks_and_mitigation": [
        {"risk": "LLM幻觉", "impact": "高", "mitigation": "RAG注入"},
    ],
    "appendix": {
        "sample_input": "让AI帮你写周报",
        "glossary": ["PRD: 产品需求文档"],
    },
}


class TestConvertToPrdMarkdown:
    def test_returns_string(self):
        result = convert_to_prd_markdown(SAMPLE_PRD_JSON)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_key_sections(self):
        result = convert_to_prd_markdown(SAMPLE_PRD_JSON)
        assert "# 产品需求文档（PRD）" in result
        assert "AI写作助手" in result
        assert "## 一、项目背景与目标" in result
        assert "## 二、用户角色与核心场景" in result
        assert "## 三、功能需求" in result
        assert "## 四、非功能性需求" in result
        assert "## 五、技术架构" in result
        assert "## 六、数据埋点与迭代规划" in result
        assert "## 七、风险与缓解" in result
        assert "## 八、附录" in result

    def test_contains_mermaid(self):
        result = convert_to_prd_markdown(SAMPLE_PRD_JSON)
        assert "```mermaid" in result

    def test_handles_minimal_json(self):
        minimal = {
            "version_record": {},
            "background_and_goals": {},
            "user_personas": [],
            "functional_requirements": [],
            "non_functional_requirements": {},
            "tech_architecture": {},
            "analytics_and_iteration": {},
            "risks_and_mitigation": [],
            "appendix": {},
        }
        result = convert_to_prd_markdown(minimal)
        assert len(result) > 50
