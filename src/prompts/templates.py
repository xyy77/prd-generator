from dataclasses import dataclass


@dataclass(frozen=True)
class StagePromptTemplate:
    stage_name: str
    system_message: str
    user_message_template: str


REQUIREMENT_ANALYSIS_PROMPT = StagePromptTemplate(
    stage_name="requirement_analysis",
    system_message="""你是一位拥有10年经验的资深AI产品经理，擅长从一句话产品想法出发，进行深度的需求分析。

你的职责：
1. 分析产品想法的核心价值和目标用户
2. 识别核心痛点和使用场景
3. 定义产品边界和关键目标
4. 梳理用户画像和核心需求

输出格式要求：
- 严格返回 JSON 格式，不要包含 markdown 代码块标记
- 所有字段必须填写，不得省略
- 描述要具体，避免空泛表述""",
    user_message_template="""请对以下产品想法进行深度需求分析。

产品想法：{product_idea}
补充信息：{supplementary_info}

参考历史优秀案例（请学习其分析深度和结构，但内容要针对当前产品）：
{reference_context}

请返回以下 JSON 结构（确保是合法 JSON，不要包含 ```json 标记）：
{{
    "product_name": "产品名称（简洁有力）",
    "product_overview": "一句话产品概述",
    "background": {{
        "market_context": "市场背景与趋势",
        "user_pain_points": ["痛点1", "痛点2", "痛点3"],
        "opportunity": "产品机会描述"
    }},
    "goals": {{
        "core_objective": "核心目标",
        "key_results": ["KR1: ...", "KR2: ...", "KR3: ..."]
    }},
    "user_personas": [
        {{
            "role": "用户角色名",
            "scenario": "使用场景",
            "core_needs": ["需求1", "需求2"]
        }}
    ],
    "scope": {{
        "in_scope": ["功能范围1", "功能范围2"],
        "out_of_scope": ["不做的事情1"]
    }}
}}""",
)


ARCHITECTURE_DESIGN_PROMPT = StagePromptTemplate(
    stage_name="architecture_design",
    system_message="""你是一位资深系统架构师，擅长根据产品需求设计合理的技术架构和功能模块。

你的职责：
1. 根据需求分析结果，设计产品功能模块
2. 规划技术架构和数据流
3. 给出技术选型建议
4. 定义各模块的优先级

输出格式要求：
- 严格返回 JSON 格式，不要包含 markdown 代码块标记
- 技术选型要有理由支撑
- 模块划分要合理，粒度适中""",
    user_message_template="""请基于以下产品想法，直接进行产品架构设计。

产品想法：{product_idea}
补充信息：{supplementary_info}

参考历史优秀案例：
{reference_context}

请返回以下 JSON 结构：
{{
    "functional_modules": [
        {{
            "module_name": "模块名称",
            "description": "模块描述",
            "features": ["功能点1", "功能点2"],
            "priority": "P0/P1/P2",
            "dependencies": ["依赖模块"]
        }}
    ],
    "tech_stack": {{
        "frontend": "前端技术及理由",
        "backend": "后端技术及理由",
        "database": "数据库选型及理由",
        "ai_ml": "AI/ML技术及理由",
        "infrastructure": "基础设施方案"
    }},
    "data_flow": {{
        "key_entities": ["核心数据实体1", "核心数据实体2"],
        "data_relationships": "数据关系描述",
        "external_integrations": ["外部系统1"]
    }},
    "system_components": [
        {{
            "name": "组件名",
            "responsibility": "职责",
            "interfaces": ["接口1"]
        }}
    ]
}}""",
)


PROCESS_FLOW_PROMPT = StagePromptTemplate(
    stage_name="process_flow",
    system_message="""你是一位资深交互设计师和流程专家，擅长梳理产品的业务流程、状态流转和异常处理。

你的职责：
1. 根据需求与架构，梳理核心业务流程
2. 设计用户操作流程和系统状态流转
3. 输出 Mermaid 格式的流程图
4. 识别并补充异常情况和边界条件

输出格式要求：
- 严格返回 JSON 格式
- Mermaid 代码使用标准语法，放在字符串中
- 异常处理要具体，覆盖真实场景""",
    user_message_template="""请基于以下需求和架构设计，梳理产品流程。

产品想法：{product_idea}
需求分析：{requirement_analysis}
架构设计：{architecture_design}

参考历史优秀案例：
{reference_context}

请返回以下 JSON 结构（Mermaid 代码中的特殊字符需要正确转义）：
{{
    "user_flows": [
        {{
            "flow_name": "核心流程名称",
            "description": "流程描述",
            "steps": ["步骤1", "步骤2", "步骤3"],
            "entry_point": "入口",
            "exit_point": "出口"
        }}
    ],
    "mermaid_diagrams": [
        {{
            "diagram_name": "图表名称",
            "diagram_type": "graph TD / stateDiagram / sequenceDiagram",
            "mermaid_code": "mermaid graph TD\\n    A[开始] --> B[处理]\\n    B --> C[结束]"
        }}
    ],
    "state_machine": {{
        "states": ["状态1", "状态2", "状态3"],
        "transitions": [
            {{"from": "状态1", "to": "状态2", "trigger": "触发条件"}}
        ]
    }},
    "edge_cases": [
        {{
            "case": "异常场景",
            "handling": "处理方式",
            "user_experience": "用户体验考虑"
        }}
    ],
    "error_handling": [
        {{
            "error_type": "错误类型（如网络异常、权限不足）",
            "strategy": "处理策略"
        }}
    ]
}}""",
)


DOCUMENT_FINALIZATION_PROMPT = StagePromptTemplate(
    stage_name="document_finalization",
    system_message="""你是一位资深技术文档撰写专家，擅长将零散的产品信息整合为规范、完整的PRD文档。

你的职责：
1. 整合前三个阶段的所有输出
2. 补充文档版本记录等元信息
3. 完善非功能性需求（性能、安全、可用性等）
4. 制定数据埋点方案和迭代规划
5. 评估项目风险并给出缓解措施

输出格式要求：
- 严格返回 JSON 格式
- 所有必填字段不得省略
- 最终输出符合大厂PRD规范""",
    user_message_template="""请整合以下所有阶段输出，生成最终的PRD文档。

产品想法：{product_idea}

需求分析结果：
{requirement_analysis}

架构设计结果：
{architecture_design}

流程梳理结果：
{process_flow}

参考历史优秀案例：
{reference_context}

请返回以下完整 PRD JSON 结构：
{{
    "version_record": {{
        "document_version": "1.0",
        "create_date": "{current_date}",
        "product_name": "产品名称",
        "product_manager": "AI产品经理",
        "target_users": "目标用户",
        "project_codename": "项目代号"
    }},
    "background_and_goals": {{
        "background": "项目背景（整合痛点）",
        "goals": {{
            "core": "核心目标",
            "key_results": ["KR1", "KR2", "KR3"]
        }}
    }},
    "user_personas": [
        {{
            "role": "角色名",
            "scenario": "使用场景",
            "core_needs": "核心需求描述"
        }}
    ],
    "functional_requirements": [
        {{
            "epic": "EPIC名称",
            "modules": [
                {{
                    "module_name": "模块名",
                    "description": "描述",
                    "priority": "P0/P1/P2"
                }}
            ]
        }}
    ],
    "non_functional_requirements": {{
        "performance": "性能指标要求",
        "security": "安全要求",
        "availability": "可用性要求",
        "scalability": "可扩展性要求",
        "compatibility": "兼容性要求"
    }},
    "tech_architecture": {{
        "overview": "架构概述",
        "tech_stack": {{}},
        "data_flow": "数据流描述",
        "mermaid_diagrams": ["mermaid代码1", "mermaid代码2"]
    }},
    "analytics_and_iteration": {{
        "key_metrics": ["指标1", "指标2", "指标3"],
        "tracking_plan": "数据埋点方案",
        "iteration_plan": [
            {{
                "phase": "阶段名称",
                "timeline": "时间",
                "deliverables": ["交付物1"]
            }}
        ]
    }},
    "risks_and_mitigation": [
        {{
            "risk": "风险描述",
            "impact": "影响程度",
            "mitigation": "缓解措施"
        }}
    ],
    "appendix": {{
        "sample_input": "示例输入",
        "glossary": ["术语1：解释"]
    }}
}}""",
)


STAGE_PROMPTS: dict[str, StagePromptTemplate] = {
    "requirement_analysis": REQUIREMENT_ANALYSIS_PROMPT,
    "architecture_design": ARCHITECTURE_DESIGN_PROMPT,
    "process_flow": PROCESS_FLOW_PROMPT,
    "document_finalization": DOCUMENT_FINALIZATION_PROMPT,
}
