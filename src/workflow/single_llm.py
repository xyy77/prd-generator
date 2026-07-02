"""Single-LLM PRD generation: one call, no pipeline, no agents, no reflection.

This is the simplest baseline — a single comprehensive prompt that asks
the LLM to generate a complete PRD JSON in one shot. Used for ablation
comparison against the classic pipeline and multi-agent workflow.
"""

import time

from src.output.json_to_markdown import convert_to_prd_markdown
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

SINGLE_LLM_SYSTEM_PROMPT = """你是一位拥有10年经验的资深产品经理。你的任务是根据用户提供的产品想法，一次性生成完整的产品需求文档（PRD）。

要求：
1. 输出严格的 JSON 格式（不要用 Markdown 代码块包裹）
2. 所有章节必须包含，不能省略
3. 内容要具体、可执行，不要空洞的套话
4. 产品名称自拟，与产品想法相关即可"""

SINGLE_LLM_USER_TEMPLATE = """请根据以下产品想法，生成一份完整的产品需求文档（PRD）。

产品想法：{product_idea}
{supplementary_section}

请输出以下 JSON 结构（每个字段都必须填写，不能留空）：

{{
  "version_record": {{
    "product_name": "产品名称（自拟）",
    "document_version": "1.0",
    "create_date": "{date}",
    "product_manager": "PM",
    "target_users": "目标用户群体描述",
    "project_codename": "项目代号"
  }},
  "background_and_goals": {{
    "background": "项目背景，2-3句话说明为什么要做这个产品，解决什么问题",
    "goals": {{
      "core": "核心目标，一句话",
      "key_results": ["关键结果1（可量化）", "关键结果2（可量化）", "关键结果3（可量化）"]
    }}
  }},
  "user_personas": [
    {{
      "role": "用户角色名",
      "scenario": "使用场景描述",
      "core_needs": ["核心需求1", "核心需求2", "核心需求3"]
    }}
  ]（至少3个用户画像）,
  "functional_requirements": [
    {{
      "epic": "功能模块名称（如：用户管理、内容发布、数据分析等）",
      "modules": [
        {{
          "module_name": "子功能名",
          "description": "功能描述",
          "priority": "P0/P1/P2"
        }}
      ]
    }}
  ]（至少3个Epic，每个Epic至少2个module）,
  "non_functional_requirements": {{
    "performance": "性能要求（如首屏加载<2s，API响应<200ms）",
    "security": "安全要求（如数据加密、身份认证方式）",
    "availability": "可用性要求（如99.9% uptime）",
    "scalability": "可扩展性要求（如支持水平扩展）",
    "compatibility": "兼容性要求（如支持主流浏览器）"
  }},
  "tech_architecture": {{
    "overview": "技术架构概述，2-3句话",
    "tech_stack": {{
      "前端": "推荐前端技术栈",
      "后端": "推荐后端技术栈",
      "数据库": "推荐数据库",
      "缓存": "推荐缓存方案",
      "部署": "推荐部署方案"
    }},
    "mermaid_diagrams": [
      "graph TD\\n    A[用户] --> B[前端]\\n    B --> C[后端API]\\n    C --> D[数据库]"
    ]
  }},
  "analytics_and_iteration": {{
    "key_metrics": ["核心指标1（如DAU）", "核心指标2（如留存率）", "核心指标3（如转化率）"],
    "tracking_plan": "数据埋点方案简述",
    "iteration_plan": [
      {{
        "phase": "Phase 1: MVP",
        "timeline": "第1-4周",
        "deliverables": ["交付物1", "交付物2"]
      }},
      {{
        "phase": "Phase 2: 迭代",
        "timeline": "第5-8周",
        "deliverables": ["交付物1", "交付物2"]
      }}
    ]
  }},
  "risks_and_mitigation": [
    {{
      "risk": "风险描述",
      "impact": "影响程度（高/中/低）",
      "mitigation": "缓解措施"
    }}
  ]（至少3个风险）,
  "appendix": {{
    "sample_input": "示例用户输入或场景描述",
    "glossary": ["术语1：解释", "术语2：解释"]
  }}
}}

注意：
- 所有数组字段至少包含2个元素
- Mermaid 图表使用 graph TD 或 sequenceDiagram，确保语法正确（使用英文括号和箭头，不要用中文标点）
- 优先级使用 P0（必须有）/ P1（应该有）/ P2（锦上添花）
- 内容应针对「{product_idea}」这个具体场景，不要泛泛而谈"""


def run_single_llm_workflow(
    product_idea: str,
    supplementary_info: str = "",
    selected_model: str | None = None,
) -> dict:
    """Generate a complete PRD with a single LLM call.

    No pipeline stages, no multi-agent, no reflection. This is the
    simplest possible baseline for ablation comparison.

    Returns a dict compatible with run_workflow() / run_multi_agent_workflow():
        {
            "product_idea": str,
            "final_prd_json": dict,
            "final_prd_markdown": str,
            "elapsed_seconds": float,
            "error_message": str | None,
        }
    """
    import json
    from datetime import date

    start = time.time()

    supplementary_section = ""
    if supplementary_info:
        supplementary_section = f"补充信息：{supplementary_info}"

    user_prompt = SINGLE_LLM_USER_TEMPLATE.format(
        product_idea=product_idea,
        supplementary_section=supplementary_section,
        date=date.today().isoformat(),
    )

    messages = [
        {"role": "system", "content": SINGLE_LLM_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        client = LLMClient()
        raw = client.chat_with_json_mode(messages, model=selected_model)
        final_json = _parse_json(raw)
        prd_md = convert_to_prd_markdown(final_json)
        elapsed = round(time.time() - start, 1)
        logger.info("Single-LLM PRD generated: %d chars, %.1fs", len(prd_md), elapsed)
        return {
            "product_idea": product_idea,
            "final_prd_json": final_json,
            "final_prd_markdown": prd_md,
            "elapsed_seconds": elapsed,
            "error_message": None,
        }
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        logger.error("Single-LLM PRD failed: %s", e)
        return {
            "product_idea": product_idea,
            "final_prd_json": {},
            "final_prd_markdown": "",
            "elapsed_seconds": elapsed,
            "error_message": str(e),
        }


def _parse_json(raw: str) -> dict:
    """Parse LLM JSON output, with markdown fence stripping."""
    import json

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    return json.loads(raw)
