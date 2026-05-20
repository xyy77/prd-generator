import json

from src.utils.llm_client import LLMClient

COMPLETENESS_PROMPT = """你是一位PRD质量评估专家。请检查以下PRD文档是否包含10个必备章节。

必备章节：
1. 版本记录（version_record）
2. 项目背景与目标（background_and_goals）
3. 用户画像（user_personas）
4. 功能需求（functional_requirements）
5. 非功能性需求（non_functional_requirements）
6. 技术架构（tech_architecture）
7. 数据分析/埋点（analytics_and_iteration）
8. 风险与缓解（risks_and_mitigation）
9. 附录（appendix）
10. 至少包含一个Mermaid流程图或架构图

对每个章节，判断是否存在且内容充实（非空）。每存在1个得1分，满分10分。

PRD文档：
{prd_content}

请严格返回JSON：
{{"score": <0-10>, "details": {{"章节名": {{"present": true/false, "comment": "说明"}}}}}}"""

FEASIBILITY_PROMPT = """你是一位资深技术架构师。请评估以下PRD文档中技术方案的可行性。

检查维度：
1. 技术栈是否与功能需求匹配
2. 是否有明显不可实现的要求（如"用1天实现ChatGPT级对话"）
3. 是否有遗漏的关键技术组件
4. 性能/安全方案是否合理

请给出0-10的可行性评分。

产品想法：{product_idea}
PRD文档：
{prd_content}

请严格返回JSON：
{{"score": <0-10>, "issues": ["问题1", "问题2"], "overall_assessment": "评估总结"}}"""

CONSISTENCY_PROMPT = """你是一位产品逻辑审查专家。请检查以下PRD文档的内部一致性。

检查项：
1. 功能列表是否覆盖了所有用户故事
2. 技术方案是否能支撑功能需求
3. 是否有前后矛盾的描述（如功能列表提到"支持离线模式"但技术方案没有本地存储）
4. 用户画像与功能设计是否匹配

请给出0-10的一致性评分。

产品想法：{product_idea}
PRD文档：
{prd_content}

请严格返回JSON：
{{"score": <0-10>, "contradictions": [{{"issue": "矛盾描述", "parts": ["部分A", "部分B"]}}], "assessment": "评估总结"}}"""

RELEVANCE_PROMPT = """你是一位产品分析专家。请评估以下PRD文档与原始产品想法的相关性。

检查项：
1. PRD内容是否紧扣产品想法
2. 是否有跑题或过度设计的内容
3. 功能设计是否围绕核心价值主张
4. 用户场景是否与目标用户匹配

产品想法：{product_idea}
PRD文档：
{prd_content}

请严格返回JSON：
{{"score": <0-10>, "on_topic": true/false, "digressions": ["跑题内容"], "assessment": "评估总结"}}"""


def _format_prd_for_eval(prd_markdown: str) -> str:
    max_chars = 8000
    if len(prd_markdown) > max_chars:
        return prd_markdown[:max_chars] + "\n\n... (truncated)"
    return prd_markdown


def _call_evaluator(prompt: str, model: str | None = None) -> dict:
    client = LLMClient()
    messages = [
        {"role": "system", "content": "你是一位严格的评估专家。请严格按照JSON格式返回评分结果。"},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = client.chat_with_json_mode(messages, model=model)
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        return json.loads(raw)
    except Exception:
        return {"score": 0, "error": "Evaluation failed"}


def completeness_score(prd_markdown: str, model: str | None = None) -> int:
    prompt = COMPLETENESS_PROMPT.format(prd_content=_format_prd_for_eval(prd_markdown))
    result = _call_evaluator(prompt, model=model)
    return int(result.get("score", 0))


def feasibility_score(prd_markdown: str, product_idea: str, model: str | None = None) -> int:
    prompt = FEASIBILITY_PROMPT.format(
        product_idea=product_idea,
        prd_content=_format_prd_for_eval(prd_markdown),
    )
    result = _call_evaluator(prompt, model=model)
    return int(result.get("score", 0))


def consistency_score(prd_markdown: str, product_idea: str, model: str | None = None) -> int:
    prompt = CONSISTENCY_PROMPT.format(
        product_idea=product_idea,
        prd_content=_format_prd_for_eval(prd_markdown),
    )
    result = _call_evaluator(prompt, model=model)
    return int(result.get("score", 0))


def relevance_score(prd_markdown: str, product_idea: str, model: str | None = None) -> int:
    prompt = RELEVANCE_PROMPT.format(
        product_idea=product_idea,
        prd_content=_format_prd_for_eval(prd_markdown),
    )
    result = _call_evaluator(prompt, model=model)
    return int(result.get("score", 0))


def evaluate_prd(
    prd_markdown: str,
    product_idea: str,
    model: str | None = None,
) -> dict:
    return {
        "completeness": completeness_score(prd_markdown, model=model),
        "feasibility": feasibility_score(prd_markdown, product_idea, model=model),
        "consistency": consistency_score(prd_markdown, product_idea, model=model),
        "relevance": relevance_score(prd_markdown, product_idea, model=model),
    }
