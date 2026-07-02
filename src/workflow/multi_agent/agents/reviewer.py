import json

from src.utils.llm_client import LLMClient
from src.prompts.manager import PromptManager
from src.workflow.node_utils import safe_json_extract
from src.workflow.multi_agent.node_utils import _build_correction_messages
from src.utils.logger import get_logger

logger = get_logger(__name__)


def reviewer_node(state: dict, reference_context: str = "") -> dict:
    client = LLMClient()
    prompt_mgr = PromptManager()
    model = state.get("selected_model") or None
    agent_errors = _collect_agent_errors(state)

    # Run deterministic checks before LLM review
    det_checks = deterministic_prd_checks(state)

    messages = prompt_mgr.get_agent_prompt(
        agent="reviewer",
        product_idea=state.get("product_idea", ""),
        supplementary_info=state.get("supplementary_info", ""),
        reference_context=reference_context or state.get("retrieved_context", ""),
        requirement_analysis=json.dumps(state.get("requirement_analysis", {}), ensure_ascii=False),
        feature_plan=json.dumps(state.get("feature_plan", {}), ensure_ascii=False),
        ux_design=json.dumps(state.get("ux_design", {}), ensure_ascii=False),
        tech_advice=json.dumps(state.get("tech_advice", {}), ensure_ascii=False),
        image_analysis=json.dumps(state.get("image_analysis", {}), ensure_ascii=False),
        deterministic_result=json.dumps(det_checks, ensure_ascii=False),
    )

    try:
        parsed = _call_reviewer_with_retry(client, messages, model)
        return _build_reviewer_result(parsed, agent_errors, det_checks)
    except Exception as e:
        logger.error("Reviewer failed after retry: %s", e)
        return {
            "reviewer_score": 75,
            "reviewer_scores": {},
            "reviewer_feedback": {},
            "reviewer_summary": "评审官评估失败，已跳过评审直接生成文档",
            "current_stage": "reviewer",
            "error_message": str(e),
            "_agent_errors": agent_errors,
            "_reviewer_failed": True,
            "_deterministic_checks": det_checks,
        }


def _call_reviewer_with_retry(
    client: LLMClient, messages: list[dict], model: str | None
) -> dict:
    raw = ""
    try:
        raw = client.chat_with_json_mode(messages, model=model)
        logger.info("Reviewer complete, response length: %d", len(raw))
        return safe_json_extract(raw)
    except json.JSONDecodeError as e:
        logger.warning("Reviewer JSON parse failed: %s. Retrying with correction...", e)
        try:
            correction_msgs = _build_correction_messages(messages, raw, str(e))
            raw2 = client.chat(correction_msgs, model=model)
            logger.info("Reviewer correction response length: %d", len(raw2))
            return safe_json_extract(raw2)
        except Exception as e2:
            logger.error("Reviewer correction also failed: %s", e2)
            raise
    except Exception:
        raise


def _build_reviewer_result(parsed: dict, agent_errors: list[str], det_checks: dict | None = None) -> dict:
    result: dict = {
        "reviewer_score": parsed.get("overall_score", 0),
        "reviewer_scores": parsed.get("scores", {}),
        "reviewer_feedback": parsed.get("feedback", {}),
        "reviewer_summary": parsed.get("summary", ""),
        "current_stage": "reviewer",
    }

    if det_checks:
        result["_deterministic_checks"] = det_checks

    feedback = parsed.get("feedback", {})
    agents_to_revise = [name for name, fb in feedback.items() if fb and str(fb).lower() != "null"]
    if agents_to_revise:
        result["agents_to_revise"] = agents_to_revise

    consistency_issues = parsed.get("consistency_issues", [])
    completeness_gaps = parsed.get("completeness_gaps", [])
    if consistency_issues or completeness_gaps:
        result["_reviewer_detail"] = {
            "consistency_issues": consistency_issues,
            "completeness_gaps": completeness_gaps,
        }

    if agent_errors:
        result.setdefault("_agent_errors", agent_errors)

    return result


def deterministic_prd_checks(state: dict) -> dict:
    """Run deterministic, code-based quality checks before LLM review.

    These checks are 100% code, 0% LLM — they provide an engineering-level
    quality gate that the LLM reviewer can incorporate into its assessment.
    """
    checks = []

    req = state.get("requirement_analysis", {}) or {}
    feat = state.get("feature_plan", {}) or {}
    ux = state.get("ux_design", {}) or {}
    tech = state.get("tech_advice", {}) or {}

    # 1. Section completeness
    sections_present = sum(1 for s in [req, feat, ux, tech] if isinstance(s, dict) and s)
    checks.append({
        "name": "章节完整性",
        "pass": sections_present >= 4,
        "detail": f"4 个核心章节中 {sections_present} 个有输出",
    })

    # 2. JSON Schema completeness — requirement_analysis
    personas = req.get("user_personas", [])
    stories = req.get("user_stories", [])
    metrics = req.get("success_metrics", [])
    checks.append({
        "name": "JSON Schema — 需求分析",
        "pass": len(personas) >= 3 and len(stories) >= 5 and len(metrics) >= 3,
        "detail": f"用户画像 {len(personas)}/3, 用户故事 {len(stories)}/5, 成功指标 {len(metrics)}/3",
    })

    # 3. JSON Schema — feature_plan
    features = feat.get("feature_list", [])
    has_mvp = bool(feat.get("mvp_scope", {}))
    checks.append({
        "name": "JSON Schema — 功能规划",
        "pass": len(features) > 0 and has_mvp,
        "detail": f"功能数量 {len(features)}, MVP定义 {'有' if has_mvp else '缺失'}",
    })

    # 4. Mermaid existence
    mermaid = ux.get("mermaid_user_flow", "")
    checks.append({
        "name": "Mermaid 流程图",
        "pass": bool(mermaid and len(str(mermaid)) > 20),
        "detail": f"Mermaid 代码 {'存在' if mermaid else '缺失'}（{len(str(mermaid))} 字符）",
    })

    # 5. API endpoint count
    endpoints = tech.get("api_endpoints", [])
    checks.append({
        "name": "API 端点数量",
        "pass": len(endpoints) > 0,
        "detail": f"定义了 {len(endpoints)} 个 API 端点",
    })

    # 6. Internal consistency — product name
    names = set()
    for section in [req, feat, ux, tech]:
        n = section.get("product_name", "")
        if n:
            names.add(str(n))
    checks.append({
        "name": "产品名称一致性",
        "pass": len(names) <= 1,
        "detail": f"各章节产品名称: {list(names) if names else ['无']}",
    })

    passed = sum(1 for c in checks if c["pass"])
    failed = len(checks) - passed
    det_score = round(passed / len(checks) * 100) if checks else 0

    return {
        "checks": checks,
        "passed": passed,
        "failed": failed,
        "total": len(checks),
        "deterministic_score": det_score,
    }


def _collect_agent_errors(state: dict) -> list[str]:
    errors: list[str] = []
    for key in ("requirement_analysis", "feature_plan", "ux_design", "tech_advice"):
        val = state.get(key)
        if isinstance(val, dict) and not val:
            errors.append(f"{key} 输出为空（可能解析失败）")
    return errors
