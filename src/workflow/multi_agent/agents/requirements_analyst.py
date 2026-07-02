import json
import os

from src.utils.llm_client import MultiProviderLLMClient
from src.prompts.manager import PromptManager
from src.workflow.multi_agent.node_utils import run_agent_with_reflexion, run_agent_with_tools
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEARCH_COMPETITORS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_competitors",
        "description": "搜索互联网上的竞品信息和市场现状。当你不熟悉某类产品的竞品格局、或需要了解市场上有哪些类似产品时，调用此工具获取实时信息。调用后返回相关文章的标题、链接和摘要。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，例如 'AI口语练习App 竞品分析 2026'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最多返回结果数，默认5",
                },
            },
            "required": ["query"],
        },
    },
}


def _search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo for competitor info. Free, no API key needed."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        if not results:
            return json.dumps({"results": [], "message": "未搜索到相关结果"})
        return json.dumps({"results": results, "source": "DuckDuckGo"}, ensure_ascii=False)
    except ImportError:
        return json.dumps({"error": "DuckDuckGo search library not available (pip install duckduckgo_search)", "results": []})
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return json.dumps({"error": str(e), "results": []})


def _search_via_mcp(query: str, max_results: int = 5) -> str:
    """Search via Tavily MCP Server."""
    try:
        from src.mcp.mcp_client import connect_mcp_client

        mcp_client = connect_mcp_client("npx", ["-y", "tavily-mcp-server"])
        try:
            result = mcp_client.call_tool_sync("search", {"query": query, "max_results": max_results})
            return result
        finally:
            mcp_client.close_sync()
    except Exception as e:
        logger.warning("Tavily MCP search failed, falling back to DuckDuckGo: %s", e)
        return _search_duckduckgo(query, max_results)


def _search_competitors(query: str, max_results: int = 5) -> str:
    """Search competitors — MCP (Tavily) preferred, DuckDuckGo fallback."""
    from src.mcp.mcp_client import is_tavily_available

    if is_tavily_available():
        return _search_via_mcp(query, max_results)
    return _search_duckduckgo(query, max_results)


def requirements_analyst_node(state: dict, reference_context: str = "") -> dict:
    prompt_mgr = PromptManager()
    model = state.get("selected_model") or None

    messages = prompt_mgr.get_agent_prompt(
        agent="requirements_analyst",
        product_idea=state.get("product_idea", ""),
        supplementary_info=state.get("supplementary_info", ""),
        reference_context=reference_context or state.get("retrieved_context", ""),
        image_analysis=json.dumps(state.get("image_analysis", {}), ensure_ascii=False),
        product_type=state.get("planner_output", {}).get("product_type", ""),
        user_feedback=state.get("user_feedback", ""),
    )

    try:
        client = MultiProviderLLMClient()
        tools = [SEARCH_COMPETITORS_TOOL]
        tool_fns = {"search_competitors": _search_competitors}

        logger.info("Requirements analyst running with tools (search_competitors available)")
        return run_agent_with_tools(
            client, messages, tools, tool_fns,
            "requirements_analyst", "requirement_analysis", model=model,
        )
    except Exception as e:
        logger.error("Requirements analyst with tools failed: %s, falling back to reflexion mode", e)
        from src.utils.llm_client import LLMClient
        client = LLMClient()
        return run_agent_with_reflexion(client, messages, "requirements_analyst", "requirement_analysis", model=model)
