"""MCP Server tools wrapping the PRD knowledge base."""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Cache the retriever to avoid re-initializing per call
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.rag.embedder import EmbeddingService
        from src.rag.retriever import Retriever
        from src.rag.store import ChromaStore

        store = ChromaStore()
        embedder = EmbeddingService()
        _retriever = Retriever(store, embedder)
        _retriever.ensure_methodology_loaded()
        if store.count() > 0:
            try:
                _retriever.build_graph_index()
            except Exception:
                pass
        logger.info("MCP retriever initialized, %d docs in store", store.count())
    return _retriever


def search_prd_knowledge(query: str) -> str:
    """搜索 PRD 知识库，返回向量检索 + 知识图谱的结构化上下文。

    适用场景：
    - 查找与某个产品想法相关的历史 PRD 案例
    - 查询特定功能模块的依赖关系
    - 了解某个产品类型的最佳实践

    Args:
        query: 搜索查询，例如"社交 App 的用户画像怎么写"
    """
    if not query or not query.strip():
        return "错误：查询内容不能为空"

    try:
        retriever = _get_retriever()
        context = retriever.search_as_context(query.strip())
        if not context or context == "暂无相关知识库内容":
            return "未找到与查询相关的 PRD 知识。知识库可能为空，请先上传 PRD 文档。"
        return context
    except Exception as e:
        logger.error("MCP search_prd_knowledge failed: %s", e)
        return f"知识库搜索失败：{e}"


def list_methodologies() -> str:
    """列出预置的产品方法论和 PRD 写作框架。

    适用场景：
    - 了解系统内置了哪些产品分析方法论
    - 选择适合当前产品类型的方法论框架
    """
    methodologies = [
        "### 需求分析\n"
        "- 用户故事映射 (User Story Mapping)：从用户视角梳理完整的使用流程\n"
        "- KANO 模型：区分基本型、期望型、兴奋型需求\n"
        "- 用户画像 (Persona)：多维度用户角色建模\n\n"
        "### 增长与指标\n"
        "- AARRR 漏斗：获取→激活→留存→收入→传播\n"
        "- HEART 框架：Google 用户体验度量模型\n"
        "- North Star Metric：北极星指标拆解\n\n"
        "### 优先级排序\n"
        "- MoSCoW 优先级：Must/Should/Could/Won't\n"
        "- RICE 评分：Reach × Impact × Confidence ÷ Effort\n"
        "- 价值/复杂度四象限矩阵\n\n"
        "### PRD 写作\n"
        "- 大厂标准 PRD 模板：14 章节完整结构\n"
        "- 功能规格说明 (FRS)：功能描述 + 验收标准 + 边界条件\n"
        "- 非功能性需求 (NFR)：性能、安全、可扩展性指标",
    ]
    return "\n".join(methodologies)
