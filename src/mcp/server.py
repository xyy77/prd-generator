#!/usr/bin/env python
"""MCP Server for PRD Knowledge Base.

Provides tools for querying the PRD knowledge base via the
Model Context Protocol (MCP). Compatible with Claude Desktop
and other MCP clients.

Usage:
    python -m src.mcp.server

Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "prd-knowledge-base": {
          "command": "python",
          "args": ["-m", "src.mcp.server"],
          "cwd": "/path/to/prd-generator"
        }
      }
    }
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server

from src.mcp.tools import list_methodologies, search_prd_knowledge
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = Server("prd-knowledge-base")


@app.tool()
async def tool_search_prd_knowledge(query: str) -> str:
    """搜索 PRD 知识库（向量检索 + 知识图谱），查找与查询相关的历史案例和功能依赖关系。

    Args:
        query: 搜索查询，例如"社交 App 功能规划"、"登录模块的技术方案"
    """
    return search_prd_knowledge(query)


@app.tool()
async def tool_list_methodologies() -> str:
    """列出系统预置的产品方法论和 PRD 写作框架，帮助你理解可用的分析工具。"""
    return list_methodologies()


async def main():
    logger.info("Starting MCP server: prd-knowledge-base")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
