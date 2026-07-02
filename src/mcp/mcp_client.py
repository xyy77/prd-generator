"""MCP Client — connect to external MCP servers and call their tools.

Uses the MCP SDK's ClientSession + stdio_client to communicate with
external MCP servers via JSON-RPC over stdio.

Supports:
- Node.js MCP servers (npx):  MCPClient("npx", ["-y", "tavily-mcp-server"])
- Python MCP servers:         MCPClient("python", ["-m", "some_mcp_server"])
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCPClient:
    """Connect to an external MCP server via stdio and call its tools."""

    def __init__(self, command: str, args: list[str] | None = None):
        self.command = command
        self.args = args or []
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list[dict] = []
        self._connected = False

    @property
    def tools(self) -> list[dict]:
        """Tools exposed by the MCP server (OpenAI Function Calling format)."""
        return list(self._tools)

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> list[dict]:
        """Start the MCP server subprocess, complete MCP handshake, list tools.

        Returns tool definitions in OpenAI Function Calling format.
        """
        self._exit_stack = AsyncExitStack()

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=None,
        )

        try:
            transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = transport
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            logger.info(
                "MCPClient connected to %s %s", self.command, " ".join(self.args)
            )

            response = await self._session.list_tools()
            self._tools = self._convert_tools(response.tools)
            self._connected = True
            logger.info("MCPClient loaded %d tools", len(self._tools))
            return self._tools

        except Exception:
            logger.error("MCPClient connection failed for %s %s", self.command, self.args)
            await self._cleanup()
            raise

    def _convert_tools(self, mcp_tools) -> list[dict]:
        """Convert MCP Tool objects to OpenAI Function Calling format."""
        converted = []
        for tool in mcp_tools:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema if tool.inputSchema else {
                        "type": "object",
                        "properties": {},
                    },
                },
            })
        return converted

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the connected MCP server. Returns the tool's output as a string."""
        if not self._session or not self._connected:
            return json.dumps({"error": "MCP client not connected"})

        try:
            result = await self._session.call_tool(tool_name, arguments)
            if hasattr(result, "content") and result.content:
                parts = []
                for c in result.content:
                    if hasattr(c, "text"):
                        parts.append(c.text)
                    elif hasattr(c, "data"):
                        parts.append(str(c.data))
                    else:
                        parts.append(str(c))
                return "\n".join(parts)
            return str(result)
        except Exception as e:
            logger.warning("MCPClient tool call '%s' failed: %s", tool_name, e)
            return json.dumps({"error": str(e), "tool": tool_name})

    async def _cleanup(self):
        self._connected = False
        self._session = None
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            self._exit_stack = None

    async def close(self):
        """Close the MCP connection and terminate the subprocess."""
        await self._cleanup()

    def call_tool_sync(self, tool_name: str, arguments: dict) -> str:
        """Synchronous wrapper for call_tool."""
        return _run_async(self.call_tool(tool_name, arguments))

    def close_sync(self):
        """Synchronous wrapper for close."""
        try:
            _run_async(self.close())
        except Exception:
            pass


def _run_async(coro):
    """Run an async coroutine synchronously, returning its result."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        # Already inside an event loop — create a new loop in a thread
        import concurrent.futures
        import threading
        result_holder = []
        error_holder = []

        def _run_in_new_loop():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result_holder.append(new_loop.run_until_complete(coro))
                new_loop.close()
            except Exception as e:
                error_holder.append(e)

        t = threading.Thread(target=_run_in_new_loop)
        t.start()
        t.join()
        if error_holder:
            raise error_holder[0]
        return result_holder[0] if result_holder else None


def connect_mcp_client(command: str, args: list[str] | None = None) -> MCPClient:
    """Create and connect an MCPClient synchronously.

    Returns a connected MCPClient, or raises an exception on failure.
    """
    client = MCPClient(command, args)
    _run_async(client.connect())
    return client


def _check_tavily_available() -> bool:
    """Check if Tavily MCP server can be launched (Node.js + npx available)."""
    if not os.getenv("TAVILY_API_KEY"):
        return False
    try:
        result = subprocess.run(
            ["npx", "--version"], capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return result.returncode == 0
    except Exception:
        return False


_TAVILY_CACHE: bool | None = None


def is_tavily_available() -> bool:
    global _TAVILY_CACHE
    if _TAVILY_CACHE is None:
        _TAVILY_CACHE = _check_tavily_available()
    return _TAVILY_CACHE
