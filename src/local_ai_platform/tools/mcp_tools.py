"""MCP (Model Context Protocol) tool discovery and invocation."""
from __future__ import annotations

import json
import os
from urllib import request as urllib_request

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class MCPQueryInput(BaseModel):
    prompt: str = Field(..., description="Task/query to send to MCP server")


def mcp_query(prompt: str) -> str:
    """Call a configured MCP server tool endpoint."""
    endpoint = os.getenv("MCP_SERVER_URL", "").strip()
    method = os.getenv("MCP_TOOL_METHOD", "tools/call").strip() or "tools/call"
    if not endpoint:
        return "MCP server not configured. Set MCP_SERVER_URL."

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {"input": prompt},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=20) as resp:  # noqa: S310
            return resp.read().decode("utf-8")
    except Exception as exc:
        return f"MCP request failed: {exc}"


async def discover_mcp_server_tools(server_config: dict) -> list[dict]:
    """Discover tools from an MCP server using langchain-mcp-adapters."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        transport = server_config.get("transport", "stdio")
        name = server_config.get("name", "server")

        if transport == "stdio":
            config = {
                name: {
                    "transport": "stdio",
                    "command": server_config.get("command", ""),
                    "args": server_config.get("args", []),
                    "env": server_config.get("env"),
                }
            }
        else:
            config = {
                name: {
                    "transport": "sse",
                    "url": server_config.get("endpoint", ""),
                }
            }

        async with MultiServerMCPClient(config) as client:
            tools = client.get_tools()
            return [
                {
                    "tool_name": t.name,
                    "description": t.description or "",
                    "input_schema": t.args_schema.schema() if hasattr(t, "args_schema") and t.args_schema else {},
                }
                for t in tools
            ]
    except ImportError:
        return []
    except Exception as exc:
        return [{"error": str(exc)}]


async def invoke_mcp_tool(server_config: dict, tool_name: str, arguments: dict) -> dict:
    """Invoke a specific tool on an MCP server.

    Uses the same MultiServerMCPClient as discovery but calls the named tool.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        transport = server_config.get("transport", "stdio")
        name = server_config.get("name", "server")

        if transport == "stdio":
            config = {
                name: {
                    "transport": "stdio",
                    "command": server_config.get("command", ""),
                    "args": server_config.get("args", []),
                    "env": server_config.get("env"),
                }
            }
        else:
            config = {
                name: {
                    "transport": "sse",
                    "url": server_config.get("endpoint", ""),
                }
            }

        async with MultiServerMCPClient(config) as client:
            tools = client.get_tools()
            target = next((t for t in tools if t.name == tool_name), None)
            if target is None:
                return {"error": f"Tool '{tool_name}' not found on server '{name}'"}
            result = await target.ainvoke(arguments)
            return {"result": str(result)}
    except ImportError:
        return {"error": "langchain-mcp-adapters not installed"}
    except Exception as exc:
        return {"error": str(exc)}


def get_mcp_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=mcp_query,
            name="mcp_query",
            description="Call a configured MCP server tool endpoint.",
            args_schema=MCPQueryInput,
        ),
    ]
