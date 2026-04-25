"""Tools + MCP servers router.

[IMPROVE-1] Commit 5 — fourth router.

Endpoints (12):
  GET    /tools                                       list saved + runtime tools
  POST   /tools                                       create a tool (DB + orch)
  DELETE /tools/{tool_id}                             remove from DB
  GET    /tools/tavily/status                         is Tavily key configured?
  POST   /tools/{tool_id}/test                        invoke tool with sample input
  GET    /tools/categories                            tools grouped by category
  POST   /mcp/servers/json                            create MCP server (stdio)
  GET    /mcp/servers                                 list MCP servers
  PUT    /mcp/servers/{server_id}                     update MCP config
  DELETE /mcp/servers/{server_id}                     delete server + discovered tools
  POST   /mcp/servers/{server_id}/discover            discover tools from MCP server
  POST   /mcp/servers/{server_id}/tools/{tool_name}/invoke   invoke specific MCP tool

GET /tools is graceful (returns [] when orchestrator not yet ready)
because Flutter polls it on cold boot — a 503 there would flash an
error banner every cold start. POST /tools/{id}/test is strict (404
when the tool isn't loaded — that's the right answer).

Heavy MCP imports (mcp_tools.discover_mcp_server_tools, invoke_mcp_tool)
stay inline inside handlers — the langchain-mcp-adapters package is
optional, so importing it lazily lets the rest of the router boot
without it.

References (2025–2026):
* FastAPI APIRouter — https://fastapi.tiangolo.com/tutorial/bigger-applications/
* Model Context Protocol (MCP) spec — https://modelcontextprotocol.io/
* langchain-mcp-adapters — https://github.com/langchain-ai/langchain-mcp-adapters
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.api.deps import (
    get_orchestrator,
    get_orchestrator_or_none,
)
from local_ai_platform.config import get_settings
from local_ai_platform.repositories.tools_repo import (
    delete_mcp_server,
    delete_tool_db,
    list_tools_db,
    upsert_mcp_server,
    upsert_tool,
)

router = APIRouter()


# ── Tools ─────────────────────────────────────────────────────────


@router.get("/tools")
async def get_tools(
    orchestrator: AgentOrchestrator | None = Depends(get_orchestrator_or_none),
):
    """Return tools in the format Flutter expects: {items: [...]}."""
    # Graceful: Flutter polls /tools on cold boot; returning [] while the
    # orchestrator is still initializing is better than flashing a 503.
    runtime = orchestrator.get_tool_names() if orchestrator else []
    saved = list_tools_db()
    items = []
    for name in runtime:
        items.append({"name": name, "type": "builtin", "is_enabled": True})
    for tool in saved:
        items.append(tool)
    return {"items": items}


@router.post("/tools")
async def create_tool(
    body: dict[str, Any],
    orchestrator: AgentOrchestrator | None = Depends(get_orchestrator_or_none),
):
    tool_type = body.get("type", "custom")
    name = body.get("name", "").strip()
    description = body.get("description", "").strip()
    config = body.get("config_json", {})

    # Persist to DB
    result = upsert_tool(
        tool_id=None,
        name=name,
        tool_type=tool_type,
        description=description,
        config=config,
        is_enabled=body.get("is_enabled", True),
    )

    # Also register as a runtime tool in the orchestrator. Graceful: the
    # DB row is the source of truth; runtime registration is a warm-cache
    # optimization that the next orchestrator boot will pick up from DB.
    if orchestrator and name:
        if tool_type == "instruction":
            # Instruction tool: wraps an LLM call with a custom system prompt
            instructions = config.get("instructions", description)
            orchestrator.add_instruction_tool(name, instructions)
        elif tool_type == "agent_tool":
            # Agent delegation tool
            target = config.get("target_agent", "")
            if target:
                orchestrator.add_agent_delegate_tool(name, target)

    return result


@router.delete("/tools/{tool_id}")
async def remove_tool(tool_id: str):
    delete_tool_db(tool_id)
    return {"status": "deleted"}


@router.get("/tools/tavily/status")
async def tavily_status():
    # [IMPROVE-69] Routed through AppSettings so .env values are
    # honored consistently with tools/web.py's web_search path.
    key = get_settings().tavily_api_key.strip()
    return {"present": bool(key)}


@router.post("/tools/{tool_id}/test")
async def test_tool(
    tool_id: str,
    body: dict[str, Any],
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Test a tool with sample input."""
    for tool in orchestrator.tools:
        if tool.name == tool_id:
            try:
                result = tool.invoke(body.get("input", ""))
                return {"result": result}
            except Exception as exc:
                return {"error": str(exc)}

    raise HTTPException(404, f"Tool '{tool_id}' not found")


# ── MCP Servers ───────────────────────────────────────────────────


@router.post("/mcp/servers/json")
async def create_mcp_server(body: dict[str, Any]):
    return upsert_mcp_server(
        server_id=None,
        name=body.get("name", ""),
        transport="stdio",
        command=body.get("config_json", {}).get("command", ""),
    )


@router.get("/tools/categories")
async def get_tool_categories():
    """Return tools grouped by category."""
    from local_ai_platform.tools import get_tools_by_category
    return {"categories": get_tools_by_category()}


@router.post("/mcp/servers/{server_id}/discover")
async def discover_mcp_tools(server_id: str):
    """Discover tools from an MCP server."""
    from local_ai_platform.repositories.tools_repo import list_mcp_servers, upsert_mcp_discovered_tools
    servers = list_mcp_servers()
    server = next((s for s in servers if s["id"] == server_id), None)
    if not server:
        raise HTTPException(404, f"MCP server '{server_id}' not found")

    try:
        from local_ai_platform.tools.mcp_tools import discover_mcp_server_tools
        tools = await discover_mcp_server_tools(server)
        if tools and not any("error" in t for t in tools):
            upsert_mcp_discovered_tools(server_id, tools)
        return {"items": tools}
    except Exception as exc:
        return {"items": [], "error": str(exc)}


@router.post("/mcp/servers/{server_id}/tools/{tool_name}/invoke")
async def invoke_mcp_tool_endpoint(server_id: str, tool_name: str, body: dict[str, Any]):
    """Invoke a specific MCP tool."""
    from local_ai_platform.repositories.tools_repo import list_mcp_servers
    servers = list_mcp_servers()
    server = next((s for s in servers if s["id"] == server_id), None)
    if not server:
        raise HTTPException(404, f"MCP server '{server_id}' not found")
    from local_ai_platform.tools.mcp_tools import invoke_mcp_tool
    result = await invoke_mcp_tool(server, tool_name, body.get("arguments", body))
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


@router.get("/mcp/servers")
async def list_mcp_servers_endpoint():
    """List all configured MCP servers."""
    from local_ai_platform.repositories.tools_repo import list_mcp_servers as _list, list_mcp_discovered_tools
    servers = _list()
    for s in servers:
        s["discovered_tools"] = list_mcp_discovered_tools(s["id"])
    return {"items": servers}


@router.put("/mcp/servers/{server_id}")
async def update_mcp_server(server_id: str, body: dict[str, Any]):
    """Update an MCP server configuration."""
    return upsert_mcp_server(
        server_id=server_id,
        name=body.get("name", ""),
        transport=body.get("transport", "stdio"),
        endpoint=body.get("endpoint") or "",
        command=body.get("command") or "",
        args=body.get("args"),
        env=body.get("env"),
    )


@router.delete("/mcp/servers/{server_id}")
async def delete_mcp_server_endpoint(server_id: str):
    """Delete an MCP server and its discovered tools."""
    from local_ai_platform.repositories.tools_repo import delete_mcp_discovered_tools
    delete_mcp_discovered_tools(server_id)
    delete_mcp_server(server_id)
    return {"status": "deleted"}
