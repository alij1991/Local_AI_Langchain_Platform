"""Tool library for the Local AI Platform.

Re-exports `build_default_tools` so that existing code like
`from .tools import build_default_tools` continues to work.
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool

from .builtin import get_builtin_tools
from .code_exec import get_code_tools
from .file_ops import get_file_tools
from .image_tools import get_image_tools
from .mcp_tools import get_mcp_tools
from .memory_tools import get_memory_tools
from .rag_tools import get_rag_tools
from .web import get_web_tools

# ── Tool categories ──────────────────────────────────────────────

TOOL_CATEGORIES: dict[str, dict] = {
    "utility": {
        "label": "Utilities",
        "icon": "build",
        "description": "Basic utilities: math, date/time, calculator",
        "getter": get_builtin_tools,
    },
    "file_ops": {
        "label": "File Operations",
        "icon": "folder_open",
        "description": "Read, write, list, and search files in the workspace",
        "getter": get_file_tools,
    },
    "code_exec": {
        "label": "Code Execution",
        "icon": "terminal",
        "description": "Run Python code and shell commands (sandboxed)",
        "getter": get_code_tools,
    },
    "web": {
        "label": "Web & Research",
        "icon": "search",
        "description": "Search the web, fetch webpages",
        "getter": get_web_tools,
    },
    "image": {
        "label": "Image Generation",
        "icon": "image",
        "description": "Generate and edit images using local diffusion models",
        "getter": get_image_tools,
    },
    "memory": {
        "label": "Memory",
        "icon": "psychology",
        "description": "Save and recall information across conversations",
        "getter": get_memory_tools,
    },
    "knowledge": {
        "label": "Knowledge & RAG",
        "icon": "library_books",
        "description": "Index and search documents for knowledge retrieval",
        "getter": get_rag_tools,
    },
    "mcp": {
        "label": "MCP Servers",
        "icon": "hub",
        "description": "Connect to Model Context Protocol servers",
        "getter": get_mcp_tools,
    },
}


def build_default_tools() -> list[StructuredTool]:
    """Build and return all available tools across all categories."""
    tools: list[StructuredTool] = []
    for cat in TOOL_CATEGORIES.values():
        try:
            tools.extend(cat["getter"]())
        except Exception:
            pass
    return tools


def get_tools_by_category() -> list[dict]:
    """Return tool info grouped by category, suitable for API responses."""
    result = []
    for cat_id, cat in TOOL_CATEGORIES.items():
        try:
            tools = cat["getter"]()
        except Exception:
            tools = []
        result.append({
            "id": cat_id,
            "label": cat["label"],
            "icon": cat["icon"],
            "description": cat["description"],
            "tools": [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "dangerous": bool((t.metadata or {}).get("dangerous")),
                }
                for t in tools
            ],
        })
    return result
