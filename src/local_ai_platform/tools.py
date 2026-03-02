from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from urllib import request

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Max Tavily results")


class MCPQueryInput(BaseModel):
    prompt: str = Field(..., description="Task/query to send to MCP server")


class GenerateImageInput(BaseModel):
    session_id: str = Field(..., description="Image session id")
    model_id: str = Field(..., description="Hugging Face image model id")
    prompt: str = Field(..., description="Prompt to generate image")


class EditImageInput(BaseModel):
    session_id: str = Field(..., description="Image session id")
    base_image_id: str = Field(..., description="Existing image id")
    model_id: str = Field(..., description="Hugging Face image model id")
    instruction: str = Field(..., description="Natural-language image edit instruction")


def multiply_numbers(a: float, b: float) -> str:
    return f"{a} * {b} = {a * b}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tavily_web_search(query: str, max_results: int = 5) -> str:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return "Tavily API key missing. Set TAVILY_API_KEY in backend environment/.env."

    try:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=max_results)
        result = tool.invoke({"query": query})
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Tavily search unavailable: {exc}"




def _post_json(url: str, payload: dict) -> str:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def generate_image(session_id: str, model_id: str, prompt: str) -> str:
    api_base = os.getenv("LOCAL_AI_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        return _post_json(f"{api_base}/images/generate", {"session_id": session_id, "model_id": model_id, "prompt": prompt})
    except Exception as exc:  # noqa: BLE001
        return f"Image generation failed: {exc}"


def edit_image(session_id: str, base_image_id: str, model_id: str, instruction: str) -> str:
    api_base = os.getenv("LOCAL_AI_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        return _post_json(
            f"{api_base}/images/edit",
            {
                "session_id": session_id,
                "base_image_id": base_image_id,
                "model_id": model_id,
                "instruction": instruction,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return f"Image edit failed: {exc}"


def mcp_query(prompt: str) -> str:
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
    req = request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=20) as resp:  # noqa: S310
            return resp.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"MCP request failed: {exc}"


def build_default_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=multiply_numbers,
            name="multiply_numbers",
            description="Multiply two numbers together.",
            args_schema=MultiplyInput,
        ),
        StructuredTool.from_function(
            func=utc_now,
            name="utc_now",
            description="Get the current UTC time in ISO format.",
        ),
        StructuredTool.from_function(
            func=tavily_web_search,
            name="tavily_web_search",
            description="Search the web using Tavily.",
            args_schema=WebSearchInput,
        ),
        StructuredTool.from_function(
            func=mcp_query,
            name="mcp_query",
            description="Call a configured MCP server tool endpoint.",
            args_schema=MCPQueryInput,
        ),
        StructuredTool.from_function(
            func=generate_image,
            name="generate_image",
            description="Generate an image in an image session using configured image model.",
            args_schema=GenerateImageInput,
        ),
        StructuredTool.from_function(
            func=edit_image,
            name="edit_image",
            description="Edit an existing image version in an image session.",
            args_schema=EditImageInput,
        ),
    ]
