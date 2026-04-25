"""Web search and fetching tools."""
from __future__ import annotations

import re

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..config import get_settings
from ..http_client import get_sync_client


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Maximum results to return")


class FetchWebpageInput(BaseModel):
    url: str = Field(..., description="URL to fetch")
    max_chars: int = Field(5000, description="Maximum characters to return")


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web. Uses Tavily if API key available, otherwise DuckDuckGo."""
    # Try Tavily first
    tavily_key = get_settings().tavily_api_key.strip()
    if tavily_key:
        try:
            from langchain_tavily import TavilySearch
            tool = TavilySearch(max_results=max_results)
            result = tool.invoke({"query": query})
            return str(result)
        except Exception as exc:
            pass  # Fall through to DuckDuckGo

    # DuckDuckGo fallback (no API key needed)
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No results found for: {query}"
        lines = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"**{title}**\n{body}\n{href}")
        return "\n\n".join(lines)
    except ImportError:
        return "Web search unavailable: install duckduckgo-search (pip install duckduckgo-search)"
    except Exception as exc:
        return f"Web search failed: {exc}"


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_webpage(url: str, max_chars: int = 5000) -> str:
    """Fetch a webpage and extract its text content.

    [IMPROVE-7] Uses the shared httpx sync client. The legacy
    ``ImportError`` fallback to ``urllib.request`` was dead code:
    httpx is a hard runtime dependency since Commit 1/6, so the
    fallback can never trigger. Removing it shrinks the surface
    and lets a missing httpx fail loudly instead of silently
    degrading to a less-capable transport.
    """
    try:
        # 15s timeout — fetch_webpage is a user-facing tool, not a
        # background scrape, so we don't want a slow site to pin the
        # whole agent turn. ``follow_redirects`` is on by default
        # via the shared client config.
        resp = get_sync_client().get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
        )
        resp.raise_for_status()
        text = _strip_html(resp.text)
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n... (truncated, {len(text)} total chars)"
        return text
    except Exception as exc:
        return f"Error fetching {url}: {exc}"


def get_web_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=web_search,
            name="web_search",
            description="Search the web for information. Uses Tavily if API key is set, otherwise DuckDuckGo.",
            args_schema=WebSearchInput,
        ),
        StructuredTool.from_function(
            func=fetch_webpage,
            name="fetch_webpage",
            description="Fetch and extract text content from a URL.",
            args_schema=FetchWebpageInput,
        ),
    ]
