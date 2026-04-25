"""Tests for the [IMPROVE-7] Commit 6/6 tools/* httpx migration.

The final urllib.request consumers in the codebase live in three
LangChain ``StructuredTool`` wrappers:

* ``tools/web.py::fetch_webpage`` — agents use this to read pages by
  URL. Pre-migration it tried httpx via top-level ``import httpx``
  inside the function and fell back to ``urllib.request`` on
  ImportError. The fallback was dead code (httpx is a hard
  runtime dependency since Commit 1/6) and has been removed.
* ``tools/image_tools.py::_post_json`` — the indirect HTTP path used
  when the agent runs out-of-process from the API server.
* ``tools/mcp_tools.py::mcp_query`` — JSON-RPC ``tools/call`` to a
  configured MCP endpoint.

After this commit the codebase contains zero direct
``urllib.request`` consumers — only comments and docstrings still
reference it for historical context.

References (2025–2026):
* httpx Client API — https://www.python-httpx.org/api/#client
* httpx MockTransport — https://www.python-httpx.org/advanced/transports/#mock-transports
* MCP JSON-RPC spec — https://modelcontextprotocol.io/specification
"""
from __future__ import annotations

import json

import httpx
import pytest

from local_ai_platform.http_client import (
    reset_clients,
    set_test_clients,
)


@pytest.fixture(autouse=True)
def _isolated_singletons():
    reset_clients()
    yield
    reset_clients()


# ── tools/web.py::fetch_webpage ─────────────────────────────────────


def test_fetch_webpage_returns_extracted_text():
    """The tool must strip HTML and return the visible text. Pin
    that the User-Agent header is forwarded — many sites refuse
    requests without one — and that the truncation marker fires when
    the body exceeds ``max_chars``.
    """
    captured: dict = {}

    html_body = (
        "<html><head><title>x</title>"
        "<script>tracking()</script>"
        "<style>body{}</style></head>"
        "<body><h1>Hello</h1><p>World example body content.</p></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["user_agent"] = request.headers.get("user-agent")
        return httpx.Response(200, content=html_body.encode())

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.tools.web import fetch_webpage

    text = fetch_webpage("https://example.test/page", max_chars=5000)

    assert captured["url"] == "https://example.test/page"
    assert captured["user_agent"] == "Mozilla/5.0"
    # Script + style stripped, visible text retained.
    assert "Hello" in text
    assert "World example body content" in text
    assert "tracking()" not in text


def test_fetch_webpage_truncates_long_bodies():
    """Bodies past ``max_chars`` get cut, with a footer announcing
    the original length so the agent can decide whether to re-fetch
    or accept the partial.
    """
    long_text = "abcde " * 5000  # ~30k chars of visible content

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=f"<p>{long_text}</p>".encode())

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.tools.web import fetch_webpage

    out = fetch_webpage("https://example.test/long", max_chars=200)
    assert len(out) > 200  # footer pushes it past the cap
    assert "(truncated," in out
    assert "total chars)" in out


def test_fetch_webpage_returns_error_string_on_failure():
    """The tool wraps failures in a returned string (rather than
    raising) so the agent can incorporate the error into its plan.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("DNS")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.tools.web import fetch_webpage

    out = fetch_webpage("https://example.test/down")
    assert out.startswith("Error fetching ")
    assert "https://example.test/down" in out


# ── tools/image_tools.py::_post_json ─────────────────────────────────


def test_image_tools_post_json_sends_json_body_and_returns_text():
    """The indirect path posts JSON to the API server and returns the
    raw response body for the agent to interpret. Pin the JSON body
    shape because a regression here would break the contract with
    ``/images/generate`` / ``/images/edit``.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, content=b'{"status": "ok", "image_id": "abc123"}')

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.tools.image_tools import _post_json

    out = _post_json(
        "http://api.test/images/generate",
        {"session_id": "s1", "model_id": "flux", "prompt": "a cat"},
    )
    assert captured["url"] == "http://api.test/images/generate"
    assert captured["method"] == "POST"
    assert captured["body"] == {"session_id": "s1", "model_id": "flux", "prompt": "a cat"}
    # Raw body returned as text — caller (the agent) parses it itself.
    assert out == '{"status": "ok", "image_id": "abc123"}'


def test_image_tools_post_json_raises_on_5xx():
    """A 5xx must surface so ``generate_image`` / ``edit_image``'s
    outer ``try/except`` can wrap it into a user-readable error. The
    pre-migration urllib path raised ``HTTPError``; httpx raises
    ``HTTPStatusError`` from ``raise_for_status``.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"upstream is sad")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.tools.image_tools import _post_json

    with pytest.raises(httpx.HTTPStatusError):
        _post_json("http://api.test/images/generate", {"x": 1})


# ── tools/mcp_tools.py::mcp_query ───────────────────────────────────


def test_mcp_query_sends_jsonrpc_envelope(monkeypatch):
    """``mcp_query`` wraps the user's prompt in the standard JSON-RPC
    envelope (``jsonrpc=2.0``, ``method=tools/call``, ``params.input``)
    before posting. Pin both the envelope and the URL so a regression
    can't silently re-shape the request.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, content=b'{"jsonrpc":"2.0","id":1,"result":"ok"}')

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    # Override settings via the module's import binding — the
    # project's settings_customise_sources puts ``.env`` ahead of
    # shell env, and ``.env`` ships with ``MCP_SERVER_URL=`` blanked,
    # so ``monkeypatch.setenv`` alone won't take. Patching the
    # ``get_settings`` reference inside the tool module is cleaner.
    from local_ai_platform.tools import mcp_tools

    class _FakeSettings:
        mcp_server_url = "http://mcp.test/rpc"
        mcp_tool_method = "tools/call"

    monkeypatch.setattr(mcp_tools, "get_settings", lambda: _FakeSettings())

    out = mcp_tools.mcp_query("describe a sunset")

    assert captured["url"] == "http://mcp.test/rpc"
    body = captured["body"]
    assert body["jsonrpc"] == "2.0"
    assert body["method"] == "tools/call"
    assert body["params"] == {"input": "describe a sunset"}
    # Raw body returned for the agent to parse.
    assert "result" in out


def test_mcp_query_returns_error_string_on_failure(monkeypatch):
    """Connect failures must wrap into a returned string, not raise —
    same contract as the urllib path, since LangChain tools
    catch + report errors themselves but leak less context if we
    raise vs return.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("MCP server down")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.tools import mcp_tools

    class _FakeSettings:
        mcp_server_url = "http://mcp.test/rpc"
        mcp_tool_method = "tools/call"

    monkeypatch.setattr(mcp_tools, "get_settings", lambda: _FakeSettings())

    out = mcp_tools.mcp_query("hello")
    assert out.startswith("MCP request failed:")


def test_mcp_query_returns_helpful_error_when_url_unset(monkeypatch):
    """Empty MCP_SERVER_URL must produce a guidance string rather
    than a confusing connection error — the tool may be wired into
    an agent on a deployment that hasn't set the env var.
    """
    from local_ai_platform.tools import mcp_tools

    class _FakeSettings:
        mcp_server_url = ""
        mcp_tool_method = "tools/call"

    monkeypatch.setattr(mcp_tools, "get_settings", lambda: _FakeSettings())

    out = mcp_tools.mcp_query("hello")
    assert "MCP_SERVER_URL" in out


# ── Closing invariant: zero direct urllib consumers in src/ ─────────


def test_no_direct_urllib_request_consumers_remain_in_src():
    """[IMPROVE-7] guardrail. The whole point of the migration is
    a single httpx surface; this test fails if a future change
    re-introduces a stdlib ``urllib.request`` call site. Comments
    + docstrings are allowed (they reference the old code for
    historical context). Module-level imports from
    ``urllib.request`` are not.
    """
    import re
    from pathlib import Path

    src_root = Path(__file__).resolve().parent.parent / "src"
    bad: list[tuple[str, int, str]] = []
    # Match an actual import or call, but not a comment or docstring
    # mention. Docstring references use backticks (``urllib.request.X``)
    # and won't have a trailing call paren — requiring ``(`` after the
    # attribute access lets us catch real calls without false-positives
    # on comments / docstrings. ``urllib.parse`` is fine; only
    # ``urllib.request`` is the network surface.
    code_pattern = re.compile(
        r"(^\s*import\s+urllib\.request|"
        r"^\s*from\s+urllib(?:\s+import\s+request|\.request\s+import)|"
        r"\burllib\.request\.\w+\s*\(|"
        r"\burlretrieve\s*\()",
        re.MULTILINE,
    )
    for py in src_root.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        for match in code_pattern.finditer(text):
            # Skip if the match starts inside a comment.
            line_start = text.rfind("\n", 0, match.start()) + 1
            line = text[line_start:text.find("\n", match.start())]
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            line_no = text.count("\n", 0, match.start()) + 1
            bad.append((str(py.relative_to(src_root)), line_no, line.strip()))

    assert not bad, (
        "Direct urllib.request consumers re-introduced. [IMPROVE-7] "
        "requires every HTTP call to go through the shared httpx "
        "client (``http_client.get_sync_client`` / "
        "``get_async_client``). Offending lines:\n"
        + "\n".join(f"  {p}:{ln}: {src}" for p, ln, src in bad)
    )
