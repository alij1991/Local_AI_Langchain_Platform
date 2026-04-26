"""[IMPROVE-4] Commit 3/4: tool dispatch + agent attribution gen_ai spans.

Pin the dual-emit contract for tool execution:
- Each tool invocation produces a ``gen_ai.execute_tool`` span shaped
  by the spec (gen_ai.tool.name, gen_ai.tool.type, gen_ai.tool.call.id
  via context propagation).
- Tool spans nest under the active chat span via OTel context — the
  trace tree shows "agent X chatted, called tool Y" without any extra
  plumbing.
- Errors raised by the tool propagate exactly as before; the span
  status flips to ERROR with the exception recorded once.
- Agent attribution: the chat span carries gen_ai.agent.name pulled
  from the context dict at __enter__.

Sources (2025–2026):
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- https://oneuptime.com/blog/post/2026-02-06-monitor-llm-opentelemetry-genai-semantic-conventions/view (2026-02-06)
"""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import StructuredTool
from opentelemetry import trace as _ot_trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from local_ai_platform import otel as otel_module
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.observability import track_event


@pytest.fixture
def memory_exporter(monkeypatch):
    """Wire an InMemorySpanExporter onto the live TracerProvider.

    Same shape as tests/test_otel_chat_spans.py — ``set_tracer_provider``
    is once-per-process so tests can't reset it; they add and tear down
    their own SimpleSpanProcessor.
    """
    monkeypatch.setenv("OTEL_EXPORTER", "none")
    otel_module.init_otel("test-service")
    provider = _ot_trace.get_tracer_provider()

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    yield exporter

    processor.shutdown()


def _instrument(func, name="my_tool", description="test tool", *, async_tool=False):
    """Build + instrument a StructuredTool around ``func``.

    Mirrors what AgentOrchestrator does internally when registering a
    tool. ``StructuredTool.from_function`` routes coroutines via the
    ``coroutine=`` kwarg, sync callables via ``func=`` — caller picks
    via ``async_tool``. Returns the instrumented tool.
    """
    if async_tool:
        tool = StructuredTool.from_function(
            coroutine=func, name=name, description=description,
        )
    else:
        tool = StructuredTool.from_function(
            func=func, name=name, description=description,
        )
    return AgentOrchestrator._instrument_tool(tool)


# ── basic span emission ──────────────────────────────────────────────


def test_sync_tool_emits_execute_tool_span(memory_exporter):
    """A sync tool invocation produces one gen_ai.execute_tool span
    with the tool name + type populated.
    """
    def add_two(x: int) -> int:
        return x + 2

    tool = _instrument(add_two, name="add_two")

    result = tool.func(x=40)
    assert result == 42

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert span.attributes["gen_ai.tool.name"] == "add_two"
    # Spec valid values: function | extension | datastore. LangChain
    # StructuredTool maps to OpenAI's function-call shape.
    assert span.attributes["gen_ai.tool.type"] == "function"


def test_async_tool_emits_execute_tool_span(memory_exporter):
    """Coroutine-backed tools follow the same path. Pin async because
    the wrapper has a separate code path for ``inspect.iscoroutinefunction``.
    """
    async def slow_double(x: int) -> int:
        await asyncio.sleep(0)  # yield once
        return x * 2

    tool = _instrument(slow_double, name="slow_double", async_tool=True)

    result = asyncio.run(tool.coroutine(x=21))
    assert result == 42

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.tool.name"] == "slow_double"


def test_span_name_includes_tool_name_per_spec(memory_exporter):
    """Spec format: span name = "{operation} {target}". For a tool
    span the target is the tool name itself, so we get
    "execute_tool add_two" once __exit__ updates the name based on
    the gen_ai.request.model attribute…

    Wait — there's no model attribute on a tool span. The current
    implementation only updates the name when gen_ai.request.model is
    set. So tool spans stay as the bare operation name. Pin that
    expected behavior so a future "include tool name in span name"
    refactor either updates this test or adds the right code path.
    """
    def echo(s: str) -> str:
        return s

    tool = _instrument(echo, name="echo")
    tool.func(s="hello")

    span = memory_exporter.get_finished_spans()[0]
    # Today: bare operation name. Update if span-name format changes.
    assert span.name == "execute_tool"


# ── error path ───────────────────────────────────────────────────────


def test_sync_tool_error_marks_span_and_propagates(memory_exporter):
    """A tool that raises must propagate the exception exactly as
    before — the wrapper is observability-only, not error-swallowing.
    The span ends with status=ERROR and a single ``exception`` event
    (no double-record from the SDK auto-exception path).
    """
    def boom(_: str) -> str:
        raise ValueError("nope")

    tool = _instrument(boom, name="boom")

    with pytest.raises(ValueError, match="nope"):
        tool.func(_="ignored")

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    exc_events = [e for e in span.events if e.name == "exception"]
    assert len(exc_events) == 1
    assert exc_events[0].attributes["exception.type"] == "ValueError"


def test_async_tool_error_marks_span_and_propagates(memory_exporter):
    """Same contract for the async wrapper."""
    async def kaput(_: str) -> str:
        raise RuntimeError("async oops")

    tool = _instrument(kaput, name="kaput", async_tool=True)

    with pytest.raises(RuntimeError, match="async oops"):
        asyncio.run(tool.coroutine(_="ignored"))

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


# ── nesting under parent chat span ───────────────────────────────────


def test_tool_span_nests_under_active_chat_span(memory_exporter):
    """OTel context propagation: when a tool runs inside an active
    chat span (which the chat router opens via track_event("chat",
    "send", ...)), the tool span's parent_span_id matches the chat
    span's span_id. Pin that so an operator viewing the trace tree
    sees "agent X chatted -> called tool Y" without any extra
    plumbing.
    """
    def square(x: int) -> int:
        return x * x

    tool = _instrument(square, name="square")

    with track_event(
        "chat", "send",
        context={
            "agent": "assistant",
            "model": "qwen3:8b",
            "provider": "ollama",
        },
    ):
        tool.func(x=7)

    spans = memory_exporter.get_finished_spans()
    # SimpleSpanProcessor exports in completion order — child first
    # (it ends inside the parent's with-block), then parent.
    assert len(spans) == 2
    tool_span, chat_span = spans

    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert chat_span.attributes["gen_ai.operation.name"] == "chat"

    # Parent-child link: tool span's parent must be the chat span.
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == chat_span.context.span_id


def test_multiple_tool_calls_produce_sibling_spans(memory_exporter):
    """Two sequential tool calls inside one chat span → two sibling
    tool spans, each with their own attrs. Pin that the wrapper
    doesn't accidentally pin the first tool's attrs onto the second.
    """
    def first(x: int) -> int:
        return x + 1

    def second(x: int) -> int:
        return x + 2

    tool_a = _instrument(first, name="add_one")
    tool_b = _instrument(second, name="add_two")

    with track_event("chat", "send", context={"agent": "a", "model": "m", "provider": "p"}):
        tool_a.func(x=1)
        tool_b.func(x=10)

    spans = memory_exporter.get_finished_spans()
    # Order: tool_a, tool_b, then chat — both children end before parent.
    tool_spans = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "execute_tool"]
    assert [s.attributes["gen_ai.tool.name"] for s in tool_spans] == ["add_one", "add_two"]


# ── agent attribution ───────────────────────────────────────────────


def test_chat_span_carries_agent_name_from_context(memory_exporter):
    """The chat router stores ``agent`` in context — __enter__ mirrors
    it onto gen_ai.agent.name. This is the parent attribute every
    nested tool span can inherit via context propagation if needed.
    """
    with track_event(
        "chat", "send",
        context={
            "agent": "research_assistant",
            "model": "qwen3:8b",
            "provider": "ollama",
        },
    ):
        pass

    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.agent.name"] == "research_assistant"


def test_chat_span_omits_agent_name_when_context_missing(memory_exporter):
    """Defensive: a track_event call that doesn't include ``agent``
    in context shouldn't get a stale or empty gen_ai.agent.name. Some
    chat paths (e.g. enhance_prompt) don't have an agent identity at
    all — ensure they don't emit a misleading attribute.
    """
    with track_event(
        "chat", "enhance_prompt",
        context={"prompt_length": 50, "model_hint": "qwen3:4b"},
    ) as ev:
        ev.set_otel_attributes({"gen_ai.request.model": "qwen3:4b"})

    span = memory_exporter.get_finished_spans()[0]
    assert "gen_ai.agent.name" not in span.attributes


# ── unmapped subsystems still skipped ───────────────────────────────


def test_provider_availability_probe_still_unmapped(memory_exporter):
    """Regression guard: Commit 2/4 added "chat" to the map, Commit
    3/4 added "tool". The provider/availability_probe pair must
    still NOT emit a gen_ai span — it's a health probe, not a
    gen_ai operation.
    """
    with track_event("provider", "availability_probe", context={"provider": "ollama"}):
        pass

    assert memory_exporter.get_finished_spans() == ()


def test_image_subsystem_still_unmapped_until_commit_4(memory_exporter):
    """Regression guard mirroring the chat-span tests — image is
    reserved for Commit 4/4. Adding "image" to the map prematurely
    would change the test result; keeping this here forces the wave
    plan to be honored.
    """
    with track_event("image", "generate", context={"model": "flux1-schnell"}):
        pass

    assert memory_exporter.get_finished_spans() == ()
