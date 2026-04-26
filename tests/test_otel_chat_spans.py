"""[IMPROVE-4] Commit 2/4: chat-path gen_ai span emission.

These tests pin the dual-emit contract: existing ``emit()`` rows still
land in ``app_events``, and an additional OTel span shaped by the
``gen_ai.*`` semantic conventions is produced for every (subsystem,
action) pair that's mapped via ``_OTEL_OPERATION_MAP``.

Test strategy: spin up an ``InMemorySpanExporter`` on top of the
provider that ``init_otel`` returns, then drive ``track_event`` directly
(skip the chat router — that's an FastAPI integration test for [IMPROVE-68]
unification later). The chat router's set_otel_attributes call sites
are exercised by reading the same source file's enhance/non-stream/stream
paths and re-emitting the same attribute dicts via ``ev.set_otel_attributes``
in these tests.

Sources (2025-2026):
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- https://www.datadoghq.com/blog/llm-otel-semantic-convention/ (2025)
"""
from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from opentelemetry import trace as _ot_trace

from local_ai_platform import otel as otel_module
from local_ai_platform.observability import track_event


@pytest.fixture
def memory_exporter(monkeypatch):
    """Wire an InMemorySpanExporter onto the live TracerProvider.

    OTel's ``trace.set_tracer_provider`` is once-per-process — calling
    ``init_otel`` again silently no-ops if a provider is already set,
    even after our wrapper state resets. So tests work with whatever
    provider is currently active and just add their own exporter on
    top via ``add_span_processor``. The existing processor list (from
    init_otel's exporter selection) is left untouched.

    We tear the added processor down explicitly so spans from a
    later test don't leak into a previous test's exporter buffer
    (which would be a confusing failure mode if a test runs second
    accidentally captures the first's spans).
    """
    monkeypatch.setenv("OTEL_EXPORTER", "none")
    otel_module.init_otel("test-service")
    provider = _ot_trace.get_tracer_provider()

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    yield exporter

    processor.shutdown()


# ── basic span emission ──────────────────────────────────────────────


def test_chat_send_emits_gen_ai_span_with_default_context(memory_exporter):
    """A ``track_event("chat", "send", {...})`` block produces a span
    with operation.name=chat and the standard attributes pulled from
    the context dict at __enter__.
    """
    with track_event(
        "chat", "send",
        context={
            "agent": "assistant",
            "model": "qwen3:8b",
            "provider": "ollama",
            "conversation_id": "conv-123",
            "run_id": "run-abc",
        },
    ):
        pass

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.system"] == "ollama"
    assert span.attributes["gen_ai.request.model"] == "qwen3:8b"
    assert span.attributes["gen_ai.conversation.id"] == "conv-123"


def test_chat_enhance_prompt_emits_gen_ai_span(memory_exporter):
    """``chat/enhance_prompt`` is also mapped to operation.name=chat
    (it's a chat completion under the hood, just with a system prompt
    that asks the model to rewrite the user's prompt).
    """
    with track_event(
        "chat", "enhance_prompt",
        context={
            "prompt_length": 42,
            "model_hint": "qwen3:4b",
            "detected_type": "image",
        },
    ) as ev:
        # Mirror what api/routers/chat.py does after the resolver
        # picks a concrete model for the "auto" hint.
        ev.set_otel_attributes({
            "gen_ai.request.model": "qwen3:4b",
            "gen_ai.system": "ollama",
        })

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.system"] == "ollama"
    assert span.attributes["gen_ai.request.model"] == "qwen3:4b"


def test_unmapped_subsystem_emits_no_span(memory_exporter):
    """Provider availability probes, lifespan markers, and other
    non-gen_ai events MUST NOT pollute the OTel trace graph. Pin
    that ``track_event`` doesn't emit a span when the (subsystem,
    action) pair is missing from _OTEL_OPERATION_MAP.

    (The emit() row in app_events still lands — that's a separate
    code path and not what this test cares about.)
    """
    with track_event("provider", "availability_probe", context={"provider": "ollama"}):
        pass

    assert memory_exporter.get_finished_spans() == ()


def test_image_subsystem_unmapped_until_commit_4(memory_exporter):
    """Sanity check for the wave plan: ``image`` is reserved for
    Commit 4/4. Until then, image events emit their app_events row
    but no OTel span. Pin that so a careless extension to the map
    in this commit fails this test.
    """
    with track_event("image", "generate", context={"model": "flux1-schnell"}):
        pass

    assert memory_exporter.get_finished_spans() == ()


# ── status mapping ───────────────────────────────────────────────────


def test_clean_completion_leaves_status_unset_meaning_ok(memory_exporter):
    """Per the OTel spec, a span with status=UNSET is interpreted as
    OK by all consumers. Setting StatusCode.OK is reserved for "we
    explicitly know it succeeded" — for the chat path, "no exception
    propagated" is the same signal so we leave UNSET.
    """
    with track_event("chat", "send", context={"model": "qwen3:8b", "provider": "ollama"}):
        pass

    span = memory_exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.UNSET


def test_exception_in_block_marks_span_error_and_records_exception(memory_exporter):
    """The streaming chat path's ``except Exception`` block calls
    ``ev.mark_error`` after yielding the SSE error frame; the
    non-streaming path lets the exception propagate. Both must end
    with span.status=ERROR. Test the propagation case here.
    """
    with pytest.raises(RuntimeError):
        with track_event("chat", "send", context={"model": "qwen3:8b", "provider": "ollama"}):
            raise RuntimeError("model timed out")

    span = memory_exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    # record_exception attaches the exception type/message as a span event
    # — pin that the SDK saw and recorded our raise.
    exc_events = [e for e in span.events if e.name == "exception"]
    assert len(exc_events) == 1
    assert exc_events[0].attributes["exception.type"] == "RuntimeError"
    assert "model timed out" in exc_events[0].attributes["exception.message"]


def test_mark_error_without_propagation_still_marks_span(memory_exporter):
    """The streaming path catches exceptions to yield SSE error frames
    and uses ``ev.mark_error(exc)`` to record the outcome without
    re-raising. The span's status must still flip to ERROR — this is
    the corner case the old ``__exit__`` would silently mark "ok"
    before [IMPROVE-4].
    """
    with track_event("chat", "send", context={"model": "qwen3:8b", "provider": "ollama"}) as ev:
        try:
            raise ValueError("partial stream")
        except ValueError as exc:
            ev.mark_error(exc)

    span = memory_exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR


def test_cancellation_marks_span_error_with_cancelled_message(memory_exporter):
    """Client disconnect → GeneratorExit → mark_cancelled in the
    streaming path. Per the spec, cancellation is ERROR with a
    "cancelled" status message (UNSET would be wrong — UNSET means
    "we don't know what happened"; we know exactly: the client gave up).
    """
    with track_event("chat", "send", context={"model": "qwen3:8b", "provider": "ollama"}) as ev:
        ev.mark_cancelled("client_disconnect")

    span = memory_exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "cancel" in (span.status.description or "").lower()


# ── set_otel_attributes contract ─────────────────────────────────────


def test_set_otel_attributes_attaches_token_usage(memory_exporter):
    """Mirror the streaming chat path: token_count is set on the
    span via set_otel_attributes after the stream completes. The
    spec key is ``gen_ai.usage.output_tokens``.
    """
    with track_event("chat", "send", context={"model": "qwen3:8b", "provider": "ollama"}) as ev:
        # Pretend we just streamed 247 tokens.
        ev.set_otel_attributes({
            "gen_ai.usage.output_tokens": 247,
            "gen_ai.response.id": "run-zzz",
            "gen_ai.agent.name": "assistant",
        })

    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.usage.output_tokens"] == 247
    assert span.attributes["gen_ai.response.id"] == "run-zzz"
    assert span.attributes["gen_ai.agent.name"] == "assistant"


def test_set_otel_attributes_skips_none_values(memory_exporter):
    """OTel rejects ``None`` attributes outright. set_otel_attributes
    swallows None silently so the chat router can pass
    ``{"gen_ai.usage.input_tokens": prompt_tokens or None}`` without
    extra guards at the call site.
    """
    with track_event("chat", "send", context={"model": "qwen3:8b", "provider": "ollama"}) as ev:
        ev.set_otel_attributes({
            "gen_ai.usage.output_tokens": 100,
            "gen_ai.usage.input_tokens": None,  # unknown — must be skipped
            "gen_ai.response.id": None,
        })

    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.usage.output_tokens"] == 100
    assert "gen_ai.usage.input_tokens" not in span.attributes
    assert "gen_ai.response.id" not in span.attributes


def test_set_otel_attributes_safe_when_unmapped(memory_exporter):
    """Calling ev.set_otel_attributes on an unmapped subsystem must
    be a silent no-op. Lets generic helpers attach attrs without
    knowing whether the wrapping track_event is gen_ai or not.
    """
    with track_event("provider", "availability_probe", context={"provider": "ollama"}) as ev:
        # Should not raise; attrs go nowhere because no span exists.
        ev.set_otel_attributes({"gen_ai.system": "ollama"})

    assert memory_exporter.get_finished_spans() == ()


def test_set_otel_attributes_overrides_context_defaults(memory_exporter):
    """The context dict at __enter__ is the cheap path; explicit
    set_otel_attributes is the override path. Last-write-wins —
    enhance_prompt resolves "auto" to a concrete model name *inside*
    the with-block, and this is the call site that pins it.
    """
    with track_event(
        "chat", "enhance_prompt",
        context={"model_hint": "auto", "prompt_length": 42},
    ) as ev:
        # Initially no model attribute set (model_hint == "auto" is
        # filtered out by __enter__). Override with the resolved name.
        ev.set_otel_attributes({"gen_ai.request.model": "qwen3:8b"})

    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.request.model"] == "qwen3:8b"


# ── span name format ─────────────────────────────────────────────────


def test_span_name_includes_model_when_attached(memory_exporter):
    """Per the spec: span name = "{operation} {model}". Our __exit__
    updates the name from the bare operation to include the model
    if a gen_ai.request.model attribute has been attached.
    """
    with track_event(
        "chat", "send",
        context={"model": "qwen3:8b", "provider": "ollama"},
    ):
        pass

    span = memory_exporter.get_finished_spans()[0]
    assert span.name == "chat qwen3:8b"


def test_span_name_falls_back_to_operation_when_model_unknown(memory_exporter):
    """When the model can't be determined (no context.model + no
    set_otel_attributes call), the span name stays as the bare
    operation. The spec allows this fallback.
    """
    with track_event("chat", "send", context={"provider": "ollama"}):
        pass

    span = memory_exporter.get_finished_spans()[0]
    assert span.name == "chat"


# ── auto-loaded context defaults ─────────────────────────────────────


def test_model_hint_auto_is_filtered_out_at_enter(memory_exporter):
    """``model_hint="auto"`` is the chat router's "ask the resolver"
    sentinel — it's not a real model name. Pin that __enter__ doesn't
    eagerly stamp "auto" onto gen_ai.request.model. The resolver's
    set_otel_attributes call later will pin the resolved name.
    """
    with track_event(
        "chat", "enhance_prompt",
        context={"model_hint": "auto", "prompt_length": 100},
    ):
        pass

    span = memory_exporter.get_finished_spans()[0]
    assert "gen_ai.request.model" not in span.attributes


def test_provider_attribute_passed_through_unchanged(memory_exporter):
    """The chat router stores the raw provider name (ollama,
    huggingface, llamacpp, openai_compatible) in context["provider"].
    __enter__ writes it to gen_ai.system as-is — no normalization, no
    mapping to spec well-known values, because half our providers don't
    have a spec-canonical name. This is the load-bearing decision
    flagged in the proposal: pass-through wins over invented mapping.
    """
    for provider in ("ollama", "huggingface", "llamacpp", "openai_compatible"):
        with track_event(
            "chat", "send",
            context={"model": "test:1b", "provider": provider},
        ):
            pass

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 4
    systems = [s.attributes["gen_ai.system"] for s in spans]
    assert systems == ["ollama", "huggingface", "llamacpp", "openai_compatible"]
