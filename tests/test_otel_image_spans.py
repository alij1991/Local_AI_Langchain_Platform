"""[IMPROVE-4] Commit 4/4: image generation + image edit gen_ai spans.

Pin the dual-emit contract for image-producing operations:
- ``track_event("image", "generate", ...)`` produces a
  ``gen_ai.image_generation`` span with ``gen_ai.output.type="image"``,
  ``gen_ai.system``, ``gen_ai.request.model`` (from ``model_id`` in
  context), and any caller-supplied ``gen_ai.usage.output_images``.
- ``track_event("editor", "edit", ...)`` produces a ``gen_ai.image_edit``
  span with the same auto-attached attributes.
- ``track_event("image", "enhance_prompt", ...)`` is a chat completion
  under the hood (the model rewrites the user's prompt) — it gets a
  ``gen_ai.chat`` span, not an image one.

Per the spec, image-producing operations use ``gen_ai.output.type``
(valid values: text|json|image|speech). ``__enter__`` auto-attaches
"image" for image_generation / image_edit ops so individual call sites
don't repeat the boilerplate.

Sources (2025-2026):
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- https://oneuptime.com/blog/post/2026-02-06-monitor-llm-opentelemetry-genai-semantic-conventions/view (2026-02-06)
"""
from __future__ import annotations

import pytest
from opentelemetry import trace as _ot_trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from local_ai_platform import otel as otel_module
from local_ai_platform.observability import track_event


@pytest.fixture
def memory_exporter(monkeypatch):
    """Same shape as tests/test_otel_chat_spans.py — see that file
    for fixture rationale (set_tracer_provider is once-per-process).
    """
    monkeypatch.setenv("OTEL_EXPORTER", "none")
    otel_module.init_otel("test-service")
    provider = _ot_trace.get_tracer_provider()

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    yield exporter

    processor.shutdown()


# ── image_generation span ───────────────────────────────────────────


def test_image_generate_emits_image_generation_span(memory_exporter):
    """A ``track_event("image", "generate", ...)`` block produces one
    span with operation.name=image_generation and the standard
    image attributes.
    """
    with track_event(
        "image", "generate",
        context={
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "prompt_length": 64,
            "steps": 20,
            "width": 1024,
            "height": 1024,
        },
    ) as ev:
        ev.set_otel_attributes({
            "gen_ai.system": "diffusers",
            "gen_ai.usage.output_images": 1,
        })

    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.operation.name"] == "image_generation"
    # model_id (image-gen convention) is auto-mirrored to gen_ai.request.model.
    assert span.attributes["gen_ai.request.model"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert span.attributes["gen_ai.system"] == "diffusers"
    assert span.attributes["gen_ai.usage.output_images"] == 1


def test_image_generation_span_auto_sets_output_type(memory_exporter):
    """Image-producing operations get ``gen_ai.output.type="image"``
    automatically — the spec valid values are text|json|image|speech
    and image is the obvious default for image_generation /
    image_edit. Pin so a future refactor that drops the auto-set
    fails this test, not silently downstream where dashboards filter
    by output type.
    """
    with track_event(
        "image", "generate",
        context={"model_id": "flux1-schnell"},
    ):
        pass

    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.output.type"] == "image"


def test_image_generate_span_name_includes_model(memory_exporter):
    """Span name format follows the spec: "{operation} {model}".
    Image gen uses model_id, which __enter__ pulls into
    gen_ai.request.model — __exit__ then composes the span name.
    """
    with track_event(
        "image", "generate",
        context={"model_id": "flux-dev"},
    ):
        pass

    span = memory_exporter.get_finished_spans()[0]
    assert span.name == "image_generation flux-dev"


def test_image_generate_failure_marks_span_error(memory_exporter):
    """A failure inside the generate block (e.g. CUDA OOM) propagates
    exactly as before — span ends with status=ERROR and a single
    exception event recorded.
    """
    with pytest.raises(RuntimeError, match="OOM"):
        with track_event(
            "image", "generate",
            context={"model_id": "flux-dev", "steps": 50},
        ):
            raise RuntimeError("CUDA OOM during inference")

    span = memory_exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    exc_events = [e for e in span.events if e.name == "exception"]
    assert len(exc_events) == 1


# ── image_edit span ─────────────────────────────────────────────────


def test_editor_edit_emits_image_edit_span(memory_exporter):
    """``track_event("editor", "edit", ...)`` produces a span with
    operation.name=image_edit and gen_ai.output.type="image".
    """
    with track_event(
        "editor", "edit",
        context={
            "session_id": "sess-abc",
            "operation": "remove_background",
            "param_count": 2,
        },
    ) as ev:
        ev.set_otel_attributes({
            "editor.operation": "remove_background",
            "gen_ai.system": "diffusers",
        })

    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.operation.name"] == "image_edit"
    assert span.attributes["gen_ai.output.type"] == "image"
    assert span.attributes["editor.operation"] == "remove_background"


def test_editor_edit_no_model_in_context(memory_exporter):
    """Editor ops don't always have a model attached (analyze /
    classical CV ops run without one). Pin that the span still
    emits cleanly with no gen_ai.request.model attribute.
    """
    with track_event(
        "editor", "edit",
        context={"session_id": "sess-xyz", "operation": "denoise"},
    ):
        pass

    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.operation.name"] == "image_edit"
    assert "gen_ai.request.model" not in span.attributes


def test_editor_edit_span_name_falls_back_to_operation(memory_exporter):
    """Without a gen_ai.request.model attribute the span name stays
    as the bare operation per the existing __exit__ logic.
    """
    with track_event(
        "editor", "edit",
        context={"session_id": "s", "operation": "rotate"},
    ):
        pass

    span = memory_exporter.get_finished_spans()[0]
    assert span.name == "image_edit"


# ── image enhance_prompt is a chat span, not image ──────────────────


def test_image_enhance_prompt_is_chat_operation_not_image(memory_exporter):
    """``image/enhance_prompt`` is a chat completion (the model rewrites
    the user's image prompt). Per the spec, the gen_ai.operation.name
    must be "chat" — NOT "image_generation" — because the OTel side
    cares about the actual operation shape (it's a text-out chat call,
    not an image-out generation). Two separate dashboards, two separate
    operations.
    """
    with track_event(
        "image", "enhance_prompt",
        context={
            "model_hint": "qwen3:8b",
            "prompt_length": 50,
            "model_family": "sdxl",
        },
    ) as ev:
        ev.set_otel_attributes({"gen_ai.request.model": "qwen3:8b"})

    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.operation.name"] == "chat"
    # Crucially NOT "image_generation"
    assert span.attributes["gen_ai.operation.name"] != "image_generation"
    # And NO image output type — this is a text-out operation.
    assert "gen_ai.output.type" not in span.attributes


# ── unmapped image actions stay unmapped ────────────────────────────


def test_image_load_event_still_emits_no_span(memory_exporter):
    """``emit("image", "load", ...)`` is the per-stage marker inside
    images/service.py for pipeline loads — it's an internal stage,
    NOT a request-level operation, so it must not produce its own
    gen_ai span. The map only contains image/generate at the request
    level. This guards against a "while we're at it, let's map every
    image action" regression.
    """
    with track_event("image", "load", context={"model_id": "flux"}):
        pass

    assert memory_exporter.get_finished_spans() == ()


def test_image_postprocess_event_still_emits_no_span(memory_exporter):
    """Same regression guard for postprocess — it's a sub-stage of
    a generation, captured as an app_events row, not as a separate
    gen_ai span. [IMPROVE-68] is the right place to wire stage spans
    as children of the request span.
    """
    with track_event("image", "postprocess", context={}):
        pass

    assert memory_exporter.get_finished_spans() == ()


def test_editor_undo_redo_export_emit_no_span(memory_exporter):
    """Editor's history operations (undo / redo / export) are not
    image_edit operations per the spec — they're session-management
    actions. Pin that the operation map only matches editor/edit
    and leaves the others as plain app_events rows.
    """
    with track_event("editor", "undo", context={"session_id": "s"}):
        pass
    with track_event("editor", "redo", context={"session_id": "s"}):
        pass
    with track_event("editor", "export", context={"session_id": "s"}):
        pass

    assert memory_exporter.get_finished_spans() == ()


# ── nesting under parent span ───────────────────────────────────────


def test_image_generate_span_nests_under_parent_chat_span(memory_exporter):
    """If a chat agent invokes image generation as a tool (rare today
    but plausible — see image_tools), the image_generation span
    inherits the chat span as parent automatically via OTel context
    propagation. Pin so a regression that drops start_as_current_span
    in favour of start_span fails here.
    """
    with track_event(
        "chat", "send",
        context={"agent": "assistant", "model": "qwen3:8b", "provider": "ollama"},
    ):
        with track_event("image", "generate", context={"model_id": "flux"}):
            pass

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 2
    image_span, chat_span = spans
    assert image_span.attributes["gen_ai.operation.name"] == "image_generation"
    assert chat_span.attributes["gen_ai.operation.name"] == "chat"
    assert image_span.parent is not None
    assert image_span.parent.span_id == chat_span.context.span_id
