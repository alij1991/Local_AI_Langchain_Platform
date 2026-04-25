"""Foundation tests for the [IMPROVE-4] OTel GenAI semconv migration.

This file pins the *bootstrap* contract — exporter selection, idempotency,
semconv opt-in, lifespan teardown. Spans aren't emitted yet (Commits
2-4/4 wire the call sites); these tests cover the "the SDK is wired
correctly" half of the work so a regression in init / shutdown shape
fails loudly here rather than silently breaking spans further out.

Sources (2025-2026):
- https://opentelemetry.io/docs/specs/semconv/gen-ai/
- https://oneuptime.com/blog/post/2026-02-06-monitor-llm-opentelemetry-genai-semantic-conventions/view
"""
from __future__ import annotations

import os

import pytest

from local_ai_platform import otel as otel_module


@pytest.fixture(autouse=True)
def _reset_otel():
    """Each test gets a clean module state.

    ``trace.set_tracer_provider`` only fires once per process (the SDK
    logs a warning and ignores subsequent calls). Tests that need a
    fresh ``init_otel`` call must reset our wrapper state AND also
    accept that the underlying global TracerProvider may already be
    set from a previous test — which is fine, ``init_otel`` returns
    the existing provider on second call.

    The env mutations we do in tests are scoped via ``monkeypatch.setenv``
    in the body of each test so they unwind cleanly.
    """
    otel_module._reset_for_tests()
    yield
    otel_module._reset_for_tests()


# ── exporter selection ──────────────────────────────────────────────


def test_default_mode_registers_provider_with_no_processors(monkeypatch):
    """Default ``OTEL_EXPORTER=none`` (or unset) — the TracerProvider
    is wired but has zero ``SpanProcessor`` instances attached. Spans
    are still constructed (so ``with tracer.start_as_current_span(...)``
    works and tests can install an InMemorySpanExporter on top), but
    they're discarded — zero overhead, zero network. This is the
    desktop default, the load-bearing decision called out in the
    proposal.
    """
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)

    provider = otel_module.init_otel("test-service")

    # SDK invariant: the active span processor is a multi-wrapper that
    # holds a tuple of registered processors. Empty tuple == no-op mode.
    multi = provider._active_span_processor
    assert multi._span_processors == (), (
        "Default mode must register ZERO SpanProcessors so spans get "
        "discarded with zero overhead. Got: %r" % (multi._span_processors,)
    )


def test_console_mode_attaches_console_exporter(monkeypatch):
    """``OTEL_EXPORTER=console`` wires a BatchSpanProcessor backed by
    ConsoleSpanExporter. Useful for ``OTEL_EXPORTER=console uvicorn ...``
    during dev — spans go to stderr, no infra required.
    """
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    monkeypatch.setenv("OTEL_EXPORTER", "console")

    provider = otel_module.init_otel("test-service")

    multi = provider._active_span_processor
    procs = multi._span_processors
    assert len(procs) == 1
    assert isinstance(procs[0], BatchSpanProcessor)
    # The exporter is private but stable in 1.40 — pin the type so a
    # regression that swaps to SimpleSpanExporter or similar fails here.
    assert isinstance(procs[0].span_exporter, ConsoleSpanExporter)


def test_otlp_mode_attaches_otlp_exporter(monkeypatch):
    """``OTEL_EXPORTER=otlp`` wires the gRPC OTLP exporter. We don't
    test the actual network call — that's chromadb's transitive
    dependency contract — but pin that the right exporter type lands.
    """
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    monkeypatch.setenv("OTEL_EXPORTER", "otlp")
    # Standard OTel env — the OTLPSpanExporter constructor reads it.
    # Set to a clearly-fake value so a stray network call fails fast
    # if a future refactor breaks the no-network invariant.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-test.invalid:4317")

    provider = otel_module.init_otel("test-service")

    multi = provider._active_span_processor
    procs = multi._span_processors
    assert len(procs) == 1
    assert isinstance(procs[0], BatchSpanProcessor)
    assert isinstance(procs[0].span_exporter, OTLPSpanExporter)


def test_unknown_mode_falls_back_to_no_op_with_warning(monkeypatch, caplog):
    """A typo'd ``OTEL_EXPORTER=jaeger`` (we don't support that)
    must not crash boot — fall back to no-op and log a warning so the
    operator can grep the warning out of the logs.
    """
    import logging

    monkeypatch.setenv("OTEL_EXPORTER", "jaeger")
    caplog.set_level(logging.WARNING, logger="local_ai_platform.otel")

    provider = otel_module.init_otel("test-service")

    multi = provider._active_span_processor
    assert multi._span_processors == ()
    assert any(
        "Unknown OTEL_EXPORTER" in r.getMessage() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


# ── semconv opt-in ──────────────────────────────────────────────────


def test_semconv_opt_in_set_when_unset(monkeypatch):
    """Per the spec, ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental``
    is required during the experimental period so the SDK emits the
    ``gen_ai.*`` attributes. ``init_otel`` sets it on our behalf if
    the caller hasn't.
    """
    monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)

    otel_module.init_otel("test-service")

    assert os.environ.get("OTEL_SEMCONV_STABILITY_OPT_IN") == "gen_ai_latest_experimental"


def test_semconv_opt_in_not_clobbered_when_user_set(monkeypatch):
    """If the caller has already opted into a specific semconv level
    (e.g. they want to test the future stable shape), ``init_otel`` must
    leave it alone. Uses ``setdefault``, not ``[]=``.
    """
    monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "user_chosen_value")

    otel_module.init_otel("test-service")

    assert os.environ.get("OTEL_SEMCONV_STABILITY_OPT_IN") == "user_chosen_value"


# ── idempotency ─────────────────────────────────────────────────────


def test_init_otel_is_idempotent(monkeypatch):
    """Lifespan + tests both call ``init_otel`` — second call must
    return the same provider, not raise, not re-register processors.
    The SDK warns on duplicate ``set_tracer_provider`` calls; this test
    exists so a refactor that drops the early-return guard fails here
    instead of polluting the test log.
    """
    monkeypatch.setenv("OTEL_EXPORTER", "console")

    p1 = otel_module.init_otel("test-service")
    p2 = otel_module.init_otel("test-service")

    assert p1 is p2
    # Still exactly one processor — guard against "re-init double-wires
    # the exporter" regressions.
    assert len(p1._active_span_processor._span_processors) == 1


def test_shutdown_otel_clears_state_and_is_idempotent(monkeypatch):
    """Lifespan teardown calls ``shutdown_otel`` — and pytest reruns
    the lifespan via TestClient many times, so it must be safe to call
    twice in a row. Pre-init shutdown must be a no-op (defensive).
    """
    # Pre-init shutdown — must not raise.
    otel_module.shutdown_otel()

    monkeypatch.setenv("OTEL_EXPORTER", "console")
    otel_module.init_otel("test-service")
    assert otel_module._provider is not None

    otel_module.shutdown_otel()
    assert otel_module._provider is None
    assert otel_module._initialized is False

    # Second shutdown — no exception.
    otel_module.shutdown_otel()


# ── module-level singleton invariant ────────────────────────────────


def test_api_server_does_not_expose_otel_provider_at_module_scope(monkeypatch):
    """Mirrors ``test_api_server_has_no_stateful_singletons``: the OTel
    provider is a process-singleton inside the ``otel`` module, plus
    aliased onto ``app.state.otel_tracer_provider`` for any router that
    wants to introspect it. It must NOT live as a module attribute on
    ``api_server`` itself — that path was the source of every stale-None
    bug [IMPROVE-5] Commit 3 cleaned up.
    """
    import api_server

    forbidden = {
        "otel_tracer_provider", "tracer_provider", "_otel_provider",
    }
    present = forbidden & set(vars(api_server).keys())
    assert not present, (
        "api_server must not expose OTel state at module scope: "
        f"{sorted(present)}. Use the app.state.otel_tracer_provider "
        "alias from the lifespan instead."
    )


# ── tracer factory ──────────────────────────────────────────────────


def test_get_tracer_returns_tracer_after_init(monkeypatch):
    """``get_tracer()`` returns a tracer whose ``start_as_current_span``
    is callable — the contract Commits 2-4/4 lean on. Pre-init it
    returns the SDK's NoOp tracer (verified by call shape, not type
    name — the SDK rebrands NoOpTracer between versions).
    """
    monkeypatch.setenv("OTEL_EXPORTER", "none")
    otel_module.init_otel("test-service")

    tracer = otel_module.get_tracer()
    with tracer.start_as_current_span("foo") as span:
        span.set_attribute("gen_ai.system", "test")
    # No assertion needed — if the tracer didn't honor the context
    # protocol, the ``with`` block would have raised. The point of
    # this test is that the call doesn't blow up when the no-op
    # exporter is wired and the span is set+discarded immediately.


def test_get_tracer_works_before_init():
    """The SDK's lazy global means ``trace.get_tracer`` always returns
    *something* — without ``init_otel``, that something is a NoOp
    tracer. Code in Commits 2-4/4 calls ``get_tracer()`` at module
    import time (before lifespan); this test pins the no-init-required
    contract.
    """
    # Don't call init_otel here — _reset_for_tests cleared our state.
    tracer = otel_module.get_tracer()
    with tracer.start_as_current_span("foo"):
        pass
