"""OpenTelemetry foundation — tracer factory + lifespan hooks.

[IMPROVE-4] Commit 1/4. Sets up an ``opentelemetry-sdk`` ``TracerProvider``
that subsequent commits will hang ``gen_ai.*`` semantic-convention spans
from. This module deliberately stops short of emitting any spans itself
— that lives in commits 2/4 (chat path), 3/4 (tool calls + agents),
4/4 (image gen / edit). Today it only owns:

* ``init_otel(service_name)`` — idempotent SDK setup. Reads
  ``OTEL_EXPORTER`` to pick an exporter (default ``none`` → register a
  ``TracerProvider`` with no ``SpanProcessor`` so spans are created and
  immediately discarded; zero overhead, zero network). Sets
  ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`` if the
  caller hasn't already, per the spec's experimental-period guidance.

* ``get_tracer()`` — module tracer for whichever caller wants to start
  a span. Always safe to call (returns a no-op tracer if ``init_otel``
  hasn't run yet, matching the SDK's lazy-init behaviour).

* ``shutdown_otel()`` — flushes pending spans and tears down processors.
  Wired into the FastAPI lifespan teardown alongside
  ``http_client.aclose_clients`` so a clean shutdown doesn't leave
  background batcher threads alive.

The "default no-op" choice is the load-bearing one. This is a desktop
platform — most installs don't run an observability stack, so a pre-wired
exporter that fails to reach localhost:4317 every minute would be pure
noise. ``OTEL_EXPORTER=console`` is the dev mode (prints spans to stderr);
``OTEL_EXPORTER=otlp`` + the standard ``OTEL_EXPORTER_OTLP_ENDPOINT``
env var is the production opt-in.

Sources (2025–2026):
- https://opentelemetry.io/docs/specs/semconv/gen-ai/
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- https://www.datadoghq.com/blog/llm-otel-semantic-convention/ (2025)
- https://oneuptime.com/blog/post/2026-02-06-monitor-llm-opentelemetry-genai-semantic-conventions/view (2026-02-06)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)


# Module state. Kept private so callers go through the helpers — that
# keeps the SDK's "set the global TracerProvider exactly once" rule in
# this file rather than smeared across the lifespan + tests.
_provider: Optional[TracerProvider] = None
_initialized: bool = False


# Tracer name used for spans created by this codebase. Kept as a single
# constant so a future split (one tracer per subsystem) only touches one
# spot and doesn't drift between call sites.
_TRACER_NAME = "local_ai_platform"


def _select_exporter(mode: str) -> Optional[BatchSpanProcessor]:
    """Resolve ``OTEL_EXPORTER`` to a configured ``BatchSpanProcessor``.

    Returns ``None`` for ``"none"`` (the default — TracerProvider stays
    bare, spans are created and discarded). The caller is responsible
    for ``add_span_processor()`` if a non-None value comes back.
    """
    mode = (mode or "none").strip().lower()
    if mode == "none":
        # No exporter — spans are created (so context propagation still
        # works for code that reads them in tests via InMemorySpanExporter)
        # but never serialized. This is the desktop default.
        return None
    if mode == "console":
        # Useful for `OTEL_EXPORTER=console uvicorn api_server:app` —
        # spans go to stderr in JSON-ish form, no infra required.
        return BatchSpanProcessor(ConsoleSpanExporter())
    if mode == "otlp":
        # Standard production path. Endpoint comes from the OTel-defined
        # OTEL_EXPORTER_OTLP_ENDPOINT (e.g. http://collector:4317), which
        # the OTLPSpanExporter constructor reads if we don't pass it
        # explicitly. We import lazily so a non-OTLP install still works
        # (the gRPC exporter pulls grpcio, which is heavy and platform-
        # specific — fine because chromadb already pulls it transitively
        # but we want a clean ImportError rather than a surprise).
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as exc:
            logger.warning(
                "OTEL_EXPORTER=otlp but opentelemetry-exporter-otlp-proto-grpc "
                "isn't installed: %s. Falling back to no-op (default).", exc,
            )
            return None
        return BatchSpanProcessor(OTLPSpanExporter())
    logger.warning(
        "Unknown OTEL_EXPORTER=%r — falling back to no-op (default). "
        "Valid values: none, console, otlp.", mode,
    )
    return None


def init_otel(service_name: str = "local-ai-platform") -> TracerProvider:
    """Idempotently set up the global ``TracerProvider``.

    Safe to call multiple times — subsequent calls return the existing
    provider untouched (the SDK's ``set_tracer_provider`` warns if you
    try to replace, so we don't). Returns the provider so a caller that
    wants to install a custom processor (e.g. an ``InMemorySpanExporter``
    in tests) can do so against the same instance.

    The semconv opt-in env is set here rather than in code that reads
    the gen_ai_* attributes because the SDK reads it during span export,
    not during attribute construction — putting it next to the SDK
    bootstrap keeps the cause + effect adjacent.
    """
    global _provider, _initialized
    if _initialized and _provider is not None:
        return _provider

    # Per the spec: setting this env var enables emitting the
    # ``gen_ai.*`` attributes during the experimental period. Don't
    # clobber if the caller already set it (they may want a different
    # opt-in level once the spec stabilises).
    os.environ.setdefault(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental",
    )

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    processor = _select_exporter(os.environ.get("OTEL_EXPORTER", "none"))
    if processor is not None:
        provider.add_span_processor(processor)
        logger.info(
            "[OTEL] Tracing enabled — exporter=%s service=%s",
            os.environ.get("OTEL_EXPORTER", "none").lower(), service_name,
        )
    else:
        # Don't log at INFO for the default — would be noise on every
        # process boot. DEBUG is enough for the curious operator.
        logger.debug(
            "[OTEL] Tracing initialized in no-op mode "
            "(set OTEL_EXPORTER=console|otlp to activate).",
        )

    trace.set_tracer_provider(provider)
    _provider = provider
    _initialized = True
    return provider


def get_tracer() -> trace.Tracer:
    """Return the module tracer.

    Always safe — if ``init_otel`` hasn't run, the SDK returns a NoOp
    tracer that swallows all span calls. This means call sites in
    Commits 2-4/4 don't need a guard before ``with tracer.start_as_current_span``.
    """
    return trace.get_tracer(_TRACER_NAME)


def shutdown_otel() -> None:
    """Flush + tear down. Idempotent.

    Called from the FastAPI lifespan teardown. The ``BatchSpanProcessor``
    runs a daemon thread that batches spans every ~5s; without this call
    you'd see "Event loop is closed" warnings on uvicorn shutdown
    (same shape as the [IMPROVE-7] httpx teardown we already wire).
    """
    global _provider, _initialized
    if _provider is None:
        return
    try:
        _provider.shutdown()
    except Exception as exc:
        # ``shutdown`` shouldn't raise but a custom processor might —
        # we never want to mask the real shutdown error from lifespan.
        logger.debug("[OTEL] shutdown raised %s", exc)
    finally:
        _provider = None
        _initialized = False


def _reset_for_tests() -> None:
    """Drop module state so a test can re-init with a different exporter.

    ``trace.set_tracer_provider`` only fires once per process — once set,
    a second call logs a warning and is ignored. Tests that need to swap
    providers mid-run reach into the trace module's globals; this helper
    drops *our* mirror state so subsequent ``init_otel`` calls actually
    run their setup again. Not for production use.
    """
    global _provider, _initialized
    _provider = None
    _initialized = False
