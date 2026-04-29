"""[IMPROVE-NEW-16] Telemetry event-name registry.

Wave 6 added 8 new observability event types (``wave_parallel``,
``wave_parallel_fallback``, ``optimization_plan``,
``oom_ladder_start`` / ``oom_ladder_done`` / ``oom_stage_attempt``,
``routing_decision`` plus its SSE variant, enhancer fallback
context). The codebase now has 36 unique ``(subsystem, action)``
pairs in ``emit()`` calls, all bare strings. A typo in either
field produces a row in ``app_events`` with a misspelled name —
never caught at lint time, only by an operator noticing the
wrong name in a dashboard query weeks later.

This module is the single source of truth for the registered
event names. Two surfaces:

  * ``KNOWN_EVENT_NAMES`` — flat ``frozenset[str]`` of
    ``"subsystem.action"`` strings. Tests assert their callsites
    use registered events; the keystone test
    ``test_registry_covers_every_emit_callsite_in_codebase``
    walks the source tree and fails CI on any unregistered emit.

  * ``emit_typed(subsystem, action, ...)`` — typed front door.
    ``subsystem`` is a ``Literal[...]`` so mypy catches misspelled
    subsystem strings at lint time; ``action`` is validated at
    runtime against the per-subsystem registry. Existing
    ``emit()`` callsites keep working unchanged so this rolls in
    incrementally — new code uses ``emit_typed``, old code
    migrates opportunistically.

Why a separate module rather than baking into observability.py:
``observability.py`` is imported very early (the config bootstrap
uses it). Keeping the registry in its own file means future
growth — per-event schemas, deprecation markers, OTel attribute
mappings — can land here without thrashing the hot
``observability.emit`` import path.

Why per-subsystem ``frozenset`` rather than one big ``Literal``
of all action names: actions overlap in name across subsystems
(``"load"`` exists for ``image`` AND ``instruct_edit``; they're
different events with different context shapes). A flat union
would lose that distinction, and a per-subsystem ``Literal`` +
``overload`` set would add a lot of typing machinery for marginal
benefit. The runtime check in ``emit_typed`` catches the
misspelled-action case at first call.

Sources (2025-2026):
  * docs/features/09-observability.md — internal observability
    surface this module formalises.
  * Wave 6 IMPROVE-36/40/44/35 telemetry commits — the 8 new
    event names that motivated this registry.
  * OpenTelemetry semantic conventions for GenAI (otel-genai
    2025-12 release): events similarly registered as constants
    in ``opentelemetry.semconv.gen_ai.events``.
"""
from __future__ import annotations

from typing import Any, Literal

from .observability import emit as _emit


# ── Per-subsystem name catalogues ──────────────────────────────

# Each tuple is sorted for stable diff in code review when a new
# event lands. The grouping comments document the subsystem.

# agent: orchestrator-level lifecycle (tool calls, fallbacks,
# auto-resume after dangerous-tool interrupts) plus
# [IMPROVE-NEW-18] /agents/* validation rejections so operators can
# see "30% of agent creates fail with invalid_tool" without log
# scraping.
_AGENT_ACTIONS = (
    "fallback",
    "protected_delete_blocked",
    "tool_auto_resume",
    "tool_call",
    "tool_result",
    "validation_rejected",
)

# editor: image-editor session ops (apply_edit, undo, redo,
# blend, export). Wave 5 [IMPROVE-52/53/56/57] all wrote here.
_EDITOR_ACTIONS = (
    "blend_with_previous",
    "export",
    "op",
    "redo",
    "undo",
)

# image: image generation pipeline. Wave 5/6 added several
# observability events (oom_ladder_*, optimization_plan).
_IMAGE_ACTIONS = (
    "infer",
    "load",
    "oom_ladder_done",
    "oom_ladder_start",
    "oom_stage_attempt",
    "optimization_plan",
    "plan",
    "postprocess",
    "warmup",
)

# images (plural — one historical inconsistency from
# [IMPROVE-39] / [IMPROVE-47] detection telemetry). Don't
# normalise here; the event-name drift is real and a future
# cleanup would touch every consumer query/dashboard.
_IMAGES_ACTIONS = (
    "detect_hints",
)

# instruct_edit: editor model-load / inference pipeline
# (separate subsystem from "editor" because the load is
# expensive and surfaced separately for the user-visible
# spinner).
_INSTRUCT_EDIT_ACTIONS = (
    "load",
    "run",
)

# partner: voice/persona partner.
_PARTNER_ACTIONS = (
    "chat",
    "emotion_detect",
    "fact_extract",
    "stt",
    "tts",
    "voice_init",
)

# provider: LLM provider availability + routing.
_PROVIDER_ACTIONS = (
    "availability_probe",
)

# system: DAG executor lifecycle. Wave 5/6 added
# routing_decision (IMPROVE-35 SSE), wave_parallel +
# wave_parallel_fallback (IMPROVE-36 telemetry), validate
# (IMPROVE-31).
_SYSTEM_ACTIONS = (
    "node_end",
    "node_start",
    "routing_decision",
    "run_done",
    "validate",
    "wave_parallel",
    "wave_parallel_fallback",
)

# tool: built-in tool execution.
_TOOL_ACTIONS = (
    "calculator_eval",
)


# ── Frozen registry ────────────────────────────────────────────

# Per-subsystem map. Frozen at import; the registry doesn't grow
# at runtime — every new event must be registered here as part
# of its commit so the keystone test passes.
KNOWN_EVENTS: dict[str, frozenset[str]] = {
    "agent": frozenset(_AGENT_ACTIONS),
    "editor": frozenset(_EDITOR_ACTIONS),
    "image": frozenset(_IMAGE_ACTIONS),
    "images": frozenset(_IMAGES_ACTIONS),
    "instruct_edit": frozenset(_INSTRUCT_EDIT_ACTIONS),
    "partner": frozenset(_PARTNER_ACTIONS),
    "provider": frozenset(_PROVIDER_ACTIONS),
    "system": frozenset(_SYSTEM_ACTIONS),
    "tool": frozenset(_TOOL_ACTIONS),
}

# Flat "subsystem.action" set for fast membership lookup. Used by
# ``emit_typed`` and by the keystone callsite-coverage test.
KNOWN_EVENT_NAMES: frozenset[str] = frozenset(
    f"{subsystem}.{action}"
    for subsystem, actions in KNOWN_EVENTS.items()
    for action in actions
)


# ── Literal type alias ─────────────────────────────────────────

# ``Literal`` of registered subsystem names. mypy catches a typo
# in the subsystem arg of ``emit_typed`` at lint time. The
# action arg is intentionally typed as plain ``str`` because per-
# subsystem Literals would require ``overload`` + a redeclaration
# per subsystem — significant typing machinery for marginal
# payoff (the runtime check in ``emit_typed`` catches the
# misspelled-action case at first call).
SubsystemName = Literal[
    "agent",
    "editor",
    "image",
    "images",
    "instruct_edit",
    "partner",
    "provider",
    "system",
    "tool",
]


# ── emit_typed wrapper ─────────────────────────────────────────


class UnknownEventNameError(ValueError):
    """Raised by ``emit_typed`` when ``(subsystem, action)`` is
    not in ``KNOWN_EVENT_NAMES``. Distinct from plain
    ``ValueError`` so test assertions can target this specifically
    without false positives from unrelated value validation
    elsewhere in the call stack.
    """


def emit_typed(
    subsystem: SubsystemName,
    action: str,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None:
    """[IMPROVE-NEW-16] Type-checked + runtime-validated wrapper
    around ``observability.emit``.

    Why:
      * mypy catches misspelled subsystem strings at lint time
        via the ``SubsystemName`` Literal.
      * Runtime validation catches misspelled action strings — a
        per-subsystem Literal would need ``overload``, too much
        machinery for now. An unknown name raises
        ``UnknownEventNameError`` so a CI grep-and-test catches it
        before the wrong event lands in production app_events.
      * Behaviour-equivalent to ``emit`` for every kwarg the
        wrapped function accepts. Use ``emit_typed`` for new
        callsites; existing ``emit`` callsites can migrate
        opportunistically — the keystone test
        ``test_registry_covers_every_emit_callsite_in_codebase``
        ensures every event name is registered regardless of
        which front door called it.

    Args mirror ``observability.emit`` exactly so this is a
    drop-in replacement at the call site.
    """
    full_name = f"{subsystem}.{action}"
    if full_name not in KNOWN_EVENT_NAMES:
        known = sorted(KNOWN_EVENTS.get(subsystem, frozenset()))
        raise UnknownEventNameError(
            f"Unknown event {full_name!r}. Register it in "
            f"observability_events.py before emitting. Known "
            f"actions for subsystem {subsystem!r}: {known or '(none)'}"
        )
    _emit(
        subsystem,
        action,
        status,
        duration_ms=duration_ms,
        error_code=error_code,
        error_message=error_message,
        context=context,
        perf=perf,
    )
