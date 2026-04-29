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

# config: settings/env bootstrap. Single ``load`` event today;
# [IMPROVE-89] registered the subsystem during the bulk
# emit_typed migration since the pre-existing kwarg-shape emit
# in config.py used to fly under the keystone test's regex.
_CONFIG_ACTIONS = (
    "load",
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
# ``infer.start``/``load.start`` are literal-string companions to
# ``infer``/``load`` end-events — they fire from explicit
# ``emit("image", "infer.start", ...)`` calls (NOT from the
# Recorder class's f-string ``f"{action}.start"`` shape, which is
# dynamic-action and ignored by the keystone). [IMPROVE-89]
# registered them after tightening the keystone regex revealed
# they'd been firing without coverage.
_IMAGE_ACTIONS = (
    "infer",
    "infer.start",
    "load",
    "load.start",
    "oom_ladder_done",
    "oom_ladder_start",
    "oom_stage_attempt",
    "optimization_plan",
    "plan",
    "postprocess",
    # [IMPROVE-87] Pre-flight VRAM probe for the diffusers-based
    # upscalers (latent / sdxl_x4). The probe fires once per
    # ``upscale_image`` call when the method is one of the
    # diffusers options. Carries method + available_gb + required_gb
    # + reason so dashboards can chart "% of upscale calls that
    # pre-flight rejected the diffusers path" without grepping
    # logs. Mirror of the IMPROVE-79 commit's spawned-followup #2.
    "vram_probe",
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
# spinner). ``run.start`` is the literal-string companion to
# ``run``; [IMPROVE-89] surfaced it during keystone-regex
# tightening.
_INSTRUCT_EDIT_ACTIONS = (
    "load",
    "run",
    "run.start",
)

# model: HF download lifecycle (snapshot + hf_hub_download).
# Actions use a dotted shape (``download.start`` etc.) because
# the model.* namespace previously used dotted action names that
# the pre-[IMPROVE-89] keystone regex couldn't match — the
# events fired but were invisible to coverage. Registering them
# here closes that gap; the regex was tightened in the same
# commit to allow ``[a-z_]+(?:\.[a-z_]+)*`` action names so
# future dotted actions are pinned too.
_MODEL_ACTIONS = (
    "download.done",
    "download.error",
    "download.progress",
    "download.start",
)

# partner: voice/persona partner. ``chat.start`` /
# ``voice_init.start`` are literal-string companions to the
# corresponding end events; ``stt.partial`` is a separate
# mid-stream event for STT chunk-level results; ``mem0_init``
# fires on Mem0/ChromaDB cold-start. [IMPROVE-89] registered
# all four after keystone-regex tightening revealed they'd been
# firing without coverage (digit in ``mem0_init`` and dotted
# names in the ``.start``/``.partial`` shapes both escaped the
# old regex).
_PARTNER_ACTIONS = (
    "chat",
    "chat.start",
    "emotion_detect",
    "fact_extract",
    "mem0_init",
    "stt",
    "stt.partial",
    "tts",
    "voice_init",
    "voice_init.start",
)

# provider: LLM provider availability + routing.
_PROVIDER_ACTIONS = (
    "availability_probe",
)

# system: DAG executor lifecycle. Wave 5/6 added
# routing_decision (IMPROVE-35 SSE), wave_parallel +
# wave_parallel_fallback (IMPROVE-36 telemetry), validate
# (IMPROVE-31). ``run.start`` is the literal-string companion to
# ``run_done``; [IMPROVE-89] registered it after the keystone
# regex tightening surfaced the gap.
_SYSTEM_ACTIONS = (
    "node_end",
    "node_start",
    "routing_decision",
    "run.start",
    "run_done",
    "validate",
    # [IMPROVE-85] Mirror of [IMPROVE-82]'s
    # ``agent.validation_rejected``. Pre-IMPROVE-85 the
    # /systems/* boundary fired ``system.validate`` with
    # ``status="error"`` for both Pydantic schema-invalid and
    # Kahn-cycle-detected rejections — that conflates "user
    # posted bad JSON" with "validation completed OK" because
    # both share the event name. Splitting the rejection case
    # off into its own action lets dashboards chart 400-rate
    # separately from total-system-runs without a SQL filter.
    "validation_rejected",
    "wave_parallel",
    "wave_parallel_fallback",
)

# tool: built-in tool execution. ``file_ops.path_rejected`` was
# registered by [IMPROVE-89] — tools/file_ops.py and
# tools/rag_tools.py both fire it from their workspace-sandbox
# rejection paths, but the dotted action name didn't match the
# pre-[IMPROVE-89] keystone regex.
_TOOL_ACTIONS = (
    "calculator_eval",
    "file_ops.path_rejected",
)


# ── Frozen registry ────────────────────────────────────────────

# Per-subsystem map. Frozen at import; the registry doesn't grow
# at runtime — every new event must be registered here as part
# of its commit so the keystone test passes.
KNOWN_EVENTS: dict[str, frozenset[str]] = {
    "agent": frozenset(_AGENT_ACTIONS),
    "config": frozenset(_CONFIG_ACTIONS),
    "editor": frozenset(_EDITOR_ACTIONS),
    "image": frozenset(_IMAGE_ACTIONS),
    "images": frozenset(_IMAGES_ACTIONS),
    "instruct_edit": frozenset(_INSTRUCT_EDIT_ACTIONS),
    "model": frozenset(_MODEL_ACTIONS),
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
    "config",
    "editor",
    "image",
    "images",
    "instruct_edit",
    "model",
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
