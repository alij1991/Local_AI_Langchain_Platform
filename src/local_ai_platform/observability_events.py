"""[IMPROVE-NEW-16] Telemetry event-name registry.

Wave 6 added 8 new observability event types (``wave_parallel``,
``wave_parallel_fallback``, ``optimization_plan``,
``oom_ladder_start`` / ``oom_ladder_done`` / ``oom_stage_attempt``,
``routing_decision`` plus its SSE variant, enhancer fallback
context). The codebase grew to 51 unique ``(subsystem, action)``
pairs after [IMPROVE-89]'s bulk migration. A typo in either
field used to produce a row in ``app_events`` with a misspelled
name — never caught at lint time, only by an operator noticing
the wrong name in a dashboard query weeks later.

This module is the single source of truth for the registered
event names. Three surfaces:

  * Per-subsystem ``Literal`` types (``AgentAction``,
    ``ImageAction``, etc.) — the source of truth that mypy
    pattern-matches against. Tuples + frozensets are DERIVED
    from the Literals via ``typing.get_args`` so the registry
    can never drift from the type aliases.

  * ``KNOWN_EVENT_NAMES`` — flat ``frozenset[str]`` of
    ``"subsystem.action"`` strings. Tests assert their callsites
    use registered events; the keystone test
    ``test_registry_covers_every_emit_callsite_in_codebase``
    walks the source tree and fails CI on any unregistered emit.

  * ``emit_typed(subsystem, action, ...)`` — typed front door
    with per-subsystem ``@overload`` signatures. mypy sees a
    distinct overload per subsystem so ``emit_typed("agent",
    "tool_calll", ...)`` (typo) fails at lint without ever
    running. Runtime validation in the impl is preserved as a
    safety net for dynamic callers (test fixtures, future
    metaprogramming).

Why a separate module rather than baking into observability.py:
``observability.py`` is imported very early (the config bootstrap
uses it). Keeping the registry in its own file means future
growth — per-event schemas (IMPROVE-92), deprecation markers,
OTel attribute mappings — can land here without thrashing the
hot ``observability.emit`` import path.

[IMPROVE-91] history note: pre-this-commit the action arg was
typed as plain ``str`` because per-subsystem Literals + overload
were rejected as "too much machinery for marginal benefit."
After Wave 8's four real-world ``emit_typed`` callsites
validated the pattern AND the [IMPROVE-89] keystone tightening
revealed three pre-existing coverage gaps (digit-bearing actions,
dotted actions, kwarg-shape calls), the cost-benefit flipped:
mypy catching action typos at lint is now worth the ~130 LoC of
overload signatures.

Sources (2025-2026):
  * docs/features/09-observability.md — internal observability
    surface this module formalises.
  * Wave 6 IMPROVE-36/40/44/35 telemetry commits — the 8 new
    event names that motivated this registry.
  * Wave 7 [IMPROVE-80] commit (25b851e) — ``emit_typed`` front
    door this commit extends with per-subsystem overloads.
  * Wave 9 [IMPROVE-89] commit — bulk migration that proved the
    pattern across 100 callsites.
  * OpenTelemetry semantic conventions for GenAI (otel-genai
    2025-12 release): events similarly registered as constants
    in ``opentelemetry.semconv.gen_ai.events``.
  * Python typing module + ``Literal`` + ``@overload`` patterns
    (PEP 586 + PEP 484, plus 2025 typing-extensions notes):
    https://docs.python.org/3/library/typing.html#typing.Literal
"""
# [IMPROVE-92] No ``from __future__ import annotations`` —
# deferred annotation strings break ``TypedDict.__required_keys__``
# / ``__optional_keys__`` introspection of ``NotRequired[...]``
# wrappers used by the per-event context schemas below. Python
# 3.11+ PEP 604 union syntax (``int | None``) and PEP 585
# generic-types (``dict[str, X]``) work in this module without
# the future import.

from typing import Any, Literal, get_args, overload

# [IMPROVE-92] ``typing_extensions.TypedDict`` is required for
# pydantic 2.x ``TypeAdapter`` support on Python < 3.12 (the
# stdlib version isn't fully introspectable). Importing here keeps
# the schema section below clean and centralised.
from typing_extensions import NotRequired, TypedDict

from pydantic import ConfigDict

from .observability import emit as _emit


# ── Per-subsystem Literal types ────────────────────────────────

# The Literal types are the SOURCE OF TRUTH. The action tuples +
# the ``KNOWN_EVENTS`` dict + ``KNOWN_EVENT_NAMES`` frozenset are
# derived. A new event lands here as a new entry in the relevant
# Literal; tests + downstream tuples + overloads pick it up
# automatically.

# agent: orchestrator-level lifecycle (tool calls, fallbacks,
# auto-resume after dangerous-tool interrupts) plus
# [IMPROVE-NEW-18] /agents/* validation rejections so operators can
# see "30% of agent creates fail with invalid_tool" without log
# scraping.
AgentAction = Literal[
    "fallback",
    "protected_delete_blocked",
    "tool_auto_resume",
    "tool_call",
    "tool_result",
    "validation_rejected",
]

# chat: LLM chat boundary events fired via ``track_event`` in
# api/routers/chat.py. ``send`` / ``send.start`` cover the main
# chat completion path; ``enhance_prompt`` / ``enhance_prompt.start``
# cover the prompt-enhancement preprocessing call. [IMPROVE-96]
# registered the subsystem after the Recorder enumeration AST
# walker surfaced these dynamic-action emits as unregistered
# (the keystone callsite test only catches literal-string
# emit/emit_typed calls; track_event's f-string-based emit chain
# fell under the regex's blind spot).
ChatAction = Literal[
    "enhance_prompt",
    "enhance_prompt.start",
    "send",
    "send.start",
]

# config: settings/env bootstrap. Single ``load`` event today;
# [IMPROVE-89] registered the subsystem during the bulk
# emit_typed migration since the pre-existing kwarg-shape emit
# in config.py used to fly under the keystone test's regex.
ConfigAction = Literal[
    "load",
]

# editor: image-editor session ops (apply_edit, undo, redo,
# blend, export). Wave 5 [IMPROVE-52/53/56/57] all wrote here.
# [IMPROVE-96] added ``edit`` / ``edit.start`` after the
# Recorder enumeration AST walker surfaced the
# ``track_event("editor", "edit", ...)`` callsite at
# api/routers/editor.py:432 as unregistered.
EditorAction = Literal[
    "blend_with_previous",
    "edit",
    "edit.start",
    "export",
    "op",
    "redo",
    "undo",
]

# image: image generation pipeline. Wave 5/6 added several
# observability events (oom_ladder_*, optimization_plan).
# ``infer.start``/``load.start`` are literal-string companions to
# ``infer``/``load`` end-events — they fire from explicit
# ``emit("image", "infer.start", ...)`` calls (NOT from the
# Recorder class's f-string ``f"{action}.start"`` shape, which is
# dynamic-action and ignored by the keystone). [IMPROVE-89]
# registered them after tightening the keystone regex revealed
# they'd been firing without coverage.
# [IMPROVE-87] Pre-flight VRAM probe for the diffusers-based
# upscalers (latent / sdxl_x4) carries method + available_gb +
# required_gb + reason so dashboards can chart "% of upscale
# calls that pre-flight rejected the diffusers path" without
# grepping logs.
ImageAction = Literal[
    "enhance_prompt",
    "enhance_prompt.start",
    "generate",
    "generate.start",
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
    "vram_probe",
    "warmup",
]

# images (plural — one historical inconsistency from
# [IMPROVE-39] / [IMPROVE-47] detection telemetry). Don't
# normalise here; the event-name drift is real and a future
# cleanup would touch every consumer query/dashboard.
ImagesAction = Literal[
    "detect_hints",
]

# instruct_edit: editor model-load / inference pipeline
# (separate subsystem from "editor" because the load is
# expensive and surfaced separately for the user-visible
# spinner). ``run.start`` is the literal-string companion to
# ``run``; [IMPROVE-89] surfaced it during keystone-regex
# tightening.
InstructEditAction = Literal[
    "load",
    "run",
    "run.start",
]

# model: HF download lifecycle (snapshot + hf_hub_download).
# Actions use a dotted shape (``download.start`` etc.) because
# the model.* namespace previously used dotted action names that
# the pre-[IMPROVE-89] keystone regex couldn't match — the
# events fired but were invisible to coverage. Registering them
# here closes that gap; the regex was tightened in the same
# commit to allow ``[a-z_]+(?:\.[a-z_]+)*`` action names so
# future dotted actions are pinned too.
ModelAction = Literal[
    "download.done",
    "download.error",
    "download.progress",
    "download.start",
]

# partner: voice/persona partner. ``chat.start`` /
# ``voice_init.start`` are literal-string companions to the
# corresponding end events; ``stt.partial`` is a separate
# mid-stream event for STT chunk-level results; ``mem0_init``
# fires on Mem0/ChromaDB cold-start. [IMPROVE-89] registered
# all four after keystone-regex tightening revealed they'd been
# firing without coverage (digit in ``mem0_init`` and dotted
# names in the ``.start``/``.partial`` shapes both escaped the
# old regex).
PartnerAction = Literal[
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
]

# provider: LLM provider availability + routing.
ProviderAction = Literal[
    "availability_probe",
]

# system: DAG executor lifecycle. Wave 5/6 added
# routing_decision (IMPROVE-35 SSE), wave_parallel +
# wave_parallel_fallback (IMPROVE-36 telemetry), validate
# (IMPROVE-31). ``run.start`` is the literal-string companion to
# ``run_done``; [IMPROVE-89] registered it after the keystone
# regex tightening surfaced the gap.
# [IMPROVE-85] Mirror of [IMPROVE-82]'s
# ``agent.validation_rejected``. Pre-IMPROVE-85 the
# /systems/* boundary fired ``system.validate`` with
# ``status="error"`` for both Pydantic schema-invalid and
# Kahn-cycle-detected rejections — that conflates "user posted
# bad JSON" with "validation completed OK" because both share
# the event name. Splitting the rejection case off into its own
# action lets dashboards chart 400-rate separately from
# total-system-runs without a SQL filter.
SystemAction = Literal[
    "node_end",
    "node_start",
    "routing_decision",
    "run.start",
    "run_done",
    "validate",
    "validation_rejected",
    "wave_parallel",
    "wave_parallel_fallback",
]

# tool: built-in tool execution. ``file_ops.path_rejected`` was
# registered by [IMPROVE-89] — tools/file_ops.py and
# tools/rag_tools.py both fire it from their workspace-sandbox
# rejection paths, but the dotted action name didn't match the
# pre-[IMPROVE-89] keystone regex. [IMPROVE-96] added ``invoke``
# / ``invoke.start`` after the Recorder enumeration AST walker
# surfaced the two ``track_event("tool", "invoke", ...)``
# callsites in agents.py (the agent-loop tool dispatcher).
ToolAction = Literal[
    "calculator_eval",
    "file_ops.path_rejected",
    "invoke",
    "invoke.start",
]


# ── Derived per-subsystem tuples ───────────────────────────────

# Each tuple is derived from the corresponding Literal via
# ``typing.get_args``. New events land in ONE place (the Literal)
# and propagate everywhere automatically. Order matches the
# Literal's declaration order (Python preserves it since 3.7+);
# the ``frozenset`` cast in ``KNOWN_EVENTS`` strips ordering.

_AGENT_ACTIONS: tuple[str, ...] = get_args(AgentAction)
_CHAT_ACTIONS: tuple[str, ...] = get_args(ChatAction)
_CONFIG_ACTIONS: tuple[str, ...] = get_args(ConfigAction)
_EDITOR_ACTIONS: tuple[str, ...] = get_args(EditorAction)
_IMAGE_ACTIONS: tuple[str, ...] = get_args(ImageAction)
_IMAGES_ACTIONS: tuple[str, ...] = get_args(ImagesAction)
_INSTRUCT_EDIT_ACTIONS: tuple[str, ...] = get_args(InstructEditAction)
_MODEL_ACTIONS: tuple[str, ...] = get_args(ModelAction)
_PARTNER_ACTIONS: tuple[str, ...] = get_args(PartnerAction)
_PROVIDER_ACTIONS: tuple[str, ...] = get_args(ProviderAction)
_SYSTEM_ACTIONS: tuple[str, ...] = get_args(SystemAction)
_TOOL_ACTIONS: tuple[str, ...] = get_args(ToolAction)


# ── Frozen registry ────────────────────────────────────────────

# Per-subsystem map. Frozen at import; the registry doesn't grow
# at runtime — every new event must be registered here as part
# of its commit so the keystone test passes.
KNOWN_EVENTS: dict[str, frozenset[str]] = {
    "agent": frozenset(_AGENT_ACTIONS),
    "chat": frozenset(_CHAT_ACTIONS),
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


# ── Subsystem Literal alias ────────────────────────────────────

# ``Literal`` of registered subsystem names. mypy catches a typo
# in the subsystem arg of ``emit_typed`` at lint time.
SubsystemName = Literal[
    "agent",
    "chat",
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


# [IMPROVE-91] Per-subsystem ``@overload`` signatures so that
# mypy / pyright catch a misspelled action AT LINT TIME — e.g.
# ``emit_typed("agent", "tool_calll", ...)`` (typo: extra ``l``)
# fails type-check without running. The runtime check below stays
# as a safety net for dynamic callers (test fixtures using
# ``setattr``, future metaprogramming).
#
# The 11 overloads are alphabetised by subsystem name. Each
# mirrors the impl signature exactly: same kwargs, same defaults.
# When a new subsystem is added, three coordinated edits are
# required: (1) the new ``Literal`` type definition above, (2) a
# new entry in ``KNOWN_EVENTS``, (3) a new ``@overload`` here. The
# keystone test ``test_per_subsystem_overload_count_matches_known_events``
# fails CI if (3) is missed.


@overload
def emit_typed(
    subsystem: Literal["agent"],
    action: AgentAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


@overload
def emit_typed(
    subsystem: Literal["chat"],
    action: ChatAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


@overload
def emit_typed(
    subsystem: Literal["config"],
    action: ConfigAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


@overload
def emit_typed(
    subsystem: Literal["editor"],
    action: EditorAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


@overload
def emit_typed(
    subsystem: Literal["image"],
    action: ImageAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


@overload
def emit_typed(
    subsystem: Literal["images"],
    action: ImagesAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


@overload
def emit_typed(
    subsystem: Literal["instruct_edit"],
    action: InstructEditAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


@overload
def emit_typed(
    subsystem: Literal["model"],
    action: ModelAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


@overload
def emit_typed(
    subsystem: Literal["partner"],
    action: PartnerAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


@overload
def emit_typed(
    subsystem: Literal["provider"],
    action: ProviderAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


@overload
def emit_typed(
    subsystem: Literal["system"],
    action: SystemAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


@overload
def emit_typed(
    subsystem: Literal["tool"],
    action: ToolAction,
    status: str = "ok",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
) -> None: ...


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
    """[IMPROVE-NEW-16 + IMPROVE-91] Type-checked + runtime-validated
    wrapper around ``observability.emit``.

    Why:
      * mypy catches misspelled subsystem strings at lint time
        via the ``SubsystemName`` Literal.
      * mypy catches misspelled action strings at lint time via
        the per-subsystem ``Literal`` types (e.g. ``AgentAction``)
        in the ``@overload`` signatures above.
      * Runtime validation catches dynamic-action typos that
        bypass the static check — test fixtures using
        ``setattr``, future metaprogramming, etc. An unknown
        name raises ``UnknownEventNameError`` so a CI grep-and-
        test catches it before the wrong event lands in
        production app_events.
      * Behaviour-equivalent to ``emit`` for every kwarg the
        wrapped function accepts. The keystone test
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


# ── [IMPROVE-92] Per-event context schemas ─────────────────────

# Per Q2=C in the Wave 9 plan: TypedDict for the static front,
# pydantic ``TypeAdapter`` validation in the audit test only —
# never on the emit hot path. Callers who want runtime validation
# can opt in by importing the relevant TypedDict and constructing
# their context dict against it; the schemas themselves do NOT
# add per-emit overhead (they only run when a test imports them).
#
# Why a subset, not all 54 events: each schema is one source of
# truth for a callsite shape. Pinning ALL events would require
# spelunking every callsite to enumerate its keys — bulk work
# better suited to incremental commits as schemas mature. This
# commit pins six well-shaped events that surfaced from Wave 8/9
# typed-event work; future commits add more opportunistically.
# The audit test
# ``test_emit_typed_callsite_keys_match_pinned_schema`` enforces
# that EVERY callsite for an event WITH a schema matches; events
# WITHOUT a pinned schema are silently skipped (no regression).
#
# ``__pydantic_config__ = ConfigDict(extra="forbid")`` on each
# TypedDict makes pydantic's ``TypeAdapter`` reject unknown keys
# at runtime (used by the audit test). Without it pydantic
# defaults to allowing extras, which would let typo'd context
# keys slip through.

_FORBID_EXTRA = ConfigDict(extra="forbid")


class AgentValidationRejectedContext(TypedDict):
    """[IMPROVE-92] Context schema for ``agent.validation_rejected``.

    Wave 7 [IMPROVE-82] introduced this event with three
    distinct shapes depending on which validator rejected the
    request: ``invalid_tool`` (rejected_tool_ids + submitted_tool_ids
    + known_tool_count), ``protected_delete_blocked``
    (agent_name + reason), and ``DuplicateAgent``-style
    conflicts. Total=False with NotRequired captures the union
    so each callsite type-checks under the same schema.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    rejected_tool_ids: NotRequired[list[str]]
    submitted_tool_ids: NotRequired[list[str]]
    known_tool_count: NotRequired[int]
    agent_name: NotRequired[str]
    reason: NotRequired[str]
    conflict: NotRequired[str]


class SystemValidationRejectedContext(TypedDict):
    """[IMPROVE-92] Context schema for ``system.validation_rejected``.

    Wave 8 [IMPROVE-85] / [IMPROVE-88] fire this event with
    three error_code variants: SchemaInvalid (errors list +
    node_count + edge_count), CycleDetected (cyclic_nodes), and
    OrphanLlmRouterEdge (errors). ``system_name`` is the only
    always-present key.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    system_name: str
    errors: NotRequired[list[Any]]
    cyclic_nodes: NotRequired[list[str]]
    node_count: NotRequired[int]
    edge_count: NotRequired[int]


class ImageVramProbeContext(TypedDict):
    """[IMPROVE-92] Context schema for ``image.vram_probe``.

    Wave 8 [IMPROVE-87] introduced this event at all 5 probe
    exit paths with a uniform shape. [IMPROVE-93] added the
    ``tile_mode`` field as required across ALL callsites so
    dashboards can chart "% of probes that ran in tiled mode"
    alongside the existing per-reason breakdown. Pre-IMPROVE-93
    callsites all updated to pass ``tile_mode=False`` explicitly
    (no NotRequired — the field is always present).
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    method: str
    available_gb: float
    required_gb: float
    reason: str
    ok: bool
    tile_mode: bool


class SystemWaveParallelContext(TypedDict):
    """[IMPROVE-92] Context schema for ``system.wave_parallel``.

    Wave 5 [IMPROVE-36] / Wave 8 [IMPROVE-83] fire this event
    when a parallel-wave runs. ``streaming`` distinguishes
    sync vs. streaming executor paths (added by IMPROVE-83 so
    dashboards can chart parallel engagement separately).
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    run_id: str
    system_name: str
    step: int
    node_count: int
    agents: list[str]
    errors: int
    streaming: NotRequired[bool]


class SystemWaveParallelFallbackContext(TypedDict):
    """[IMPROVE-92] Context schema for ``system.wave_parallel_fallback``.

    Fires when the parallel-wave pre-pass detects duplicate
    agents in the same wave — falls back to sequential to
    protect shared ``_smart_memories[agent]`` state per the
    [IMPROVE-36] safety constraint. The ``streaming`` flag
    mirrors the parallel event's sync vs. streaming
    distinction added by [IMPROVE-83]. ``agents`` lists the
    duplicated agent names; ``reason`` carries the bypass
    cause string (today always ``"duplicate_agents"`` but
    future fallback reasons will land here).
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    run_id: str
    system_name: str
    step: int
    node_count: int
    agents: list[str]
    reason: str
    streaming: NotRequired[bool]


class PartnerMem0InitContext(TypedDict):
    """[IMPROVE-92] Context schema for ``partner.mem0_init``.

    Fires on Mem0/ChromaDB cold-start (success + failure paths
    via [IMPROVE-89]'s migration to emit_typed).
    Status='ok' carries llm_model + embed_model + retry;
    status='error' carries retry + retry_in_sec.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    retry: bool
    llm_model: NotRequired[str]
    embed_model: NotRequired[str]
    retry_in_sec: NotRequired[float]


# ── [IMPROVE-95] Wave 10 batch: 12 high-traffic event schemas ─

# Closes 25% of the IMPROVE-92 schema-coverage gap (6 → 18 of
# 54 events pinned). Schemas land alphabetised by (subsystem,
# action) so a future addition slots into the obvious place.
# Each schema's docstring names the source file:line(s) so the
# audit failure messages map straight to the callsite.


class AgentToolCallContext(TypedDict):
    """[IMPROVE-95] Context schema for ``agent.tool_call``.

    Fires from agents.py:1073 on every model-emitted tool_call
    that the agent loop dispatches. ``args_preview`` is the
    string-cast first 200 chars of the tool args (debug-friendly,
    log-safe).
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    agent: str
    tool: str
    call_id: str
    args_preview: str
    thread_id: str


class ConfigLoadContext(TypedDict):
    """[IMPROVE-95] Context schema for ``config.load``.

    Fires once at boot from config.py:415 after settings + .env
    resolution. ``env_file_path`` is None when no .env file was
    found; ``override_count`` reports how many env-var entries
    actually overrode a default.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    env_file_found: bool
    env_file_path: str | None
    override_count: int


class ImageOomLadderDoneContext(TypedDict):
    """[IMPROVE-95] Context schema for ``image.oom_ladder_done``.

    Fires from images/service.py:4646 (helper
    ``_emit_oom_ladder_done``) at the close of every OOM retry
    ladder. ``successful_stage`` is None when ALL stages
    exhausted. Tier 1's ``test_oom_retry_ladder.py`` exercises
    this event end-to-end.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    successful_stage: str | None
    stages_tried: list[str]
    stage_count: int


class ImageOomLadderStartContext(TypedDict):
    """[IMPROVE-95] Context schema for ``image.oom_ladder_start``.

    Fires from images/service.py:9379 at ladder entry with the
    full plan. ``stages_planned`` is the ordered list of retry
    stages the ladder will attempt. ``error_code`` carries the
    triggering OOM type (cuda_oom / system_oom / etc.).
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    error_code: str
    original_width: int
    original_height: int
    original_steps: int
    stages_planned: list[str]
    allow_cpu: bool


class ImageOomStageAttemptContext(TypedDict):
    """[IMPROVE-95] Context schema for ``image.oom_stage_attempt``.

    Fires from images/service.py:9461 (ok) and 9487 (error) for
    every individual ladder stage. ``retry_*`` keys describe the
    attempted parameters at that stage; ok/error variants share
    the same key set (status + error_code/error_message handled
    by the emit_typed signature itself, not the context dict).
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    stage_name: str
    retry_width: int
    retry_height: int
    retry_steps: int
    retry_timeout_s: int
    retry_device: str


class ImageOptimizationPlanContext(TypedDict):
    """[IMPROVE-95] Context schema for ``image.optimization_plan``.

    Fires from images/service.py:3934 once per image generation
    request after the optimization-rules layer resolves. The
    ``rules_*`` lists carry the plan-vs-applied decision trail
    so dashboards can chart "rules suppressed by family" etc.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    backend: str
    family: str
    quality_tier: str
    steps: int
    is_few_step: bool
    is_cpu: bool
    rules_fired: list[str]
    rules_suppressed: list[str]
    rules_suppressed_by: dict[str, Any]


class ImageWarmupContext(TypedDict):
    """[IMPROVE-95] Context schema for ``image.warmup``.

    Fires from images/service.py:7390 (ok) and 7401 (error) on
    image-pipeline warmup. error_code/error_message land on the
    emit_typed signature itself (not the context dict) so the
    schema is identical across both variants.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    mode: str
    device: str


class ProviderAvailabilityProbeContext(TypedDict):
    """[IMPROVE-95] Context schema for ``provider.availability_probe``.

    Fires from providers/router.py:102 (ok with available bool)
    and 116 (error path — ``available`` omitted because the
    probe failed before resolving). NotRequired captures the
    union without forcing the error path to fabricate a value.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    provider: str
    available: NotRequired[bool]


class SystemNodeEndContext(TypedDict):
    """[IMPROVE-95] Context schema for ``system.node_end``.

    Fires from 6 callsites in systems/executor.py covering ok /
    error / cached-preloaded-ok / cached-preloaded-error / stream
    / skipped paths (executor.py:805/818/1107/1081/1212/1234) +
    the agent-not-found skipped path (executor.py:757) which
    adds a ``reason`` key. The skipped/skip-reason path is the
    only variant that carries ``reason``.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    run_id: str
    system_name: str
    node_id: str
    agent: str
    role: str
    reason: NotRequired[str]


class SystemNodeStartContext(TypedDict):
    """[IMPROVE-95] Context schema for ``system.node_start``.

    Fires from systems/executor.py:780 (sync) and 1146
    (streaming) with the same six core keys. The streaming
    preloaded variant at executor.py:1062 adds a ``preloaded``
    bool flag; NotRequired captures the union.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    run_id: str
    system_name: str
    node_id: str
    agent: str
    role: str
    step: int
    preloaded: NotRequired[bool]


class SystemRoutingDecisionContext(TypedDict):
    """[IMPROVE-95] Context schema for ``system.routing_decision``.

    Fires from systems/executor.py:1275 once per llm_router
    node. ``chosen_option`` is None when no rule matched (the
    router fell through to default routing). ``candidates`` is
    the list of router-edge target node_ids the rule layer
    considered.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    run_id: str
    system_name: str
    node_id: str
    chosen_option: str | None
    candidates: list[str]
    rule_count: int


class SystemRunDoneContext(TypedDict):
    """[IMPROVE-95] Context schema for ``system.run_done``.

    Fires from systems/executor.py:853 (sync) and 1306
    (streaming) at the end of every system run. ``streaming``
    is a literal True only on the streaming-executor path
    (mirrors IMPROVE-83's flag on ``system.wave_parallel``).
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    run_id: str
    system_name: str
    conversation_id: str
    streaming: NotRequired[bool]


# ── [IMPROVE-101] Wave 11 batch: 10 high-traffic event schemas ─

# Closes another ~19% of the IMPROVE-92 schema-coverage gap (18
# → 28 of 54 events pinned, 52%). Tier-A picks per the Wave 11
# plan: high-traffic events with stable shapes that surfaced
# during Wave 10 work but weren't in the IMPROVE-95 mechanical
# batch (download lifecycle, partner-engine chat, image-pipeline
# inference, instruct-edit run, system-validate). Each schema's
# docstring names the source file:line(s) so the audit failure
# messages map straight to the callsite, matching IMPROVE-95's
# convention. 8 distinct TypedDict classes cover 10 (sub, act)
# tuples — partner.chat shares its shape with chat.start (the
# Recorder companion event) and image.infer shares with
# infer.start; both are the same _chat_ctx / _infer_ctx variable
# spread at start + end emits.


class ImageInferContext(TypedDict):
    """[IMPROVE-101] Context schema for ``image.infer`` and
    ``image.infer.start``.

    Fires from images/service.py:10008 (start) and 10027 (end).
    Both spread the same ``_infer_ctx`` dict assembled at
    service.py:10000, so a single schema covers both events.
    ``mode`` is one of "txt2img" / "img2img" / "inpaint";
    ``device`` is the resolved preferred-device name. perf
    payload carries width/height/steps/image_bytes — those land
    on emit_typed's ``perf=`` arg, NOT in this context dict.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    model_id: str
    model_source: str
    device: str
    mode: str
    scheduler: str


class InstructEditRunContext(TypedDict):
    """[IMPROVE-101] Context schema for ``instruct_edit.run``.

    Fires from 7 callsites in images/ai_enhance.py covering
    kontext/nunchaku ok (2667), value/runtime errors (2679),
    other exceptions (2689); cosxl ok (3214), value/runtime
    errors (3223), other exceptions (3231); plus the terminal
    UnknownModel branch (3239). All callsites spread
    ``_ie_ctx`` from line 2169 — nine always-present keys.
    ``backend`` is added by branch ("nunchaku" / "kontext" /
    "cosxl") on every path EXCEPT the terminal UnknownModel
    branch which fires the bare ctx, hence NotRequired.
    ``gguf_quant_requested`` is always present per IMPROVE-49
    but ``str | None`` to capture the env-default fallback.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    model: str
    requested_steps: int
    requested_guidance: float
    has_negative_prompt: bool
    true_cfg_scale: float
    input_width: int
    input_height: int
    seed_set: bool
    gguf_quant_requested: str | None
    backend: NotRequired[str]


class ModelDownloadDoneContext(TypedDict):
    """[IMPROVE-101] Context schema for ``model.download.done``.

    Fires from api/routers/models.py:836 (Ollama success path)
    and 2303 (HuggingFace success path). Both carry the
    provider + model_id pair; the HF callsite additionally
    carries the GGUF filename + download_key so the
    ``downloads_state`` map (keyed by download_key) can be
    located post-completion. ``gguf_filename`` is nullable
    (passed even when the download wasn't a GGUF file) to
    match the start event's shape.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    provider: str
    model_id: str
    gguf_filename: NotRequired[str | None]
    download_key: NotRequired[str]


class ModelDownloadErrorContext(TypedDict):
    """[IMPROVE-101] Context schema for ``model.download.error``.

    Fires from api/routers/models.py:846 (Ollama error path) and
    2315 (HuggingFace error path). Same shape as
    ``model.download.done`` — the error_code/error_message land
    on emit_typed's signature itself (not the context dict) so
    the schema parallels the success-path shape exactly.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    provider: str
    model_id: str
    gguf_filename: NotRequired[str | None]
    download_key: NotRequired[str]


class ModelDownloadProgressContext(TypedDict):
    """[IMPROVE-101] Context schema for ``model.download.progress``.

    Fires from api/routers/models.py:816 (Ollama progress
    bucket-emitter) only — HF progress is reported via the
    sibling per-byte filesystem watcher (IMPROVE-86) and lands
    on a different event surface. ``phase`` echoes Ollama's
    free-form status string ("downloading" / "verifying" /
    etc.); pin as ``str`` rather than Literal because the
    upstream enum is not stable. perf payload (pct,
    completed_bytes, total_bytes) lands on emit_typed's
    ``perf=`` arg, NOT in this context dict.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    provider: str
    model_id: str
    phase: str


class ModelDownloadStartContext(TypedDict):
    """[IMPROVE-101] Context schema for ``model.download.start``.

    Fires from api/routers/models.py:783 (Ollama pull start) and
    2122 (HuggingFace snapshot_download start). HF callsite adds
    gguf_filename / download_key / has_token; the Ollama path
    omits all three because they don't apply (no GGUF selection,
    no auth-token requirement). ``has_token`` reports whether a
    HF token was resolved at the call site without leaking the
    token value.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    provider: str
    model_id: str
    gguf_filename: NotRequired[str | None]
    download_key: NotRequired[str]
    has_token: NotRequired[bool]


class PartnerChatContext(TypedDict):
    """[IMPROVE-101] Context schema for ``partner.chat`` and
    ``partner.chat.start``.

    Fires from partner/engine.py:330 (start), 334 (error), 409
    (ok). All three callsites spread the same ``_chat_ctx`` dict
    assembled at engine.py:328 — three required keys cover the
    union. The end-event reply_length / emotion fields land on
    emit_typed's ``perf=`` arg, NOT in this context dict.
    ``streaming`` is currently always False at the sole
    assignment site; type left as ``bool`` because the streaming
    sibling path (``astream_chat``) may grow a parallel
    ``_chat_ctx`` later.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    model: str
    streaming: bool
    user_input_length: int


class SystemValidateContext(TypedDict):
    """[IMPROVE-101] Context schema for ``system.validate``.

    Fires from api/routers/systems.py:368 on the success path
    only — the rejection branches (269/304/342) fire the
    sibling ``system.validation_rejected`` event per IMPROVE-85
    so dashboards can chart 400-rate separately from
    total-validation-runs without a SQL filter. ``node_count``
    + ``edge_count`` mirror ``system.validation_rejected``'s
    SchemaInvalid variant for cross-event correlation.
    """
    __pydantic_config__ = _FORBID_EXTRA  # type: ignore[misc]
    system_name: str
    node_count: int
    edge_count: int


# Map (subsystem, action) → TypedDict schema class. The audit
# test ``test_emit_typed_callsite_keys_match_pinned_schema``
# walks emit_typed callsites and validates each against the
# corresponding schema's ``__required_keys__`` /
# ``__optional_keys__`` introspection. New schemas land here
# alongside their TypedDict definition above.
EVENT_CONTEXT_SCHEMAS: dict[tuple[str, str], type] = {
    ("agent", "tool_call"): AgentToolCallContext,
    ("agent", "validation_rejected"): AgentValidationRejectedContext,
    ("config", "load"): ConfigLoadContext,
    ("image", "infer"): ImageInferContext,
    ("image", "infer.start"): ImageInferContext,
    ("image", "oom_ladder_done"): ImageOomLadderDoneContext,
    ("image", "oom_ladder_start"): ImageOomLadderStartContext,
    ("image", "oom_stage_attempt"): ImageOomStageAttemptContext,
    ("image", "optimization_plan"): ImageOptimizationPlanContext,
    ("image", "vram_probe"): ImageVramProbeContext,
    ("image", "warmup"): ImageWarmupContext,
    ("instruct_edit", "run"): InstructEditRunContext,
    ("model", "download.done"): ModelDownloadDoneContext,
    ("model", "download.error"): ModelDownloadErrorContext,
    ("model", "download.progress"): ModelDownloadProgressContext,
    ("model", "download.start"): ModelDownloadStartContext,
    ("partner", "chat"): PartnerChatContext,
    ("partner", "chat.start"): PartnerChatContext,
    ("partner", "mem0_init"): PartnerMem0InitContext,
    ("provider", "availability_probe"): ProviderAvailabilityProbeContext,
    ("system", "node_end"): SystemNodeEndContext,
    ("system", "node_start"): SystemNodeStartContext,
    ("system", "routing_decision"): SystemRoutingDecisionContext,
    ("system", "run_done"): SystemRunDoneContext,
    ("system", "validate"): SystemValidateContext,
    ("system", "validation_rejected"): SystemValidationRejectedContext,
    ("system", "wave_parallel"): SystemWaveParallelContext,
    ("system", "wave_parallel_fallback"): SystemWaveParallelFallbackContext,
}
