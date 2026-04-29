"""[IMPROVE-92] Tests for per-event context schemas.

Wave 7 [IMPROVE-80] introduced the ``KNOWN_EVENTS`` registry +
the keystone test
``test_registry_covers_every_emit_callsite_in_codebase``. Wave 9
[IMPROVE-91] added per-subsystem ``Literal`` types + ``@overload``
signatures so mypy catches misspelled actions at lint time.

Both layers cover the EVENT NAME side of the contract — they
fail when an emit fires an unregistered ``(subsystem, action)``
pair. Neither covers the CONTEXT SHAPE side: a callsite passing
``context={"agent": ..., "tooll": ...}`` (typo'd ``tool`` key)
silently writes garbage into ``app_events`` even though the
event name is correct.

This commit pins six well-shaped events with TypedDict context
schemas (``EVENT_CONTEXT_SCHEMAS`` in observability_events.py).
The audit test below walks every ``emit_typed(...)`` callsite,
extracts the literal-dict context keys, and validates against
the schema's required/optional key sets via TypedDict
introspection. Pydantic ``TypeAdapter`` is used at test time
only — no per-emit overhead at runtime, per Q2=C in the Wave 9
plan.

Why a SUBSET of the 54 registered events: pinning all events
would require spelunking every callsite to enumerate keys —
that's bulk work better suited to incremental commits as
schemas mature. Six events with stable shapes are pinned now;
future commits add more opportunistically. Events without a
schema in ``EVENT_CONTEXT_SCHEMAS`` are silently skipped by the
audit (no regression).

Sources (2025-2026):
  * docs/features/09-observability.md — internal observability
    surface this commit extends.
  * Wave 7 [IMPROVE-80] commit (25b851e) — typed front door +
    registry that this builds on.
  * Wave 9 [IMPROVE-91] commit (578d1d0) — per-subsystem
    ``Literal`` types this commit's TypedDicts complement (
    name-side + shape-side together).
  * Pydantic 2 ``TypeAdapter`` + ``TypedDict`` validation
    (pydantic docs 2025):
    https://docs.pydantic.dev/latest/concepts/types/#typeddict
  * PEP 655 ``Required`` / ``NotRequired`` for fine-grained
    optional-field control:
    https://peps.python.org/pep-0655/
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from local_ai_platform.observability_events import (
    EVENT_CONTEXT_SCHEMAS,
    AgentFallbackContext,
    AgentProtectedDeleteBlockedContext,
    AgentToolAutoResumeContext,
    AgentToolCallContext,
    AgentToolResultContext,
    AgentValidationRejectedContext,
    ChatEnhancePromptContext,
    ChatSendContext,
    ConfigLoadContext,
    EditorBlendWithPreviousContext,
    EditorEditContext,
    EditorExportContext,
    EditorOpContext,
    EditorRedoContext,
    EditorUndoContext,
    ImageEnhancePromptContext,
    ImageGenerateContext,
    ImageInferContext,
    ImageLoadContext,
    ImageOomLadderDoneContext,
    ImageOomLadderStartContext,
    ImageOomStageAttemptContext,
    ImageOptimizationPlanContext,
    ImagePlanContext,
    ImagePostprocessContext,
    ImagesDetectHintsContext,
    ImageVramProbeContext,
    ImageWarmupContext,
    InstructEditLoadContext,
    InstructEditRunContext,
    ModelDownloadDoneContext,
    ModelDownloadErrorContext,
    ModelDownloadProgressContext,
    ModelDownloadStartContext,
    PartnerChatContext,
    PartnerEmotionDetectContext,
    PartnerFactExtractContext,
    PartnerMem0InitContext,
    PartnerSttContext,
    PartnerTtsContext,
    PartnerVoiceInitContext,
    ProviderAvailabilityProbeContext,
    SystemNodeEndContext,
    SystemNodeStartContext,
    SystemRoutingDecisionContext,
    SystemRunDoneContext,
    SystemRunStartContext,
    SystemValidateContext,
    SystemValidationRejectedContext,
    SystemWaveParallelContext,
    SystemWaveParallelFallbackContext,
    ToolCalculatorEvalContext,
    ToolFileOpsPathRejectedContext,
    ToolInvokeContext,
)


# ── Schema introspection helpers ───────────────────────────────


def _required_optional_keys(schema_class) -> tuple[frozenset[str], frozenset[str]]:
    """Return (required, optional) key sets for a TypedDict
    schema. Uses ``__required_keys__`` / ``__optional_keys__``
    which are populated by typing_extensions when
    ``from __future__ import annotations`` is NOT in scope at
    class-definition time."""
    return (
        frozenset(schema_class.__required_keys__),
        frozenset(schema_class.__optional_keys__),
    )


# ── Schema definitions ─────────────────────────────────────────


def test_event_context_schemas_dict_is_well_formed():
    """[IMPROVE-92] Each entry in EVENT_CONTEXT_SCHEMAS maps a
    (subsystem, action) tuple to a TypedDict class. Verify the
    structure so a future ``EVENT_CONTEXT_SCHEMAS["agent"] = ...``
    (str key) typo doesn't slip through."""
    assert isinstance(EVENT_CONTEXT_SCHEMAS, dict)
    for key, schema in EVENT_CONTEXT_SCHEMAS.items():
        assert isinstance(key, tuple), f"key {key!r} must be a tuple"
        assert len(key) == 2, f"key {key!r} must be (subsystem, action)"
        subsystem, action = key
        assert isinstance(subsystem, str)
        assert isinstance(action, str)
        # TypedDict classes have ``__required_keys__`` /
        # ``__optional_keys__`` attrs set by the metaclass.
        assert hasattr(schema, "__required_keys__"), (
            f"schema for {key!r} must be a TypedDict subclass; "
            f"got {type(schema).__name__}"
        )


def test_event_context_schemas_keys_are_registered_events():
    """[IMPROVE-92] Every (subsystem, action) in
    EVENT_CONTEXT_SCHEMAS must refer to a REGISTERED event in
    KNOWN_EVENT_NAMES. Catches a future schema-without-event typo
    where someone defines a TypedDict for a misspelled action.
    """
    from local_ai_platform.observability_events import KNOWN_EVENT_NAMES

    for subsystem, action in EVENT_CONTEXT_SCHEMAS.keys():
        full_name = f"{subsystem}.{action}"
        assert full_name in KNOWN_EVENT_NAMES, (
            f"[IMPROVE-92] EVENT_CONTEXT_SCHEMAS has entry "
            f"{(subsystem, action)!r} but {full_name!r} is not "
            f"in KNOWN_EVENT_NAMES — register the event first."
        )


# ── Per-schema introspection ───────────────────────────────────


def test_image_vram_probe_schema_is_strict():
    """[IMPROVE-87 + IMPROVE-93] The vram_probe event has a
    fixed shape — six keys (method/available_gb/required_gb/
    reason/ok/tile_mode) are all required. Pre-IMPROVE-93 the
    schema had five keys; ``tile_mode`` was added as required
    when [IMPROVE-93] introduced the tiled-mode probe path so
    dashboards can chart "% of probes that ran in tiled mode"
    via the existing per-event aggregation."""
    required, optional = _required_optional_keys(ImageVramProbeContext)
    assert required == {
        "method", "available_gb", "required_gb", "reason", "ok",
        "tile_mode",
    }
    assert optional == frozenset()


def test_system_validation_rejected_schema_minimal_required():
    """[IMPROVE-85/88] The validation_rejected event has three
    error_code variants (SchemaInvalid / CycleDetected /
    OrphanLlmRouterEdge); only ``system_name`` is always
    present. The other keys are optional via NotRequired."""
    required, optional = _required_optional_keys(
        SystemValidationRejectedContext,
    )
    assert required == {"system_name"}
    assert "errors" in optional
    assert "cyclic_nodes" in optional


def test_partner_mem0_init_schema_retry_required():
    """[IMPROVE-89] The mem0_init event always carries
    ``retry`` (bool) regardless of success/error path. Other
    keys (llm_model, embed_model, retry_in_sec) appear only on
    one path each."""
    required, optional = _required_optional_keys(
        PartnerMem0InitContext,
    )
    assert required == {"retry"}
    assert optional == {"llm_model", "embed_model", "retry_in_sec"}


def test_system_wave_parallel_schema_streaming_optional():
    """[IMPROVE-83] The streaming flag was added in Wave 8 to
    distinguish sync vs. streaming executor paths. Pre-IMPROVE-83
    callsites didn't carry it; the schema treats it as optional."""
    required, optional = _required_optional_keys(
        SystemWaveParallelContext,
    )
    assert "streaming" in optional
    assert "run_id" in required
    assert "agents" in required


# ── [IMPROVE-95] Wave 10 batch: per-schema introspection (12) ──


def test_agent_tool_call_schema_strict():
    """[IMPROVE-95] All five keys (agent / tool / call_id /
    args_preview / thread_id) are always present on the single
    callsite at agents.py:1073. No NotRequired."""
    required, optional = _required_optional_keys(AgentToolCallContext)
    assert required == {"agent", "tool", "call_id", "args_preview", "thread_id"}
    assert optional == frozenset()


def test_config_load_schema_three_required_keys():
    """[IMPROVE-95] Boot-time event with exactly three keys.
    ``env_file_path`` is explicitly nullable (str | None) so the
    'no .env found' case doesn't drop the key."""
    required, optional = _required_optional_keys(ConfigLoadContext)
    assert required == {"env_file_found", "env_file_path", "override_count"}
    assert optional == frozenset()


def test_image_oom_ladder_done_schema_successful_stage_nullable():
    """[IMPROVE-95] ``successful_stage`` is None when ALL stages
    were exhausted — explicitly nullable, NOT NotRequired (the
    key is always present, just sometimes None)."""
    required, optional = _required_optional_keys(ImageOomLadderDoneContext)
    assert required == {"successful_stage", "stages_tried", "stage_count"}
    assert optional == frozenset()


def test_image_oom_ladder_start_schema_strict():
    """[IMPROVE-95] Six required keys at ladder entry. Pin so a
    future stages_planned-as-tuple-not-list typo gets caught
    at schema-validation time."""
    required, optional = _required_optional_keys(ImageOomLadderStartContext)
    assert required == {
        "error_code", "original_width", "original_height",
        "original_steps", "stages_planned", "allow_cpu",
    }
    assert optional == frozenset()


def test_image_oom_stage_attempt_schema_identical_across_ok_error():
    """[IMPROVE-95] Both ok and error variants of the
    ``image.oom_stage_attempt`` event share the same six retry_*
    keys; error_code/error_message live on the emit_typed
    signature itself, NOT in the context dict."""
    required, optional = _required_optional_keys(ImageOomStageAttemptContext)
    assert required == {
        "stage_name", "retry_width", "retry_height", "retry_steps",
        "retry_timeout_s", "retry_device",
    }
    assert optional == frozenset()


def test_image_optimization_plan_schema_full_shape():
    """[IMPROVE-95] The optimization-plan event carries the full
    rules-fired/suppressed/suppressed_by trail. No optional
    keys — all nine fields populated on every fire."""
    required, optional = _required_optional_keys(ImageOptimizationPlanContext)
    assert required == {
        "backend", "family", "quality_tier", "steps", "is_few_step",
        "is_cpu", "rules_fired", "rules_suppressed", "rules_suppressed_by",
    }
    assert optional == frozenset()


def test_image_warmup_schema_two_required_keys():
    """[IMPROVE-95] The warmup event carries only mode + device
    in context; error_code/error_message land on emit_typed's
    signature, not the context dict, so the schema is identical
    across ok and error callsites."""
    required, optional = _required_optional_keys(ImageWarmupContext)
    assert required == {"mode", "device"}
    assert optional == frozenset()


def test_provider_availability_probe_schema_available_optional():
    """[IMPROVE-95] ``provider`` is always present;
    ``available`` only fires on the success path
    (providers/router.py:102) since the error path 116 omits it
    by design (the probe failed before resolving)."""
    required, optional = _required_optional_keys(ProviderAvailabilityProbeContext)
    assert required == {"provider"}
    assert optional == {"available"}


def test_system_node_end_schema_reason_optional():
    """[IMPROVE-95] Five core keys always present;
    ``reason`` only on the agent-not-found skipped path
    (executor.py:757)."""
    required, optional = _required_optional_keys(SystemNodeEndContext)
    assert required == {"run_id", "system_name", "node_id", "agent", "role"}
    assert optional == {"reason"}


def test_system_node_start_schema_preloaded_optional():
    """[IMPROVE-95] Six core keys present in sync + streaming
    callsites; ``preloaded`` only in the streaming-preloaded
    variant at executor.py:1062."""
    required, optional = _required_optional_keys(SystemNodeStartContext)
    assert required == {
        "run_id", "system_name", "node_id", "agent", "role", "step",
    }
    assert optional == {"preloaded"}


def test_system_routing_decision_schema_strict_chosen_nullable():
    """[IMPROVE-95] All six keys always present;
    ``chosen_option`` is explicitly nullable (str | None) for
    the no-rule-matched fallback case, NOT NotRequired."""
    required, optional = _required_optional_keys(SystemRoutingDecisionContext)
    assert required == {
        "run_id", "system_name", "node_id", "chosen_option",
        "candidates", "rule_count",
    }
    assert optional == frozenset()


def test_system_run_done_schema_streaming_optional():
    """[IMPROVE-95] Three core keys; ``streaming`` flag only on
    the streaming-executor path (executor.py:1306). Mirrors
    [IMPROVE-83]'s convention on system.wave_parallel."""
    required, optional = _required_optional_keys(SystemRunDoneContext)
    assert required == {"run_id", "system_name", "conversation_id"}
    assert optional == {"streaming"}


# ── [IMPROVE-101] Wave 11 batch: per-schema introspection (10) ──


def test_image_infer_schema_strict():
    """[IMPROVE-101] Both ``image.infer`` and ``image.infer.start``
    spread the same ``_infer_ctx`` dict at images/service.py:10000,
    so the schema covers both events with five required keys and
    no optional. perf payload (width/height/steps/image_bytes)
    lands on emit_typed's ``perf=`` arg, not the context dict."""
    required, optional = _required_optional_keys(ImageInferContext)
    assert required == {"model_id", "model_source", "device", "mode", "scheduler"}
    assert optional == frozenset()


def test_image_infer_and_start_share_same_schema():
    """[IMPROVE-101] Pin the schema-sharing convention: a single
    TypedDict class covers both the start and end events of the
    image.infer pair because both callsites spread the same
    ``_infer_ctx`` variable. A future commit splitting them apart
    surfaces here as a regression."""
    assert EVENT_CONTEXT_SCHEMAS[("image", "infer")] is ImageInferContext
    assert EVENT_CONTEXT_SCHEMAS[("image", "infer.start")] is ImageInferContext


def test_instruct_edit_run_schema_backend_optional():
    """[IMPROVE-101] Nine required keys cover the always-present
    ``_ie_ctx`` dict (ai_enhance.py:2169). ``backend`` is only on
    the kontext/nunchaku/cosxl branches; the terminal
    UnknownModel branch (3239) fires the bare ctx and omits it,
    hence NotRequired. ``gguf_quant_requested`` is required (per
    IMPROVE-49 always present) but typed as ``str | None`` for
    the env-default fallback case."""
    required, optional = _required_optional_keys(InstructEditRunContext)
    assert required == {
        "model", "requested_steps", "requested_guidance",
        "has_negative_prompt", "true_cfg_scale",
        "input_width", "input_height",
        "seed_set", "gguf_quant_requested",
    }
    assert optional == {"backend"}


def test_model_download_done_schema_provider_required():
    """[IMPROVE-101] provider + model_id are always present
    (Ollama and HF callsites both pass them). gguf_filename +
    download_key are HF-only — NotRequired so the Ollama
    callsite at models.py:836 type-checks under the same
    schema."""
    required, optional = _required_optional_keys(ModelDownloadDoneContext)
    assert required == {"provider", "model_id"}
    assert optional == {"gguf_filename", "download_key"}


def test_model_download_error_schema_mirrors_done():
    """[IMPROVE-101] error_code/error_message land on
    emit_typed's signature itself (not the context dict), so
    the error variant of model.download.* shares the same shape
    as the done variant — same required/optional split."""
    required, optional = _required_optional_keys(ModelDownloadErrorContext)
    assert required == {"provider", "model_id"}
    assert optional == {"gguf_filename", "download_key"}


def test_model_download_progress_schema_three_required_keys():
    """[IMPROVE-101] Ollama-only event (HF progress fires through
    the IMPROVE-86 filesystem watcher on a different surface).
    ``phase`` echoes Ollama's free-form status string and is
    typed ``str`` rather than Literal because the upstream
    enum is not stable."""
    required, optional = _required_optional_keys(ModelDownloadProgressContext)
    assert required == {"provider", "model_id", "phase"}
    assert optional == frozenset()


def test_model_download_start_schema_three_optional_keys():
    """[IMPROVE-101] Same provider + model_id required pair as
    download.done; HF callsite (models.py:2122) adds three
    extras — gguf_filename / download_key / has_token — that
    Ollama (783) doesn't carry. ``has_token`` reports whether
    a HF token was resolved, not its value."""
    required, optional = _required_optional_keys(ModelDownloadStartContext)
    assert required == {"provider", "model_id"}
    assert optional == {"gguf_filename", "download_key", "has_token"}


def test_partner_chat_schema_three_required_keys():
    """[IMPROVE-101] Both ``partner.chat`` and
    ``partner.chat.start`` spread the same ``_chat_ctx`` dict
    at partner/engine.py:328. Three required keys; the
    end-event reply_length / emotion fields land on
    emit_typed's ``perf=`` arg, not this context dict."""
    required, optional = _required_optional_keys(PartnerChatContext)
    assert required == {"model", "streaming", "user_input_length"}
    assert optional == frozenset()


def test_partner_chat_and_start_share_same_schema():
    """[IMPROVE-101] Pin the schema-sharing convention: one
    TypedDict class for both partner.chat and partner.chat.start
    because both callsites spread the same ``_chat_ctx`` variable
    (engine.py:328 → 330/334/409). A future commit splitting them
    apart surfaces as a regression here."""
    assert EVENT_CONTEXT_SCHEMAS[("partner", "chat")] is PartnerChatContext
    assert EVENT_CONTEXT_SCHEMAS[("partner", "chat.start")] is PartnerChatContext


def test_system_validate_schema_three_required_keys():
    """[IMPROVE-101] Success-path only event (rejection branches
    fire the sibling system.validation_rejected per IMPROVE-85).
    Three required keys mirror the SchemaInvalid variant of
    system.validation_rejected for cross-event correlation."""
    required, optional = _required_optional_keys(SystemValidateContext)
    assert required == {"system_name", "node_count", "edge_count"}
    assert optional == frozenset()


# ── [IMPROVE-102] Wave 11 batch: 6 Recorder context schemas ────


def test_chat_enhance_prompt_schema_three_required_keys():
    """[IMPROVE-102] Recorder track_event callsite at
    chat.py:358. Three required keys cover prompt length +
    target model hint + detected prompt-shape."""
    required, optional = _required_optional_keys(ChatEnhancePromptContext)
    assert required == {"prompt_length", "model_hint", "detected_type"}
    assert optional == frozenset()


def test_chat_send_schema_thread_id_optional():
    """[IMPROVE-102] Eight always-present keys cover the sync
    callsite (chat.py:720); ``thread_id`` only on the streaming
    callsite at chat.py:947 hence NotRequired. Both callsites
    share this schema via the (sub, act) + (sub, act.start)
    pairing."""
    required, optional = _required_optional_keys(ChatSendContext)
    assert required == {
        "agent", "model", "provider", "conversation_id", "run_id",
        "has_images", "image_count", "streaming",
    }
    assert optional == {"thread_id"}


def test_chat_send_and_start_share_same_schema():
    """[IMPROVE-102] Pin the schema-sharing convention for the
    Recorder pattern: a single TypedDict covers both the base
    and ``.start`` companion event because the Recorder spreads
    the same context dict on __enter__ + __exit__."""
    assert EVENT_CONTEXT_SCHEMAS[("chat", "send")] is ChatSendContext
    assert EVENT_CONTEXT_SCHEMAS[("chat", "send.start")] is ChatSendContext


def test_chat_enhance_prompt_and_start_share_same_schema():
    """[IMPROVE-102] Pin the schema-sharing convention for
    chat.enhance_prompt's Recorder pair (chat.py:358)."""
    assert EVENT_CONTEXT_SCHEMAS[
        ("chat", "enhance_prompt")
    ] is ChatEnhancePromptContext
    assert EVENT_CONTEXT_SCHEMAS[
        ("chat", "enhance_prompt.start")
    ] is ChatEnhancePromptContext


def test_editor_edit_schema_three_required_keys():
    """[IMPROVE-102] Recorder track_event callsite at
    editor.py:432. Three required keys: session_id /
    operation / param_count."""
    required, optional = _required_optional_keys(EditorEditContext)
    assert required == {"session_id", "operation", "param_count"}
    assert optional == frozenset()


def test_editor_edit_and_start_share_same_schema():
    """[IMPROVE-102] Pin the schema-sharing convention for
    editor.edit's Recorder pair (editor.py:432)."""
    assert EVENT_CONTEXT_SCHEMAS[
        ("editor", "edit")
    ] is EditorEditContext
    assert EVENT_CONTEXT_SCHEMAS[
        ("editor", "edit.start")
    ] is EditorEditContext


def test_image_enhance_prompt_schema_four_required_keys():
    """[IMPROVE-102] Recorder track_event callsite at
    images.py:643. Four required keys cover the image-prompt
    rewriter prelude (length / family / model_hint / weighting
    flag)."""
    required, optional = _required_optional_keys(ImageEnhancePromptContext)
    assert required == {
        "prompt_length", "model_family", "model_hint", "prompt_weighting",
    }
    assert optional == frozenset()


def test_image_enhance_prompt_and_start_share_same_schema():
    """[IMPROVE-102] Pin the schema-sharing convention for
    image.enhance_prompt's Recorder pair (images.py:643)."""
    assert EVENT_CONTEXT_SCHEMAS[
        ("image", "enhance_prompt")
    ] is ImageEnhancePromptContext
    assert EVENT_CONTEXT_SCHEMAS[
        ("image", "enhance_prompt.start")
    ] is ImageEnhancePromptContext


def test_image_generate_schema_scheduler_nullable():
    """[IMPROVE-102] Eight required keys cover the images.py:876
    Recorder callsite. ``scheduler`` and ``controlnet_type`` are
    explicitly nullable (``str | None``) — they come from
    body.get() WITHOUT defaults, so they're always-present-but-
    nullable rather than NotRequired (the keys themselves DO
    appear in every callsite)."""
    required, optional = _required_optional_keys(ImageGenerateContext)
    assert required == {
        "model_id", "prompt_length", "steps", "width", "height",
        "num_images", "scheduler", "controlnet_type",
    }
    assert optional == frozenset()


def test_image_generate_and_start_share_same_schema():
    """[IMPROVE-102] Pin the schema-sharing convention for
    image.generate's Recorder pair (images.py:876)."""
    assert EVENT_CONTEXT_SCHEMAS[
        ("image", "generate")
    ] is ImageGenerateContext
    assert EVENT_CONTEXT_SCHEMAS[
        ("image", "generate.start")
    ] is ImageGenerateContext


def test_tool_invoke_schema_three_required_keys():
    """[IMPROVE-102] Recorder track_event callsites at
    agents.py:578 (sync) and 590 (async). Both spread the same
    ``ctx`` dict from lines 576/588 — three required keys
    (tool / dangerous / arg_size). Both callsites use
    variable-context so the audit walker skips them — but the
    schema is pinned so a future literal-context refactor
    inherits validation automatically."""
    required, optional = _required_optional_keys(ToolInvokeContext)
    assert required == {"tool", "dangerous", "arg_size"}
    assert optional == frozenset()


def test_tool_invoke_and_start_share_same_schema():
    """[IMPROVE-102] Pin the schema-sharing convention for
    tool.invoke's Recorder pair (agents.py:578 + 590)."""
    assert EVENT_CONTEXT_SCHEMAS[
        ("tool", "invoke")
    ] is ToolInvokeContext
    assert EVENT_CONTEXT_SCHEMAS[
        ("tool", "invoke.start")
    ] is ToolInvokeContext


# ── [IMPROVE-107] Final-tier schemas: 100% coverage ────────────


def test_agent_fallback_schema_four_required_keys():
    """[IMPROVE-107] Fallback fires from agents.py:1206 with the
    four-key ``{agent, model, provider, reason}`` shape. ``reason``
    is today always ``"model_does_not_support_tools"`` but the
    field is ``str`` (not Literal) for future fallback reasons."""
    required, optional = _required_optional_keys(AgentFallbackContext)
    assert required == {"agent", "model", "provider", "reason"}
    assert optional == frozenset()


def test_agent_protected_delete_blocked_schema_single_key():
    """[IMPROVE-107] DELETE /agents/<name> rejection at
    api/routers/agents.py:441 with a single ``agent_name`` key —
    error_code/error_message land on emit_typed's signature."""
    required, optional = _required_optional_keys(AgentProtectedDeleteBlockedContext)
    assert required == {"agent_name"}
    assert optional == frozenset()


def test_agent_tool_auto_resume_schema_via_optional():
    """[IMPROVE-107] Fires from agents.py:410 (post-interrupt;
    adds ``via="resume"``) and 1179 (post-empty-response retry;
    omits ``via``). ``via`` is NotRequired to capture the union."""
    required, optional = _required_optional_keys(AgentToolAutoResumeContext)
    assert required == {"agent", "thread_id", "tool_names", "iter"}
    assert optional == {"via"}


def test_agent_tool_result_schema_four_required_keys():
    """[IMPROVE-107] Fires from agents.py:1092 after each tool
    dispatch returns. ``call_id`` is the LangGraph run_id that
    correlates tool_call → tool_result pairs."""
    required, optional = _required_optional_keys(AgentToolResultContext)
    assert required == {"agent", "tool", "call_id", "thread_id"}
    assert optional == frozenset()


def test_editor_blend_with_previous_schema_three_required_keys():
    """[IMPROVE-107] Fires from images/editor.py:965 + 1002 via
    the shared ``_blend_ctx`` dict (line 947). ``blend`` is the
    0.0-1.0 mix factor; ``current_step`` is the step index the
    blend was applied against."""
    required, optional = _required_optional_keys(EditorBlendWithPreviousContext)
    assert required == {"session_id", "blend", "current_step"}
    assert optional == frozenset()


def test_editor_export_schema_quality_optional():
    """[IMPROVE-107] Fires from images/editor.py:1102
    (SessionNotFound; ``quality`` omitted because the lookup
    failed before quality was resolved), 1117 (other-error), 1124
    (ok). ``quality`` is NotRequired."""
    required, optional = _required_optional_keys(EditorExportContext)
    assert required == {"session_id", "format"}
    assert optional == {"quality"}


def test_editor_op_schema_dispatch_optional():
    """[IMPROVE-107] Fires from images/editor.py:731 (analyze ok
    with ``dispatch``), 741 (error), 820 (apply_edit ok). All
    three callsites spread ``_edit_ctx`` (line 700). ``dispatch``
    is NotRequired — only the analyze-success branch sets it."""
    required, optional = _required_optional_keys(EditorOpContext)
    assert required == {"session_id", "operation", "param_keys", "source"}
    assert optional == {"dispatch"}


def test_editor_redo_schema_history_keys_optional():
    """[IMPROVE-107] Fires from images/editor.py:869 (NothingToRedo
    error; only session_id) and 878 (ok; full shape). The
    redone_operation + current_step keys are NotRequired because
    the error path fires before the redo stack pop."""
    required, optional = _required_optional_keys(EditorRedoContext)
    assert required == {"session_id"}
    assert optional == {"redone_operation", "current_step"}


def test_editor_undo_schema_history_keys_optional():
    """[IMPROVE-107] Fires from images/editor.py:841 (NothingToUndo
    error) and 852 (ok). Same pattern as ``editor.redo``:
    history-bearing keys NotRequired because the error path can't
    populate them."""
    required, optional = _required_optional_keys(EditorUndoContext)
    assert required == {"session_id"}
    assert optional == {"undone_operation", "current_step"}


def test_image_load_schema_low_mem_and_backend_optional():
    """[IMPROVE-107] Fires from images/service.py:7592 (.start
    with full _load_ctx), 7580 (cache_hit shortcut omitting
    ``low_mem``), 7827 (nunchaku ok with ``backend``), 8244
    (regular ok). ``low_mem`` is NotRequired because the
    cache_hit path uses a minimal context; ``backend`` is
    NotRequired because only the nunchaku branch sets it."""
    required, optional = _required_optional_keys(ImageLoadContext)
    assert required == {"model_id", "mode", "device", "dtype", "cache_hit"}
    assert optional == {"low_mem", "backend"}


def test_image_load_and_start_share_same_schema():
    """[IMPROVE-107] Pin the schema-sharing convention: a single
    TypedDict covers both ``image.load`` and ``image.load.start``
    because the .start emit at service.py:7592 spreads the same
    ``_load_ctx`` variable that the ok-path emits use."""
    assert EVENT_CONTEXT_SCHEMAS[("image", "load")] is ImageLoadContext
    assert EVENT_CONTEXT_SCHEMAS[("image", "load.start")] is ImageLoadContext


def test_image_plan_schema_nullable_planner_fields():
    """[IMPROVE-107] Fires from images/service.py:9822 once per
    image generation request after the planner resolves model +
    device + backend + dtype + LoRA / init / mask flags. The
    detector-derived fields are nullable (``str | None``) because
    ``_img_hints.get`` and ``execution_plan.get`` return None
    when the upstream stage didn't populate them — they're always
    PRESENT in the dict, just sometimes None."""
    required, optional = _required_optional_keys(ImagePlanContext)
    assert required == {
        "model_id", "model_family", "model_variant", "device_plan",
        "inference_backend", "torch_dtype", "has_lora",
        "has_init_image", "has_mask", "scheduler",
    }
    assert optional == frozenset()


def test_image_postprocess_schema_four_required_keys():
    """[IMPROVE-107] Fires from images/service.py:10224 (ok) and
    10230 (error) via the shared ``_pp_ctx`` dict at line 10216.
    ``upscale`` and ``postprocess`` are stage-flag bools; the
    output_bytes perf field lands on emit_typed's ``perf=`` arg."""
    required, optional = _required_optional_keys(ImagePostprocessContext)
    assert required == {"model_id", "upscale", "postprocess", "input_bytes"}
    assert optional == frozenset()


def test_images_detect_hints_schema_signals_optional():
    """[IMPROVE-107] Fires from images/service.py:1469 with
    spread syntax (``{**signals, ...}``) so the audit walker
    skips the callsite. The schema covers the two always-present
    keys plus the known signals flags as NotRequired — pin the
    forward-looking shape so a future literal-context refactor
    inherits validation."""
    required, optional = _required_optional_keys(ImagesDetectHintsContext)
    assert required == {"family", "variant"}
    # The signal fields are best-effort optional — five known
    # detector flags pinned today; the schema is forward-looking.
    assert "has_safetensors_metadata" in optional
    assert "has_model_index_json" in optional


def test_instruct_edit_load_schema_gguf_keys_optional():
    """[IMPROVE-107] Fires from images/ai_enhance.py:2235
    (kontext/nunchaku via ``_load_context`` at line 2229) and
    2712 (cosxl with bare ``{"backend": "cosxl"}``). The
    GGUF-specific keys are kontext-only — NotRequired so cosxl
    + nunchaku don't fabricate values."""
    required, optional = _required_optional_keys(InstructEditLoadContext)
    assert required == {"backend"}
    assert optional == {"gguf_quant", "gguf_quant_overridden"}


def test_instruct_edit_run_start_shares_run_schema():
    """[IMPROVE-107] ``instruct_edit.run.start`` (ai_enhance.py:2185)
    spreads the same ``_ie_ctx`` dict that the IMPROVE-101
    ``instruct_edit.run`` callsites spread, so the .start
    companion REUSES ``InstructEditRunContext`` rather than
    introducing a new schema. Pin the reuse so a future split
    surfaces here."""
    assert EVENT_CONTEXT_SCHEMAS[("instruct_edit", "run")] is InstructEditRunContext
    assert EVENT_CONTEXT_SCHEMAS[("instruct_edit", "run.start")] is InstructEditRunContext


def test_partner_emotion_detect_schema_two_required_keys():
    """[IMPROVE-107] Fires from partner/engine.py:513 (tag-prefix
    detection) and 586 (heuristic fallback). ``source`` is one of
    "tag_prefix" / "heuristic_fallback" — pinned as ``str`` for
    future detector additions (LLM-classifier, embedding-based)."""
    required, optional = _required_optional_keys(PartnerEmotionDetectContext)
    assert required == {"emotion", "source"}
    assert optional == frozenset()


def test_partner_fact_extract_schema_two_required_keys():
    """[IMPROVE-107] Fires from partner/engine.py:733 once per
    user message after Mem0 fact-extraction. ``life_event_detected``
    is the boolean result of the heuristic life-event scanner —
    the new_facts perf field lands on emit_typed's ``perf=`` arg."""
    required, optional = _required_optional_keys(PartnerFactExtractContext)
    assert required == {"input_length", "life_event_detected"}
    assert optional == frozenset()


def test_partner_stt_schema_samples_optional():
    """[IMPROVE-107] Fires from 5 callsites in partner/engine.py
    covering file vs. buffer paths (995/1004/1010/1107/1117) plus
    the stt.partial throttled coalesce at 1131. ``samples`` is
    NotRequired — only the buffer-path callsites carry it."""
    required, optional = _required_optional_keys(PartnerSttContext)
    assert required == {"source"}
    assert optional == {"samples"}


def test_partner_stt_and_partial_share_same_schema():
    """[IMPROVE-107] Pin the schema-sharing convention: a single
    TypedDict covers both ``partner.stt`` and
    ``partner.stt.partial`` because partial uses a strict subset
    of stt's keys (just ``source: "buffer"``). A future stt.partial
    addition inherits validation automatically."""
    assert EVENT_CONTEXT_SCHEMAS[("partner", "stt")] is PartnerSttContext
    assert EVENT_CONTEXT_SCHEMAS[("partner", "stt.partial")] is PartnerSttContext


def test_partner_tts_schema_voice_and_skipped_empty_optional():
    """[IMPROVE-107] Fires from 5 callsites in partner/engine.py
    (1391/1400/1411/1436/1444). All callsites spread ``_tts_ctx``
    (line 1386). ``voice`` is NotRequired (kokoro-only); the
    ``skipped_empty`` bool lands only on the empty-text shortcut."""
    required, optional = _required_optional_keys(PartnerTtsContext)
    assert required == {"emotion", "input_length", "path"}
    assert optional == {"voice", "skipped_empty"}


def test_partner_voice_init_schema_no_required_keys():
    """[IMPROVE-107] Fires from partner/engine.py:893 (.start) and
    986 (ok). Both pass empty ``{}`` — the per-component status
    bools live on ``perf=``. The empty-shape schema pins the "no
    required context" contract so a future addition lands as a
    deliberate NotRequired update."""
    required, optional = _required_optional_keys(PartnerVoiceInitContext)
    assert required == frozenset()
    assert optional == frozenset()


def test_partner_voice_init_and_start_share_same_schema():
    """[IMPROVE-107] Pin the schema-sharing convention: a single
    TypedDict covers both ``partner.voice_init`` and the .start
    companion. Both callsites fire empty ``{}``; sharing one
    schema lets a future addition (e.g. driver_version on .start)
    land as a deliberate NotRequired update."""
    assert EVENT_CONTEXT_SCHEMAS[("partner", "voice_init")] is PartnerVoiceInitContext
    assert EVENT_CONTEXT_SCHEMAS[("partner", "voice_init.start")] is PartnerVoiceInitContext


def test_system_run_start_schema_streaming_optional():
    """[IMPROVE-107] Fires from systems/executor.py:668 (sync) and
    950 (streaming). Five required dimensional keys; ``streaming``
    is NotRequired — only the streaming-path callsite (line 957)
    sets it to True. Sibling ``SystemRunDoneContext`` covers the
    end event with a different shape (no node_count/edge_count)."""
    required, optional = _required_optional_keys(SystemRunStartContext)
    assert required == {
        "run_id", "system_name", "conversation_id",
        "node_count", "edge_count",
    }
    assert optional == {"streaming"}


def test_tool_calculator_eval_schema_single_key():
    """[IMPROVE-107] Fires from tools/builtin.py:163 (UnsafeExpression
    error), 171 (SyntaxError), 182 (EvalError), 190 (ok). All four
    callsites carry the single ``expression_length`` key — the
    error_code / error_message / result_type fields land on
    emit_typed's signature itself."""
    required, optional = _required_optional_keys(ToolCalculatorEvalContext)
    assert required == {"expression_length"}
    assert optional == frozenset()


def test_tool_file_ops_path_rejected_schema_single_key():
    """[IMPROVE-107] Fires from tools/file_ops.py:42 and
    tools/rag_tools.py:61 (both with
    ``error_code="PathOutsideWorkspace"``). The two callsites
    share a single ``user_path`` truncated to 200 chars at the
    call site."""
    required, optional = _required_optional_keys(ToolFileOpsPathRejectedContext)
    assert required == {"user_path"}
    assert optional == frozenset()


# ── Pydantic TypeAdapter validation (audit-time) ───────────────


def test_typeadapter_validates_image_vram_probe_strict():
    """[IMPROVE-92] Pydantic ``TypeAdapter`` over a TypedDict
    with ``__pydantic_config__ = ConfigDict(extra='forbid')``
    rejects unknown keys at runtime. Audit-time — never on the
    emit hot path. Pin the strict-mode behaviour."""
    adapter = TypeAdapter(ImageVramProbeContext)
    # Valid example. [IMPROVE-93] added tile_mode as required.
    valid = {
        "method": "sdxl_x4",
        "available_gb": 4.5,
        "required_gb": 6.0,
        "reason": "insufficient_vram",
        "ok": False,
        "tile_mode": False,
    }
    adapter.validate_python(valid)  # no raise
    # Extra key — must fail.
    with pytest.raises(Exception) as ei:
        adapter.validate_python({**valid, "EXTRA_TYPO": 1})
    assert "EXTRA_TYPO" in str(ei.value)
    # Missing required — must fail.
    with pytest.raises(Exception) as ei:
        adapter.validate_python({"method": "x"})
    msg = str(ei.value).lower()
    assert "missing" in msg or "field required" in msg


def test_typeadapter_validates_partner_mem0_init_partial():
    """[IMPROVE-92] A schema with both required + NotRequired
    fields validates a partial dict (only required keys
    present)."""
    adapter = TypeAdapter(PartnerMem0InitContext)
    # Only the required key — accepted.
    adapter.validate_python({"retry": False})
    # Required + one optional — also accepted.
    adapter.validate_python({"retry": True, "llm_model": "gemma3:1b"})
    # Missing required — fails.
    with pytest.raises(Exception):
        adapter.validate_python({"llm_model": "gemma3:1b"})


# ── Audit pin: callsite key sets match schemas ─────────────────


def _extract_emit_typed_callsite_contexts() -> list[
    tuple[str, str, str, frozenset[str]]
]:
    """Walk every ``.py`` under ``src/local_ai_platform`` and
    yield a tuple ``(file, subsystem, action, context_keys)``
    for every ``emit_typed("subsys", "action", ..., context={...})``
    callsite where the context arg is a literal dict. Variable
    or computed contexts (e.g. ``context=ctx_dict``) are
    skipped — they can't be statically validated.

    Uses Python's ``ast`` module rather than regex so multi-line
    + nested dict shapes parse correctly.
    """
    src_root = (
        Path(__file__).parent.parent
        / "src" / "local_ai_platform"
    )
    found: list[tuple[str, str, str, frozenset[str]]] = []
    for py in src_root.rglob("*.py"):
        # Don't scan the registry module itself — its emit_typed
        # def + the _emit alias would match falsely.
        if py.name == "observability_events.py":
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = py.read_text(encoding="latin-1")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # Shouldn't happen for valid Python; skip if it does.
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Match emit_typed(...) calls only — bare ``emit_typed``
            # name (not e.g. ``self.emit_typed``).
            func = node.func
            if not isinstance(func, ast.Name) or func.id != "emit_typed":
                continue
            # Need at least 2 positional args (subsystem + action),
            # both literal strings.
            if len(node.args) < 2:
                continue
            sub_arg, act_arg = node.args[0], node.args[1]
            if not (isinstance(sub_arg, ast.Constant)
                    and isinstance(sub_arg.value, str)):
                continue
            if not (isinstance(act_arg, ast.Constant)
                    and isinstance(act_arg.value, str)):
                continue
            # Find the ``context=`` kwarg.
            ctx_node = None
            for kw in node.keywords:
                if kw.arg == "context":
                    ctx_node = kw.value
                    break
            if ctx_node is None:
                # No context kwarg — schema validation N/A.
                continue
            if not isinstance(ctx_node, ast.Dict):
                # Variable / computed context — can't statically
                # extract keys. Skip.
                continue
            # [IMPROVE-101] Spread-syntax dicts (``{**ctx, "x": 1}``)
            # carry unknown keys that aren't statically
            # introspectable — Python's AST represents the spread
            # as a None key. Skip the callsite entirely rather
            # than false-flag on the visible literal keys, matching
            # the variable-context skip above. The audit
            # philosophy: best-effort static analysis, never
            # false-positive.
            if any(k is None for k in ctx_node.keys):
                continue
            # Extract the literal-string keys. Non-string keys
            # (e.g. computed expressions) are skipped, which means
            # the audit is best-effort but doesn't false-flag.
            keys: set[str] = set()
            for k in ctx_node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
            found.append((
                str(py),
                sub_arg.value,
                act_arg.value,
                frozenset(keys),
            ))
    return found


def test_emit_typed_callsite_keys_match_pinned_schema():
    """[IMPROVE-92] The pay-off audit. For every event WITH a
    pinned schema in ``EVENT_CONTEXT_SCHEMAS``, every
    ``emit_typed`` callsite for that event MUST:

      * include all ``__required_keys__`` (no missing required)
      * use only keys from ``__required_keys__ |
        __optional_keys__`` (no unexpected extras — typo'd keys
        get caught here)

    [IMPROVE-109] OPT-OUT FLIP: a callsite for a (sub, act)
    tuple WITHOUT a pinned schema is now a FAILURE rather than
    a silent skip. Pre-IMPROVE-109 this test skipped unregistered
    tuples (the audit was opt-in: registering a schema activated
    checks). Post-IMPROVE-107's 100% coverage every known
    (sub, act) HAS a schema, so a missing-schema callsite is by
    definition a regression — either:
      * a new event was added to the Literal but no schema was
        defined; OR
      * the callsite uses an unregistered tuple (which would
        also fail the keystone runtime check at emit_typed,
        but a static-analysis fail at lint is faster).

    Failure messages name the file, callsite signature, and the
    specific missing/extra keys so the fix is one grep.
    """
    callsites = _extract_emit_typed_callsite_contexts()
    failures: list[str] = []
    for path, subsystem, action, callsite_keys in callsites:
        schema = EVENT_CONTEXT_SCHEMAS.get((subsystem, action))
        if schema is None:
            # [IMPROVE-109] Opt-out: missing schema is a failure,
            # not a silent skip. Every (sub, act) tuple in
            # KNOWN_EVENT_NAMES has a schema as of [IMPROVE-107]
            # — this surfaces the regression class "added a new
            # event to the Literal but forgot the schema" at
            # CI time rather than runtime.
            failures.append(
                f"{path}: emit_typed({subsystem!r}, {action!r}, "
                f"context={sorted(callsite_keys)})"
                f"\n    UNREGISTERED schema — add an entry to "
                f"EVENT_CONTEXT_SCHEMAS or remove this callsite"
            )
            continue
        required, optional = _required_optional_keys(schema)
        all_allowed = required | optional
        missing = required - callsite_keys
        extras = callsite_keys - all_allowed
        if missing or extras:
            failures.append(
                f"{path}: emit_typed({subsystem!r}, {action!r}, "
                f"context={sorted(callsite_keys)})"
                + (f"\n    missing required: {sorted(missing)}"
                   if missing else "")
                + (f"\n    unexpected extras: {sorted(extras)}"
                   if extras else "")
            )
    if failures:
        pytest.fail(
            f"[IMPROVE-92] {len(failures)} emit_typed callsite(s) "
            f"don't match the pinned context schema:\n  "
            + "\n  ".join(failures)
        )


# ── [IMPROVE-102] Audit: track_event callsite shapes ───────────


def _extract_track_event_callsite_contexts() -> list[
    tuple[str, str, str, frozenset[str]]
]:
    """[IMPROVE-102] Walk every ``.py`` under
    ``src/local_ai_platform`` and yield ``(file, subsystem,
    action, context_keys)`` for every ``track_event(subsys,
    action, ...)`` callsite where the context arg is a literal
    dict.

    Mirror of the IMPROVE-92 emit_typed walker — same skip
    rules apply (variable context, spread-syntax dicts, no
    context kwarg). The Recorder pattern uses the same
    ``context=`` kwarg as emit_typed so the same shape
    extraction works.

    track_event's signature: ``track_event(subsystem, action,
    context=None)`` — context is the third positional OR a
    kwarg. Both forms are handled.

    Skip ``observability.py`` (the track_event def + docstring
    examples live there, would match falsely).
    """
    src_root = (
        Path(__file__).parent.parent
        / "src" / "local_ai_platform"
    )
    found: list[tuple[str, str, str, frozenset[str]]] = []
    for py in src_root.rglob("*.py"):
        # Don't scan the observability module — track_event def +
        # docstring snippets would match falsely.
        if py.name == "observability.py":
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = py.read_text(encoding="latin-1")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Name) or func.id != "track_event":
                continue
            if len(node.args) < 2:
                continue
            sub_arg, act_arg = node.args[0], node.args[1]
            if not (isinstance(sub_arg, ast.Constant)
                    and isinstance(sub_arg.value, str)):
                continue
            if not (isinstance(act_arg, ast.Constant)
                    and isinstance(act_arg.value, str)):
                continue
            # Find context — either third positional or kwarg.
            ctx_node = None
            if len(node.args) >= 3:
                ctx_node = node.args[2]
            else:
                for kw in node.keywords:
                    if kw.arg == "context":
                        ctx_node = kw.value
                        break
            if ctx_node is None:
                continue
            if not isinstance(ctx_node, ast.Dict):
                # Variable / computed context — skip (audit
                # philosophy: best-effort static analysis,
                # never false-positive).
                continue
            # Spread-syntax dicts have None keys — skip per
            # IMPROVE-101 walker convention.
            if any(k is None for k in ctx_node.keys):
                continue
            keys: set[str] = set()
            for k in ctx_node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
            found.append((
                str(py),
                sub_arg.value,
                act_arg.value,
                frozenset(keys),
            ))
    return found


def test_track_event_callsite_keys_match_pinned_schema():
    """[IMPROVE-102] Audit: every track_event callsite for an
    event WITH a pinned schema must match the schema's
    required/optional key sets — same contract as the
    emit_typed audit. Recorder fires both ``sub.act`` and
    ``sub.act.start`` from the same context dict, so the schema
    pinned for either tuple is enforced against the
    track_event callsite's literal keys.

    Variable-context + spread-syntax callsites are skipped per
    walker convention; only literal-dict callsites get
    validated. This is the pay-off pin for IMPROVE-102's six
    Recorder schemas.

    [IMPROVE-109] OPT-OUT FLIP: matches the emit_typed audit —
    a literal-context callsite for a (sub, act) WITHOUT a
    pinned schema is a FAILURE (was silent skip pre-IMPROVE-109).
    """
    callsites = _extract_track_event_callsite_contexts()
    failures: list[str] = []
    for path, subsystem, action, callsite_keys in callsites:
        schema = EVENT_CONTEXT_SCHEMAS.get((subsystem, action))
        if schema is None:
            # [IMPROVE-109] Opt-out: missing schema is a failure,
            # mirror of the emit_typed audit's flip. Note that
            # Recorder events fire BOTH ``sub.act`` and
            # ``sub.act.start`` from the same context dict —
            # the audit reports the canonical (the action as
            # passed to track_event, no .start suffix) so the
            # error message points at the right callsite.
            failures.append(
                f"{path}: track_event({subsystem!r}, {action!r}, "
                f"context={sorted(callsite_keys)})"
                f"\n    UNREGISTERED schema — add an entry to "
                f"EVENT_CONTEXT_SCHEMAS or remove this callsite"
            )
            continue
        required, optional = _required_optional_keys(schema)
        all_allowed = required | optional
        missing = required - callsite_keys
        extras = callsite_keys - all_allowed
        if missing or extras:
            failures.append(
                f"{path}: track_event({subsystem!r}, {action!r}, "
                f"context={sorted(callsite_keys)})"
                + (f"\n    missing required: {sorted(missing)}"
                   if missing else "")
                + (f"\n    unexpected extras: {sorted(extras)}"
                   if extras else "")
            )
    if failures:
        pytest.fail(
            f"[IMPROVE-102] {len(failures)} track_event callsite(s) "
            f"don't match the pinned context schema:\n  "
            + "\n  ".join(failures)
        )


def test_every_known_event_has_pinned_schema():
    """[IMPROVE-109] Strict pin: every (subsystem, action) tuple
    in ``KNOWN_EVENT_NAMES`` MUST have a corresponding entry in
    ``EVENT_CONTEXT_SCHEMAS``. The opt-out flip's structural
    defence — pre-IMPROVE-109 a new event added to the Literal
    types could ship without a schema (the emit-callsite audit
    skipped events without schemas). Post-IMPROVE-109 + post-
    IMPROVE-107's 100% coverage:

      * ``KNOWN_EVENT_NAMES`` is the authoritative event-name
        registry (66 tuples as of Wave 12).
      * ``EVENT_CONTEXT_SCHEMAS`` is the authoritative
        event-shape registry.
      * Both must agree on membership: a tuple in one MUST be
        in the other.

    A failure here means one of two things:

      * A new event was added to ``XxxAction = Literal[...]``
        without a corresponding TypedDict + EVENT_CONTEXT_SCHEMAS
        entry. Fix: add the schema (mirror of IMPROVE-107's
        pattern).
      * A schema was deleted but the event_name stayed
        registered. Fix: remove the event_name from the Literal
        if the event is gone, OR re-add the schema.

    The keystone runtime check (``UnknownEventNameError`` in
    emit_typed) catches the inverse direction (a callsite using
    an UNregistered name) at runtime. This test catches the
    same regression class at CI time + extends to the schema
    side.

    Empty-context events (e.g. ``partner.voice_init``) STILL
    need a schema entry — the empty-shape TypedDict pins the
    "no required context" contract per IMPROVE-107.

    Failure messages list the missing tuples sorted so a fresh
    contributor can see exactly which schemas to add.
    """
    from local_ai_platform.observability_events import (
        KNOWN_EVENT_NAMES,
    )
    pinned = set(EVENT_CONTEXT_SCHEMAS.keys())
    known: set[tuple[str, str]] = set()
    for name in KNOWN_EVENT_NAMES:
        sub, _, act = name.partition(".")
        known.add((sub, act))

    missing = sorted(known - pinned)
    if missing:
        pytest.fail(
            f"[IMPROVE-109] {len(missing)} known event(s) lack a "
            f"pinned context schema. Every entry in "
            f"KNOWN_EVENT_NAMES MUST have a TypedDict + "
            f"EVENT_CONTEXT_SCHEMAS entry. Missing:\n  "
            + "\n  ".join(f"({s!r}, {a!r})" for s, a in missing)
        )

    # Symmetric pin: nothing in EVENT_CONTEXT_SCHEMAS should be
    # absent from KNOWN_EVENT_NAMES (a stale entry would silently
    # validate against a never-emitted event). Today this is
    # tautologically true (the audit walker test verifies the
    # forward direction); pin the reverse so a future commit
    # adding a schema for an unregistered event surfaces here.
    extras = sorted(pinned - known)
    if extras:
        pytest.fail(
            f"[IMPROVE-109] {len(extras)} pinned schema(s) reference "
            f"unregistered event tuples (stale schemas?):\n  "
            + "\n  ".join(f"({s!r}, {a!r})" for s, a in extras)
        )


def test_pinned_schema_count_grows_or_stays():
    """[IMPROVE-92] Audit pin: the number of pinned schemas in
    ``EVENT_CONTEXT_SCHEMAS`` should only ever GROW (or stay
    the same), never shrink without a deliberate baseline bump.

    Today: 66 pinned schemas (6 from [IMPROVE-92] + 12 from
    [IMPROVE-95]'s Wave 10 batch + 10 from [IMPROVE-101]'s
    Wave 11 Tier-A batch + 12 from [IMPROVE-102]'s Wave 11
    Recorder batch + 26 from [IMPROVE-107]'s Wave 12 final-tier
    batch — 100% coverage of the registered (subsystem, action)
    tuples). If a future commit needs to delete one (e.g. event
    renamed), update this baseline AND update
    ``EVENT_CONTEXT_SCHEMAS`` in the same commit so the intent
    is explicit in code review.

    [IMPROVE-109] kept this count baseline alongside the new
    ``test_every_known_event_has_pinned_schema`` strict pin
    because the two tests catch DIFFERENT regression classes:

      * count baseline: schema removed (count drops below 66)
      * strict pin: event added without schema (or schema
        removed but event-name kept — tuple-mismatch)

    Defence in depth — both pin essentials of the type-safety
    contract from different angles. The count baseline ALSO
    gives a more readable failure message ("count dropped from
    66 to 65") than the strict pin's "tuple X is missing".
    """
    pinned_count = len(EVENT_CONTEXT_SCHEMAS)
    minimum_pinned = 66  # baseline as of [IMPROVE-107] (100% coverage)
    assert pinned_count >= minimum_pinned, (
        f"[IMPROVE-92] Pinned schema count dropped below the "
        f"baseline ({pinned_count} < {minimum_pinned}). "
        f"Update the baseline in this test only as a deliberate "
        f"choice, alongside the EVENT_CONTEXT_SCHEMAS change."
    )


# ── No callsite under audit covered by Pydantic at runtime ─────


def test_emit_typed_does_not_use_pydantic_at_runtime():
    """[IMPROVE-92] Q2=C explicitly chose: TypedDict for the
    static front, pydantic ``TypeAdapter`` only at audit time.
    The emit_typed implementation MUST NOT reference pydantic
    on the hot path — that would add ~5-10ms per emit,
    multiplied by hundreds of emits per request.

    Pin the discipline by source-grepping the emit_typed body
    for ``pydantic`` / ``TypeAdapter`` references. The schema
    section IS allowed to import pydantic for the
    ``ConfigDict``, but the function body itself isn't.
    """
    src = (
        Path(__file__).parent.parent
        / "src" / "local_ai_platform"
        / "observability_events.py"
    ).read_text(encoding="utf-8")
    # Find the emit_typed *implementation* (not @overload defs)
    # — it's the only def emit_typed without a leading @overload.
    # Crude but reliable: locate the IMPLEMENTATION by finding
    # the def that doesn't have @overload on the line above.
    lines = src.splitlines()
    impl_start = None
    for i, line in enumerate(lines):
        if line.startswith("def emit_typed("):
            # Check the previous non-blank line for @overload
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j < 0 or not lines[j].strip().startswith("@overload"):
                impl_start = i
                break
    assert impl_start is not None, "emit_typed implementation not found"
    # Walk forward to find the end of the function (next
    # top-level def or end of file).
    impl_end = len(lines)
    for i in range(impl_start + 1, len(lines)):
        # A line starting with non-whitespace that's a def/class/etc
        # OR a section comment (starting with `# ──`) marks the end
        # of the function.
        line = lines[i]
        if line and not line[0].isspace():
            impl_end = i
            break
    impl_body = "\n".join(lines[impl_start:impl_end])
    # The forbidden references: a runtime pydantic call from
    # inside the implementation. ``ConfigDict`` is fine because
    # it's a class, not a runtime validator. ``TypeAdapter`` is
    # the heavy one.
    forbidden = ["TypeAdapter", "BaseModel", "model_validate"]
    for name in forbidden:
        assert name not in impl_body, (
            f"[IMPROVE-92] Forbidden pydantic-runtime reference "
            f"{name!r} found in emit_typed body. Q2=C explicitly "
            f"chose audit-time validation only — moving validation "
            f"to the hot path adds ~5-10ms per emit."
        )
