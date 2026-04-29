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
    AgentValidationRejectedContext,
    ImageVramProbeContext,
    PartnerMem0InitContext,
    SystemValidationRejectedContext,
    SystemWaveParallelContext,
    SystemWaveParallelFallbackContext,
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
    """[IMPROVE-87] The vram_probe event has a fixed shape — all
    five keys (method/available_gb/required_gb/reason/ok) are
    required. Pin the strict-shape contract."""
    required, optional = _required_optional_keys(ImageVramProbeContext)
    assert required == {
        "method", "available_gb", "required_gb", "reason", "ok",
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


# ── Pydantic TypeAdapter validation (audit-time) ───────────────


def test_typeadapter_validates_image_vram_probe_strict():
    """[IMPROVE-92] Pydantic ``TypeAdapter`` over a TypedDict
    with ``__pydantic_config__ = ConfigDict(extra='forbid')``
    rejects unknown keys at runtime. Audit-time — never on the
    emit hot path. Pin the strict-mode behaviour."""
    adapter = TypeAdapter(ImageVramProbeContext)
    # Valid example.
    valid = {
        "method": "sdxl_x4",
        "available_gb": 4.5,
        "required_gb": 6.0,
        "reason": "insufficient_vram",
        "ok": False,
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

    Events WITHOUT a pinned schema are skipped silently (no
    regression risk). Adding a schema raises the bar for that
    event going forward.

    Failure messages name the file, callsite signature, and the
    specific missing/extra keys so the fix is one grep.
    """
    callsites = _extract_emit_typed_callsite_contexts()
    failures: list[str] = []
    for path, subsystem, action, callsite_keys in callsites:
        schema = EVENT_CONTEXT_SCHEMAS.get((subsystem, action))
        if schema is None:
            continue  # event without pinned schema — skip
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


def test_pinned_schema_count_grows_or_stays():
    """[IMPROVE-92] Audit pin: the number of pinned schemas in
    ``EVENT_CONTEXT_SCHEMAS`` should only ever GROW (or stay
    the same), never shrink. A future commit removing a schema
    would silently drop coverage for that event — the audit's
    "events without schemas are skipped" semantics means a
    deletion goes unnoticed.

    Today: 6 pinned schemas. If a future commit needs to
    delete one (e.g. event renamed), update this baseline AND
    update ``EVENT_CONTEXT_SCHEMAS`` in the same commit so the
    intent is explicit in code review.
    """
    pinned_count = len(EVENT_CONTEXT_SCHEMAS)
    minimum_pinned = 6  # baseline as of [IMPROVE-92]
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
