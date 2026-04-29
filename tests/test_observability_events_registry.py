"""[IMPROVE-NEW-16] Tests for the telemetry event-name registry.

The registry's pay-off is the keystone test
``test_registry_covers_every_emit_callsite_in_codebase`` which
walks the source tree and asserts every literal-string ``emit()``
call uses a registered event. Without that test, the registry
just drifts and the typed front door becomes ceremony — those
tests + the wrapper would catch typos in NEW callsites but old
callsites would silently keep working with misspelled names.

The test scans every ``.py`` file under ``src/local_ai_platform``
and matches the regex form ``emit("subsys", "action", ...)``.
Variable-action callsites (``emit("editor", op_name, ...)``) are
intentionally ignored — those can't be statically validated.
There are 0 such callsites in the codebase today (audited
2026-04-28); if a future commit introduces one, the audit pin
``test_no_dynamic_action_emit_callsites_in_codebase`` will fail
and the author can either keep the registry honest by
materialising the action set at the callsite or grant an
explicit exemption.

Sources (2025-2026):
  * docs/features/09-observability.md — internal observability
    surface this module formalises.
  * Wave 6 IMPROVE-36/40/44/35 telemetry commits — eight new
    event names that motivated this registry.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from local_ai_platform.observability_events import (
    emit_typed,
    KNOWN_EVENT_NAMES,
    KNOWN_EVENTS,
    SubsystemName,
    UnknownEventNameError,
)


# ── Registry shape ────────────────────────────────────────────


def test_known_event_names_is_frozenset():
    assert isinstance(KNOWN_EVENT_NAMES, frozenset)


def test_known_events_per_subsystem_is_frozenset():
    """Per-subsystem entries are frozen so a runtime mutation
    can't unregister an event mid-process."""
    for subsystem, actions in KNOWN_EVENTS.items():
        assert isinstance(actions, frozenset), (
            f"KNOWN_EVENTS[{subsystem!r}] must be frozenset, "
            f"got {type(actions).__name__}"
        )


def test_known_event_names_includes_all_known_events():
    """Round-trip: the flat set is exactly the cross product of
    KNOWN_EVENTS subsystems and their actions."""
    expected = {
        f"{subsystem}.{action}"
        for subsystem, actions in KNOWN_EVENTS.items()
        for action in actions
    }
    assert KNOWN_EVENT_NAMES == frozenset(expected)


def test_no_duplicate_event_names_across_subsystems():
    """A "subsystem.action" string should be unique. This catches
    a future copy-paste where someone accidentally registers
    ``image.load`` and ``IMAGE.load`` (case-mismatch) or
    duplicates a tuple entry."""
    flat = []
    for subsystem, actions in KNOWN_EVENTS.items():
        for action in actions:
            flat.append(f"{subsystem}.{action}")
    assert len(flat) == len(set(flat)), (
        "Duplicate event name in registry: "
        f"{[x for x in flat if flat.count(x) > 1]}"
    )


def test_subsystem_name_literal_includes_all_registered_subsystems():
    """Every key in KNOWN_EVENTS must be a value of the
    SubsystemName Literal so emit_typed's mypy check matches the
    runtime check."""
    # Pull the args out of the Literal at runtime.
    from typing import get_args
    literal_subsystems = set(get_args(SubsystemName))
    registered = set(KNOWN_EVENTS.keys())
    assert registered == literal_subsystems, (
        f"SubsystemName Literal drift: "
        f"only-in-Literal={literal_subsystems - registered}, "
        f"only-in-KNOWN_EVENTS={registered - literal_subsystems}"
    )


# ── emit_typed wrapper ────────────────────────────────────────


def test_emit_typed_accepts_known_event(monkeypatch):
    """Happy path: a registered (subsystem, action) reaches the
    underlying emit() with the same args."""
    captured = []

    def fake_emit(subsystem, action, status="ok", **kw):
        captured.append((subsystem, action, status, kw))

    from local_ai_platform import observability_events as oe
    monkeypatch.setattr(oe, "_emit", fake_emit)

    emit_typed(
        "system", "wave_parallel",
        duration_ms=42,
        context={"node_count": 3},
        perf={"node_count": 3},
    )

    assert len(captured) == 1
    sub, action, status, kw = captured[0]
    assert sub == "system"
    assert action == "wave_parallel"
    assert status == "ok"
    assert kw["duration_ms"] == 42
    assert kw["context"] == {"node_count": 3}
    assert kw["perf"] == {"node_count": 3}


def test_emit_typed_rejects_unknown_action():
    """Misspelled action raises UnknownEventNameError. The
    message names the subsystem and lists registered actions so
    the engineer can fix the typo without grepping the registry."""
    with pytest.raises(UnknownEventNameError) as ei:
        emit_typed("system", "wave_paralel")  # typo: missing 'l'
    msg = str(ei.value)
    assert "system.wave_paralel" in msg
    assert "wave_parallel" in msg  # registered actions listed
    assert "Register it in observability_events.py" in msg


def test_emit_typed_rejects_unknown_subsystem():
    """Unknown subsystem also raises (Literal catches at mypy
    time but a runtime caller via getattr/dynamic could still
    pass an unknown string). The error message lists "(none)"
    when the subsystem isn't registered at all."""
    with pytest.raises(UnknownEventNameError) as ei:
        # type: ignore — testing the runtime path
        emit_typed("nonexistent", "frob")  # type: ignore[arg-type]
    msg = str(ei.value)
    assert "nonexistent.frob" in msg
    assert "(none)" in msg


def test_emit_typed_does_not_call_emit_when_unknown(monkeypatch):
    """Validation must run BEFORE the emit() call so a misspelled
    name doesn't pollute app_events. Pin via a fake emit that
    would record any call."""
    called = []

    def fake_emit(*a, **kw):
        called.append((a, kw))

    from local_ai_platform import observability_events as oe
    monkeypatch.setattr(oe, "_emit", fake_emit)

    with pytest.raises(UnknownEventNameError):
        emit_typed("system", "bogus_action_name")

    assert called == [], (
        "emit_typed must not call _emit when the action is "
        "unregistered; it leaked through with: " + repr(called)
    )


def test_emit_typed_propagates_status_and_error_fields(monkeypatch):
    """All kwargs forwarded — error path with status='error',
    error_code, error_message."""
    captured = []

    def fake_emit(subsystem, action, status="ok", **kw):
        captured.append((subsystem, action, status, kw))

    from local_ai_platform import observability_events as oe
    monkeypatch.setattr(oe, "_emit", fake_emit)

    emit_typed(
        "system", "node_end",
        status="error",
        error_code="OOM",
        error_message="cuda out of memory",
        context={"node": "img_n"},
    )

    sub, action, status, kw = captured[0]
    assert status == "error"
    assert kw["error_code"] == "OOM"
    assert kw["error_message"] == "cuda out of memory"


# ── Keystone: every emit() callsite uses a registered event ────


_EMIT_LITERAL_PATTERN = re.compile(
    # Matches both ``emit("subsys", "action", ...)`` and
    # ``emit_typed("subsys", "action", ...)``. Word boundary keeps
    # ``self.emit`` / ``recorder.subsystem_event`` etc. out — only
    # the bare ``emit`` and ``emit_typed`` front doors count.
    #
    # [IMPROVE-89] tightened the action group from ``[a-z_]+`` to
    # ``[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*`` so digit-bearing
    # actions (``mem0_init``) and dotted actions
    # (``download.start``, ``file_ops.path_rejected``) are no
    # longer invisible to coverage. Pre-tightening, those
    # callsites silently fired with unregistered names because the
    # regex skipped them — bug noticed during the bulk migration.
    r"""\bemit(?:_typed)?\(\s*["']([a-z_]+)["']\s*,\s*["']([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*)["']""",
    re.MULTILINE,
)


def _scan_emit_callsites() -> set[tuple[str, str, str]]:
    """Walk src/local_ai_platform/**.py and yield (file, subsystem,
    action) for every literal-string emit() call. Returns a set of
    triples for de-dup; the file path is included so a failing test
    can point at the unregistered callsite."""
    src_root = Path(__file__).parent.parent / "src" / "local_ai_platform"
    found: set[tuple[str, str, str]] = set()
    for py in src_root.rglob("*.py"):
        # Don't scan the registry module itself — its own _emit
        # alias counts as a call but isn't a real event.
        if py.name == "observability_events.py":
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = py.read_text(encoding="latin-1")
        for m in _EMIT_LITERAL_PATTERN.finditer(text):
            found.add((str(py), m.group(1), m.group(2)))
    return found


def test_registry_covers_every_emit_callsite_in_codebase():
    """The pay-off test. Every literal-string emit() call in the
    codebase MUST use a (subsystem, action) pair registered in
    KNOWN_EVENT_NAMES. A new event added without registration
    fails this test; the engineer adds the entry to
    observability_events.py and the test passes.

    Failure mode is informative: lists each unregistered event
    with the file path so the fix is one grep.
    """
    callsites = _scan_emit_callsites()
    unregistered = []
    for path, subsystem, action in callsites:
        if f"{subsystem}.{action}" not in KNOWN_EVENT_NAMES:
            unregistered.append((path, f"{subsystem}.{action}"))
    if unregistered:
        lines = [
            f"  {path}: emit({subsystem_action!r})"
            for path, subsystem_action in sorted(unregistered)
        ]
        pytest.fail(
            "emit() callsites use unregistered events:\n"
            + "\n".join(lines)
            + "\n\nRegister them in "
            "src/local_ai_platform/observability_events.py "
            "(KNOWN_EVENTS dict) so the typed front door + "
            "registry stay in sync."
        )


def test_registry_has_no_dead_entries():
    """Inverse pin: every registered event has at least one
    callsite in the codebase. Catches dead entries that linger
    after a feature is removed (they'd otherwise appear as
    valid-but-never-fired events in dashboards).

    Skips events from modules NOT YET migrated — when a new event
    is added to the registry as preparation for a callsite that
    lands in a follow-up commit, that's intentional. Today every
    registered event corresponds to a callsite, so the audit is
    strict.
    """
    callsites = _scan_emit_callsites()
    callsite_pairs = {
        f"{subsystem}.{action}"
        for _path, subsystem, action in callsites
    }
    dead = KNOWN_EVENT_NAMES - callsite_pairs
    if dead:
        pytest.fail(
            "Registered events with no emit() callsite "
            "(dead registry entries):\n  "
            + "\n  ".join(sorted(dead))
            + "\n\nRemove from observability_events.py if the "
            "feature is gone, or add a callsite if the event is "
            "intended to fire."
        )


def test_no_dynamic_action_emit_callsites_in_codebase():
    """Audit pin: there should be ZERO ``emit(subsystem, var, ...)``
    callsites in the codebase. A future addition would invalidate
    the keystone test's coverage guarantee — the engineer would
    need to either (a) materialise the var to a literal at the
    callsite or (b) grant an explicit exemption (and then the
    inverse test_registry_has_no_dead_entries would need to
    accommodate the dynamic action).

    Counts 0 today (audited 2026-04-28). Failure here means the
    keystone test's coverage guarantee just got weaker.
    """
    src_root = Path(__file__).parent.parent / "src" / "local_ai_platform"
    # emit( "subsys" , <something not starting with quote>
    pattern = re.compile(
        r"""emit\(\s*["'][a-z_]+["']\s*,\s*([^"',\s)][^,)]*),"""
    )
    dynamic = []
    for py in src_root.rglob("*.py"):
        if py.name == "observability_events.py":
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = py.read_text(encoding="latin-1")
        for m in pattern.finditer(text):
            # Skip comment lines + empty matches.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line = text[line_start:text.find("\n", m.end())]
            if line.lstrip().startswith("#"):
                continue
            arg = m.group(1).strip()
            # Heuristic: an unquoted arg is dynamic. But our regex
            # allows constants like None, integers, etc. — those
            # aren't realistic for an action name. Keep the audit
            # strict and let any matches fail.
            if arg and not arg.startswith(("'", '"')):
                dynamic.append((str(py), arg))
    assert not dynamic, (
        "Found dynamic-action emit() callsites that escape the "
        "registry's static coverage:\n  "
        + "\n  ".join(f"{p}: emit(..., {a}, ...)" for p, a in dynamic[:5])
    )


def test_no_bare_emit_imports_in_src_after_bulk_migration():
    """[IMPROVE-89] Pin: no ``from local_ai_platform.observability
    import emit`` (or relative form) should appear in any production
    module after the Wave 9 bulk migration to ``emit_typed``.

    Why this is stricter than the keystone callsite test: the
    keystone catches the missing-registry case, but a future revert
    that re-imports ``emit`` and calls it WITH a registered name
    would slip through the keystone (no unregistered emit). This pin
    makes the typed front door the only legitimate route, so a
    future contributor can't accidentally bypass the runtime
    UnknownEventNameError check + bypass the per-event schemas
    landing in IMPROVE-92.

    Exempts ``observability_events.py`` (defines the ``_emit`` alias)
    and ``observability.py`` (defines ``emit`` itself). Comment
    references are skipped via the "#" prefix check.
    """
    src_root = Path(__file__).parent.parent / "src" / "local_ai_platform"
    # Match: from <path>.observability import emit  (with optional
    # other names, but the bare ``emit`` must appear in the import
    # list). Skips ``emit_typed`` and ``track_event`` etc.
    pattern = re.compile(
        r"""^from\s+\S*observability\s+import\s+([^\n]+)$""",
        re.MULTILINE,
    )
    # The forbidden names list — a real import line can include any
    # of these but ``emit`` (without the ``_typed`` suffix) is
    # specifically forbidden.
    offenders: list[tuple[str, str]] = []
    EXEMPT_FILES = {"observability_events.py", "observability.py"}
    for py in src_root.rglob("*.py"):
        if py.name in EXEMPT_FILES:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = py.read_text(encoding="latin-1")
        for m in pattern.finditer(text):
            imports = m.group(1)
            # Split on commas, strip "as X" aliases.
            names = [
                n.strip().split(" as ")[0].strip()
                for n in imports.split(",")
            ]
            # Strip parentheses from multiline imports if present.
            names = [n.strip("()") for n in names if n.strip("()")]
            if "emit" in names:
                offenders.append((str(py), m.group(0)))
    assert not offenders, (
        "[IMPROVE-89] Found bare-``emit`` imports that bypass the "
        "typed front door — migrate the file to ``emit_typed``:\n  "
        + "\n  ".join(
            f"{p}: {line}" for p, line in offenders[:10]
        )
    )


# ── [IMPROVE-91] Per-subsystem Literal + @overload pins ─────────


# Sub-tuple: (Literal-type-name, KNOWN_EVENTS-key). The Literal
# type's ``get_args`` MUST equal the KNOWN_EVENTS frozenset for
# the same subsystem — derivation flows ``Literal -> tuple ->
# frozenset`` so a drift means someone bypassed get_args.
_LITERAL_KEY_PAIRS = [
    ("AgentAction", "agent"),
    ("ConfigAction", "config"),
    ("EditorAction", "editor"),
    ("ImageAction", "image"),
    ("ImagesAction", "images"),
    ("InstructEditAction", "instruct_edit"),
    ("ModelAction", "model"),
    ("PartnerAction", "partner"),
    ("ProviderAction", "provider"),
    ("SystemAction", "system"),
    ("ToolAction", "tool"),
]


@pytest.mark.parametrize("literal_name,subsystem_key", _LITERAL_KEY_PAIRS)
def test_per_subsystem_literal_args_match_known_events_frozenset(
    literal_name, subsystem_key,
):
    """[IMPROVE-91] Each per-subsystem ``Literal`` type's
    ``get_args()`` MUST match its corresponding ``KNOWN_EVENTS``
    frozenset. Pin the source-of-truth contract: a drift means
    the ``Literal -> tuple -> frozenset`` derivation chain has
    been bypassed (e.g. someone added an entry to KNOWN_EVENTS
    without updating the Literal, which would silently lose
    mypy coverage for the new action).
    """
    from typing import get_args as _get_args
    import local_ai_platform.observability_events as oe

    literal_type = getattr(oe, literal_name)
    literal_actions = frozenset(_get_args(literal_type))
    registry_actions = oe.KNOWN_EVENTS[subsystem_key]
    assert literal_actions == registry_actions, (
        f"[IMPROVE-91] Drift: {literal_name} args differ from "
        f"KNOWN_EVENTS[{subsystem_key!r}]. "
        f"Only-in-Literal: {literal_actions - registry_actions}. "
        f"Only-in-registry: {registry_actions - literal_actions}."
    )


def test_per_subsystem_literal_types_exist_for_every_subsystem():
    """[IMPROVE-91] Every key in ``KNOWN_EVENTS`` must have a
    corresponding ``<Subsystem>Action`` Literal type exported
    from the module. Pinned so a future contributor adding a new
    subsystem can't skip the per-subsystem Literal step.
    """
    import local_ai_platform.observability_events as oe

    expected = {
        # Map registry key to expected type name
        "agent": "AgentAction",
        "config": "ConfigAction",
        "editor": "EditorAction",
        "image": "ImageAction",
        "images": "ImagesAction",
        "instruct_edit": "InstructEditAction",
        "model": "ModelAction",
        "partner": "PartnerAction",
        "provider": "ProviderAction",
        "system": "SystemAction",
        "tool": "ToolAction",
    }
    registry_keys = set(oe.KNOWN_EVENTS.keys())
    expected_keys = set(expected.keys())
    assert registry_keys == expected_keys, (
        f"[IMPROVE-91] KNOWN_EVENTS keys diverged from the "
        f"expected per-subsystem Literal mapping. "
        f"Only-in-registry: {registry_keys - expected_keys}. "
        f"Only-in-mapping: {expected_keys - registry_keys}."
    )
    for subsystem_key, type_name in expected.items():
        assert hasattr(oe, type_name), (
            f"[IMPROVE-91] Subsystem {subsystem_key!r} has a "
            f"KNOWN_EVENTS entry but no {type_name} Literal "
            f"exported from observability_events.py. Add "
            f"``{type_name} = Literal[...]`` alongside the "
            f"sibling types."
        )


def test_per_subsystem_overload_count_matches_subsystem_count():
    """[IMPROVE-91] Each subsystem in ``KNOWN_EVENTS`` MUST have
    exactly one ``@overload`` signature for ``emit_typed``. If a
    new subsystem is added without an overload, mypy silently
    drops to the impl signature (which accepts ``action: str``)
    for that subsystem — the lint catch is lost.

    This pin counts ``@overload``-decorated ``emit_typed`` defs
    in the source by static parsing.
    """
    src = (
        Path(__file__).parent.parent
        / "src" / "local_ai_platform"
        / "observability_events.py"
    ).read_text(encoding="utf-8")
    # Match the @overload + def emit_typed(... pattern. Each
    # overload starts with "@overload\ndef emit_typed("
    overload_pattern = re.compile(
        r"^@overload\s*\ndef\s+emit_typed\s*\(",
        re.MULTILINE,
    )
    overload_count = len(overload_pattern.findall(src))
    import local_ai_platform.observability_events as oe
    subsystem_count = len(oe.KNOWN_EVENTS)
    assert overload_count == subsystem_count, (
        f"[IMPROVE-91] Overload count ({overload_count}) doesn't "
        f"match subsystem count ({subsystem_count}). Either an "
        f"@overload was added without a corresponding KNOWN_EVENTS "
        f"entry, or vice-versa. Adding a new subsystem requires "
        f"three coordinated edits: (1) Literal type, (2) "
        f"KNOWN_EVENTS entry, (3) @overload def."
    )


def test_emit_typed_typo_action_still_raises_at_runtime():
    """[IMPROVE-91] Static type-checking via @overload doesn't
    eliminate the runtime check — it adds a layer on top.
    Dynamic callers (test fixtures using setattr, future
    metaprogramming, modules deferred-loading the registry) can
    still pass an unknown action and MUST get
    ``UnknownEventNameError`` at runtime.

    Pin the safety-net behaviour so a future "we have static
    checks now, drop the runtime check" refactor can't slip
    through.
    """
    from local_ai_platform.observability_events import (
        emit_typed,
        UnknownEventNameError,
    )
    with pytest.raises(UnknownEventNameError) as ei:
        # Cast away static type-checking for the dynamic call.
        # The runtime check MUST still fire.
        emit_typed("agent", "tool_calll")  # type: ignore[arg-type]
    msg = str(ei.value)
    assert "agent.tool_calll" in msg
    assert "tool_call" in msg  # registered actions listed for hint


def test_emit_typed_overloads_alphabetised_by_subsystem_in_source():
    """[IMPROVE-91] The 11 ``@overload`` signatures appear in
    alphabetical order of subsystem name in the source file.
    Pin the convention so a future addition lands at the right
    place — keeps diffs reviewable.
    """
    src = (
        Path(__file__).parent.parent
        / "src" / "local_ai_platform"
        / "observability_events.py"
    ).read_text(encoding="utf-8")
    # Each overload starts with @overload then has
    # ``subsystem: Literal["X"]`` on the next line.
    overload_subsystem_pattern = re.compile(
        r"@overload\s*\ndef\s+emit_typed\s*\(\s*\n\s*subsystem:\s*Literal\[\"([a-z_]+)\"\]",
        re.MULTILINE,
    )
    subsystems_in_order = overload_subsystem_pattern.findall(src)
    assert subsystems_in_order == sorted(subsystems_in_order), (
        f"[IMPROVE-91] @overload subsystems not alphabetised. "
        f"Found order: {subsystems_in_order}. "
        f"Expected: {sorted(subsystems_in_order)}."
    )
