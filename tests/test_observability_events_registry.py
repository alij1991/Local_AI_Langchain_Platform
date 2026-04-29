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
    r"""emit\(\s*["']([a-z_]+)["']\s*,\s*["']([a-z_]+)["']""",
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
