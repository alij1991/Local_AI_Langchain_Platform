"""[IMPROVE-106] CI guard: ``observability_events.py`` must pass
``mypy --strict``.

Per Q4=A in the Wave 11 plan: scope the strict check to a single
file (the typed-event-registry module) rather than the whole
package. Smaller delta + the registry is the highest-leverage
type-safety surface — every emit_typed callsite in the codebase
type-checks against the per-subsystem ``Literal`` types declared
here. A regression that loses a Literal annotation (e.g. someone
adds a new subsystem with plain ``str`` actions) silently weakens
type-checking across all 80+ emit_typed callsites.

The test runs mypy in-process via ``mypy.api.run`` so:
  * No subprocess overhead (mypy daemon-style invocation in-test).
  * Exit code + stdout are programmatically inspectable.
  * Failure messages name the line + the error code so the fix
    is one grep.

The test is gated by ``pytest.importorskip("mypy")`` so a dev
environment without mypy installed (the package itself doesn't
depend on mypy at runtime) doesn't fail this — it just skips.
CI installs mypy explicitly; locally a contributor wanting to
run the full Tier 1 sweep needs ``pip install mypy``.

Sources (2025-2026):
  * mypy 1.13+ ``mypy.api.run`` for programmatic invocation:
    https://mypy.readthedocs.io/en/stable/extending_mypy.html
  * Wave 9 [IMPROVE-91] commit (578d1d0) — per-subsystem
    Literal types this strict check pins.
  * Wave 9 [IMPROVE-92] commit (3c29667) — TypedDict context
    schemas this strict check covers.
  * docs/features/09-observability.md — internal observability
    surface this guard pins.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_observability_events_passes_mypy_strict():
    """[IMPROVE-106] ``mypy --strict`` over
    ``src/local_ai_platform/observability_events.py`` must
    return zero errors. Pin the strict-mode contract so a
    future commit weakening any annotation surfaces here.

    A failure prints the full mypy stdout so the line + error
    code are visible without re-running.
    """
    pytest.importorskip("mypy")
    from mypy.api import run as mypy_run

    target = (
        Path(__file__).parent.parent
        / "src" / "local_ai_platform"
        / "observability_events.py"
    )
    assert target.exists(), f"target file missing: {target}"

    stdout, stderr, exit_code = mypy_run([
        "--strict",
        str(target),
    ])

    if exit_code != 0:
        pytest.fail(
            f"[IMPROVE-106] mypy --strict failed on {target.name} "
            f"(exit {exit_code}):\n"
            f"--- stdout ---\n{stdout}"
            f"--- stderr ---\n{stderr}"
        )


def test_per_subsystem_action_tuples_carry_literal_types():
    """[IMPROVE-106] The 12 ``_<SUBSYSTEM>_ACTIONS`` derivation
    tuples are typed ``tuple[<Action>, ...]`` not plain
    ``tuple[str, ...]``. Pin the literal-typing convention so
    a future commit reverting to ``str`` surfaces here without
    having to inspect mypy output.

    Inspects the source via ``ast`` (rather than runtime
    introspection — the assignment annotations are evaluated
    lazily in Python 3.11+ when ``from __future__ import
    annotations`` is in scope, which observability_events.py
    intentionally is NOT, but the source-level pin is still
    the most explicit guard).
    """
    import ast

    target = (
        Path(__file__).parent.parent
        / "src" / "local_ai_platform"
        / "observability_events.py"
    )
    tree = ast.parse(target.read_text(encoding="utf-8"))

    # Find every ``_<UPPER>_ACTIONS: <annotation> = ...``
    # AnnAssign at module level. Pull the annotation; verify
    # it's ``tuple[<Action>, ...]`` shape with ``<Action>``
    # ending in "Action".
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        target_node = node.target
        if not isinstance(target_node, ast.Name):
            continue
        name = target_node.id
        if not (name.startswith("_") and name.endswith("_ACTIONS")):
            continue
        # Render the annotation back to source for matching.
        ann_src = ast.unparse(node.annotation)
        found[name] = ann_src

    # Every per-subsystem tuple should be present + typed
    # tuple[<Foo>Action, ...].
    assert len(found) >= 12, (
        f"[IMPROVE-106] expected at least 12 per-subsystem "
        f"action tuples, found {len(found)}: {sorted(found)}"
    )

    for name, ann in found.items():
        # Crude shape check: the annotation must end with
        # ``Action, ...]`` (after the leading ``tuple[`` part).
        # Pin the literal-typing convention without coupling
        # to mypy's exact output.
        assert ann.startswith("tuple["), (
            f"[IMPROVE-106] {name} annotation must start with "
            f"``tuple[``; got {ann!r}"
        )
        assert "Action" in ann, (
            f"[IMPROVE-106] {name} annotation must reference an "
            f"<Action> Literal alias (e.g. ``tuple[AgentAction, "
            f"...]``); got {ann!r}. The literal-typing convention "
            f"is documented in the comment block above the "
            f"derivation tuples."
        )
        assert ann.endswith(", ...]"), (
            f"[IMPROVE-106] {name} annotation must end with "
            f"``, ...]`` (variadic tuple); got {ann!r}"
        )
        # Crude rejection of the pre-IMPROVE-106 ``tuple[str, ...]``
        # shape. Pin so a regression surfaces here.
        assert "tuple[str" not in ann, (
            f"[IMPROVE-106] {name} reverted to plain ``tuple[str, "
            f"...]`` — this weakens the per-subsystem Literal "
            f"propagation that mypy --strict relies on."
        )
