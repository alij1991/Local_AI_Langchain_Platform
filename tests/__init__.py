"""[IMPROVE-137] tests/ promoted to a Python package.

Wave 15's [IMPROVE-126] commit established the
``tests/_lint_helpers.py`` module shared across the lint test
files. The original ship relied on pytest's rootdir-based
collection mode adding tests/ to sys.path, allowing direct
imports via ``from _lint_helpers import ...`` without a
``tests/__init__.py``.

The Wave 15 audit named the candidate explicitly:

    Promote tests/ to a Python package (add __init__.py) +
    use ``from tests._lint_helpers import ...`` instead of
    pytest-rootdir path. ~0.15d. Hold.

The hold reasoning was "one helper module doesn't justify the
architectural change". Wave 16 grew the lint family to 4 lints
(IMPROVE-118 routes, IMPROVE-120 IMPROVE-N refs, IMPROVE-127
Wave-N refs, [IMPROVE-132] cross-endpoint naming-drift,
[IMPROVE-135] SHA-ancestor refs) — 5 sibling test files all
importing from the shared helpers. The architectural cleanup
now pays back; the package-relative import form is more
explicit and easier for future readers to locate.

This file is intentionally empty (no public exports). Its
presence promotes ``tests/`` to a package so consumers use
``from tests._lint_helpers import ...`` per Q6=A in the
Wave 16 plan.

The pytest collection mode auto-detects: with __init__.py,
test modules are imported as ``tests.<module>`` (vs the prior
rootdir-based ``<module>`` form). All Tier 1 sweep tests
collect identically — the only operator-visible change is the
canonical import path.

Sources (2025-2026):
  * Wave 15 [IMPROVE-126] commit (033b54a) — established the
    shared helpers module.
  * Wave 15 deferred queue (§10.5 of 10-improvements.md) —
    named the package promotion candidate.
  * pytest pythonpath docs (canonical 2025 reference):
    https://docs.pytest.org/en/stable/explanation/pythonpath.html
"""
