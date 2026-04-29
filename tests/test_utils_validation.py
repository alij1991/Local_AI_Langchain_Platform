"""[IMPROVE-111] Tests for ``utils.validation``.

The Wave 12 plan (Q5=A) promoted partner/export.py's
``_validate_decay_config_keys`` to a generic helper in NEW
``src/local_ai_platform/utils/validation.py``. This test file
covers the helper's contract end-to-end + pins the existing
``_validate_decay_config_keys`` callsite still works after
delegation.

Sources (2025-2026):
  * Wave 10 [IMPROVE-98] commit (a05fb46) — original
    ``_validate_decay_config_keys`` that the helper generalises.
  * Python ``inspect.signature`` + ``Parameter.kind``
    introspection (Python 3.11 docs):
    https://docs.python.org/3.11/library/inspect.html#inspect.signature
"""
from __future__ import annotations

import pytest

from local_ai_platform.utils.validation import (
    validate_kwargs_against_signature,
)


# ── Function fixtures ──────────────────────────────────────────


def _three_param_fn(x: int, y: int = 0, *, z: int = 0) -> int:
    """Reference function with 3 named params (positional +
    keyword-only). Used as the validator's target."""
    return x + y + z


def _kwargs_fn(**opts: object) -> dict:
    """Function accepting only **kwargs. Validator should be
    permissive — can't reject anything."""
    return dict(opts)


def _args_kwargs_fn(*args: object, **opts: object) -> tuple:
    """Function with both *args + **kwargs. Validator should
    still be permissive (because of **kwargs)."""
    return args, opts


def _no_params_fn() -> None:
    """Function with no parameters. Any payload key is rejected."""


def _star_args_only_fn(*args: object) -> tuple:
    """Function with only *args (no **kwargs). Validator should
    reject any kwargs since *args isn't name-addressable."""
    return args


# ── Happy-path: valid kwargs ───────────────────────────────────


def test_validate_kwargs_accepts_known_keys():
    """[IMPROVE-111] A payload containing only known parameter
    names returns without raising."""
    validate_kwargs_against_signature(
        _three_param_fn, {"x": 1, "y": 2, "z": 3},
    )


def test_validate_kwargs_accepts_subset_of_known_keys():
    """[IMPROVE-111] A payload containing a strict subset of
    parameter names returns without raising — the validator
    doesn't enforce that ALL parameters are present, only that
    no UNKNOWN keys are passed. Mirror of the
    ``_validate_decay_config_keys`` behaviour."""
    validate_kwargs_against_signature(_three_param_fn, {"x": 1})


def test_validate_kwargs_accepts_empty_payload():
    """[IMPROVE-111] An empty payload is trivially valid — no
    keys to reject. Pin so a "no config provided" path doesn't
    spuriously fail."""
    validate_kwargs_against_signature(_three_param_fn, {})


# ── Rejection: unknown keys ────────────────────────────────────


def test_validate_kwargs_rejects_unknown_key():
    """[IMPROVE-111] A single unknown key raises ValueError with
    the unknown listed + the accepted set listed."""
    with pytest.raises(ValueError) as exc_info:
        validate_kwargs_against_signature(
            _three_param_fn, {"x": 1, "w": 2},
        )
    msg = str(exc_info.value)
    assert "unknown kwarg(s): ['w']" in msg
    # accepted set lists all 3 parameters sorted.
    assert "['x', 'y', 'z']" in msg


def test_validate_kwargs_rejects_multiple_unknown_keys():
    """[IMPROVE-111] Multiple unknowns are all listed (sorted)
    so a contributor sees every key to fix in one pass — not
    just the first."""
    with pytest.raises(ValueError) as exc_info:
        validate_kwargs_against_signature(
            _three_param_fn, {"a": 1, "b": 2, "c": 3, "x": 4},
        )
    msg = str(exc_info.value)
    assert "['a', 'b', 'c']" in msg


def test_validate_kwargs_rejects_against_no_params_fn():
    """[IMPROVE-111] A function with NO parameters rejects any
    non-empty payload."""
    with pytest.raises(ValueError) as exc_info:
        validate_kwargs_against_signature(_no_params_fn, {"x": 1})
    msg = str(exc_info.value)
    assert "['x']" in msg


# ── **kwargs: permissive ──────────────────────────────────────


def test_validate_kwargs_permissive_on_var_keyword():
    """[IMPROVE-111] A function with ``**kwargs`` accepts ANY
    key — the validator can't reject anything because the
    signature doesn't capture which keys are valid. Return
    without raising. Mirror of the pre-IMPROVE-111
    ``_validate_decay_config_keys`` behaviour: best-effort."""
    validate_kwargs_against_signature(
        _kwargs_fn, {"anything_goes": 1, "really_any_key": 2},
    )


def test_validate_kwargs_permissive_on_args_kwargs_combo():
    """[IMPROVE-111] A function with BOTH ``*args`` AND
    ``**kwargs`` is permissive (because of ``**kwargs``). Pin
    the union semantic — the validator is permissive whenever
    ANY varkeyword is present."""
    validate_kwargs_against_signature(
        _args_kwargs_fn, {"random_key": "value"},
    )


def test_validate_kwargs_rejects_on_star_args_only():
    """[IMPROVE-111] A function with only ``*args`` (no
    ``**kwargs``) does NOT accept arbitrary kwargs — *args
    isn't name-addressable. Validator rejects any kwarg since
    no named parameters exist."""
    with pytest.raises(ValueError) as exc_info:
        validate_kwargs_against_signature(
            _star_args_only_fn, {"x": 1},
        )
    msg = str(exc_info.value)
    assert "['x']" in msg


# ── Custom label ───────────────────────────────────────────────


def test_validate_kwargs_uses_custom_label():
    """[IMPROVE-111] The ``label`` parameter customises the
    error message domain noun. Pin so the
    ``_validate_decay_config_keys`` delegate's ``"decay config
    key"`` label appears in errors as expected."""
    with pytest.raises(ValueError) as exc_info:
        validate_kwargs_against_signature(
            _three_param_fn, {"unknown": 1},
            label="decay config key",
        )
    msg = str(exc_info.value)
    assert "unknown decay config key(s):" in msg


def test_validate_kwargs_default_label_is_kwarg():
    """[IMPROVE-111] When ``label`` is omitted the default is
    ``"kwarg"`` (singular noun, pluralised by the helper's
    ``"(s)"`` suffix)."""
    with pytest.raises(ValueError) as exc_info:
        validate_kwargs_against_signature(
            _three_param_fn, {"unknown": 1},
        )
    msg = str(exc_info.value)
    assert "unknown kwarg(s):" in msg


# ── Sort stability ─────────────────────────────────────────────


def test_validate_kwargs_lists_are_sorted_for_stable_messages():
    """[IMPROVE-111] Both the unknown list and the accepted
    list are sorted in the error message so the output is
    stable across runs (sets are unordered in Python). Pin
    the sort so a future test asserting message equality
    doesn't flake."""
    with pytest.raises(ValueError) as exc_info:
        validate_kwargs_against_signature(
            _three_param_fn,
            {"zoo": 1, "alpha": 2, "middle": 3},
        )
    msg = str(exc_info.value)
    # Unknowns sorted ascending: alpha, middle, zoo.
    assert "['alpha', 'middle', 'zoo']" in msg


# ── Delegation: partner/export.py:_validate_decay_config_keys ──


def test_decay_config_keys_delegates_to_helper():
    """[IMPROVE-111] The pre-IMPROVE-111
    ``_validate_decay_config_keys`` callsite still raises
    ValueError on unknown keys + the message format is
    preserved (``"unknown decay config key(s):"``).

    [IMPROVE-111] FIX: pre-IMPROVE-111 the validator wrongly
    flagged LEGIT keys as unknown (because
    ``inspect.signature(set_decay_config).parameters`` returned
    ``{"_persist", "updates"}`` for the ``**kwargs``-style
    function). The refactor uses ``get_decay_config().keys()``
    as the accepted set, which matches what
    ``set_decay_config`` validates against internally. Pin
    that legit keys (e.g. ``importance_floor``, the actual
    decay-config field) are accepted + only truly-unknown
    keys raise.
    """
    from local_ai_platform.partner.export import (
        _validate_decay_config_keys,
    )
    # Known keys (in _DECAY_CONFIG) → OK. Pre-IMPROVE-111 this
    # WOULD HAVE FAILED because the buggy signature-based
    # validator rejected importance_floor too.
    _validate_decay_config_keys(
        {"importance_floor": 7},
    )
    # Multiple known keys still pass.
    _validate_decay_config_keys(
        {
            "importance_floor": 7,
            "archive_threshold": 0.85,
        },
    )
    # Unknown key → ValueError with the labelled message.
    # Legit ``importance_floor`` accompanying it IS NOT flagged
    # — only ``unknown_param`` raises.
    with pytest.raises(ValueError) as exc_info:
        _validate_decay_config_keys(
            {"importance_floor": 7, "unknown_param": 42},
        )
    msg = str(exc_info.value)
    assert "unknown decay config key(s):" in msg
    assert "['unknown_param']" in msg
    # Legit key is NOT in the unknown list (the IMPROVE-111
    # fix). Pre-IMPROVE-111 ``importance_floor`` was wrongly
    # flagged.
    assert "importance_floor" not in msg.split(";")[0]
