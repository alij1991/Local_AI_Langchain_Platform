"""[IMPROVE-111] Signature-based + explicit-keys kwarg validators.

Wave 10 [IMPROVE-98] introduced ``_validate_decay_config_keys``
in ``partner/export.py`` to validate dry-run JSON payload keys
against an accepted-keys set WITHOUT actually calling the
target function. The original implementation tried to use
``inspect.signature(set_decay_config).parameters`` to derive
the accepted set — but ``set_decay_config(**updates)`` uses
``**kwargs``, so the inspect-based approach gave only
``{"_persist", "updates"}`` as accepted. Effectively the
validator rejected EVERYTHING (false positives included) but
the existing test happened to pass because the asserted
substring ``"unknown decay config key"`` appeared regardless
of which keys were flagged.

The Wave 12 plan (Q5=A) promoted the helper extraction to
``IMPROVE-111`` with the additional intent of fixing the
``**kwargs`` blind spot. This module exposes TWO helpers:

  * ``validate_kwargs_against_signature(fn, payload, label=...)``
    — for functions with named parameters (no ``**kwargs``).
    PERMISSIVE on ``**kwargs`` functions (returns without
    raising) because the signature gives no information about
    valid keys when ``**kwargs`` is present.

  * ``validate_kwargs_against_keys(payload, accepted, label=...)``
    — for callers with an explicit accepted-keys set. Used by
    ``_validate_decay_config_keys`` (the IMPROVE-98 callsite)
    because ``set_decay_config`` validates against the
    ``_DECAY_CONFIG`` dict's keys internally — the
    signature-based variant doesn't capture this.

Per Q5=A clean-separation: this module is leaf
(``utils.validation`` imports nothing from
``local_ai_platform.*``) so it can safely be imported from
anywhere.

Sources (2025-2026):
  * Wave 10 [IMPROVE-98] commit (a05fb46) — original
    ``_validate_decay_config_keys`` that this module
    generalises (and fixes — the buggy ``**kwargs`` behaviour
    surfaced during the IMPROVE-111 refactor).
  * Python ``inspect.signature`` + ``Parameter.kind``
    introspection (Python 3.11 docs):
    https://docs.python.org/3.11/library/inspect.html#inspect.signature
    https://docs.python.org/3.11/library/inspect.html#inspect.Parameter.kind
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable


def validate_kwargs_against_signature(
    fn: Callable[..., Any],
    payload: dict[str, Any],
    *,
    label: str = "kwarg",
) -> None:
    """Validate that every key in ``payload`` corresponds to a
    parameter accepted by ``fn``. Raises ``ValueError`` listing
    the unknown keys + the accepted parameter set when
    ``payload`` contains keys ``fn`` doesn't accept.

    The accepted parameter list is pulled from ``fn`` via
    ``inspect.signature``. The two stay in sync automatically
    — a future addition to ``fn``'s signature is permitted by
    this helper without any code change.

    ``fn`` is NOT called; ``payload`` is read-only. Use this
    on dry-run paths where you want the same "unknown key"
    error a real call would raise, but without the side
    effects.

    ``**kwargs``-style functions accept ANY key (the parameter
    list won't capture which keys are valid). For those, this
    helper is permissive — returns without raising. Caller
    that wants strict rejection on **kwargs functions should
    inspect the signature themselves.

    ``*args``-style positional varargs are excluded from the
    accepted-set because they're not name-addressable.

    ``label`` customises the error message ("decay config key",
    "bundle option", etc.) without coupling the helper to a
    specific domain. Default ``"kwarg"`` is generic.

    Args:
        fn: The callable whose signature defines accepted keys.
        payload: Dict of kwargs to validate.
        label: Domain noun for the error message; pluralised
            with "(s)" automatically.

    Raises:
        ValueError: If ``payload`` contains keys not in ``fn``'s
            signature. Message format:
            ``"unknown {label}(s): {sorted unknowns}; accepted:
              {sorted accepted}"``

    Example:
        >>> def f(x: int, y: int, *, z: int = 0) -> int: ...
        >>> validate_kwargs_against_signature(f, {"x": 1})
        # OK
        >>> validate_kwargs_against_signature(f, {"x": 1, "w": 2})
        Traceback (most recent call last):
            ...
        ValueError: unknown kwarg(s): ['w']; accepted: ['x', 'y', 'z']
    """
    sig = inspect.signature(fn)
    accepted: set[str] = set()
    has_var_keyword = False
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # **kwargs — function accepts any key.
            has_var_keyword = True
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            # *args — not name-addressable, exclude from accepted.
            continue
        else:
            accepted.add(name)
    if has_var_keyword:
        # Function accepts **kwargs; can't reject anything.
        # Best-effort: return without raising. Caller wanting
        # to validate against an explicit keys set should use
        # ``validate_kwargs_against_keys`` instead.
        return
    unknown = set(payload.keys()) - accepted
    if unknown:
        raise ValueError(
            f"unknown {label}(s): {sorted(unknown)}; "
            f"accepted: {sorted(accepted)}"
        )


def validate_kwargs_against_keys(
    payload: dict[str, Any],
    accepted: Iterable[str],
    *,
    label: str = "kwarg",
) -> None:
    """Validate that every key in ``payload`` is in ``accepted``.

    Mirror of ``validate_kwargs_against_signature`` but takes
    an explicit accepted-keys iterable instead of inspecting a
    function's signature. Use this when:

      * The target function uses ``**kwargs`` and validates
        against an internal whitelist (the signature's
        ``**kwargs`` gives no information about valid keys).

      * The accepted set comes from data (a dict's keys, a
        config schema, etc.) rather than a function signature.

    The error message format matches
    ``validate_kwargs_against_signature`` so callers can switch
    between the two helpers without changing the substring
    that downstream tests / dashboards grep for.

    Args:
        payload: Dict of kwargs to validate.
        accepted: Iterable of accepted key names. Materialised
            into a set internally so a generator works.
        label: Domain noun for the error message; pluralised
            with ``"(s)"`` automatically.

    Raises:
        ValueError: If ``payload`` contains keys not in
            ``accepted``. Message format:
            ``"unknown {label}(s): {sorted unknowns}; accepted:
              {sorted accepted}"``

    Example:
        >>> validate_kwargs_against_keys(
        ...     {"x": 1, "z": 9}, ["x", "y"], label="config key",
        ... )
        Traceback (most recent call last):
            ...
        ValueError: unknown config key(s): ['z']; accepted: ['x', 'y']
    """
    accepted_set = set(accepted)
    unknown = set(payload.keys()) - accepted_set
    if unknown:
        raise ValueError(
            f"unknown {label}(s): {sorted(unknown)}; "
            f"accepted: {sorted(accepted_set)}"
        )
