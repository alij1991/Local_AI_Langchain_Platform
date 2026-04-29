"""[IMPROVE-111+114] Signature-based + explicit-keys kwarg helpers.

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

  * ``filter_kwargs_to_signature(fn, payload, exclude=None)``
    — [IMPROVE-114] sibling that returns a NEW filtered dict
    instead of raising. Used at the diffusers-pipeline-kwarg
    + classical-image-op callsites in ``images/processors.py``
    + ``images/editor.py`` where unknown keys should be
    DROPPED (not raised) because the user's params dict is
    a freeform UI payload that may carry stale keys after a
    pipeline / operation signature change.

Per Q5=A (W12) + Q2=A (W13) clean-separation: this module is
leaf (``utils.validation`` imports nothing from
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


def filter_kwargs_to_signature(
    fn: Callable[..., Any],
    payload: dict[str, Any],
    *,
    exclude: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a NEW dict containing only ``payload`` entries
    whose keys are accepted by ``fn``'s signature, MINUS any
    keys in ``exclude``.

    Mirror of ``validate_kwargs_against_signature`` but RETURNS
    a filtered dict instead of raising on unknowns. Use this at
    callsites that want to DROP unknown keys silently rather
    than reject — typically a freeform user payload (Flutter
    UI dict, JSON config) being passed into a third-party
    function whose signature varies between versions.

    Common callsite shape pre-IMPROVE-114:

        sig = inspect.signature(fn)
        valid = {
            k: v for k, v in payload.items()
            if k in sig.parameters and k != "image"
        }
        result = fn(image, **valid)

    Post-IMPROVE-114:

        valid = filter_kwargs_to_signature(fn, payload, exclude=["image"])
        result = fn(image, **valid)

    For functions with ``**kwargs``, this helper returns a
    SHALLOW COPY of ``payload`` (modulo ``exclude``) — the
    function accepts any key so there's nothing to filter
    based on the signature. This matches the
    ``validate_kwargs_against_signature`` "permissive on
    **kwargs" semantic.

    ``*args`` is excluded automatically (not name-addressable).

    The returned dict is a NEW object — mutating it doesn't
    affect ``payload``. Use this property at callsites that
    need to also pre-process the filtered values (e.g.
    ``images/processors.py``'s type-coercion second pass).

    Args:
        fn: The callable whose signature defines accepted keys.
        payload: Source dict; not mutated.
        exclude: Optional iterable of keys to drop EVEN IF
            present in ``fn``'s signature. Use for
            positional-required params already supplied
            elsewhere (typically ``"image"`` at the editor /
            processor callsites where image is the first
            positional arg). ``None`` means drop nothing
            extra; pass ``[]`` for the same effect.

    Returns:
        New dict containing the accepted keys from ``payload``.
        Empty dict if no key matches.

    Example:
        >>> def f(image, x: int, y: int = 0): ...
        >>> filter_kwargs_to_signature(
        ...     f, {"x": 1, "y": 2, "image": "img", "z": 3},
        ...     exclude=["image"],
        ... )
        {'x': 1, 'y': 2}

    Sources (2025-2026):
      * Wave 12 [IMPROVE-111] commit (4e7ce54) — sibling
        ``validate_kwargs_against_signature`` this helper
        complements with the filter-instead-of-raise variant.
      * Wave 13 plan (Q2=A) — promoted the filter helper +
        3-callsite migration (processors.py:1243 +
        editor.py:713 + editor.py:725) into IMPROVE-114.
    """
    excluded: set[str] = set(exclude) if exclude is not None else set()
    sig = inspect.signature(fn)
    accepted: set[str] = set()
    has_var_keyword = False
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        else:
            accepted.add(name)
    if has_var_keyword:
        # Function accepts any key — return shallow copy minus
        # excluded. Mirrors the validate_against_signature
        # "permissive on **kwargs" semantic.
        return {k: v for k, v in payload.items() if k not in excluded}
    return {
        k: v for k, v in payload.items()
        if k in accepted and k not in excluded
    }
