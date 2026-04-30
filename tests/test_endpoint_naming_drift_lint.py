"""[IMPROVE-132] Cross-endpoint naming-drift lint.

Wave 14 [IMPROVE-124] manually canonicalised /observability/
timeseries' ``fill_zeros`` (legacy) → ``fill_zero_time``
(canonical) to align with /observability/summary's
``fill_zero_dim``. The drift class — same semantic axis named
differently across endpoints — was caught by manual review at
the time. Wave 15 [IMPROVE-129] introduced ``FILTERS_ECHO_SCHEMA``
as a centralised registry; this lint iterates that schema and
checks for cross-endpoint naming drift automatically.

Per Q2=A in the Wave 16 plan: hardcoded prefix-allowlist (vs
Levenshtein distance threshold or curated-allowlist alternatives).
Easy to extend, low false-positive, explicit about which key sets
are intentional vs accidental.

## Detection algorithm

For each PAIR of distinct keys across the schema:
  1. Compute longest common prefix.
  2. Strip trailing underscore (``fill_zero_`` → ``fill_zero``).
  3. If the result is ≥3 chars, both keys join the
     ``prefix → set[key]`` group.

For each group with 2+ different keys, compare against the
allowlist:
  * Prefix in allowlist + group ⊆ allowlist[prefix] → OK.
  * Otherwise → drift = group - allowlist[prefix].

The allowlist enumerates KNOWN axes that legitimately span
multiple keys (the ``error_code`` vs ``error_code_prefix`` pair
established by [IMPROVE-108]; the ``fill_zero_*`` deprecation
alias trio established by [IMPROVE-115] / [IMPROVE-110] /
[IMPROVE-124]).

## What this catches

  * Future commit adding ``error_code_normalized`` to /recent
    when only ``error_code`` and ``error_code_prefix`` are
    allowlisted — fires.
  * Future commit adding ``fill_zero_count`` to /timeseries
    when only the three current variants are allowlisted —
    fires.
  * Future commit creating a new prefix group (e.g.
    ``cluster_id`` + ``cluster_name`` across 2 endpoints) with
    no allowlist entry — fires.

## What this does NOT catch

  * Single-key prefixes (e.g. just ``subsystem`` across 3
    endpoints with the SAME name) — no drift, just shared schema.
  * 1-char or 2-char shared prefixes (e.g. ``status`` vs
    ``subsystem`` share only ``s``) — below 3-char threshold.
  * Prefix overlap with an empty trailing remainder (e.g. two
    keys identical) — the algorithm requires DIFFERENT keys
    to form a group.
  * Cross-endpoint naming drift in non-filter response shapes
    (this lint is scoped to FILTERS_ECHO_SCHEMA — the items
    array shapes are pinned separately by [IMPROVE-123] tests).

## Failure-mode guidance

When drift fires, three fix paths:
  1. Rename the new key to align with an existing variant
     (most common — drift is usually a typo or naming
     oversight).
  2. Add to ``_NAMING_DRIFT_ALLOWLIST`` when the new variant
     is intentional (e.g. a new IMPROVE-N alias relationship).
  3. Choose a different prefix entirely if the overlap is
     incidental (rare — pick a name with no 3+ char overlap
     with existing keys).

## Sources (2025-2026)

  * Wave 14 [IMPROVE-124] commit (5a3649d) — manual canonical
    rename this lint catches automatically forward-going.
  * Wave 15 [IMPROVE-129] commit (5283e32) — FILTERS_ECHO_SCHEMA
    registry this lint iterates.
  * Wave 12 [IMPROVE-108] commit (bed5fd3) —
    _build_error_code_filter helper that established the
    error_code / error_code_prefix axis pair.
  * Wave 13 [IMPROVE-115] commit (89aff82) — fill_zero_dim
    on /summary.
  * Wave 12 [IMPROVE-110] commit (7d8bbb0) — fill_zeros legacy
    on /timeseries.
  * "API field naming consistency" — REST design pattern
    (Stripe API style guide 2025): https://stripe.com/docs/api
  * "Defence-in-depth lints for documentation drift" (Hyrum's
    Law adjacency 2025): https://www.hyrumslaw.com/
"""
from __future__ import annotations

from typing import Final

from local_ai_platform.api.routers.observability import (
    FILTERS_ECHO_SCHEMA,
)


# ── Allowlist of intentional cross-endpoint naming patterns ──


# Per IMPROVE-132: prefix → set of full keys allowed under it.
# Each entry documents WHY the variant exists so future operators
# extending the allowlist understand the precedent.
_NAMING_DRIFT_ALLOWLIST: Final[dict[str, frozenset[str]]] = {
    # error_code (the axis itself) vs error_code_prefix (the
    # filter-by-prefix variant). Pair established by Wave 12
    # [IMPROVE-108]'s _build_error_code_filter helper. The
    # divergence is "axis vs operation on the axis" — both are
    # always-present in the filters echo for endpoints that
    # support the error_code axis.
    "error_code": frozenset({"error_code", "error_code_prefix"}),
    # fill_zero_dim (W13 [IMPROVE-115] on /summary's per-dimension
    # rollup) vs fill_zeros (W12 [IMPROVE-110] legacy on
    # /timeseries) vs fill_zero_time (W14 [IMPROVE-124] canonical
    # on /timeseries's time-bucket axis). The /summary +
    # /timeseries dimensions have different semantics (dim vs time)
    # so fully canonicalising is not appropriate. Three variants
    # is intentional per the deprecation alias relationship.
    "fill_zero": frozenset(
        {"fill_zero_dim", "fill_zeros", "fill_zero_time"},
    ),
}


# Minimum length of a shared prefix to count as a "group". Below
# this threshold, single-character coincidences (e.g. ``status``
# vs ``subsystem`` share ``s``) would create false-positive
# groups. Three chars is the empirical sweet spot — covers all
# meaningful axis prefixes today (``error_code``, ``fill_zero``,
# hypothetical ``cluster``, ``tenant``, etc.) without firing on
# accidental overlap.
_MIN_PREFIX_LEN: Final[int] = 3


# ── Helpers ──────────────────────────────────────────────────


def _shared_prefix(a: str, b: str) -> str:
    """Return the longest common prefix between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        The longest leading substring shared by both. Empty
        string when no overlap.
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def _extract_prefix_groups(
    schema: dict[str, list[str]],
) -> dict[str, set[str]]:
    """Group schema keys by 3+ char shared prefix.

    Walks all PAIRS of distinct keys across all endpoints; for
    each pair with a 3+ char shared prefix (after stripping
    trailing underscore), adds both keys to the group keyed by
    that prefix.

    Single-key groups (a key with no prefix-sibling) are NOT
    returned — drift requires 2+ different keys sharing a
    prefix. Keys that appear identical across multiple endpoints
    don't trigger grouping (no drift, just shared schema).

    Args:
        schema: mapping of endpoint → list of filter keys.

    Returns:
        dict mapping prefix → set of keys sharing that prefix.
        Only groups with 2+ different keys are returned.
    """
    all_keys: set[str] = set()
    for keys in schema.values():
        all_keys.update(keys)

    groups: dict[str, set[str]] = {}
    keys_list = sorted(all_keys)
    for i, k1 in enumerate(keys_list):
        for k2 in keys_list[i + 1:]:
            common = _shared_prefix(k1, k2).rstrip("_")
            if len(common) >= _MIN_PREFIX_LEN:
                groups.setdefault(common, set()).update([k1, k2])
    return groups


def _detect_naming_drift(
    schema: dict[str, list[str]],
    *,
    allowlist: dict[str, frozenset[str]] | None = None,
) -> dict[str, set[str]]:
    """Detect cross-endpoint naming drift in a filters echo schema.

    For each prefix-group, compares the group membership against
    the allowlist:
      * Prefix in allowlist + group ⊆ allowlist[prefix] → OK.
      * Prefix in allowlist + group ⊄ allowlist[prefix] →
        drift = group - allowlist[prefix].
      * Prefix NOT in allowlist + group has 2+ different keys →
        drift = entire group (operator either canonicalises or
        adds to the allowlist).

    Args:
        schema: mapping of endpoint → list of filter keys.
        allowlist: prefix → set of intentional variants. Defaults
            to ``_NAMING_DRIFT_ALLOWLIST``.

    Returns:
        dict mapping prefix → set of UNEXPECTED keys. Empty when
        no drift detected.
    """
    effective = (
        _NAMING_DRIFT_ALLOWLIST if allowlist is None else allowlist
    )
    groups = _extract_prefix_groups(schema)
    drift: dict[str, set[str]] = {}
    for prefix, keys in groups.items():
        allowed = effective.get(prefix, frozenset())
        unexpected = keys - allowed
        if unexpected:
            drift[prefix] = unexpected
    return drift


# ── Helper unit tests ────────────────────────────────────────


def test_shared_prefix_basic():
    """Pin: longest common prefix on simple distinct strings."""
    assert _shared_prefix("error_code", "error_code_prefix") == "error_code"


def test_shared_prefix_no_overlap():
    """Pin: empty string when no shared leading chars."""
    assert _shared_prefix("subsystem", "action") == ""


def test_shared_prefix_full_match():
    """Pin: identical strings return the full string."""
    assert _shared_prefix("error_code", "error_code") == "error_code"


def test_shared_prefix_one_empty():
    """Pin: empty input returns empty prefix."""
    assert _shared_prefix("", "anything") == ""
    assert _shared_prefix("anything", "") == ""


def test_extract_prefix_groups_today_schema_returns_known_pairs():
    """Pin: today's FILTERS_ECHO_SCHEMA produces exactly two
    prefix groups — error_code and fill_zero."""
    groups = _extract_prefix_groups(FILTERS_ECHO_SCHEMA)
    assert set(groups.keys()) == {"error_code", "fill_zero"}
    assert groups["error_code"] == {"error_code", "error_code_prefix"}
    assert groups["fill_zero"] == {
        "fill_zero_dim",
        "fill_zeros",
        "fill_zero_time",
    }


def test_extract_prefix_groups_returns_empty_for_no_overlap():
    """Pin: no prefix overlap → no groups returned."""
    schema = {
        "/a": ["alpha"],
        "/b": ["beta"],
        "/c": ["gamma"],
    }
    assert _extract_prefix_groups(schema) == {}


def test_extract_prefix_groups_skips_single_key_prefixes():
    """Pin: a key with no prefix-sibling doesn't form a group
    (drift requires 2+ different keys)."""
    schema = {
        "/a": ["unique_axis"],
        "/b": ["unique_axis"],  # same key on 2 endpoints — no drift
    }
    assert _extract_prefix_groups(schema) == {}


def test_extract_prefix_groups_strips_trailing_underscore():
    """Pin: ``cluster_id`` and ``cluster_name`` group under
    ``cluster`` (stripped trailing underscore from common
    prefix ``cluster_``)."""
    schema = {
        "/a": ["cluster_id"],
        "/b": ["cluster_name"],
    }
    groups = _extract_prefix_groups(schema)
    assert "cluster" in groups
    assert groups["cluster"] == {"cluster_id", "cluster_name"}


def test_extract_prefix_groups_respects_3_char_threshold():
    """Pin: 1-2 char shared prefix doesn't create a group.
    ``status`` vs ``subsystem`` share only ``s`` — below
    threshold."""
    schema = {
        "/a": ["status"],
        "/b": ["subsystem"],
    }
    assert _extract_prefix_groups(schema) == {}


def test_detect_drift_today_schema_clean():
    """Pin: today's schema has no unexpected drift (allowlist
    fully covers the two known prefix groups)."""
    drift = _detect_naming_drift(FILTERS_ECHO_SCHEMA)
    assert drift == {}


def test_detect_drift_fires_on_unallowlisted_variant():
    """Pin: adding ``error_code_normalized`` to a synthetic
    schema fires drift (allowlist covers only error_code +
    error_code_prefix)."""
    schema = {
        "/a": ["error_code", "error_code_prefix"],
        "/b": ["error_code", "error_code_normalized"],
    }
    drift = _detect_naming_drift(schema)
    assert "error_code" in drift
    assert drift["error_code"] == {"error_code_normalized"}


def test_detect_drift_fires_on_unknown_prefix_with_2plus_keys():
    """Pin: a brand-new prefix group with no allowlist entry
    fires (operator must canonicalise or update allowlist)."""
    schema = {
        "/a": ["cluster_id"],
        "/b": ["cluster_name"],
    }
    drift = _detect_naming_drift(schema)
    assert "cluster" in drift
    assert drift["cluster"] == {"cluster_id", "cluster_name"}


def test_detect_drift_skips_when_group_in_allowlist():
    """Pin: when a group is fully covered by the allowlist,
    no drift fires (the today-schema case)."""
    schema = {
        "/a": ["error_code", "error_code_prefix"],
    }
    drift = _detect_naming_drift(schema)
    assert drift == {}


def test_detect_drift_returns_only_unexpected_keys():
    """Pin: drift result reports ONLY the unexpected keys,
    not the entire group (so the operator sees just what's
    new)."""
    schema = {
        "/a": ["error_code", "error_code_prefix", "error_code_canonical"],
    }
    drift = _detect_naming_drift(schema)
    # error_code + error_code_prefix are allowlisted; only
    # error_code_canonical is unexpected.
    assert drift == {"error_code": {"error_code_canonical"}}


def test_detect_drift_accepts_custom_allowlist():
    """Pin: caller can override the allowlist (used for
    synthetic-schema testing without mutating the module-
    level constant).

    Uses ``foo_alpha`` + ``foo_beta`` so the shared prefix
    cleanly resolves to ``foo`` (common prefix = ``foo_``,
    rstripped to ``foo``).
    """
    schema = {
        "/a": ["foo_alpha", "foo_beta"],
    }
    custom = {"foo": frozenset({"foo_alpha", "foo_beta"})}
    drift = _detect_naming_drift(schema, allowlist=custom)
    assert drift == {}


# ── Tier 1 lint test ─────────────────────────────────────────


def test_filters_echo_schema_no_naming_drift():
    """[IMPROVE-132] Cross-endpoint naming-drift lint on
    FILTERS_ECHO_SCHEMA.

    Walks the live registry and asserts no unexpected naming
    variants appear. When this fires, three fix paths:

      1. Rename to canonical form (most common — typo or
         naming oversight).
      2. Add to ``_NAMING_DRIFT_ALLOWLIST`` in this file when
         the new variant is intentional (document WHY in a
         comment).
      3. Choose a different prefix if the overlap is incidental
         (rare).
    """
    drift = _detect_naming_drift(FILTERS_ECHO_SCHEMA)
    assert not drift, (
        "Cross-endpoint naming drift detected in "
        "FILTERS_ECHO_SCHEMA. Either rename for cross-endpoint "
        "consistency, or update _NAMING_DRIFT_ALLOWLIST in "
        "tests/test_endpoint_naming_drift_lint.py to register "
        f"the variant. Drift: {dict(drift)}"
    )
