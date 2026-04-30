"""[IMPROVE-123] Cross-endpoint pin: the ``filters`` echo dict
shape on the four observability review endpoints.

Wave 14's audit surfaced the cross-endpoint shape variance:

  * /observability/recent — 5 keys (subsystem, status, action,
    error_code, error_code_prefix). Shipped W13 [IMPROVE-113].
  * /observability/summary — 3 keys (error_code,
    error_code_prefix, fill_zero_dim). Shipped W13 [IMPROVE-115]
    (grew the original 2-key W12 [IMPROVE-108] echo with
    fill_zero_dim).
  * /observability/timeseries — 6 keys (subsystem, action,
    error_code, error_code_prefix, fill_zeros, fill_zero_time).
    Shipped W12 [IMPROVE-110]; grew to 6 in W14 [IMPROVE-124].
  * /observability/rejections — 4 keys (subsystem, action,
    error_code, error_code_prefix). Shipped W11 [IMPROVE-103].

Four endpoints, four shapes, no cross-endpoint pin tests
pre-IMPROVE-123. A future commit dropping a key from one
endpoint's echo would silently break dashboards that assume
the always-present-key contract.

Per Q4=A in the Wave 14 plan: hardcoded expected key sets per
endpoint (4 dict literals here). Simple, explicit — vs the
alternative of a centralised registry in ``observability.py``
(B; higher abstraction) or auto-introspection from each
endpoint's signature (C; most decoupled but most fragile).

[IMPROVE-129] update (Wave 15): the deferred-queue option B
shipped — production code now sources its filters echo dicts
from ``observability.FILTERS_ECHO_SCHEMA`` (a centralised
registry) via the ``_build_filters_echo`` helper. The test-
side ``EXPECTED_*_FILTERS`` constants here STAY (they pin the
schema as test-side documentation); a NEW cross-pin test
asserts the production-side schema matches each endpoint's
test-side expectation. Drift between the two surfaces in a
single failing test rather than at silent dashboard runtime.

This file exists ALONGSIDE the per-endpoint test files (which
already test individual filter behaviour); the schema pins
here cross-cut to catch drift at the boundary.

Sources (2025-2026):
  * Wave 13 [IMPROVE-113] commit (8fa0ba6) — /recent's 5-key
    echo introduced.
  * Wave 13 [IMPROVE-115] commit (89aff82) — /summary's
    2-key → 3-key echo growth.
  * Wave 12 [IMPROVE-108] commit (bed5fd3) — _build_error_code
    _filter helper that motivated the 4-endpoint axis
    consolidation.
  * Wave 12 [IMPROVE-110] commit (7d8bbb0) — /timeseries'
    5-key echo introduced.
  * Wave 11 [IMPROVE-103] commit (3b3c50d) — /rejections
    endpoint introduced.
  * "Always-present field contracts" — REST API design
    pattern (Stripe API style guide 2025):
    https://stripe.com/docs/api
"""
from __future__ import annotations

import pytest

from local_ai_platform.api.routers.observability import (
    FILTERS_ECHO_SCHEMA,
    _build_filters_echo,
)


# ── Expected schemas (4 dict literals per Q4=A) ──────────────


# /observability/recent filters echo (post-IMPROVE-113):
# 5-key always-present dict. The ``limit`` query param is
# pagination not a filter, so it stays OUT of the echo.
EXPECTED_RECENT_FILTERS = {
    "subsystem",
    "status",
    "action",
    "error_code",
    "error_code_prefix",
}


# /observability/summary filters echo (post-IMPROVE-115):
# 3-key always-present dict. Grew from the original 2-key
# IMPROVE-108 echo (added fill_zero_dim in W13).
EXPECTED_SUMMARY_FILTERS = {
    "error_code",
    "error_code_prefix",
    "fill_zero_dim",
}


# /observability/timeseries filters echo (post-IMPROVE-124):
# 6-key always-present dict. ``fill_zeros`` (legacy from
# IMPROVE-110) coexists with ``fill_zero_time`` (canonical
# from IMPROVE-124) — deprecation alias relationship per
# Q5=A in the Wave 14 plan. No removal date set; both keys
# echo independently so dashboards can verify which name was
# in play.
EXPECTED_TIMESERIES_FILTERS = {
    "subsystem",
    "action",
    "error_code",
    "error_code_prefix",
    "fill_zeros",
    "fill_zero_time",
}


# /observability/rejections filters echo (post-IMPROVE-108
# extension to /rejections from IMPROVE-103): 4-key dict.
# No fill_zeros / fill_zero_dim — rejections are filter-scoped
# events, not aggregations.
EXPECTED_REJECTIONS_FILTERS = {
    "subsystem",
    "action",
    "error_code",
    "error_code_prefix",
}


# ── Per-endpoint pins ────────────────────────────────────────


def test_recent_filters_echo_has_canonical_5_key_shape(obs_test_client):
    """[IMPROVE-123] /observability/recent's filters echo has
    the canonical 5-key shape (subsystem, status, action,
    error_code, error_code_prefix). Pin so a future change
    that drops a key surfaces here as a deliberate update,
    not a silent regression."""
    resp = obs_test_client.get("/observability/recent")
    assert resp.status_code == 200
    body = resp.json()
    assert "filters" in body, "missing 'filters' echo key"
    assert set(body["filters"].keys()) == EXPECTED_RECENT_FILTERS, (
        f"/recent filters echo shape changed: "
        f"got {sorted(body['filters'].keys())}, "
        f"expected {sorted(EXPECTED_RECENT_FILTERS)}"
    )


def test_summary_filters_echo_has_canonical_3_key_shape(obs_test_client):
    """[IMPROVE-123] /observability/summary's filters echo has
    the canonical 3-key shape (error_code, error_code_prefix,
    fill_zero_dim). The narrower shape vs /recent / /timeseries
    reflects /summary's rejection-rollup focus — subsystem +
    action are NOT scoped at the endpoint level (they appear
    per-rollup-row in the items array)."""
    resp = obs_test_client.get("/observability/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "filters" in body, "missing 'filters' echo key"
    assert set(body["filters"].keys()) == EXPECTED_SUMMARY_FILTERS, (
        f"/summary filters echo shape changed: "
        f"got {sorted(body['filters'].keys())}, "
        f"expected {sorted(EXPECTED_SUMMARY_FILTERS)}"
    )


def test_timeseries_filters_echo_has_canonical_5_key_shape(obs_test_client):
    """[IMPROVE-123] /observability/timeseries' filters echo has
    the canonical 5-key shape (subsystem, action, error_code,
    error_code_prefix, fill_zeros). The ``fill_zeros`` key is
    the [IMPROVE-110] zero-padding flag — naming differs from
    /summary's ``fill_zero_dim`` (a Wave 14 audit observation;
    [IMPROVE-124] adds a ``fill_zero_time`` alias for symmetry
    while keeping the existing ``fill_zeros`` name)."""
    resp = obs_test_client.get("/observability/timeseries")
    assert resp.status_code == 200
    body = resp.json()
    assert "filters" in body, "missing 'filters' echo key"
    assert set(body["filters"].keys()) == EXPECTED_TIMESERIES_FILTERS, (
        f"/timeseries filters echo shape changed: "
        f"got {sorted(body['filters'].keys())}, "
        f"expected {sorted(EXPECTED_TIMESERIES_FILTERS)}"
    )


def test_rejections_filters_echo_has_canonical_4_key_shape(obs_test_client):
    """[IMPROVE-123] /observability/rejections' filters echo has
    the canonical 4-key shape (subsystem, action, error_code,
    error_code_prefix). No fill_zeros / fill_zero_dim — the
    /rejections endpoint surfaces ALL events with non-null
    error_code (no zero-padding semantic; you can't have a
    zero-row rejection)."""
    resp = obs_test_client.get("/observability/rejections")
    assert resp.status_code == 200
    body = resp.json()
    assert "filters" in body, "missing 'filters' echo key"
    assert set(body["filters"].keys()) == EXPECTED_REJECTIONS_FILTERS, (
        f"/rejections filters echo shape changed: "
        f"got {sorted(body['filters'].keys())}, "
        f"expected {sorted(EXPECTED_REJECTIONS_FILTERS)}"
    )


# ── Cross-endpoint pins ──────────────────────────────────────


def test_all_obs_endpoints_share_error_code_axis(obs_test_client):
    """[IMPROVE-123] All four obs endpoints share the
    ``error_code`` filter axis. Pin the cross-endpoint
    consistency the [IMPROVE-108] helper provides — a future
    rename or removal would surface here AND in the per-endpoint
    schema test, but this cross-cut catches divergence between
    them in a single-failing test."""
    endpoints = [
        "/observability/recent",
        "/observability/summary",
        "/observability/timeseries",
        "/observability/rejections",
    ]
    for path in endpoints:
        resp = obs_test_client.get(path)
        assert resp.status_code == 200, f"{path} returned {resp.status_code}"
        filters = resp.json().get("filters", {})
        assert "error_code" in filters, (
            f"{path} missing error_code in filters echo: "
            f"got {sorted(filters.keys())}"
        )


def test_all_obs_endpoints_share_error_code_prefix_axis(obs_test_client):
    """[IMPROVE-123] All four obs endpoints share the
    ``error_code_prefix`` filter axis. Sibling pin to the
    error_code cross-cut — both axes were extended to all
    four endpoints over Waves 11-13 (rejections shipped W11
    in [IMPROVE-103], summary shipped W12 via [IMPROVE-108],
    timeseries shipped W12 via [IMPROVE-108], recent shipped
    W13 via [IMPROVE-113])."""
    endpoints = [
        "/observability/recent",
        "/observability/summary",
        "/observability/timeseries",
        "/observability/rejections",
    ]
    for path in endpoints:
        resp = obs_test_client.get(path)
        assert resp.status_code == 200
        filters = resp.json().get("filters", {})
        assert "error_code_prefix" in filters, (
            f"{path} missing error_code_prefix in filters echo: "
            f"got {sorted(filters.keys())}"
        )


def test_all_obs_endpoints_filters_keys_are_always_present(obs_test_client):
    """[IMPROVE-123] All four obs endpoints emit ALL their
    documented filter axes as keys, with values defaulting to
    None when the corresponding query param is absent (pin the
    "always-present" contract). Dashboards rendering badges
    rely on key existence to render the "no filter applied"
    state — a missing key would break that codepath.

    The expected_schemas mapping doubles as a registry in case
    a future caller wants to introspect all 4 schemas
    programmatically."""
    expected_schemas = {
        "/observability/recent": EXPECTED_RECENT_FILTERS,
        "/observability/summary": EXPECTED_SUMMARY_FILTERS,
        "/observability/timeseries": EXPECTED_TIMESERIES_FILTERS,
        "/observability/rejections": EXPECTED_REJECTIONS_FILTERS,
    }
    for path, expected_keys in expected_schemas.items():
        resp = obs_test_client.get(path)
        assert resp.status_code == 200
        filters = resp.json().get("filters", {})
        # Every documented key must be present (key existence,
        # not value content — values are None by default).
        for key in expected_keys:
            assert key in filters, (
                f"{path} filters echo missing always-present "
                f"key {key!r}: got {sorted(filters.keys())}"
            )


# ── Filter axes value pins ───────────────────────────────────


def test_recent_filters_default_values_are_none(obs_test_client):
    """[IMPROVE-123] /recent's filter values default to None
    when no query params are passed. Pin the "always-present
    keys, None values" contract so a dashboard rendering
    "no filter active" doesn't have to check key existence."""
    resp = obs_test_client.get("/observability/recent")
    filters = resp.json()["filters"]
    for key in EXPECTED_RECENT_FILTERS:
        assert filters[key] is None, (
            f"/recent filters[{key!r}] default value changed: "
            f"got {filters[key]!r}, expected None"
        )


def test_summary_fill_zero_dim_defaults_to_false(obs_test_client):
    """[IMPROVE-123] /summary's ``fill_zero_dim`` is the only
    filter axis with a non-None default — it's a boolean flag
    (False by default). Pin the default value so a future
    change to True surfaces here."""
    resp = obs_test_client.get("/observability/summary")
    filters = resp.json()["filters"]
    assert filters["fill_zero_dim"] is False, (
        f"/summary filters['fill_zero_dim'] default changed: "
        f"got {filters['fill_zero_dim']!r}, expected False"
    )


def test_timeseries_fill_zeros_defaults_to_false(obs_test_client):
    """[IMPROVE-123] /timeseries' ``fill_zeros`` boolean flag
    defaults to False. Pin the default — same rationale as the
    /summary fill_zero_dim default pin (catch unintentional
    flips)."""
    resp = obs_test_client.get("/observability/timeseries")
    filters = resp.json()["filters"]
    assert filters["fill_zeros"] is False, (
        f"/timeseries filters['fill_zeros'] default changed: "
        f"got {filters['fill_zeros']!r}, expected False"
    )


# ── [IMPROVE-129] Centralised registry pins ──────────────────


def test_filters_echo_schema_has_all_4_endpoints():
    """[IMPROVE-129] The production-side registry registers
    exactly the 4 obs endpoints. A future endpoint addition
    bumps this count + grows the registry."""
    assert set(FILTERS_ECHO_SCHEMA.keys()) == {
        "/observability/recent",
        "/observability/summary",
        "/observability/timeseries",
        "/observability/rejections",
    }


def test_filters_echo_schema_recent_matches_test_side_expected():
    """[IMPROVE-129] Cross-pin: production-side schema for
    /recent matches the test-side EXPECTED_RECENT_FILTERS.
    A future commit changing one without the other surfaces
    here as a deliberate update, not silent drift."""
    schema_keys = set(FILTERS_ECHO_SCHEMA["/observability/recent"])
    assert schema_keys == EXPECTED_RECENT_FILTERS, (
        f"production schema drift on /recent: "
        f"got {sorted(schema_keys)}, "
        f"test expects {sorted(EXPECTED_RECENT_FILTERS)}"
    )


def test_filters_echo_schema_summary_matches_test_side_expected():
    """[IMPROVE-129] Cross-pin for /summary."""
    schema_keys = set(FILTERS_ECHO_SCHEMA["/observability/summary"])
    assert schema_keys == EXPECTED_SUMMARY_FILTERS, (
        f"production schema drift on /summary: "
        f"got {sorted(schema_keys)}, "
        f"test expects {sorted(EXPECTED_SUMMARY_FILTERS)}"
    )


def test_filters_echo_schema_timeseries_matches_test_side_expected():
    """[IMPROVE-129] Cross-pin for /timeseries (6-key shape
    post-IMPROVE-124's fill_zero_time alias)."""
    schema_keys = set(FILTERS_ECHO_SCHEMA["/observability/timeseries"])
    assert schema_keys == EXPECTED_TIMESERIES_FILTERS, (
        f"production schema drift on /timeseries: "
        f"got {sorted(schema_keys)}, "
        f"test expects {sorted(EXPECTED_TIMESERIES_FILTERS)}"
    )


def test_filters_echo_schema_rejections_matches_test_side_expected():
    """[IMPROVE-129] Cross-pin for /rejections."""
    schema_keys = set(FILTERS_ECHO_SCHEMA["/observability/rejections"])
    assert schema_keys == EXPECTED_REJECTIONS_FILTERS, (
        f"production schema drift on /rejections: "
        f"got {sorted(schema_keys)}, "
        f"test expects {sorted(EXPECTED_REJECTIONS_FILTERS)}"
    )


def test_filters_echo_schema_keys_are_unique_per_endpoint():
    """[IMPROVE-129] No duplicate keys within an endpoint's
    schema list (insertion-order list shouldn't have repeats —
    a duplicate would let _build_filters_echo silently shadow
    one of them in the resulting dict)."""
    for endpoint, keys in FILTERS_ECHO_SCHEMA.items():
        assert len(keys) == len(set(keys)), (
            f"{endpoint} has duplicate keys in schema: {keys}"
        )


def test_build_filters_echo_recent_uses_kwargs():
    """[IMPROVE-129] Helper assembles the dict from kwargs;
    each schema key takes its value from the matching kwarg."""
    result = _build_filters_echo(
        "/observability/recent",
        subsystem="kontext",
        status="error",
        action="generate",
        error_code="cuda_oom",
        error_code_prefix=None,
    )
    assert result == {
        "subsystem": "kontext",
        "status": "error",
        "action": "generate",
        "error_code": "cuda_oom",
        "error_code_prefix": None,
    }


def test_build_filters_echo_missing_kwargs_default_to_none():
    """[IMPROVE-129] When a kwarg matching a schema key is
    NOT passed, the value defaults to None — preserves the
    always-present-key contract (key is in the dict, value
    is None)."""
    result = _build_filters_echo("/observability/recent")
    # All 5 keys present, all values None.
    assert set(result.keys()) == EXPECTED_RECENT_FILTERS
    for v in result.values():
        assert v is None


def test_build_filters_echo_extra_kwargs_silently_dropped():
    """[IMPROVE-129] Kwargs NOT in the schema are silently
    dropped — the schema is the source of truth, not the
    caller's kwarg list. Pin so a typo'd kwarg like
    ``fill_zeros_typo=True`` doesn't slip into the response."""
    result = _build_filters_echo(
        "/observability/summary",
        error_code="x",
        error_code_prefix=None,
        fill_zero_dim=True,
        bogus_extra_key="should be dropped",
    )
    assert "bogus_extra_key" not in result
    assert set(result.keys()) == EXPECTED_SUMMARY_FILTERS


def test_build_filters_echo_unknown_endpoint_raises():
    """[IMPROVE-129] An endpoint NOT in the registry raises
    KeyError. Pin the strict-lookup behaviour — catches typos
    at the production boundary instead of silently emitting
    a malformed dict."""
    with pytest.raises(KeyError):
        _build_filters_echo("/observability/notreal")


def test_build_filters_echo_preserves_schema_insertion_order():
    """[IMPROVE-129] Per Q4=A: the registry uses ``list[str]``
    (ordered) so dashboards display keys in the schema-defined
    order. Pin that the helper preserves that order — Python
    dicts preserve insertion order since 3.7, so as long as the
    helper builds via dict-comp over the list, this holds."""
    result = _build_filters_echo(
        "/observability/timeseries",
        subsystem="x",
        action="y",
        error_code="z",
        error_code_prefix="p",
        fill_zeros=True,
        fill_zero_time=False,
    )
    assert list(result.keys()) == FILTERS_ECHO_SCHEMA[
        "/observability/timeseries"
    ]


def test_endpoints_consume_centralised_schema(obs_test_client):
    """[IMPROVE-129] Live integration pin: each endpoint's
    response ``filters`` keys match the centralised schema
    exactly (no inline-dict drift). This pins the post-
    IMPROVE-129 migration: production runs through
    _build_filters_echo for all 4 endpoints."""
    for endpoint, schema_keys in FILTERS_ECHO_SCHEMA.items():
        resp = obs_test_client.get(endpoint)
        assert resp.status_code == 200, (
            f"{endpoint} returned {resp.status_code}"
        )
        filters = resp.json().get("filters", {})
        assert set(filters.keys()) == set(schema_keys), (
            f"{endpoint} response filters keys {sorted(filters.keys())} "
            f"diverge from FILTERS_ECHO_SCHEMA {sorted(schema_keys)} "
            f"— [IMPROVE-129] centralised-registry contract broken"
        )
