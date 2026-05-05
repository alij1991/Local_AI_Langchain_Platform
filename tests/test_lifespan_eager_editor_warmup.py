"""[IMPROVE-161] Wave 27 — lifespan eager editor warm-up under
feature flag (Path D from the Wave 21 residue list).

When ``LIFESPAN_EAGER_EDITOR_WARMUP=1`` is set in .env, lifespan
calls ``await asyncio.to_thread(_build_editor_service)`` after the
[IMPROVE-155] hardware-profile warm-up. Trades ~21s of boot time
for hot first ``/editor/*`` calls. Default-off (False) preserves
current boot speed.

These tests pin the on/off contract:

  1. Default (flag unset) — lifespan does NOT pre-build the editor
     service. ``app.state._editor_service`` stays unset until the
     first ``/editor/*`` request triggers the lazy-init via
     ``get_editor_service``.

  2. Enabled (flag=1) — lifespan DOES pre-build.
     ``app.state._editor_service`` is non-None right after
     ``__enter__`` returns.

  3. Build failure — lifespan continues even if ``_build_editor_
     service`` raises. The lazy-init path on first request takes
     over (degraded but not broken).

Test strategy: build a ``TestClient(api_server.app)`` with the
flag patched via ``monkeypatch.setenv`` + ``reset_settings_cache``.
The lifespan reads ``app.state.settings.lifespan_eager_editor_
warmup`` directly so flipping the env-var + resetting the cache
flips the behavior across the lifespan boundary.

Sources (2025-2026):

  * pydantic-settings docs (2025):
    https://docs.pydantic.dev/latest/concepts/settings/ —
    canonical reference for env-var-driven config.

  * Wave 21 retrospective 5c79cbf — original lazy-init design
    that the eager-warm-up flag pre-builds on top of.

  * Wave 21 [IMPROVE-153] commit (5b6725f) — ``get_editor_
    service`` async + ``_build_editor_service`` sync helper.
    This wave's flag uses the same helper; the difference is
    WHEN it runs (lifespan vs first request).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api_server
from local_ai_platform.config import reset_settings_cache


@pytest.fixture
def fresh_settings(monkeypatch):
    """Reset the AppSettings cache so a fresh
    ``get_settings()`` call picks up the env-var changes this
    test sets up. Both the pre-test and post-test reset are
    needed to keep the cache from leaking across tests.
    """
    reset_settings_cache()
    yield
    monkeypatch.delenv("LIFESPAN_EAGER_EDITOR_WARMUP", raising=False)
    reset_settings_cache()


@pytest.fixture
def stub_build_editor_service(monkeypatch):
    """Replace ``_build_editor_service`` with a stub that returns
    a sentinel object — fast, no diffusers/transformers imports
    in the test path. Returns the sentinel for assertion.
    """
    sentinel = MagicMock(name="StubEditorService")
    monkeypatch.setattr(
        "local_ai_platform.api.deps._build_editor_service",
        lambda: sentinel,
    )
    return sentinel


# ── Default-off path ────────────────────────────────────────────────


def test_default_off_does_not_prebuild(monkeypatch, fresh_settings,
                                        stub_build_editor_service):
    """When ``LIFESPAN_EAGER_EDITOR_WARMUP`` is unset (default),
    lifespan must NOT call ``_build_editor_service`` at startup.
    The lazy-init path on first request handles construction.
    """
    monkeypatch.delenv("LIFESPAN_EAGER_EDITOR_WARMUP", raising=False)
    reset_settings_cache()

    # Track whether the stub was called during lifespan startup.
    calls = []
    monkeypatch.setattr(
        "local_ai_platform.api.deps._build_editor_service",
        lambda: calls.append("build") or stub_build_editor_service,
    )

    with TestClient(api_server.app) as client:
        # After lifespan startup, the editor service must NOT be
        # set (no eager pre-build) — it'll be built lazily on
        # first /editor/* request.
        assert not calls, (
            f"Default-off path called _build_editor_service "
            f"{len(calls)} time(s); expected 0. "
            f"LIFESPAN_EAGER_EDITOR_WARMUP must be opt-in."
        )

        # The flag's value on the cached settings should reflect
        # the default (False).
        assert client.app.state.settings.lifespan_eager_editor_warmup is False


# ── Enabled path ────────────────────────────────────────────────────


def test_enabled_prebuilds_at_lifespan(monkeypatch, fresh_settings,
                                        stub_build_editor_service):
    """When ``LIFESPAN_EAGER_EDITOR_WARMUP=1`` is set, lifespan
    must call ``_build_editor_service`` and cache the result on
    ``app.state._editor_service`` BEFORE returning from
    ``__enter__``. First /editor/* request finds it hot.
    """
    monkeypatch.setenv("LIFESPAN_EAGER_EDITOR_WARMUP", "1")
    reset_settings_cache()

    calls = []
    monkeypatch.setattr(
        "local_ai_platform.api.deps._build_editor_service",
        lambda: calls.append("build") or stub_build_editor_service,
    )

    # Reset any cached service from a previous test so we measure
    # fresh build behavior (without this, a prior test's run could
    # leave the service cached on app.state).
    if hasattr(api_server.app.state, "_editor_service"):
        delattr(api_server.app.state, "_editor_service")

    with TestClient(api_server.app) as client:
        # Lifespan must have called the build helper exactly once
        # during startup.
        assert len(calls) == 1, (
            f"Enabled path called _build_editor_service "
            f"{len(calls)} time(s); expected 1."
        )
        # The service must be cached on app.state for the
        # async Depends factory's hot-path return.
        assert client.app.state._editor_service is stub_build_editor_service

        # The flag's value on the cached settings should be True.
        assert client.app.state.settings.lifespan_eager_editor_warmup is True


# ── Failure-tolerance path ──────────────────────────────────────────


def test_build_failure_does_not_abort_lifespan(monkeypatch, fresh_settings):
    """If ``_build_editor_service`` raises during lifespan eager
    warm-up, the exception must NOT propagate out of the lifespan
    (would abort the FastAPI app). Instead, the warm-up logs a
    warning + continues; the lazy-init path on first request
    takes over (degraded but not broken).
    """
    monkeypatch.setenv("LIFESPAN_EAGER_EDITOR_WARMUP", "1")
    reset_settings_cache()

    def _raising_build():
        raise RuntimeError("simulated diffusers import failure")

    monkeypatch.setattr(
        "local_ai_platform.api.deps._build_editor_service",
        _raising_build,
    )

    if hasattr(api_server.app.state, "_editor_service"):
        delattr(api_server.app.state, "_editor_service")

    # If the lifespan didn't catch the exception, this would raise.
    with TestClient(api_server.app) as client:
        # The service must NOT be cached (build failed) but the
        # lifespan completed — first /editor/* request will trigger
        # lazy-init (which would also fail in this scenario, but
        # that's the lazy path's problem to handle, not the
        # warm-up's).
        editor = getattr(client.app.state, "_editor_service", None)
        assert editor is None, (
            f"Build failure path left _editor_service set to "
            f"{editor!r}; expected None (lazy fallback)."
        )

        # The flag stayed True — failure didn't reset the setting.
        assert client.app.state.settings.lifespan_eager_editor_warmup is True


# ── Setting field surface pin ───────────────────────────────────────


def test_setting_field_default_is_false():
    """The default value of ``lifespan_eager_editor_warmup`` MUST
    stay False. Flipping the default to True would silently change
    boot semantics for everyone who hasn't opted in.
    """
    from local_ai_platform.config import AppSettings

    fields = AppSettings.model_fields
    assert "lifespan_eager_editor_warmup" in fields, (
        "Field name drift detected — Path D rename without test "
        "update suggests the contract changed."
    )
    field_info = fields["lifespan_eager_editor_warmup"]
    assert field_info.default is False, (
        f"Default for lifespan_eager_editor_warmup is "
        f"{field_info.default!r}; MUST be False to preserve "
        f"current boot speed."
    )
