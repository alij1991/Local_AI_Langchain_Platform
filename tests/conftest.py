"""[IMPROVE-122] Shared pytest fixtures for the test suite.

This file is auto-loaded by pytest for any test under ``tests/``
— pytest's standard ``conftest.py`` discovery semantics. Any
fixture defined here is visible to all tests below it without
explicit imports.

## obs_test_client

[IMPROVE-115]'s post-startup ``DELETE FROM app_events``
truncation pattern was duplicated across
``test_observability_recent.py`` (the [IMPROVE-113] file) and
``test_observability_summary_rejections.py`` (the [IMPROVE-90]
+ [IMPROVE-115] file). Both fixtures were nearly identical:

  * monkey-patch ``DB_PATH`` to a tmp file
  * ``db_mod.init_db()``
  * ``with TestClient(api_server.app) as c:``
  * ``DELETE FROM app_events`` (clear startup-emitted noise so
    GROUP BY / count assertions land deterministically)
  * attach ``c._db_mod = db_mod`` for direct DB access in tests
  * ``yield c``

This conftest extracts that pattern as the ``obs_test_client``
fixture. Each test file's existing ``client`` fixture now
delegates to it via the standard pytest fixture-chaining
pattern:

  ``@pytest.fixture
    def client(obs_test_client):
        return obs_test_client``

Per Q3=A in the Wave 14 plan: extract just the truncate-after-
startup pattern (no event-prepopulation parameterisation). If a
4th obs-test file surfaces with prepopulated-events needs, add
that as a sibling fixture (``obs_test_client_with_events`` or
similar). The current 2 callsites don't justify the higher
abstraction.

Sources (2025-2026):
  * Wave 13 [IMPROVE-115] commit (89aff82) — surfaced the
    truncation pattern as a recurring need + named the
    shared-fixture extraction as a Wave 14 candidate.
  * Wave 12 [IMPROVE-108] commit (bed5fd3) — the SQLite
    LIKE/escape semantics that the obs endpoints share via
    the helper this fixture's tests exercise.
  * pytest fixture chaining docs (canonical 2025 reference):
    https://docs.pytest.org/en/stable/how-to/fixtures.html#fixtures-can-request-other-fixtures
  * "DRY test fixtures: when to extract" (Hypothesis +
    pytest tutorial 2025):
    https://docs.pytest.org/en/stable/explanation/fixtures.html
"""
from __future__ import annotations

import pytest


@pytest.fixture
def obs_test_client(monkeypatch, tmp_path):
    """[IMPROVE-122] Shared TestClient + tmp DB + post-startup
    ``app_events`` truncation for observability endpoint tests.

    Setup sequence:
      1. Monkey-patch ``local_ai_platform.db.DB_PATH`` to a
         per-test temp file so events don't pollute the dev
         DB.
      2. ``db_mod.init_db()`` — creates schema tables.
      3. ``TestClient(api_server.app)`` — starts the FastAPI
         app's lifespan (image warmup, agent setup, etc.).
         Startup emits events into ``app_events``.
      4. ``DELETE FROM app_events`` — clear startup-emitted
         events so each test starts with a clean event log.
         This matters for endpoints whose response shape is
         driven by aggregations of all events (``/recent``,
         ``/summary``'s items rollup) — without truncation,
         startup events show up as extra rows / tuples and
         skew count assertions.
      5. Attach ``c._db_mod = db_mod`` so tests can insert
         events directly via ``client._db_mod.get_conn()``.
       6. ``yield c`` — test runs.
       7. (Implicit cleanup via ``with`` and ``tmp_path``.)

    Why truncate AFTER startup (not before):
      The lifespan startup runs WHEN the TestClient context
      manager enters. If we truncated before entering the
      context, startup events would re-fire during the lifespan
      and re-pollute the table. Truncating inside the context
      (after the lifespan completes) ensures the event log is
      clean for the test.

    Yields:
        ``TestClient`` instance with ``_db_mod`` attribute set
        to the patched ``local_ai_platform.db`` module.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod
    from local_ai_platform.partner import memory as _partner_memory

    path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()

    # [IMPROVE-156] Wave 22 — the lifespan fires
    # ``_async_warmup_partner_memory`` as an
    # ``asyncio.create_task`` that emits
    # ``partner.mem0_embed_warmup`` (Phase 1) and
    # ``partner.mem0_init`` (Phase 2) events AFTER the TestClient
    # context manager's lifespan __aenter__ returns — i.e. AFTER
    # the ``DELETE FROM app_events`` truncation below. Without
    # neutralising the warmup, those events land in the table
    # mid-test and skew count assertions in /observability/recent
    # and /observability/summary. Replace with a no-op coroutine
    # for obs tests; the warmup function itself is exercised
    # directly in tests/test_partner_mem0_warmup.py against an
    # injected MockTransport, not via the lifespan path.
    async def _noop_warmup() -> None:
        return None
    monkeypatch.setattr(
        _partner_memory, "_async_warmup_partner_memory", _noop_warmup
    )

    import api_server
    with TestClient(api_server.app) as c:
        # [IMPROVE-115] Clear startup-emitted events so each
        # test starts with a deterministic event log. Without
        # this truncation, image warmup / agent setup / default
        # agent creation events would appear as extra rows in
        # /observability/recent and as extra (subsystem, action)
        # tuples in /observability/summary's items rollup,
        # skewing count assertions.
        conn = db_mod.get_conn()
        try:
            conn.execute("DELETE FROM app_events")
            conn.commit()
        finally:
            conn.close()
        # Make the active DB module easily accessible so each
        # test can insert events directly without re-resolving
        # DB_PATH.
        c._db_mod = db_mod
        yield c
