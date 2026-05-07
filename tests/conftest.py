"""[IMPROVE-122] Shared pytest fixtures for the test suite.

This file is auto-loaded by pytest for any test under ``tests/``
— pytest's standard ``conftest.py`` discovery semantics. Any
fixture defined here is visible to all tests below it without
explicit imports.

## reset_settings_cache

[IMPROVE-173] Wave 37 — HF_HOME-isolated test fixture for HF
cache scan tests in ``test_images_service.py`` and
``test_huggingface.py``.

These tests use ``monkeypatch.setenv('HF_HOME', str(tmp_path))``
to redirect HF cache paths to a per-test temp dir. Production
code reads ``get_settings().hf_home`` (NOT ``os.environ['HF_HOME']``
directly), and ``get_settings()`` is module-level singleton-cached
in ``local_ai_platform.config._SETTINGS = AppSettings()`` on
first call. Once ``AppSettings`` is constructed (any earlier
``get_settings()`` call in the process — e.g. via a different
test, fixture, or import-time side effect), the cached instance
carries whatever ``HF_HOME`` was visible at construction time.
``monkeypatch.setenv`` AFTER first construction has no effect:
the cached ``AppSettings.hf_home`` doesn't re-read environ.

The fixture invalidates the cache via
``monkeypatch.setattr(cfg_mod, '_SETTINGS', None)`` so the next
``get_settings()`` call re-reads the monkeypatched env. Because
the reset goes through ``monkeypatch.setattr``, it's automatically
reverted to the pre-test value when the test ends — no manual
teardown.

Why test-side rather than production-side: a production-side
fix (e.g. having ``_hf_cache_dir`` read ``os.environ['HF_HOME']``
first and fall back to ``get_settings().hf_home``) would invert
the project's documented ".env priority > shell env" convention
per ``AppSettings.settings_customise_sources`` at
``config.py:340-359``. Test-side fix follows the W36 IMPROVE-171
"test-only fix shape" pattern: when production code is correct
but tests are stale, update the tests.

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
def reset_settings_cache(monkeypatch):
    """[IMPROVE-173] Wave 37 — Reset the AppSettings singleton so
    ``monkeypatch.setenv('HF_HOME', tmp_path)`` propagates to all
    ``get_settings().hf_home`` call sites in production.

    ``local_ai_platform.config._SETTINGS`` is a module-level cache
    populated by ``get_settings()`` on first call; subsequent calls
    return the same instance regardless of subsequent env-var
    changes. This fixture clears the cache via ``monkeypatch.setattr``
    so the next ``get_settings()`` call re-reads ``os.environ`` —
    including any HF_HOME override the test sets via
    ``monkeypatch.setenv``.

    monkeypatch.setattr restores ``_SETTINGS`` to its pre-test
    value when the test ends, so cross-test state doesn't leak.

    Tests opt in by adding ``reset_settings_cache`` to their
    parameter list; the fixture body runs before the test body
    even though the test doesn't reference the returned value.

    Pairs with ``monkeypatch.setenv('HF_HOME', str(tmp_path))`` —
    the env-var change is the active redirection; this fixture
    just ensures the change actually takes effect.
    """
    import local_ai_platform.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "_SETTINGS", None)
    monkeypatch.setattr(cfg_mod, "_SETTINGS_EMITTED", False)


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


# ── [IMPROVE-185] Wave 45 — Per-feature smoke fixtures ────────────


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """[IMPROVE-185] Wave 45 — Shared `tmp_db` fixture.

    Redirects ``local_ai_platform.db.DB_PATH`` to a per-test
    tmp file + runs ``db_mod.init_db()`` so consumers can use
    the schema without polluting the dev database.

    Returns the tmp DB path so tests that need to assert against
    the file can pass it explicitly (e.g. to ``AsyncDB(...)``).

    Pre-Wave-45 this fixture was duplicated across 4 test files
    (`test_agents_from_template_rename.py`,
    `test_conversations_thread_id.py`, `test_context_compactor.py`,
    `test_editor_user_presets.py`) — all the same 4-line pattern.
    W45 IMPROVE-185 extracts the duplicate into this conftest
    fixture; consumers drop their local copy and inherit via
    pytest's name-resolution rules.

    NOT used by ``test_db_pragmas.py`` — its `tmp_db` is a
    distinct shape (no ``init_db()`` + adds a
    ``_journal_mode_logged`` reset). That fixture stays local.
    """
    from local_ai_platform import db as db_mod

    path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()
    return path


@pytest.fixture
def tmp_editor_env(monkeypatch, tmp_path):
    """[IMPROVE-185] Wave 45 — Shared `tmp_editor_env` fixture.

    Redirects both ``db.DB_PATH`` and
    ``images.editor.EDITOR_DATA_DIR`` to per-test tmp paths so
    archive moves and DB writes don't touch the dev environment.

    Returns ``(editor_data_dir, archive_root)`` tuple. The
    archive root is derived dynamically by
    ``_editor_archive_root()`` so monkeypatching
    ``EDITOR_DATA_DIR`` is enough — no second patch needed
    (regression-pinned by
    ``test_archive_root_follows_editor_data_dir``).

    Pre-Wave-45 this fixture was duplicated across 2 test files
    (`test_editor_archive_on_close.py` +
    `test_editor_session_ttl_cleanup.py`) — same shape, same
    return tuple. W45 IMPROVE-185 extracts the duplicate; both
    consumers inherit via pytest's name-resolution rules.
    """
    from local_ai_platform import db as db_mod
    from local_ai_platform.images import editor as editor_mod

    db_path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    db_mod.init_db()

    editor_dir = tmp_path / "editor"
    editor_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", editor_dir)

    archive_root = editor_dir / "_archive"
    # [IMPROVE-185] Pre-W45 the archive_on_close.py fixture
    # didn't pre-create archive_root (production code does it
    # via `mkdir(parents=True, exist_ok=True)` on first archive),
    # but the session_ttl_cleanup.py fixture DID. Pre-create
    # here so both consumer profiles work — archive_on_close
    # tests are unaffected (the prod code's mkdir is idempotent
    # via exist_ok=True), and session_ttl_cleanup tests that
    # use direct `bucket.mkdir()` (without parents=True)
    # against pre-existing-archive_root succeed.
    archive_root.mkdir(parents=True, exist_ok=True)
    return editor_dir, archive_root
