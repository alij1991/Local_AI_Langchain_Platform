"""[IMPROVE-163] Wave 29 — partner voice persistence (Tranche B
from the Wave 18 deferred queue).

Pre-Wave-29 the PartnerEngine fields ``_voice_id`` /
``_voice_gender`` / ``_tts_mode`` were module-instance state lost
on every backend restart. Wave 29 introduces
``data/partner/voice_settings.json`` so a user picking
``af_bella`` via the partner voice-id setter survives the next
uvicorn cycle.

These tests pin both halves of the contract:

  * ``voice_settings`` module: load returns defaults when no file
    exists, roundtrips a saved dataclass exactly, falls back on
    corrupt JSON, and ignores unknown keys (forward compat).

  * ``PartnerEngine`` integration: the 3 setters
    (``set_voice_id`` / ``set_voice_gender`` / ``set_tts_mode``)
    write the file; ``_restore_persisted_voice_settings`` reads
    the file and applies fields to the engine; engine __init__
    invokes the restore path so a fresh ``PartnerEngine()`` reads
    the file.

Test strategy mirrors W24's ``test_partner_phrase_streaming.py``
+ W26's ``test_startup_timing_benchmarks.py`` patterns:

  * ``monkeypatch.setattr`` redirects ``_VOICE_SETTINGS_PATH`` to
    a tmp directory so tests don't clobber any real
    ``data/partner/voice_settings.json``.

  * Engine tests use the W24 ``__new__`` bypass (no SQLite /
    Mem0 / Kokoro init) — the persistence path is independent
    of the heavy init collaborators.

  * Module-constant pin (mirrors W24/W26/W27/W28 patterns).

Sources (2025-2026):

  * docs/features/10-improvements.md §10.5 Wave 29 — wave-shape
    spec.

  * docs/features/08-partner.md §IMPROVE-63 — chapter-side
    documentation for the voice-picker surface.

  * IMPROVE-NEW-12 prior art at
    src/local_ai_platform/partner/memory.py — the memory_decay
    pattern this wave mirrors.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── voice_settings module: load / save contract ───────────────────────


def _redirect_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect ``_VOICE_SETTINGS_PATH`` to a tmp file so tests
    don't clobber any real ``data/partner/voice_settings.json``.
    Returns the redirected path so the test can read/write it
    directly to set up fixtures.
    """
    target = tmp_path / "voice_settings.json"
    from local_ai_platform.partner import voice_settings
    monkeypatch.setattr(voice_settings, "_VOICE_SETTINGS_PATH", target)
    return target


def test_load_returns_defaults_when_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """First-run / never-customised case: missing file yields the
    hardcoded defaults without raising.
    """
    _redirect_path(monkeypatch, tmp_path)
    from local_ai_platform.partner.voice_settings import load_voice_settings

    s = load_voice_settings()
    assert s.voice_id is None
    assert s.voice_gender == "female"
    assert s.tts_mode == "kokoro"


def test_save_then_load_preserves_all_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Roundtrip pin: save → load returns the same dataclass."""
    _redirect_path(monkeypatch, tmp_path)
    from local_ai_platform.partner.voice_settings import (
        VoiceSettings, load_voice_settings, save_voice_settings,
    )

    save_voice_settings(VoiceSettings(
        voice_id="af_bella",
        voice_gender="female",
        tts_mode="chatterbox",
    ))
    s = load_voice_settings()
    assert s.voice_id == "af_bella"
    assert s.voice_gender == "female"
    assert s.tts_mode == "chatterbox"


def test_load_corrupt_json_falls_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Corrupt JSON file: load returns defaults + does not raise.
    Best-effort policy — a corrupt file shouldn't block the
    partner engine from booting.
    """
    target = _redirect_path(monkeypatch, tmp_path)
    target.write_text("{not valid json")
    from local_ai_platform.partner.voice_settings import load_voice_settings

    s = load_voice_settings()
    assert s.voice_id is None
    assert s.voice_gender == "female"
    assert s.tts_mode == "kokoro"


def test_load_non_dict_falls_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Valid JSON but wrong shape (e.g., a top-level list): load
    returns defaults. Same best-effort policy as corrupt JSON.
    """
    target = _redirect_path(monkeypatch, tmp_path)
    target.write_text(json.dumps(["not", "a", "dict"]))
    from local_ai_platform.partner.voice_settings import load_voice_settings

    s = load_voice_settings()
    assert s.voice_id is None
    assert s.voice_gender == "female"
    assert s.tts_mode == "kokoro"


def test_load_invalid_voice_gender_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Persisted voice_gender outside ``{female, male}``: that
    field falls back to default; valid sibling fields still apply.
    """
    target = _redirect_path(monkeypatch, tmp_path)
    target.write_text(json.dumps({
        "voice_id": "af_bella",
        "voice_gender": "neutral",  # invalid
        "tts_mode": "chatterbox",
    }))
    from local_ai_platform.partner.voice_settings import load_voice_settings

    s = load_voice_settings()
    assert s.voice_gender == "female"  # fell back
    assert s.voice_id == "af_bella"  # sibling still applied
    assert s.tts_mode == "chatterbox"


def test_load_invalid_tts_mode_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Persisted tts_mode outside ``{kokoro, chatterbox}``: that
    field falls back to default; valid sibling fields still apply.
    """
    target = _redirect_path(monkeypatch, tmp_path)
    target.write_text(json.dumps({
        "voice_id": "af_heart",
        "voice_gender": "male",
        "tts_mode": "espeak",  # invalid
    }))
    from local_ai_platform.partner.voice_settings import load_voice_settings

    s = load_voice_settings()
    assert s.tts_mode == "kokoro"  # fell back
    assert s.voice_id == "af_heart"
    assert s.voice_gender == "male"


def test_load_unknown_keys_silently_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Forward compat: a future field added in a newer build then
    downgraded shouldn't crash the older build's loader. Unknown
    keys are silently dropped.
    """
    target = _redirect_path(monkeypatch, tmp_path)
    target.write_text(json.dumps({
        "voice_id": "af_bella",
        "voice_gender": "female",
        "tts_mode": "kokoro",
        "future_field": {"nested": True},  # unknown
        "another_future_field": 42,
    }))
    from local_ai_platform.partner.voice_settings import load_voice_settings

    s = load_voice_settings()
    assert s.voice_id == "af_bella"
    assert s.voice_gender == "female"
    assert s.tts_mode == "kokoro"


def test_save_creates_parent_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """First-run case: save to a path whose parent doesn't exist
    yet should mkdir parents=True. The data/partner/ directory is
    created lazily on first save.
    """
    target = tmp_path / "fresh" / "subdir" / "voice_settings.json"
    from local_ai_platform.partner import voice_settings
    monkeypatch.setattr(voice_settings, "_VOICE_SETTINGS_PATH", target)
    from local_ai_platform.partner.voice_settings import (
        VoiceSettings, save_voice_settings,
    )

    assert not target.parent.exists()
    save_voice_settings(VoiceSettings(voice_id="af_heart"))
    assert target.exists()


# ── PartnerEngine integration: setters + restore ──────────────────────


def _bare_engine_with_voice_state(monkeypatch: pytest.MonkeyPatch):
    """Construct a minimally-initialised PartnerEngine with just
    the voice-related state populated. Bypasses __init__ via
    ``__new__`` (no SQLite / Mem0 / Kokoro init) — the persistence
    path is independent of those collaborators.

    Patterned after W24's ``test_partner_phrase_streaming.py`` +
    ``test_partner_engine_httpx.py`` __new__ bypass + minimal stub.
    """
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._tts = None
    eng._tts_emotional = None
    eng._tts_mode = "kokoro"
    eng._voice_gender = "female"
    eng._voice_id = None
    eng._chatterbox_variant = None
    return eng


def test_set_voice_id_persists_to_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """``set_voice_id("af_bella")`` writes the file with the new
    catalog id and the synced gender from the catalog entry.
    """
    target = _redirect_path(monkeypatch, tmp_path)
    eng = _bare_engine_with_voice_state(monkeypatch)

    eng.set_voice_id("af_bella")

    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["voice_id"] == "af_bella"
    assert payload["voice_gender"] == "female"
    assert payload["tts_mode"] == "kokoro"


def test_set_voice_gender_persists_to_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """``set_voice_gender("male")`` writes the file with the new
    gender and ``voice_id: null`` (gender-change clears the
    explicit catalog override per [IMPROVE-63] semantics).
    """
    target = _redirect_path(monkeypatch, tmp_path)
    eng = _bare_engine_with_voice_state(monkeypatch)
    eng._voice_id = "af_bella"  # pre-set state

    eng.set_voice_gender("male")

    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["voice_gender"] == "male"
    assert payload["voice_id"] is None  # cleared by set_voice_gender


def test_set_tts_mode_persists_to_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """``set_tts_mode`` writes the file on the success path. The
    early-return failure paths leave the file unchanged because
    ``_tts_mode`` itself is unchanged.
    """
    target = _redirect_path(monkeypatch, tmp_path)
    eng = _bare_engine_with_voice_state(monkeypatch)
    eng._tts_emotional = "http://stub.local:8282"  # so chatterbox path is allowed
    eng._tts = object()  # so kokoro path is allowed

    eng.set_tts_mode("chatterbox")

    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["tts_mode"] == "chatterbox"


def test_set_tts_mode_failure_does_not_persist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """When ``set_tts_mode("chatterbox")`` is called on a system
    without the Chatterbox sidecar, the early return prevents
    state mutation AND prevents the persistence write — no file
    is created on the failure path.
    """
    target = _redirect_path(monkeypatch, tmp_path)
    eng = _bare_engine_with_voice_state(monkeypatch)
    eng._tts_emotional = None  # chatterbox unavailable

    result = eng.set_tts_mode("chatterbox")

    assert "chatterbox not available" in result
    assert eng._tts_mode == "kokoro"  # state unchanged
    assert not target.exists()  # file not written on failure path


def test_restore_applies_persisted_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """``_restore_persisted_voice_settings`` reads the file and
    populates the engine's voice/gender/mode fields. Mirrors what
    happens at the tail of ``__init__``.
    """
    target = _redirect_path(monkeypatch, tmp_path)
    target.write_text(json.dumps({
        "voice_id": "af_bella",
        "voice_gender": "female",
        "tts_mode": "chatterbox",
    }))
    eng = _bare_engine_with_voice_state(monkeypatch)

    eng._restore_persisted_voice_settings()

    assert eng._voice_id == "af_bella"
    assert eng._voice_gender == "female"
    assert eng._tts_mode == "chatterbox"


def test_restore_with_no_file_keeps_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """No file: restore is a no-op (defaults remain). The engine
    starts up cleanly when the data/partner/voice_settings.json
    has never been written (first-run case)."""
    _redirect_path(monkeypatch, tmp_path)
    eng = _bare_engine_with_voice_state(monkeypatch)
    eng._voice_id = "preset_to_something"  # ensure restore is what changes it

    eng._restore_persisted_voice_settings()

    # load_voice_settings returned defaults; engine fields reflect those.
    assert eng._voice_id is None
    assert eng._voice_gender == "female"
    assert eng._tts_mode == "kokoro"


def test_restore_with_chatterbox_persisted_falls_back_at_apply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """A persisted ``tts_mode: chatterbox`` is loaded as-is by
    restore (this module doesn't validate against runtime
    availability — that's set_tts_mode's job). The engine still
    has ``_tts_mode = "chatterbox"`` after restore even if
    Chatterbox isn't running; the synthesize path's existing
    "if self._tts_emotional is not None" guards handle the
    fallback at apply time.

    This pin documents that restore is INTENTIONALLY agnostic to
    runtime availability — keeps the load layer simple, the
    apply layer already has the right guards.
    """
    target = _redirect_path(monkeypatch, tmp_path)
    target.write_text(json.dumps({"tts_mode": "chatterbox"}))
    eng = _bare_engine_with_voice_state(monkeypatch)
    eng._tts_emotional = None  # Chatterbox sidecar NOT running

    eng._restore_persisted_voice_settings()

    assert eng._tts_mode == "chatterbox"  # loaded as-is
    # Apply-time guard (set_tts_mode) catches the unavailable case;
    # synthesize() branches on ``self._tts_emotional is not None``
    # so a stale "chatterbox" value silently falls through to
    # Kokoro at synth time.


# ── Module-constant pin (mirrors W24/W26/W27/W28 pattern) ─────────────


def test_voice_settings_module_constants_match_design_values():
    """Pin the module-level constants against drift. Mirrors the
    W24 ``test_module_constants_match_design_values`` + W26
    ``test_threshold_constants_match_design_values`` + W27
    ``test_setting_field_default_is_false`` + W28
    ``test_schema_version_constant_matches_design_value`` patterns.

    Design values:
      * ``_VALID_GENDERS`` = ``("female", "male")``
      * ``_VALID_TTS_MODES`` = ``("kokoro", "chatterbox")``
      * Default ``VoiceSettings`` = (None, "female", "kokoro")

    A change to any of these is a behavioural change requiring
    documentation + cross-checks against partner/engine.py's
    ``_VOICE_MAP`` (gender keys) and ``set_tts_mode`` (mode
    accepted values).
    """
    from local_ai_platform.partner import voice_settings
    from local_ai_platform.partner.voice_settings import VoiceSettings

    assert voice_settings._VALID_GENDERS == ("female", "male"), (
        f"_VALID_GENDERS = {voice_settings._VALID_GENDERS}; "
        f"expected ('female', 'male'). A change must align with "
        f"PartnerEngine._VOICE_MAP keys."
    )
    assert voice_settings._VALID_TTS_MODES == ("kokoro", "chatterbox"), (
        f"_VALID_TTS_MODES = {voice_settings._VALID_TTS_MODES}; "
        f"expected ('kokoro', 'chatterbox'). A change must align "
        f"with PartnerEngine.set_tts_mode's accepted values."
    )

    default = VoiceSettings()
    assert default.voice_id is None
    assert default.voice_gender == "female"
    assert default.tts_mode == "kokoro"
