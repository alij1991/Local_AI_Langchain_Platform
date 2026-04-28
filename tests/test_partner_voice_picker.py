"""[IMPROVE-63] Voice picker with samples (not just female/male).

Pre-IMPROVE-63 the partner's voice was a binary female/male toggle
hardcoded in ``_VOICE_MAP``: female → ``af_heart``, male →
``am_adam``. Kokoro ships with ~9 distinct voices, but the user
couldn't pick — they got the gender default and that was it.

Doc proposal at ``docs/features/08-partner.md:604-608``: expose a
catalog endpoint, a "set voice" endpoint, and a sample endpoint
for previews. This commit:

  * Adds ``_VOICE_CATALOG`` static class-level list of 9 Kokoro
    voices (5 American female, 2 American male, 2 British
    female).
  * Adds ``get_voice_catalog`` / ``get_voice_id`` /
    ``set_voice_id`` / ``synthesize_voice_sample`` methods on
    ``PartnerEngine``. Validates against the catalog; unknown
    voice_id raises ValueError so the route can map to 400.
  * Updates ``_get_voice_for_emotion`` priority: explicit
    ``_voice_id`` (set via ``set_voice_id``) wins over the
    gender default. Pre-IMPROVE-63 callers using only
    ``set_voice_gender`` keep getting the gender's default voice.
  * Updates ``set_voice_gender`` to clear ``_voice_id`` so
    "back to male" actually reverts to ``am_adam`` instead of
    sticking with the previously-picked Bella.
  * Adds 4 routes::

        GET  /partner/voice/catalog            list voices
        GET  /partner/voice/id                 current selection
        POST /partner/voice/id  {voice_id}     set voice
        GET  /partner/voice/sample/{voice_id}  preview WAV bytes

Sources (2025-2026):
  * ``docs/features/08-partner.md:604-608`` — internal proposal
  * Kokoro TTS Voices Online 2026 (Readio):
    https://readiolabs.org/kokoro-tts
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── Catalog shape ───────────────────────────────────────────────────


def test_voice_catalog_contains_documented_ids():
    """The catalog must include the IDs the doc explicitly names
    (af_heart, af_sky, am_adam, bf_isabella). Pin so a future
    catalog edit can't silently drop one of the marquee voices."""
    from local_ai_platform.partner.engine import PartnerEngine

    catalog = PartnerEngine._VOICE_CATALOG
    ids = {v["id"] for v in catalog}
    for required in ("af_heart", "af_sky", "am_adam", "bf_isabella"):
        assert required in ids, f"missing voice: {required}"


def test_voice_catalog_entries_have_required_fields():
    """Each entry must carry id / display_name / gender /
    language / description so the Flutter picker can render
    rich cards."""
    from local_ai_platform.partner.engine import PartnerEngine

    required = {"id", "display_name", "gender", "language", "description"}
    for entry in PartnerEngine._VOICE_CATALOG:
        assert set(entry.keys()) >= required, (
            f"entry {entry.get('id')} missing fields: "
            f"{required - set(entry.keys())}"
        )


def test_voice_catalog_ids_are_unique():
    """Duplicate IDs would break ``set_voice_id`` lookup."""
    from local_ai_platform.partner.engine import PartnerEngine

    ids = [v["id"] for v in PartnerEngine._VOICE_CATALOG]
    assert len(ids) == len(set(ids))


def test_voice_catalog_genders_are_valid():
    """Every entry must report ``female`` or ``male`` so the
    backward-compat ``set_voice_gender`` sync stays sensible."""
    from local_ai_platform.partner.engine import PartnerEngine

    for entry in PartnerEngine._VOICE_CATALOG:
        assert entry["gender"] in ("female", "male")


# ── get_voice_catalog returns a copy ────────────────────────────────


def test_get_voice_catalog_returns_independent_copies():
    """A route handler that mutates the returned dicts shouldn't
    poison subsequent calls. Pin so a future "return self._VOICE_
    CATALOG directly" optimisation is rejected."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)  # bypass __init__
    a = eng.get_voice_catalog()
    a[0]["display_name"] = "MUTATED"
    b = eng.get_voice_catalog()
    assert b[0]["display_name"] != "MUTATED"


# ── get_voice_id priority ──────────────────────────────────────────


def test_get_voice_id_falls_back_to_gender_default():
    """When ``_voice_id`` is None, the gender default applies.
    Default gender ``female`` → ``af_heart``."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._voice_gender = "female"
    eng._voice_id = None
    assert eng.get_voice_id() == "af_heart"

    eng._voice_gender = "male"
    assert eng.get_voice_id() == "am_adam"


def test_get_voice_id_explicit_overrides_gender():
    """``_voice_id`` set via ``set_voice_id`` wins over the
    gender default. Mirrors the priority used in
    ``_get_voice_for_emotion`` so the route's reported value
    matches what TTS will use."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._voice_gender = "female"
    eng._voice_id = "bf_isabella"
    assert eng.get_voice_id() == "bf_isabella"


# ── set_voice_id ───────────────────────────────────────────────────


def test_set_voice_id_valid_persists():
    """``set_voice_id("af_bella")`` updates the engine state and
    returns the id."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._voice_gender = "female"
    eng._voice_id = None
    eng._tts_emotional = None  # skip Chatterbox sync

    out = eng.set_voice_id("af_bella")
    assert out == "af_bella"
    assert eng._voice_id == "af_bella"


def test_set_voice_id_syncs_gender_to_catalog():
    """Picking a male voice flips ``_voice_gender`` to "male"
    so the Chatterbox sync (gender-keyed) stays consistent.
    Pin both directions: female-pick → gender="female",
    male-pick → gender="male"."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._voice_gender = "female"  # start female
    eng._voice_id = None
    eng._tts_emotional = None

    eng.set_voice_id("am_michael")
    assert eng._voice_gender == "male"

    eng.set_voice_id("af_sky")
    assert eng._voice_gender == "female"


def test_set_voice_id_unknown_raises_valueerror():
    """Unknown voice_id → ValueError. Caller maps to 400."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._voice_gender = "female"
    eng._voice_id = None
    eng._tts_emotional = None

    with pytest.raises(ValueError) as ei:
        eng.set_voice_id("nonexistent_voice")
    # Error message should list valid IDs so the user can self-fix.
    assert "nonexistent_voice" in str(ei.value)


# ── set_voice_gender clears voice_id ───────────────────────────────


def test_set_voice_gender_clears_explicit_voice_id():
    """[IMPROVE-63] Toggling the gender clears any previously-set
    explicit voice — without this, "switch to male" after picking
    Bella would still hear Bella, which is surprising."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._voice_gender = "female"
    eng._voice_id = "af_bella"
    eng._tts_emotional = None

    eng.set_voice_gender("male")
    assert eng._voice_id is None
    assert eng.get_voice_id() == "am_adam"


# ── _get_voice_for_emotion priority ────────────────────────────────


def test_emotion_voice_lookup_uses_explicit_voice_id():
    """The TTS path reads ``_get_voice_for_emotion`` per-call.
    Pin that the explicit voice_id wins over the gender default
    inside that helper too — not just in ``get_voice_id``."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._voice_gender = "female"
    eng._voice_id = "bf_emma"
    assert eng._get_voice_for_emotion("happy") == "bf_emma"
    assert eng._get_voice_for_emotion("sad") == "bf_emma"


def test_emotion_voice_lookup_falls_back_when_voice_id_unset():
    """Pre-IMPROVE-63 callers (no ``set_voice_id`` ever called)
    keep getting the gender default. Pin the no-regression path."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._voice_gender = "male"
    eng._voice_id = None
    assert eng._get_voice_for_emotion("happy") == "am_adam"


# ── synthesize_voice_sample ────────────────────────────────────────


def test_synthesize_voice_sample_validates_voice_id():
    """Unknown voice_id → ValueError BEFORE we even try to
    generate. Saves a TTS round-trip on bad input."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._voice_gender = "female"
    eng._voice_id = None
    eng._tts = None

    with pytest.raises(ValueError):
        eng.synthesize_voice_sample("does_not_exist")


def test_synthesize_voice_sample_does_not_mutate_active_voice(monkeypatch):
    """Critical UX invariant: clicking "play sample" for af_bella
    while currently set to af_heart MUST NOT switch the active
    voice. The synthesize call passes ``voice=...`` directly,
    bypassing ``_get_voice_for_emotion``, but pin the contract
    so a future refactor can't accidentally call set_voice_id."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._voice_gender = "female"
    eng._voice_id = "af_heart"
    eng._tts_emotional = None

    # Stub synthesize to capture args + return canned bytes.
    captured = {}

    def _fake_synthesize(text, voice=None, emotion="neutral"):
        captured["text"] = text
        captured["voice"] = voice
        captured["emotion"] = emotion
        return b"FAKE_WAV_BYTES"

    eng.synthesize = _fake_synthesize

    eng.synthesize_voice_sample("af_bella")
    # Sample for af_bella was rendered.
    assert captured["voice"] == "af_bella"
    # But the engine's active voice is still af_heart.
    assert eng._voice_id == "af_heart"


def test_synthesize_voice_sample_uses_short_fixed_text():
    """The sample text is a single fixed phrase — not the user's
    last message — so previews are consistent across voices."""
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._voice_gender = "female"
    eng._voice_id = None
    eng._tts_emotional = None

    captured = {}

    def _fake_synthesize(text, voice=None, emotion="neutral"):
        captured["text"] = text
        return b"FAKE"

    eng.synthesize = _fake_synthesize
    eng.synthesize_voice_sample("af_heart")
    assert captured["text"] == PartnerEngine._VOICE_SAMPLE_TEXT
    assert "Hello" in captured["text"]


# ── Route integration ─────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch, tmp_path):
    """In-process TestClient with tmp DB. Many partner routes
    fan out to TTS / STT / Mem0 — we don't init any of them; the
    voice-picker routes only need ``get_voice_catalog`` etc.
    which are pure dict operations."""
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "app.db")
    db_mod.init_db()

    import api_server
    with TestClient(api_server.app) as c:
        yield c


def test_route_catalog_lists_voices(client):
    """``GET /partner/voice/catalog`` returns the full catalog
    plus current selection."""
    resp = client.get("/partner/voice/catalog")
    assert resp.status_code == 200
    body = resp.json()
    assert "voices" in body and len(body["voices"]) >= 9
    assert "current_voice_id" in body
    assert "fallback_gender" in body
    assert body["sample_endpoint"] == "/partner/voice/sample/{voice_id}"


def test_route_id_get_returns_current(client):
    """``GET /partner/voice/id`` returns the current voice."""
    resp = client.get("/partner/voice/id")
    assert resp.status_code == 200
    assert "voice_id" in resp.json()


def test_route_id_set_valid_persists(client):
    """``POST /partner/voice/id`` with a valid id → 200 with the
    new state. ``GET`` afterward returns the new id."""
    resp = client.post(
        "/partner/voice/id", json={"voice_id": "af_sky"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["voice_id"] == "af_sky"
    # Round-trip: GET reflects the change.
    resp_get = client.get("/partner/voice/id")
    assert resp_get.json()["voice_id"] == "af_sky"


def test_route_id_set_unknown_returns_400(client):
    """Unknown voice_id → 400 with the engine's error message."""
    resp = client.post(
        "/partner/voice/id", json={"voice_id": "no_such_voice"},
    )
    assert resp.status_code == 400


def test_route_id_set_missing_body_returns_400(client):
    """Missing ``voice_id`` field → 400."""
    resp = client.post("/partner/voice/id", json={})
    assert resp.status_code == 400


def test_route_sample_unknown_voice_returns_400(client):
    """Unknown voice_id in the path → 400."""
    resp = client.get("/partner/voice/sample/no_such_voice")
    assert resp.status_code == 400


def test_route_sample_returns_503_when_kokoro_unavailable(client):
    """Sample render requires Kokoro to be loaded. In the test
    environment Kokoro isn't initialised (no model files), so the
    route must surface that as 503 with a clear message."""
    resp = client.get("/partner/voice/sample/af_heart")
    # Either 503 (Kokoro not loaded — typical CI) or 200 (Kokoro
    # is loaded — the dev env has the models). Pin both branches
    # so the test is robust.
    assert resp.status_code in (200, 503)
    if resp.status_code == 503:
        body = resp.json()
        # Error message is helpful — points the user at the fix.
        detail = body.get("detail") or body
        assert "Kokoro" in str(detail) or "TTS" in str(detail)
    else:
        # If it did succeed, the response is WAV-typed bytes.
        assert resp.headers.get("content-type", "").startswith("audio/")


def test_route_voice_picker_does_not_disturb_legacy_gender_route(client):
    """The legacy ``POST /partner/voice/gender`` still works after
    the picker endpoints land. Backward compat pin."""
    resp = client.post(
        "/partner/voice/gender", json={"gender": "male"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["gender"] == "male"
