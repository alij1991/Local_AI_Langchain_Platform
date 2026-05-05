"""[IMPROVE-163] Wave 29 — partner voice settings persistence.

Pre-Wave-29 the PartnerEngine fields ``_voice_id`` /
``_voice_gender`` / ``_tts_mode`` were module-instance state lost
on every backend restart. A user picking ``af_bella`` via the
partner voice-id setter would have to re-pick after every
uvicorn cycle. Wave 29 introduces this module so the picks
survive across restarts.

Design follows the [IMPROVE-NEW-12] memory-decay pattern (sibling
of ``data/partner/profile.json`` / ``data/partner/user_profile.json``
/ ``data/partner/memory_decay.json``):

  * ``load_voice_settings()`` returns a ``VoiceSettings`` dataclass
    populated from disk, or the default if the file is missing /
    corrupt / contains invalid values. Best-effort.

  * ``save_voice_settings(s)`` writes the dataclass to disk.
    Best-effort: a write failure (disk full, permission denied)
    logs a warning but does not raise — the in-memory state has
    already taken effect, so the user's runtime tweak still
    works for the current session.

PartnerEngine integrates this via:

  * ``__init__`` calls ``load_voice_settings`` after the in-class
    defaults are set, so persisted state overrides defaults.
  * ``set_voice_id`` / ``set_voice_gender`` / ``set_tts_mode`` call
    ``save_voice_settings`` on success.

Cross-machine portability follows the Wave 28 identity-mint-on-
import pattern's spirit: voice_id (Kokoro catalog id like
``af_heart`` / ``af_bella``) IS portable since it maps to the
same entry on every Kokoro install. tts_mode is borderline —
a persisted ``chatterbox`` value falls back to ``kokoro`` at
apply time on machines without the Chatterbox sidecar running
(via the existing ``set_tts_mode`` "chatterbox not available"
guard, not via this module).

Sources (2025-2026):

  * docs/features/10-improvements.md §10.5 Wave 29 — the wave-shape
    spec this module implements.

  * docs/features/08-partner.md §IMPROVE-63 — chapter-side
    documentation for the voice-picker surface this module makes
    persistent.

  * IMPROVE-NEW-12 prior art:
    src/local_ai_platform/partner/memory.py — the memory_decay
    persistence pattern this module mirrors.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_VOICE_SETTINGS_PATH = Path("data/partner/voice_settings.json")

_VALID_GENDERS = ("female", "male")
_VALID_TTS_MODES = ("kokoro", "chatterbox")


@dataclass
class VoiceSettings:
    """Partner voice configuration that survives backend restart.

    Fields:
      voice_id: Kokoro catalog id (e.g. ``af_heart`` / ``af_bella``)
        or None to fall back to the gender's default voice. Picked
        via the partner voice-id setter; takes priority over the
        gender default in ``_get_voice_for_emotion``.
      voice_gender: ``female`` or ``male``. Used both as the fallback
        when voice_id is None and to drive Chatterbox sidecar gender
        sync.
      tts_mode: ``kokoro`` (fast, CPU) or ``chatterbox`` (emotional,
        GPU/CPU). A persisted ``chatterbox`` value falls back to
        ``kokoro`` at apply time on machines without the Chatterbox
        sidecar running.
    """
    voice_id: Optional[str] = None
    voice_gender: str = "female"
    tts_mode: str = "kokoro"


def load_voice_settings() -> VoiceSettings:
    """[IMPROVE-163] Load persisted voice settings from disk.

    Returns a default ``VoiceSettings`` if:
      * the file does not exist (first run / never customised)
      * the file is corrupt JSON (warn + default)
      * any field has an invalid value (default for that field;
        valid fields still apply)
      * unknown keys are present (silently ignored — forward
        compat: a future field added in a newer build then
        downgraded should not crash the old build)
    """
    if not _VOICE_SETTINGS_PATH.exists():
        return VoiceSettings()
    try:
        data = json.loads(_VOICE_SETTINGS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "[IMPROVE-163] Could not load voice settings from %s "
            "(using defaults): %s", _VOICE_SETTINGS_PATH, exc,
        )
        return VoiceSettings()
    if not isinstance(data, dict):
        logger.warning(
            "[IMPROVE-163] Voice settings at %s is not a dict "
            "(using defaults)", _VOICE_SETTINGS_PATH,
        )
        return VoiceSettings()

    settings = VoiceSettings()

    raw_voice_id = data.get("voice_id")
    if raw_voice_id is None:
        settings.voice_id = None
    elif isinstance(raw_voice_id, str) and raw_voice_id.strip():
        settings.voice_id = raw_voice_id
    else:
        logger.debug(
            "[IMPROVE-163] Persisted voice_id %r invalid (using "
            "default None)", raw_voice_id,
        )

    raw_gender = data.get("voice_gender")
    if raw_gender in _VALID_GENDERS:
        settings.voice_gender = raw_gender
    elif raw_gender is not None:
        logger.debug(
            "[IMPROVE-163] Persisted voice_gender %r invalid "
            "(using default female)", raw_gender,
        )

    raw_mode = data.get("tts_mode")
    if raw_mode in _VALID_TTS_MODES:
        settings.tts_mode = raw_mode
    elif raw_mode is not None:
        logger.debug(
            "[IMPROVE-163] Persisted tts_mode %r invalid (using "
            "default kokoro)", raw_mode,
        )

    return settings


def save_voice_settings(settings: VoiceSettings) -> None:
    """[IMPROVE-163] Persist voice settings to disk. Best-effort:
    a write failure logs a warning but does not raise — the
    in-memory state has already taken effect, so the user's
    runtime tweak still works for the current session.
    """
    try:
        _VOICE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "voice_id": settings.voice_id,
            "voice_gender": settings.voice_gender,
            "tts_mode": settings.tts_mode,
        }
        _VOICE_SETTINGS_PATH.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        logger.warning(
            "[IMPROVE-163] Could not persist voice settings to %s: %s",
            _VOICE_SETTINGS_PATH, exc,
        )
