#!/usr/bin/env python3
"""Standalone Chatterbox TTS server with voice gender support.

Setup:
    python -m venv .venv_chatterbox
    .venv_chatterbox\Scripts\Activate.ps1
    pip install chatterbox-tts fastapi uvicorn

Run:
    python scripts/chatterbox_server.py
"""
import io
import struct
import sys
import time
from pathlib import Path

try:
    import torch
    import numpy as np
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError as e:
    print(f"Missing: {e}")
    print("pip install chatterbox-tts fastapi uvicorn")
    sys.exit(1)

app = FastAPI(title="Chatterbox TTS Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Chatterbox on {device}...")

model = None
model_type = "unknown"
try:
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device=device)
    model_type = "base"
    print(f"Loaded Chatterbox base model on {device}")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

# ── Voice gender support via reference audio ──────────────────────
# Chatterbox clones any voice from a short reference clip.
# We generate female/male references using Kokoro at first startup.
VOICES_DIR = Path("data/partner/voices")
_current_gender = "female"
_gender_conds = {}  # Cache: {"female": conds, "male": conds}


def _get_ref_path(gender: str) -> Path:
    return VOICES_DIR / f"{gender}_ref.wav"


def _prepare_voice(gender: str) -> None:
    """Pre-load voice conditionals for the given gender."""
    global _current_gender
    ref_path = _get_ref_path(gender)
    if not ref_path.exists():
        print(f"[WARN] No reference audio for {gender}: {ref_path}")
        return
    if gender not in _gender_conds:
        print(f"Preparing {gender} voice conditionals from {ref_path}...")
        start = time.time()
        model.prepare_conditionals(str(ref_path))
        _gender_conds[gender] = model.conds
        print(f"{gender} voice ready ({time.time() - start:.1f}s)")
    else:
        model.conds = _gender_conds[gender]
    _current_gender = gender


# Pre-load female voice at startup (default)
try:
    _prepare_voice("female")
    print("Default voice: female")
except Exception as e:
    print(f"[WARN] Failed to prepare female voice: {e}")

# Also pre-load male for fast switching
try:
    _prepare_voice("male")
    _prepare_voice("female")  # Switch back to female default
    print("Male voice also cached")
except Exception as e:
    print(f"[WARN] Failed to prepare male voice: {e}")


def wav_bytes(samples: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert float32 samples to WAV bytes."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + len(pcm) * 2))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', len(pcm) * 2))
    buf.write(pcm.tobytes())
    return buf.getvalue()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "model": model_type,
        "current_gender": _current_gender,
        "voices_available": [g for g in ("female", "male") if _get_ref_path(g).exists()],
    }


@app.post("/gender")
async def set_gender(request: Request):
    """Switch voice gender. Swaps pre-cached conditionals (instant, no re-encode)."""
    body = await request.json()
    gender = body.get("gender", "female").lower().strip()
    if gender not in ("female", "male"):
        return JSONResponse({"error": f"Invalid gender: {gender}"}, status_code=400)
    _prepare_voice(gender)
    return {"gender": _current_gender}


@app.post("/synthesize")
async def synthesize(request: Request):
    """Synthesize text to WAV audio.

    Body: {"text": "...", "exaggeration": 0.5, "cfg_weight": 0.5, "gender": "female"}
    """
    body = await request.json()
    text = body.get("text", "")
    exaggeration = float(body.get("exaggeration", 0.7))
    cfg = float(body.get("cfg_weight", 0.5))
    gender = body.get("gender", _current_gender)

    if not text:
        return Response(content=b"", media_type="audio/wav")

    # Switch voice if needed (uses cached conds, instant)
    if gender != _current_gender and gender in _gender_conds:
        _prepare_voice(gender)

    start = time.time()
    temp = float(body.get("temperature", 0.6))
    wav_tensor = model.generate(text, exaggeration=exaggeration, cfg_weight=cfg, temperature=temp)
    elapsed = time.time() - start

    samples = wav_tensor.cpu().numpy().flatten()
    audio = wav_bytes(samples)

    print(f"Synthesized {len(text)} chars in {elapsed:.1f}s ({len(audio)//1024}KB) [{_current_gender}]")
    return Response(content=audio, media_type="audio/wav")


@app.post("/synthesize_sentence")
async def synthesize_sentence(request: Request):
    """Synthesize a single sentence for streaming TTS.

    Body: {"text": "...", "exaggeration": 0.7, "gender": "female"}
    """
    body = await request.json()
    text = body.get("text", "").strip()
    exaggeration = float(body.get("exaggeration", 0.7))
    gender = body.get("gender", _current_gender)

    if not text or len(text) < 3:
        return Response(content=b"", media_type="audio/wav")

    # Switch voice if needed
    if gender != _current_gender and gender in _gender_conds:
        _prepare_voice(gender)

    start = time.time()
    wav_tensor = model.generate(text, exaggeration=exaggeration, cfg_weight=0.5, temperature=0.6)
    elapsed = time.time() - start

    samples = wav_tensor.cpu().numpy().flatten()
    audio = wav_bytes(samples)

    print(f"Sentence ({len(text)} chars) in {elapsed:.1f}s [{_current_gender}]")
    return Response(content=audio, media_type="audio/wav")


if __name__ == "__main__":
    print("Chatterbox TTS server on http://127.0.0.1:8282")
    uvicorn.run(app, host="127.0.0.1", port=8282)
