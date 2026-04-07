#!/usr/bin/env python3
"""Standalone Chatterbox TTS server with Turbo mode.

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

# Try Turbo first (1-step diffusion = MUCH faster), fallback to base
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
    return {"status": "ok", "device": device, "model": model_type}


@app.post("/synthesize")
async def synthesize(request: Request):
    """Synthesize text → WAV audio.

    Body: {"text": "...", "exaggeration": 0.5, "cfg_weight": 0.5}
    """
    body = await request.json()
    text = body.get("text", "")
    exaggeration = float(body.get("exaggeration", 0.7))
    cfg = float(body.get("cfg_weight", 0.5))

    if not text:
        return Response(content=b"", media_type="audio/wav")

    start = time.time()
    # Lower temperature = faster convergence (less randomness in autoregressive decoding)
    temp = float(body.get("temperature", 0.6))
    wav_tensor = model.generate(text, exaggeration=exaggeration, cfg_weight=cfg, temperature=temp)
    elapsed = time.time() - start

    samples = wav_tensor.cpu().numpy().flatten()
    audio = wav_bytes(samples)

    print(f"Synthesized {len(text)} chars in {elapsed:.1f}s ({len(audio)//1024}KB)")
    return Response(content=audio, media_type="audio/wav")


@app.post("/synthesize_sentence")
async def synthesize_sentence(request: Request):
    """Synthesize a single sentence — optimized for streaming TTS.

    Called per-sentence as LLM generates. Short text = faster synthesis.
    """
    body = await request.json()
    text = body.get("text", "").strip()
    exaggeration = float(body.get("exaggeration", 0.7))

    if not text or len(text) < 3:
        return Response(content=b"", media_type="audio/wav")

    start = time.time()
    wav_tensor = model.generate(text, exaggeration=exaggeration, cfg_weight=0.5, temperature=0.6)
    elapsed = time.time() - start

    samples = wav_tensor.cpu().numpy().flatten()
    audio = wav_bytes(samples)

    print(f"Sentence ({len(text)} chars) in {elapsed:.1f}s")
    return Response(content=audio, media_type="audio/wav")


if __name__ == "__main__":
    print("Chatterbox TTS server on http://127.0.0.1:8282")
    uvicorn.run(app, host="127.0.0.1", port=8282)
