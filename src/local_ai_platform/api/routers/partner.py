"""Partner router — voice + chat + memory for the AI Partner persona.

[IMPROVE-1] Commit 11 — final router. The last surface left in
api_server.py: 27 ``/partner/*`` HTTP endpoints plus the two
WebSockets (``/partner/voice/tts-stream`` and
``/partner/voice/stream-transcribe``). Brings the
``_free_gpu_for_partner`` helper along — it's only called from
``/partner/voice/init``.

Endpoints (29 incl. 2 WS):
  GET    /partner/profile                          — persona profile
  PUT    /partner/profile                          — update persona
  GET    /partner/stats                            — usage counters
  GET    /partner/memories                         — facts + memories
  POST   /partner/memories/facts                   — add fact
  DELETE /partner/memories/facts/{key}             — remove fact
  POST   /partner/memories/key                     — add memory
  DELETE /partner/memories/key/{memory_id}         — remove memory
  GET    /partner/knowledge-graph                  — graph triples
  GET    /partner/memories/facts/history/{key}     — temporal fact history
  GET    /partner/memories/archived                — archived (decayed) memories
  POST   /partner/chat                             — sync chat
  POST   /partner/chat/stream                      — SSE typed events
  GET    /partner/history                          — recent messages
  GET    /partner/user-profile                     — full profile dashboard
  DELETE /partner/user-profile                     — one-click reset
  POST   /partner/voice/init                       — boot ASR + TTS + VAD
  GET    /partner/voice/status                     — pipeline status
  POST   /partner/voice/synthesize-sentence        — single-sentence TTS
  WS     /partner/voice/tts-stream                 — streaming PCM16 TTS
  POST   /partner/voice/mode                       — kokoro / chatterbox
  POST   /partner/voice/gender                     — female / male
  GET    /partner/voice/gender                     — current voice gender
  POST   /partner/voice/transcribe                 — file ASR
  WS     /partner/voice/stream-transcribe          — streaming PCM16 ASR
  POST   /partner/voice/synthesize                 — full-text TTS
  POST   /partner/voice/chat                       — text → reply + audio
  POST   /partner/voice/upload                     — audio → ASR → LLM → TTS

WebSocket DI gotcha
-------------------
WebSocket endpoints can't use the HTTP-Request-typed
``Depends(get_partner_engine)`` chain — Starlette's WS lifecycle
doesn't run the same dependency machinery that resolves
``Request``-typed deps. Both WS handlers therefore inline the same
lazy-init pattern that ``get_partner_engine`` uses, reaching the same
cached engine on ``app.state._partner_engine`` so HTTP and WS handlers
share state.

References (2025–2026):
* FastAPI APIRouter — https://fastapi.tiangolo.com/tutorial/bigger-applications/
* FastAPI WebSockets — https://fastapi.tiangolo.com/advanced/websockets/
* Starlette WebSocket dependency limitation — https://www.starlette.io/websockets/
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.responses import Response, StreamingResponse

from local_ai_platform.api.deps import (
    get_image_service_or_none,
    get_partner_engine,
)
from local_ai_platform.images.service import ImageGenerationService
from local_ai_platform.observability import track_event
from local_ai_platform.tracing import trace_run

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Profile / stats / memory ─────────────────────────────────────


@router.get("/partner/profile")
async def partner_get_profile(partner=Depends(get_partner_engine)):
    return partner.get_profile()


@router.put("/partner/profile")
async def partner_update_profile(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    return partner.update_profile(body)


@router.get("/partner/stats")
async def partner_stats(partner=Depends(get_partner_engine)):
    return partner.get_stats()


@router.get("/partner/memories")
async def partner_memories(partner=Depends(get_partner_engine)):
    return partner.get_memories()


@router.post("/partner/memories/facts")
async def partner_add_fact(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    partner.add_fact(body.get("key", ""), body.get("value", ""), body.get("category", "general"))
    return {"status": "ok"}


@router.delete("/partner/memories/facts/{key}")
async def partner_remove_fact(
    key: str,
    partner=Depends(get_partner_engine),
):
    partner.remove_fact(key)
    return {"status": "ok"}


@router.post("/partner/memories/key")
async def partner_add_memory(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    mid = partner.add_memory(body.get("content", ""), body.get("tone", "neutral"), body.get("importance", 5))
    return {"id": mid}


@router.delete("/partner/memories/key/{memory_id}")
async def partner_remove_memory(
    memory_id: int,
    partner=Depends(get_partner_engine),
):
    partner.remove_memory(memory_id)
    return {"status": "ok"}


@router.get("/partner/knowledge-graph")
async def partner_knowledge_graph(entity: str = "user"):
    """Get knowledge graph triples for an entity."""
    from local_ai_platform.partner.memory import get_entity_triples, search_graph
    return {
        "direct": get_entity_triples(entity),
        "extended": search_graph(entity, depth=2),
    }


@router.get("/partner/memories/facts/history/{key}")
async def partner_fact_history(key: str):
    """Get temporal history of a fact (all values over time)."""
    from local_ai_platform.partner.memory import get_fact_history
    return get_fact_history(key)


@router.get("/partner/memories/archived")
async def partner_archived_memories(limit: int = 50):
    """Get archived (decayed) memories."""
    from local_ai_platform.partner.memory import get_archived_memories
    return get_archived_memories(limit)


# ── Chat (sync + SSE) ────────────────────────────────────────────


@router.post("/partner/chat")
async def partner_chat_sync(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    message = body.get("message", "")
    model = body.get("model")
    if not message:
        raise HTTPException(400, "message is required")

    # [IMPROVE-68] Commit 4/5: wrap the sync partner chat in trace_run
    # so the engine's stage emits (chat.start / chat / emotion_detect /
    # fact_extract at engine.py:294/356/419/492/639) flow into the
    # trace JSON. ``conversation_id`` is intentionally None — partner
    # chats are persona-scoped, not conversation-scoped (the persona
    # has its own memory store).
    #
    # ``partner.chat`` is a sync method called directly from this
    # ``async def`` handler — i.e. it runs in the event loop, not a
    # threadpool worker — so the ContextVar set by trace_run is
    # visible to the engine's emit() calls without a copy_context
    # dance. (This is also the historical "blocks the loop" bug for
    # long Ollama generates, but that's a separate concern from
    # observability.)
    with trace_run(
        subsystem="partner",
        agent_name=partner.profile.name,
        # Partner routes through ProviderRouter; "ollama" is the
        # default backend. Per-call provider varies based on the
        # actual model used and is recorded on the engine's emits.
        model_provider="ollama",
        model_id=model,
    ):
        reply = partner.chat(message, model)
        return {"reply": reply}


@router.post("/partner/chat/stream")
async def partner_chat_stream(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """SSE streaming partner chat with typed events.

    Events: thinking_pause, token, sentence_complete, done, error
    """
    message = body.get("message", "")
    model = body.get("model")
    enable_pause = body.get("thinking_pause", True)
    if not message:
        raise HTTPException(400, "message is required")

    async def stream_gen():
        yield f"event: start\ndata: {json.dumps({'partner': partner.profile.name})}\n\n"
        obs_ctx = {"partner": partner.profile.name,
                   "model": model, "streaming": True,
                   "thinking_pause": enable_pause,
                   "message_length": len(message)}
        # [IMPROVE-68] Commit 4/5: trace_run lives INSIDE the generator,
        # not outside it. Starlette consumes ``stream_gen()`` lazily
        # AFTER the route function returns — the recorder needs to be
        # active for the duration of the SSE iterator's actual run, not
        # just the route's setup. Placing the ``with trace_run`` here
        # is what keeps ``_active_recorder`` set while events stream
        # and what makes the engine's per-event emits inside
        # ``partner.astream_chat`` (chat.start / chat / emotion_detect /
        # fact_extract) flow into the trace JSON.
        #
        # ContextVars survive yield/resume across an async generator —
        # asyncio's task context machinery preserves the active context
        # when the generator pauses on yield and restores it on resume.
        # No copy_context dance is needed for this path because the
        # generator runs in the same event loop task as the request.
        with trace_run(
            subsystem="partner",
            agent_name=partner.profile.name,
            model_provider="ollama",
            model_id=model,
        ):
            with track_event("partner", "chat", context=obs_ctx) as ev:
                try:
                    async for event in partner.astream_chat(
                        message, model, enable_thinking_pause=enable_pause,
                    ):
                        etype = event.get("type", "")
                        if etype == "thinking_pause":
                            yield f"event: thinking\ndata: {json.dumps({'duration_ms': event.get('duration_ms', 0)})}\n\n"
                        elif etype == "emotion":
                            yield f"event: emotion\ndata: {json.dumps({'emotion': event.get('emotion', 'neutral')})}\n\n"
                        elif etype == "token":
                            yield f"event: token\ndata: {json.dumps({'text': event.get('text', '')})}\n\n"
                        elif etype == "sentence_complete":
                            yield f"event: sentence\ndata: {json.dumps({'sentence': event.get('sentence', '')})}\n\n"
                        elif etype == "done":
                            yield f"event: end\ndata: {json.dumps({'full_reply': event.get('full_reply', '')[:200]})}\n\n"
                        elif etype == "_metrics":
                            # Engine's final metrics — copy onto ev.perf so the
                            # track_event end-emit records token count + length.
                            ev.perf = {
                                "reply_length": event.get("reply_length", 0),
                                "token_count": event.get("token_count", 0),
                                "emotion_detected": event.get("emotion_detected", False),
                            }
                except Exception as exc:
                    # Yield the error event to the client THEN re-raise
                    # so trace_run + track_event both record the
                    # failure. Pre-IMPROVE-68 this branch ate the
                    # exception (mark_error + yield, no re-raise),
                    # which left the trace dict at success=True even
                    # for failed streams. Re-raising lets trace_run
                    # save success=False so /runs?subsystem=partner
                    # shows the failed run in red. Starlette closes
                    # the connection cleanly after a re-raised
                    # exception in an async generator — the bytes we
                    # already yielded reach the client; the close
                    # handshake just stops here.
                    yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
                    raise
                except BaseException as exc:
                    # Client disconnect (GeneratorExit), task cancel
                    # (asyncio.CancelledError, which is BaseException
                    # in Python 3.8+), KeyboardInterrupt. mark_cancelled
                    # records the cancelled status in app_events via
                    # track_event; trace_run's except-Exception clause
                    # deliberately does NOT catch these (matching
                    # chat.py's semantics from Commit 1/5), so the
                    # trace JSON is not saved for cancelled streams —
                    # that's by design for now; widen if /runs gains
                    # cancellation visualization later.
                    ev.mark_cancelled(type(exc).__name__)
                    raise

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


@router.get("/partner/history")
async def partner_history(limit: int = 50):
    from local_ai_platform.partner.memory import get_recent_messages
    return {"messages": get_recent_messages(limit)}


@router.get("/partner/user-profile")
async def partner_user_profile(partner=Depends(get_partner_engine)):
    """Return the full user profile (profile dashboard)."""
    return partner.get_user_profile()


@router.delete("/partner/user-profile")
async def partner_reset_user_profile(partner=Depends(get_partner_engine)):
    """One-click profile reset (ethical requirement from research).

    [IMPROVE-67] This single-scope endpoint is kept for backward
    compat with existing Flutter clients. Prefer the scoped variant
    ``DELETE /partner/profile/user_profile`` which produces the same
    result via the unified scope vocabulary.
    """
    return partner.reset_user_profile()


# ── [IMPROVE-67] Scoped reset + export ───────────────────────────


@router.get("/partner/export")
async def partner_export(partner=Depends(get_partner_engine)):
    """[IMPROVE-67] Bundle all partner state into a ZIP download.

    Returns:
      * ``profile.json`` — AI persona
      * ``user_profile.json`` — BigFive + emotional trajectory
      * ``facts.jsonl`` / ``key_memories.jsonl`` / ``archived.jsonl`` /
        ``journal.jsonl`` / ``messages.jsonl`` /
        ``knowledge_graph.jsonl`` — SQLite tables as JSONL
      * ``README.md`` — schema notes + export timestamp

    Maps to GDPR-style data portability (Article 20): users get
    their data BEFORE choosing to nuke any scope via
    ``DELETE /partner/profile/{scope}``. Read-only — the source
    data is untouched.
    """
    from local_ai_platform.partner.export import build_export_bundle

    bundle = build_export_bundle(partner)
    return Response(
        content=bundle,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="partner-export.zip"',
            "X-Export-Bytes": str(len(bundle)),
        },
    )


@router.delete("/partner/profile/{scope}")
async def partner_reset_scope(
    scope: str,
    partner=Depends(get_partner_engine),
):
    """[IMPROVE-67] Scoped reset of partner state.

    ``scope`` is one of:
      * ``profile`` — AI persona (``data/partner/profile.json``)
      * ``user_profile`` — BigFive + emotional trajectory
        (``data/partner/user_profile.json``)
      * ``facts`` — partner_core_facts table
      * ``key_memories`` — partner_key_memories table
      * ``archived`` — partner_memories_archive table
      * ``journal`` — partner_journal table
      * ``messages`` — partner_conversations table
      * ``knowledge_graph`` — partner_knowledge_graph table
      * ``all`` — every scope above, with per-scope breakdown in
        the response

    Returns a summary like::

        {"scope": "facts", "rows_cleared": 42, "files_cleared": 0,
         "engine_state_refreshed": false}

    Pre-IMPROVE-67 the only reset was ``DELETE /partner/user-profile``
    which left facts / key memories / knowledge graph intact —
    a partial reset users typically didn't expect. Per the doc
    complaint at ``08-partner.md:432``, scoped reset surfaces the
    full reset semantics users want.
    """
    from local_ai_platform.partner.reset import (
        RESET_SCOPES,
        reset_scope,
    )

    if scope not in RESET_SCOPES:
        raise HTTPException(
            400,
            f"Unknown scope: {scope!r}. Valid: {sorted(RESET_SCOPES)}",
        )
    return reset_scope(partner, scope)


# ── Voice init + VRAM cleanup helper ─────────────────────────────


@router.post("/partner/voice/init")
async def partner_voice_init(
    partner=Depends(get_partner_engine),
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Initialize voice pipeline (ASR + TTS + VAD).

    Also frees GPU VRAM by unloading image generation pipelines.
    Partner mode needs GPU memory for Ollama LLM inference.
    """
    # Free VRAM from image editing/generation pipelines
    freed = _free_gpu_for_partner(image_service)
    result = partner.init_voice()
    if freed:
        result["vram_freed"] = True
    return result


def _free_gpu_for_partner(
    image_service: ImageGenerationService | None,
) -> bool:
    """Unload all cached image pipelines to free GPU VRAM for LLM inference.

    [IMPROVE-5] ``image_service`` is passed in explicitly (was a module
    global before Commit 3). Pass ``None`` when no image service is up
    — the editing-pipeline cleanup still runs.
    """
    freed = False
    try:
        import torch
        # 1. Unload image editing pipelines (CosXL, Kontext, IP2P, ControlNet)
        from local_ai_platform.images.ai_enhance import _instruct_pipes
        if _instruct_pipes:
            for key in list(_instruct_pipes.keys()):
                pipe = _instruct_pipes.pop(key, None)
                if pipe is not None:
                    del pipe
            logger.info("Freed VRAM: unloaded %d image editing pipeline(s)", len(_instruct_pipes) + 1)
            freed = True

        # 2. Unload image generation pipelines
        try:
            if image_service and hasattr(image_service, '_pipelines') and image_service._pipelines:
                count = len(image_service._pipelines)
                for _key, _pipe in list(image_service._pipelines.items()):
                    try:
                        del _pipe
                    except Exception:
                        pass
                image_service._pipelines.clear()
                logger.info("Freed VRAM: unloaded %d image generation pipeline(s)", count)
                freed = True
        except Exception:
            pass

        if freed and torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            logger.info("GPU VRAM after cleanup: %.1f GB free", free_mem)
    except Exception as e:
        logger.debug("VRAM cleanup failed: %s", e)
    return freed


# ── Voice status / TTS HTTP ──────────────────────────────────────


@router.get("/partner/voice/status")
async def partner_voice_status(partner=Depends(get_partner_engine)):
    return partner.get_voice_status()


@router.post("/partner/voice/synthesize-sentence")
async def partner_voice_synthesize_sentence(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Synthesize a single sentence — for streaming TTS during chat."""
    sentence = body.get("sentence", body.get("text", ""))
    emotion = body.get("emotion", "neutral")
    if not sentence:
        raise HTTPException(400, "sentence is required")
    t0 = time.monotonic()
    wav_bytes = await asyncio.get_event_loop().run_in_executor(
        None, lambda: partner.synthesize_sentence(sentence, emotion)
    )
    elapsed = time.monotonic() - t0
    if wav_bytes is None:
        raise HTTPException(422, "TTS not available")
    logger.info("TTS sentence: %.0fms, %dKB, mode=%s, gender=%s — '%s'",
                elapsed * 1000, len(wav_bytes) // 1024, partner._tts_mode,
                partner._voice_gender, sentence[:50])
    return Response(content=wav_bytes, media_type="audio/wav")


# ── WebSocket: streaming TTS ─────────────────────────────────────


@router.websocket("/partner/voice/tts-stream")
async def partner_voice_tts_stream(websocket: WebSocket):
    """WebSocket streaming TTS: client sends text, server streams PCM16 audio chunks.

    Replaces per-sentence HTTP POST with a persistent connection.
    Protocol:
    - Client sends: JSON {"text": "...", "emotion": "..."}
    - Server sends: JSON {"type": "start", "sample_rate": 24000}
    - Server sends: binary PCM16 chunks (24kHz mono 16-bit)
    - Server sends: JSON {"type": "done"}
    - Client sends: JSON {"action": "close"} to disconnect
    """
    from fastapi.websockets import WebSocketDisconnect
    await websocket.accept()

    # [IMPROVE-5] WebSocket endpoints can't use HTTP-Request-typed
    # Depends (``get_partner_engine`` takes ``Request``). Use the
    # same lazy-init pattern against ``websocket.app.state`` so
    # the cached engine is shared with the HTTP handlers.
    partner = getattr(websocket.app.state, "_partner_engine", None)
    if partner is None:
        from local_ai_platform.partner.engine import PartnerEngine
        partner = PartnerEngine(
            websocket.app.state.router,
            websocket.app.state.config,
        )
        websocket.app.state._partner_engine = partner

    if partner._tts is None and partner._tts_emotional is None:
        await websocket.send_json({"error": "TTS not initialized. Call /partner/voice/init first."})
        await websocket.close()
        return

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                data = json.loads(message["text"])

                if data.get("action") == "close":
                    break

                text = data.get("text", "")
                emotion = data.get("emotion", "neutral")

                if not text or len(text.strip()) < 3:
                    await websocket.send_json({"type": "done"})
                    continue

                try:
                    t0 = time.monotonic()
                    # Send start marker with audio params
                    await websocket.send_json({"type": "start", "sample_rate": 24000})

                    # Stream PCM16 chunks
                    chunk_count = 0
                    async for chunk in partner.stream_synthesize(text, emotion):
                        await websocket.send_bytes(chunk)
                        chunk_count += 1

                    # Signal sentence complete
                    await websocket.send_json({"type": "done"})
                    logger.info("TTS-WS: %.0fms, %d chunks — '%s'",
                                (time.monotonic() - t0) * 1000, chunk_count, text[:50])
                except Exception as e:
                    if str(e):
                        logger.error("TTS stream synthesis error: %s", e)
                    try:
                        await websocket.send_json({"type": "error", "error": str(e)})
                    except Exception:
                        pass  # Client may have already closed

    except WebSocketDisconnect:
        pass
    except Exception as e:
        if "close message" not in str(e).lower():
            logger.error("TTS stream WebSocket error: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── Voice mode / gender / transcribe ─────────────────────────────


@router.post("/partner/voice/mode")
async def partner_voice_mode(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Switch TTS mode: 'kokoro' (fast, CPU) or 'chatterbox' (emotional, GPU/CPU)."""
    mode = body.get("mode", "kokoro")
    result = partner.set_tts_mode(mode)
    return {"status": result, "mode": partner._tts_mode}


@router.post("/partner/voice/gender")
async def partner_voice_gender(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Set voice gender: 'female' or 'male'."""
    gender = body.get("gender", "female")
    result = partner.set_voice_gender(gender)
    return {"status": result, "gender": partner.get_voice_gender()}


@router.get("/partner/voice/gender")
async def partner_voice_gender_get(partner=Depends(get_partner_engine)):
    return {"gender": partner.get_voice_gender()}


@router.post("/partner/voice/transcribe")
async def partner_voice_transcribe(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Transcribe audio to text via faster-whisper."""
    audio_path = body.get("audio_path", "")
    if not audio_path:
        raise HTTPException(400, "audio_path is required")
    try:
        text = partner.transcribe(audio_path)
        return {"text": text}
    except RuntimeError as e:
        raise HTTPException(422, str(e))


# ── WebSocket: streaming ASR ─────────────────────────────────────


@router.websocket("/partner/voice/stream-transcribe")
async def partner_voice_stream_transcribe(websocket: WebSocket):
    """WebSocket streaming STT: client sends PCM16 audio chunks, server returns partial transcriptions.

    Protocol:
    - Client sends: raw bytes (PCM16, 16kHz, mono) while user holds mic button
    - Client sends: text message "END" when user releases mic button
    - Server sends: JSON {"partial": "transcribed text so far"} as speech segments complete
    - Server sends: JSON {"final": "complete transcription", "done": true} at the end
    """
    from fastapi.websockets import WebSocketDisconnect
    await websocket.accept()

    # [IMPROVE-5] Same ``app.state`` lazy-init pattern as
    # partner_voice_tts_stream — WebSocket can't use HTTP-Request
    # Depends helpers.
    partner = getattr(websocket.app.state, "_partner_engine", None)
    if partner is None:
        from local_ai_platform.partner.engine import PartnerEngine
        partner = PartnerEngine(
            websocket.app.state.router,
            websocket.app.state.config,
        )
        websocket.app.state._partner_engine = partner

    if partner._asr is None:
        await websocket.send_json({"error": "ASR not initialized. Call /partner/voice/init first."})
        await websocket.close()
        return

    import numpy as np
    audio_buffer = bytearray()       # Full recording (for final transcription)
    new_bytes_count = 0              # Bytes received since last partial transcription
    last_partial = ""                # Last partial text sent to client
    MAX_AUDIO_BUFFER = 5 * 1024 * 1024  # 5MB (~5 minutes at 16kHz 16-bit)
    TRIGGER_BYTES = 24000            # ~1.5s of new audio triggers a partial transcription
    # Window size: use full buffer up to 10 seconds, then cap at 10s.
    # 10s is the sweet spot: Whisper handles it in ~600-800ms and gets great accuracy.
    MAX_WINDOW_BYTES = 320000        # 10s at 16kHz 16-bit

    # [IMPROVE-65] Silence detection now lives on the partner engine
    # (partner.is_speech) so the loaded Silero VAD model is actually
    # consulted. Falls back to RMS when Silero didn't load. Reduces
    # false-positive transcriptions on ambient noise and catches
    # whisper-level speech that the old 500-RMS threshold missed.
    silent_chunk_count = 0

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                raw = message["bytes"]
                if len(audio_buffer) + len(raw) > MAX_AUDIO_BUFFER:
                    await websocket.send_json({"error": "Audio buffer limit exceeded (5MB)"})
                    break

                audio_buffer.extend(raw)

                # Track speech vs silence
                if partner.is_speech(raw):
                    new_bytes_count += len(raw)
                    silent_chunk_count = 0
                else:
                    silent_chunk_count += 1
                    new_bytes_count += len(raw)
                    if silent_chunk_count > 6:  # ~3s of silence — skip transcription
                        continue

                # Transcribe: use full buffer when short, sliding window when long.
                # For recordings < 10s: transcribe everything (most accurate).
                # For recordings > 10s: transcribe last 10s only (capped latency).
                # Final transcription (on END) always uses the full buffer.
                if new_bytes_count >= TRIGGER_BYTES:
                    new_bytes_count = 0
                    try:
                        if len(audio_buffer) <= MAX_WINDOW_BYTES:
                            # Short recording — transcribe entire buffer
                            pcm16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                        else:
                            # Long recording — transcribe last 10 seconds only
                            pcm16 = np.frombuffer(bytes(audio_buffer[-MAX_WINDOW_BYTES:]), dtype=np.int16)
                        audio_f32 = pcm16.astype(np.float32) / 32768.0

                        text = await asyncio.get_event_loop().run_in_executor(
                            None, partner.transcribe_buffer, audio_f32
                        )

                        if text and text != last_partial:
                            last_partial = text
                            await websocket.send_json({"partial": text})
                    except Exception as e:
                        logger.debug("Stream transcribe chunk error: %s", e)

            elif "text" in message:
                text_msg = message["text"]
                if text_msg == "END":
                    # User released mic — final transcription on FULL buffer for best quality.
                    # The full-buffer pass gives Whisper complete context = most accurate result.
                    best_text = last_partial
                    if audio_buffer:
                        try:
                            pcm16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                            audio_f32 = pcm16.astype(np.float32) / 32768.0
                            final_text = await asyncio.get_event_loop().run_in_executor(
                                None, partner.transcribe_buffer, audio_f32
                            )
                            if final_text:
                                best_text = final_text
                        except Exception as e:
                            logger.warning("Final transcription error: %s", e)

                    try:
                        await websocket.send_json({"final": best_text, "done": True})
                    except Exception:
                        pass  # Client may have already closed
                    break

    except Exception as e:
        if "close message" not in str(e).lower():
            logger.error("WebSocket stream-transcribe error: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── Voice synth + voice chat + upload ────────────────────────────


@router.post("/partner/voice/synthesize")
async def partner_voice_synthesize(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Synthesize text to speech via Kokoro TTS. Returns WAV audio."""
    text = body.get("text", "")
    voice = body.get("voice")
    emotion = body.get("emotion", "neutral")
    if not text:
        raise HTTPException(400, "text is required")
    wav_bytes = await asyncio.get_event_loop().run_in_executor(
        None, lambda: partner.synthesize(text, voice=voice, emotion=emotion)
    )
    if wav_bytes is None:
        raise HTTPException(422, "TTS not available. Call /partner/voice/init first.")
    return Response(content=wav_bytes, media_type="audio/wav")


@router.post("/partner/voice/chat")
async def partner_voice_chat(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Full voice loop: text message → LLM reply → TTS audio.

    Accepts text (already transcribed by client) and returns both
    the text reply and the synthesized audio as base64 WAV.
    """
    message = body.get("message", "")
    voice = body.get("voice", "af_heart")
    model = body.get("model")
    if not message:
        raise HTTPException(400, "message is required")

    # Get text reply from LLM (emotion tag extracted internally)
    reply = partner.chat(message, model)

    # Use the emotion extracted from the LLM's [HAPPY]/[SAD] tag (most accurate)
    emotion = partner._last_detected_emotion

    # Synthesize reply with emotion-matched voice
    audio_b64 = None
    used_voice = voice
    if partner._tts is not None or partner._tts_emotional is not None:
        wav_bytes = partner.synthesize(reply, voice=None, emotion=emotion)
        used_voice = partner._get_voice_for_emotion(emotion)
        if wav_bytes:
            import base64
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    return {
        "reply": reply,
        "audio_base64": audio_b64,
        "voice": used_voice,
        "emotion": emotion,
        "has_audio": audio_b64 is not None,
    }


@router.post("/partner/voice/upload")
async def partner_voice_upload(
    request: Request,
    partner=Depends(get_partner_engine),
):
    """Upload audio file for transcription + chat + TTS response.

    Full pipeline: audio upload → ASR → LLM → TTS → audio response.
    """
    # Read raw audio bytes from request body
    body = await request.body()
    if not body:
        raise HTTPException(400, "No audio data")

    # Save to temp file for faster-whisper
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(body)
        temp_path = f.name

    try:
        # ASR: transcribe
        if partner._asr is None:
            partner.init_voice()
        if partner._asr is None:
            raise HTTPException(422, "ASR not available. Install faster-whisper.")

        user_text = partner.transcribe(temp_path)
        if not user_text.strip():
            return {"user_text": "", "reply": "", "has_audio": False}

        # LLM: generate reply (emotion tag extracted internally)
        reply = partner.chat(user_text)
        emotion = partner._last_detected_emotion

        # TTS: synthesize with emotion-matched voice
        audio_b64 = None
        if partner._tts is not None or partner._tts_emotional is not None:
            wav_bytes = partner.synthesize(reply, voice=None, emotion=emotion)
            if wav_bytes:
                import base64
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        return {
            "user_text": user_text,
            "reply": reply,
            "audio_base64": audio_b64,
            "has_audio": audio_b64 is not None,
        }
    finally:
        os.unlink(temp_path)
