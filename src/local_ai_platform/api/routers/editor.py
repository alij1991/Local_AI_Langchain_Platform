"""Image editor router — sessions, edits, undo/redo, history, export.

[IMPROVE-1] Commit 8 — seventh router. All 13 /editor/* endpoints sit
on the lazy-init ``Depends(get_editor_service)`` (the singleton lives on
``app.state._editor_service`` from [IMPROVE-5]) — booting the editor is
expensive (loads CV / classical kernels), so it's done on first hit
rather than at lifespan startup.

Endpoints (13):
  POST   /editor/enhance-prompt              — model-aware instruction enhance
  GET    /editor/operations/list             — available ops + status
  POST   /editor/{session_id}/analyze        — quality + suggested tools
  POST   /editor/open                        — open image (file or generated)
  GET    /editor/files/{session_id}/{file}   — serve session file
  GET    /editor/{session_id}                — session state
  DELETE /editor/{session_id}                — close session
  POST   /editor/{session_id}/edit           — apply operation
  POST   /editor/{session_id}/undo
  POST   /editor/{session_id}/redo
  GET    /editor/{session_id}/history
  GET    /editor/{session_id}/compare        — diff two history steps
  POST   /editor/{session_id}/export         — write final file (PNG/JPEG/WEBP)

The /editor/files/{session_id}/{filename} handler does explicit path
traversal hardening: rejects ``..`` and slash characters in path
components, then resolves the full path and asserts it stays inside
``data/images/editor``. This mirrors the protections in /images/files/*
but is intentionally inlined here (the editor's session/file layout is
distinct from the images session layout — sharing the helper would
couple the two roots).

The /editor/enhance-prompt handler runs the (potentially blocking) LLM
call via ``loop.run_in_executor`` so the event loop stays responsive
when a slow Ollama generate happens. Same pattern as /editor/{id}/edit.

References (2025–2026):
* FastAPI APIRouter — https://fastapi.tiangolo.com/tutorial/bigger-applications/
* FastAPI FileResponse — https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse
* OWASP path traversal — https://owasp.org/www-community/attacks/Path_Traversal
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from local_ai_platform.api.deps import (
    get_app_config,
    get_editor_service,
    get_router,
)
from local_ai_platform.config import AppConfig
from local_ai_platform.observability import track_event
from local_ai_platform.providers import ProviderRouter
from local_ai_platform.tracing import trace_run

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/editor/enhance-prompt")
async def editor_enhance_prompt(
    body: dict[str, Any],
    router: ProviderRouter = Depends(get_router),
    config: AppConfig = Depends(get_app_config),
):
    """Enhance an image editing instruction for better results.

    Body:
        instruction (str, required): the user's original instruction
        model (str, optional): target model — one of 'kontext', 'cosxl',
            'pix2pix', 'controlnet'. Defaults to 'pix2pix' for backward
            compat. The enhancer produces different output formats for
            different models (target-state for kontext/controlnet,
            imperative for cosxl/pix2pix).

    Returns {original, enhanced, model}.
    """
    instruction = body.get("instruction", "")
    model = (body.get("model") or "pix2pix").lower().strip()
    if not instruction:
        raise HTTPException(400, "instruction is required")
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt
    loop = asyncio.get_event_loop()
    try:
        enhanced = await loop.run_in_executor(
            None,
            lambda: enhance_edit_prompt(instruction, router=router, config=config, model=model),
        )
    except Exception as e:
        logger.error("enhance-prompt failed: %s", e)
        raise HTTPException(500, f"Prompt enhancement failed: {e}")
    return {"original": instruction, "enhanced": enhanced, "model": model}


@router.get("/editor/operations/list")
async def editor_list_operations(
    editor=Depends(get_editor_service),
):
    """List all available edit operations (classical + AI + CV composite) with status."""
    return {"operations": editor.get_available_operations()}


@router.post("/editor/{session_id}/analyze")
async def editor_analyze(
    session_id: str,
    editor=Depends(get_editor_service),
):
    """Analyze image quality and get AI-powered tool suggestions."""
    session = editor.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    from local_ai_platform.images.ai_models import analyze_image_quality
    from PIL import Image
    image = Image.open(session["current_path"])
    return analyze_image_quality(image)


@router.post("/editor/open")
async def editor_open(
    body: dict[str, Any],
    editor=Depends(get_editor_service),
):
    """Open an image for editing. Accepts image_path, or session_id + image_id from generation."""
    image_path = body.get("image_path", "")
    source_type = body.get("source_type", "file")
    source_session_id = body.get("source_session_id")
    source_image_id = body.get("source_image_id")

    if not image_path:
        raise HTTPException(400, "image_path is required")

    try:
        return editor.open_image(image_path, source_type, source_session_id, source_image_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get("/editor/files/{session_id}/{filename}")
async def editor_serve_file(session_id: str, filename: str):
    """Serve editor image files."""
    # Security: prevent path traversal
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise HTTPException(400, "Invalid session ID")
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")
    file_path = Path(f"data/images/editor/{session_id}/{filename}")
    # Double-check resolved path is within editor directory
    editor_root = Path("data/images/editor").resolve()
    if not file_path.resolve().is_relative_to(editor_root):
        raise HTTPException(400, "Invalid path")
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    # Detect media type from extension
    suffix = file_path.suffix.lower()
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    media_type = media_types.get(suffix, "image/png")
    return FileResponse(str(file_path), media_type=media_type)


@router.get("/editor/{session_id}")
async def editor_get_session(
    session_id: str,
    editor=Depends(get_editor_service),
):
    session = editor.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Editor session '{session_id}' not found")
    return session


@router.delete("/editor/{session_id}")
async def editor_close(
    session_id: str,
    editor=Depends(get_editor_service),
):
    editor.close_session(session_id)
    return {"status": "closed"}


@router.post("/editor/{session_id}/edit")
async def editor_apply_edit(
    session_id: str,
    body: dict[str, Any],
    editor=Depends(get_editor_service),
):
    """Apply an edit operation. Body: {operation: str, params: {}}"""
    operation = body.get("operation", "")
    params = body.get("params", {})

    if not operation:
        raise HTTPException(400, "operation is required")

    # [IMPROVE-4] Commit 4/4: request-level image_edit OTel span. The
    # per-op emits inside images/editor.py keep being plain app_events
    # rows — see _OTEL_OPERATION_MAP comments. The edit op (e.g.
    # "remove_bg", "upscale", "instruct_pix2pix") is a custom
    # ``editor.operation`` attribute, NOT gen_ai.tool.name — tool
    # spans are reserved for LLM-tool calls, while editor ops are
    # the "what" of an image_edit operation.
    #
    # [IMPROVE-68] Commit 3/5: wrap the request in a TraceRecorder via
    # ``trace_run`` so the per-op emits inside editor.apply_edit
    # (``emit("editor", "op", ...)`` at editor.py:293/303/345) flow
    # into the trace JSON automatically. ``conversation_id=session_id``
    # ties every edit to its editor session, so /runs?subsystem=editor
    # groups by session naturally.
    #
    # CRITICAL: this route is ``async def``. The editor work runs via
    # ``loop.run_in_executor``, which does NOT inherit the calling
    # task's contextvars by default — that's the documented behavior of
    # threadpool executors and the load-bearing difference from the
    # image route (which is sync ``def`` and gets context propagation
    # for free via anyio's run_in_threadpool wrapper).
    #
    # ``contextvars.copy_context().run`` snapshots the current context
    # (including the ``_active_recorder`` set by trace_run on the line
    # above) and runs ``editor.apply_edit`` with that snapshot active
    # in the worker thread. Without it, every emit() inside apply_edit
    # would see ``recorder=None`` and the trace JSON would only show
    # the bracketing editor.edit start/end emits with zero per-op
    # detail.
    #
    # ``model_id=None``: editor classical CV ops (rotate, crop,
    # remove_bg) have no AI model; instruct ops (cosxl, kontext) have
    # one but it varies per call. The custom ``editor.operation`` OTel
    # attribute below is the better discriminator.
    with trace_run(
        subsystem="editor",
        agent_name="image_editor",
        model_provider="diffusers",
        model_id=None,
        conversation_id=session_id,
    ):
        with track_event("editor", "edit", context={
            "session_id": session_id,
            "operation": operation,
            "param_count": len(params or {}),
        }) as ev:
            try:
                # Snapshot the current context (with _active_recorder set)
                # and pass ctx.run as the executor target so the worker
                # thread runs editor.apply_edit with that context active.
                ctx = contextvars.copy_context()
                result = await asyncio.get_event_loop().run_in_executor(
                    None, ctx.run, editor.apply_edit,
                    session_id, operation, params,
                )
                ev.set_otel_attributes({
                    # Custom: which editor op was applied (e.g. "remove_bg",
                    # "upscale", "instruct_pix2pix"). Useful for filtering
                    # latency by op type.
                    "editor.operation": operation,
                    "gen_ai.system": "diffusers",
                })
                return result
            except ValueError as e:
                raise HTTPException(400, str(e))
            except RuntimeError as e:
                raise HTTPException(422, str(e))


@router.post("/editor/{session_id}/undo")
async def editor_undo(
    session_id: str,
    editor=Depends(get_editor_service),
):
    try:
        return editor.undo(session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/editor/{session_id}/redo")
async def editor_redo(
    session_id: str,
    editor=Depends(get_editor_service),
):
    try:
        return editor.redo(session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/editor/{session_id}/history")
async def editor_history(
    session_id: str,
    editor=Depends(get_editor_service),
):
    return {"steps": editor.get_history(session_id)}


@router.get("/editor/{session_id}/compare")
async def editor_compare(
    session_id: str,
    a: int = -1,
    b: int = -1,
    editor=Depends(get_editor_service),
):
    try:
        return editor.compare(session_id, a, b)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/editor/{session_id}/export")
async def editor_export(
    session_id: str,
    body: dict[str, Any],
    editor=Depends(get_editor_service),
):
    fmt = body.get("format", "PNG")
    quality = body.get("quality", 95)
    try:
        return editor.export(session_id, fmt, quality)
    except ValueError as e:
        raise HTTPException(400, str(e))
