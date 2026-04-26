"""Chat router — direct, agent, supervisor, resume, enhance, in-conversation image gen.

[IMPROVE-1] Commit 4 — third router. Brings the chat-only helpers
(`_pick_small_ollama_model`, `_ollama_generate_via_router`,
`_SMALL_OLLAMA_KEYWORDS`) along because they're invoked from
`/chat/enhance-prompt` and `/chat/generate-image` only.

Endpoints (7):
  POST /chat/enhance-prompt           — type-aware LLM prompt rewrite
  POST /chat/generate-image           — in-conversation image gen + msg log
  POST /chat/direct                   — provider-direct (no agent layer)
  POST /chat                          — agent chat (non-streaming)
  POST /chat/stream                   — agent chat (SSE)
  POST /chat/supervisor/{name}        — supervisor agent
  POST /chat/resume                   — resume after human-in-the-loop interrupt

The two router-mediated Ollama helpers ([IMPROVE-14]) replaced
hand-rolled urllib calls to /api/generate. Tests reach them via
``api_server._pick_small_ollama_model`` and
``api_server._ollama_generate_via_router`` — api_server.py re-exports
them under those names so the test suite (test_chat_enhance_router.py)
keeps working post-split.

References (2025–2026):
* FastAPI APIRouter — https://fastapi.tiangolo.com/tutorial/bigger-applications/
* Server-Sent Events (SSE) over StreamingResponse —
  https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
* LangGraph thread_id checkpoint pattern — https://langchain-ai.github.io/langgraph/concepts/persistence/
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.api.deps import (
    get_image_service,
    get_orchestrator,
    get_router,
    get_trace_store,
)
from local_ai_platform.images.service import ImageGenerationService
from local_ai_platform.observability import track_event
from local_ai_platform.providers import (
    ChatMessage,
    GenerationSettings,
    ProviderRouter,
)
from local_ai_platform.repositories.conversations import (
    add_message,
    create_conversation,
    get_conversation,
    list_messages,
    set_conversation_thread_id,
)
from local_ai_platform.token_counting import count_tokens
from local_ai_platform.tracing import (
    LocalTraceCallbackHandler,
    TraceRecorder,
    TraceStore,
    load_trace_config,
)

logger = logging.getLogger("api_server")

router = APIRouter()


# ── [IMPROVE-17] Stream cancel-detection seam ────────────────────


async def _is_client_gone(request: Request) -> bool:
    """[IMPROVE-17] Cancel-detection seam for ``/chat/stream``.

    Wraps Starlette's ``request.is_disconnected()`` with an
    exception-swallowing guard — the SSE inner loop must never
    crash because the disconnect probe failed; if the probe is
    flaky (rare ASGI receive-channel error), keep streaming and
    rely on the natural completion path.

    Pre-IMPROVE-17 ``/chat/stream`` had no disconnect check at all
    — Starlette buffers the SSE generator into a discarded sink
    after the client closes the connection, so the orchestrator
    ran to completion and persisted a full assistant message the
    user never asked for. The poll inside the inner loop is the
    standard pattern from "How We Used SSE to Stream LLM Responses
    at Scale" (Akabani, Medium); per-frame cost is microseconds
    against an asyncio receive channel.

    Tests monkeypatch this helper to simulate disconnect
    deterministically (tests/test_chat_stream_cancellation.py)
    instead of poking Starlette's receive channel directly.
    """
    try:
        return await request.is_disconnected()
    except Exception:
        return False


# ── Request models ───────────────────────────────────────────────


class ChatRequest(BaseModel):
    agent: str | None = None
    agent_name: str | None = None
    message: str = Field(..., min_length=1, max_length=50000)
    conversation_id: str | None = None
    image_paths: list[str] | None = None
    stream: bool = False
    settings: dict[str, Any] | None = None
    model: str | None = None
    provider: str | None = None
    thread_id: str | None = None

    @property
    def resolved_agent(self) -> str:
        return self.agent or self.agent_name or "assistant"


class DirectChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    settings: dict[str, Any] | None = None
    stream: bool = False


# ── Router-mediated Ollama helpers ([IMPROVE-14]) ─────────────────
# Both /chat/enhance-prompt and /chat/generate-image used to hand-roll
# urllib.request.urlopen calls to http://localhost:11434/api/generate.
# These helpers route through ProviderRouter.achat instead so they
# honor OLLAMA_BASE_URL, share the [IMPROVE-12] availability cache,
# and surface errors through the unified HTTPException contract.


_SMALL_OLLAMA_KEYWORDS = ("1b", "2b", "3b", "tiny", "mini", "small", "phi", "qwen2")


def _pick_small_ollama_model(router: ProviderRouter) -> str | None:
    """Return the first small chat-capable Ollama model name, or None.

    Replaces the open-coded OllamaProvider() instantiation that used to
    live inline in both endpoints. Goes through router.get_provider so
    the existing base_url / timeout config is respected.

    [IMPROVE-5] Router is now passed in (Depends-injected at the
    endpoint layer) rather than read from the module global.
    """
    prov = router.get_provider("ollama")
    if prov is None:
        return None
    try:
        models = prov.list_models()
    except Exception as exc:
        logger.debug("_pick_small_ollama_model list_models failed: %s", exc)
        return None
    names = [m.name for m in models if getattr(m.capabilities, "supports_chat", True)]
    if not names:
        return None
    for kw in _SMALL_OLLAMA_KEYWORDS:
        for n in names:
            if kw in n.lower():
                return n
    return names[0]


async def _ollama_generate_via_router(
    router: ProviderRouter,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout_sec: int = 120,
) -> str:
    """Single-shot Ollama completion through ProviderRouter.achat.

    Wraps one ChatMessage(user, prompt) call with a GenerationSettings.
    Maps connection/timeout failures to HTTPException so the endpoint
    layer can propagate them without extra try/except boilerplate.

    [IMPROVE-5] Router is now passed in explicitly (Depends-injected
    at the endpoint layer) rather than read from the module global.
    """
    settings = GenerationSettings(temperature=temperature, max_tokens=max_tokens)
    messages = [ChatMessage(role="user", content=prompt)]

    try:
        response = await asyncio.wait_for(
            router.achat(f"ollama:{model}", messages, settings),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, f"Ollama generation timed out after {timeout_sec}s")
    except HTTPException:
        raise
    except Exception as exc:
        # Check both the message and the exception class name — some
        # errors (e.g. ConnectionRefusedError) stringify to just the
        # `errno` message without any connection-related keyword.
        low = (str(exc) + " " + type(exc).__name__).lower()
        if "connection" in low or "refused" in low or "connecterror" in low:
            raise HTTPException(503, "Cannot connect to Ollama. Is it running? Start with: ollama serve")
        raise HTTPException(500, f"Ollama generation failed: {exc}")

    return (getattr(response, "content", "") or "").strip()


# ── Endpoints ─────────────────────────────────────────────────────


@router.post("/chat/enhance-prompt")
async def enhance_chat_prompt(
    body: dict[str, Any],
    router: ProviderRouter = Depends(get_router),
):
    """Use a local LLM to detect prompt type (text/image/code) and enhance accordingly."""
    user_prompt = (body.get("prompt") or "").strip()
    if not user_prompt:
        raise HTTPException(400, "prompt is required")
    ollama_model = (body.get("ollama_model") or "").strip()
    timeout_sec = int(body.get("timeout_sec") or 120)

    # [IMPROVE-14] Model picker goes through the router now.
    if not ollama_model:
        ollama_model = _pick_small_ollama_model(router) or ""
    if not ollama_model:
        raise HTTPException(503, "No Ollama model available. Install one with: ollama pull gemma3:1b")

    # ── Step 1: Classify prompt intent ──
    # Heuristic first — fast, no LLM call needed for obvious cases
    _lower = user_prompt.lower()
    _image_keywords = [
        "photo of", "picture of", "image of", "portrait of", "painting of",
        "illustration of", "render of", "scene of", "landscape of",
        "generate an image", "generate a photo", "create an image", "draw ",
        "realistic photo", "cinematic", "4k", "8k", "masterpiece",
        "best quality", "highly detailed", "a man ", "a woman ",
        "a girl ", "a boy ", "a cat ", "a dog ", "beautiful ",
        "photorealistic", "digital art", "concept art", "anime ",
        "studio lighting", "bokeh", "depth of field",
    ]
    _code_keywords = [
        "write a function", "write code", "implement", "create a class",
        "write a script", "python code", "javascript code", "fix this code",
        "debug this", "refactor", "write a program", "api endpoint",
        "def ", "function(", "class ", "import ", "```",
    ]

    # Count keyword matches
    _img_score = sum(1 for kw in _image_keywords if kw in _lower)
    _code_score = sum(1 for kw in _code_keywords if kw in _lower)

    # Determine type from heuristics
    if _img_score >= 2 or (_img_score >= 1 and _code_score == 0 and len(user_prompt.split()) <= 30):
        prompt_type = "image"
    elif _code_score >= 1:
        prompt_type = "code"
    else:
        # Ambiguous — ask the LLM to classify
        prompt_type = "text"  # default
        try:
            # [IMPROVE-14] Router-mediated classification call.
            classify_prompt = f"""/no_think
Classify this user prompt into exactly one category. Reply with ONLY the category word, nothing else.

Categories:
- IMAGE: if the user wants to generate/create/describe a visual image, photo, illustration, artwork, or scene
- CODE: if the user wants code, programming, debugging, or technical implementation
- TEXT: if the user wants text writing, questions answered, explanations, emails, essays, or general conversation

User prompt: {user_prompt}

Category:"""
            _cls = (await _ollama_generate_via_router(
                router,
                ollama_model,
                classify_prompt,
                temperature=0.1,
                max_tokens=10,
                timeout_sec=15,
            )).upper()
            if "IMAGE" in _cls:
                prompt_type = "image"
            elif "CODE" in _cls:
                prompt_type = "code"
        except Exception:
            pass  # fall back to "text"

    # ── Step 2: Enhance with type-specific system prompt ──
    if prompt_type == "image":
        system_prompt = """/no_think
You are an expert Stable Diffusion / Flux prompt engineer.
Rewrite the user's request into an optimized image generation prompt.

Rules:
1. Output a vivid, descriptive prompt (max 60 words) with subject, setting, lighting, style, and quality tags
2. Use comma-separated descriptive phrases (not sentences)
3. Include quality boosters: masterpiece, best quality, highly detailed, sharp focus
4. Include style/mood: cinematic lighting, golden hour, studio lighting, etc.
5. If the subject is a person, describe pose, expression, clothing, and hair
6. Output ONLY the prompt text, nothing else (no explanations, no "Here is...", no quotes)
7. Do NOT include negative prompt — only the positive prompt

User's request:
"""
        max_tokens = 200
    elif prompt_type == "code":
        system_prompt = """/no_think
You are a senior software engineer and prompt engineer.
Rewrite the user's coding request into a clear, specific technical prompt that will get the best code output.

Rules:
1. Clarify the programming language, framework, and version if inferable
2. Specify input/output types, edge cases, and error handling expectations
3. Mention coding standards (type hints, docstrings, clean code) if appropriate
4. Add "include example usage" or "include tests" if the request is for a function/class
5. Keep it structured and concise — use numbered requirements if helpful
6. Output ONLY the improved prompt text, nothing else (no explanations, no quotes)

User's request:
"""
        max_tokens = 512
    else:
        system_prompt = """/no_think
You are a prompt engineering expert. The user has a request they want to send to an AI assistant.
Your job is to rewrite it into a clear, detailed, well-structured prompt that will get the best response.

Rules:
1. Preserve the user's intent exactly
2. Add clarity, structure, and specificity
3. If the request is vague, add reasonable context
4. Break complex asks into numbered steps if helpful
5. Keep it concise — don't pad with fluff
6. Output ONLY the improved prompt text, nothing else (no explanations, no "Here is...", no quotes)

User's original request:
"""
        max_tokens = 1024

    # [IMPROVE-14] Main enhancement now routes through the provider
    # layer; wrap in track_event so /observability/summary captures
    # enhance-prompt latency + error rate alongside other chat events.
    with track_event("chat", "enhance_prompt", context={
        "prompt_length": len(user_prompt),
        "model_hint": body.get("ollama_model") or "auto",
        "detected_type": prompt_type,
    }) as ev:
        content = await _ollama_generate_via_router(
            router,
            ollama_model,
            system_prompt + user_prompt,
            temperature=0.7,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )

        if not content or len(content) < 5:
            ev.perf = {"resolved_model": ollama_model, "output_length": 0,
                       "fallback": "empty_response"}
            return {"prompt": user_prompt, "original_prompt": user_prompt,
                    "error": "LLM response was empty", "model": ollama_model,
                    "prompt_type": prompt_type}

        # Clean up common LLM wrapping
        for prefix in ("Here is", "Here's", "Improved prompt:", "Enhanced prompt:",
                        "Image prompt:", "Prompt:", "```"):
            if content.lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip().lstrip(":\n")
        content = content.strip().strip("`").strip('"').strip("'").strip()

        ev.perf = {"resolved_model": ollama_model, "output_length": len(content)}
        # [IMPROVE-4] enhance_prompt resolves "auto" model_hint to a
        # concrete model name inside the with-block; pin it onto the
        # gen_ai span so downstream observability tools can group
        # enhance-prompt latency by actual underlying model.
        ev.set_otel_attributes({
            "gen_ai.request.model": ollama_model,
            "gen_ai.system": "ollama",
        })
        return {
            "prompt": content,
            "original_prompt": user_prompt,
            "model": ollama_model,
            "prompt_type": prompt_type,
        }


# ── Chat Image Generation ───────────────────────────────────────


@router.post("/chat/generate-image")
async def chat_generate_image(
    body: dict[str, Any],
    image_service: ImageGenerationService = Depends(get_image_service),
    router: ProviderRouter = Depends(get_router),
):
    """Generate an image within a conversation and store it as a message.

    Optionally uses conversation context + an LLM to build a better prompt.
    Returns the image URL and message info.
    """
    prompt = (body.get("prompt") or "").strip()
    conversation_id = (body.get("conversation_id") or "").strip()
    use_context = body.get("use_context", True)
    img_steps = int(body.get("steps") or 20)
    img_guidance = float(body.get("guidance_scale") or 7.5)
    img_width = int(body.get("width") or 768)
    img_height = int(body.get("height") or 768)
    img_negative = (body.get("negative_prompt") or "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text").strip()

    if not prompt:
        raise HTTPException(400, "prompt is required")

    # Create conversation if needed
    if not conversation_id:
        conv = create_conversation(title=prompt[:50])
        conversation_id = conv["id"]

    # Save the user's image request as a message
    add_message(conversation_id, "user", prompt)

    # Optionally enhance the prompt using conversation context + LLM
    enhanced_prompt = prompt
    if use_context:
        try:
            # Load last few messages for context
            db_msgs = list_messages(conversation_id)
            context_lines = []
            for msg in db_msgs[-6:-1]:  # last 5 messages excluding current
                role = msg.get("role", "")
                content = (msg.get("content") or "")[:200]
                if content:
                    context_lines.append(f"{role}: {content}")
            context_str = "\n".join(context_lines)

            # [IMPROVE-14] Router-mediated prompt enhancement.
            if context_str:
                ollama_model = _pick_small_ollama_model(router)
                if ollama_model:
                    enhance_prompt_text = f"""/no_think
Based on this conversation context and the user's image request, write a concise Stable Diffusion image prompt (max 50 words). Include quality tags. Output ONLY the prompt text, nothing else.

Conversation:
{context_str}

Image request: {prompt}"""
                    llm_prompt = await _ollama_generate_via_router(
                        router,
                        ollama_model,
                        enhance_prompt_text,
                        temperature=0.7,
                        max_tokens=200,
                        timeout_sec=30,
                    )
                    if llm_prompt and len(llm_prompt) > 10:
                        enhanced_prompt = llm_prompt
        except Exception as exc:
            logger.warning("Chat image context enhancement failed, using raw prompt: %s", exc)

    # Pick the first available image model
    available_models = image_service.list_models()
    if not available_models:
        raise HTTPException(503, "No image generation models available. Install one via the Images page.")
    # Prefer configured/loaded models
    model_id = None
    for m in available_models:
        if isinstance(m, dict) and m.get("status") in ("ready", "loaded", "configured"):
            model_id = m.get("model_id") or m.get("id")
            break
    if not model_id:
        # Fall back to first available
        m0 = available_models[0]
        model_id = m0.get("model_id") or m0.get("id") or str(m0) if isinstance(m0, dict) else str(m0)

    # Use conversation_id as session_id for image storage
    session_id = f"chat-{conversation_id}"

    try:
        result = image_service.generate(
            model_id=model_id,
            prompt=enhanced_prompt,
            negative_prompt=img_negative,
            steps=img_steps,
            guidance_scale=img_guidance,
            width=img_width,
            height=img_height,
            timeout_sec=300,
        )

        # Handle result
        if isinstance(result, list):
            result = result[0]

        # Save image to disk
        image_id = str(uuid.uuid4())
        from local_ai_platform.repositories.images_repo import image_output_path
        out_path = image_output_path(session_id, image_id)
        if hasattr(result, "image_bytes") and result.image_bytes:
            out_path.write_bytes(result.image_bytes)
        elif hasattr(result, "image") and result.image:
            result.image.save(str(out_path))

        image_url = f"/images/files/{session_id}/{image_id}.png"

        # Save assistant message with image attachment
        attachments = [{
            "type": "generated_image",
            "image_id": image_id,
            "image_url": image_url,
            "filename": f"{image_id}.png",
            "prompt_used": enhanced_prompt,
            "model_id": model_id,
        }]
        add_message(
            conversation_id, "assistant",
            f"Generated image for: {prompt}",
            model=model_id,
            attachments=attachments,
        )

        return {
            "status": "ok",
            "conversation_id": conversation_id,
            "image_id": image_id,
            "image_url": image_url,
            "prompt_used": enhanced_prompt,
            "original_prompt": prompt,
            "was_enhanced": enhanced_prompt != prompt,
            "model_id": model_id,
        }
    except Exception as exc:
        # Save error as assistant message
        add_message(conversation_id, "assistant", f"Image generation failed: {exc}", model=model_id)
        raise HTTPException(500, f"Image generation failed: {exc}")


# ── Direct Chat (no agent) ───────────────────────────────────────


@router.post("/chat/direct")
async def direct_chat(
    req: DirectChatRequest,
    router: ProviderRouter = Depends(get_router),
):
    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]
    settings = GenerationSettings.from_dict(req.settings)

    if req.stream:
        async def stream_gen():
            async for chunk in router.astream(req.model, messages, settings):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    response = await router.achat(req.model, messages, settings)
    return {
        "content": response.content,
        "model": response.model,
        "provider": response.provider,
        "usage": response.usage,
    }


# ── Agent Chat ────────────────────────────────────────────────────


@router.post("/chat")
async def agent_chat(
    req: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    trace_store: TraceStore = Depends(get_trace_store),
):
    """Non-streaming agent chat. Flutter sends {agent, message, conversation_id}."""
    agent_name = req.resolved_agent
    if agent_name not in orchestrator.definitions:
        raise HTTPException(404, f"Agent '{agent_name}' not found")

    # Apply model override if provided
    original_model = orchestrator.definitions[agent_name].model_name
    original_provider = orchestrator.definitions[agent_name].provider
    if req.model:
        orchestrator.set_agent_model(agent_name, req.model, req.provider)
        if req.provider:
            orchestrator.definitions[agent_name].provider = req.provider

    # Create or get conversation
    conv_id = req.conversation_id
    if not conv_id:
        conv = create_conversation(title=req.message[:50])
        conv_id = conv["id"]

    # Save user message
    add_message(conv_id, "user", req.message)

    # Set up tracing
    run_id = str(uuid.uuid4())
    trace_cfg = load_trace_config()
    recorder = TraceRecorder(
        trace_cfg, run_id, conv_id,
        agent_name,
        orchestrator.definitions[agent_name].provider,
        orchestrator.definitions[agent_name].model_name,
    )
    callbacks = [LocalTraceCallbackHandler(recorder)]

    # Non-streaming
    try:
        with track_event("chat", "send", context={
            "agent": agent_name,
            "model": orchestrator.definitions[agent_name].model_name,
            "provider": orchestrator.definitions[agent_name].provider,
            "conversation_id": conv_id,
            "run_id": run_id,
            "has_images": bool(req.image_paths),
            "image_count": len(req.image_paths or []),
            "streaming": False,
        }) as ev:
            # [IMPROVE-19] shared helper — same trim/convert semantics
            # previously duplicated here and in /chat/stream below.
            chat_history = orchestrator.load_chat_history(conv_id)

            response = orchestrator.chat_with_agent(
                agent_name,
                req.message,
                image_paths=req.image_paths,
                history_override=chat_history,
                callbacks=callbacks,
                run_id=run_id,
                settings_override=req.settings,
            )

            add_message(
                conv_id, "assistant", response,
                agent=agent_name,
                model=orchestrator.definitions[agent_name].model_name,
                run_id=run_id,
            )
            trace_data = recorder.finalize(success=True)
            if trace_store:
                trace_store.save(trace_data)

            ev.perf["response_length"] = len(response) if response else 0

            # [IMPROVE-4] Attach the run_id as gen_ai.response.id and
            # mirror agent identity. Token counts on the non-streaming
            # path are best pulled from the recorder's events but the
            # callback shape varies per provider — leaving a richer
            # token-usage extraction to the [IMPROVE-13] / [IMPROVE-16]
            # tokenizer-accurate work later in Wave 3. For now, the
            # response_length proxy mirrors what /observability already
            # surfaces.
            ev.set_otel_attributes({
                "gen_ai.response.id": run_id,
                "gen_ai.agent.name": agent_name,
            })

            return {
                "assistant_reply": response,
                "response": response,
                "conversation_id": conv_id,
                "agent": agent_name,
                "run_id": run_id,
            }
    except Exception as exc:
        trace_data = recorder.finalize(success=False, error=str(exc))
        if trace_store:
            trace_store.save(trace_data)
        raise HTTPException(500, str(exc))
    finally:
        # Restore original model if overridden
        if req.model:
            orchestrator.definitions[agent_name].model_name = original_model
            orchestrator.definitions[agent_name].provider = original_provider


@router.post("/chat/stream")
async def agent_chat_stream(
    req: ChatRequest,
    request: Request,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    trace_store: TraceStore = Depends(get_trace_store),
):
    """SSE streaming agent chat. Flutter expects events: start, token, end, error.

    [IMPROVE-17] Client disconnect mid-stream halts generation —
    ``stream_gen`` polls ``_is_client_gone(request)`` after each
    token event and raises ``asyncio.CancelledError`` when the
    Starlette receive channel reports a disconnect. The existing
    ``except BaseException`` block then finalizes the trace as
    cancelled and (per the open-question default in
    docs/features/10-improvements.md:455) does NOT persist a
    partial assistant message. Pre-IMPROVE-17 the route ran to
    completion regardless of the client closing the tab.
    """
    agent_name = req.resolved_agent
    if agent_name not in orchestrator.definitions:
        raise HTTPException(404, f"Agent '{agent_name}' not found")

    # Apply model override if provided
    original_model = orchestrator.definitions[agent_name].model_name
    original_provider = orchestrator.definitions[agent_name].provider
    if req.model:
        orchestrator.set_agent_model(agent_name, req.model, req.provider)
        if req.provider:
            orchestrator.definitions[agent_name].provider = req.provider

    conv_id = req.conversation_id
    if not conv_id:
        conv = create_conversation(title=req.message[:50])
        conv_id = conv["id"]

    add_message(conv_id, "user", req.message)

    run_id = str(uuid.uuid4())
    trace_cfg = load_trace_config()
    recorder = TraceRecorder(
        trace_cfg, run_id, conv_id,
        agent_name,
        orchestrator.definitions[agent_name].provider,
        orchestrator.definitions[agent_name].model_name,
    )

    # [IMPROVE-19] shared helper — same trim/convert semantics as /chat.
    chat_history = orchestrator.load_chat_history(conv_id)

    # [IMPROVE-18] Resolve a stable thread_id per conversation so
    # LangGraph SqliteSaver checkpoints actually get reused across
    # turns. Priority: client-supplied > persisted on conversation row
    # > mint-and-persist (lazy migration for pre-IMPROVE-18 rows).
    # Resolved outside stream_gen so it's available to trace context
    # above and persisted BEFORE the response streams — otherwise a
    # client reconnecting mid-stream wouldn't see the same thread_id.
    resolved_thread_id = req.thread_id
    if not resolved_thread_id:
        try:
            _conv_row = get_conversation(conv_id)
            resolved_thread_id = (_conv_row or {}).get("thread_id") or None
        except Exception:
            resolved_thread_id = None
    if not resolved_thread_id:
        resolved_thread_id = uuid.uuid4().hex
        try:
            set_conversation_thread_id(conv_id, resolved_thread_id)
        except Exception as _pt_err:
            # Don't fail the chat turn if persistence fails — the
            # feature degrades to pre-IMPROVE-18 behavior (per-request
            # UUID) rather than breaking the whole handler.
            logger.debug("thread_id persist failed for %s: %s", conv_id, _pt_err)

    async def stream_gen():
        thread_id = resolved_thread_id
        # Send start event
        yield f"event: start\ndata: {json.dumps({'conversation_id': conv_id, 'run_id': run_id, 'thread_id': thread_id})}\n\n"

        full_response = ""
        stream_start_time = time.monotonic()
        first_token_time: float | None = None
        token_count = 0
        obs_ctx = {
            "agent": agent_name,
            "model": orchestrator.definitions[agent_name].model_name,
            "provider": orchestrator.definitions[agent_name].provider,
            "conversation_id": conv_id,
            "run_id": run_id,
            "thread_id": thread_id,
            "has_images": bool(req.image_paths),
            "image_count": len(req.image_paths or []),
            "streaming": True,
        }
        # track_event handles start.emit + end.emit + duration automatically.
        # The except clauses below use ev.mark_error/mark_cancelled because we
        # yield an SSE error event to the client instead of re-raising.
        try:
            with track_event("chat", "send", context=obs_ctx) as ev:
                try:
                    async for event in orchestrator.astream_chat_with_agent(
                        agent_name, req.message,
                        history_override=chat_history,
                        settings_override=req.settings,
                        thread_id=thread_id,
                    ):
                        etype = event.get("type", "")

                        if etype == "token":
                            text = event.get("text", "")
                            if text:
                                if first_token_time is None:
                                    first_token_time = time.monotonic()
                                full_response += text
                                yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"
                                # [IMPROVE-17] Probe between tokens.
                                # CancelledError here is the deliberate
                                # signal to halt; LangGraph's
                                # ``astream_events`` honors cooperative
                                # cancellation, so the orchestrator
                                # stops at the next await point. The
                                # existing ``except BaseException``
                                # block below finalizes the trace as
                                # cancelled and re-raises (PEP 342).
                                if await _is_client_gone(request):
                                    raise asyncio.CancelledError(
                                        "client_disconnected"
                                    )

                        elif etype == "tool_call":
                            yield f"event: tool_call\ndata: {json.dumps({'name': event.get('name', ''), 'args': event.get('args', ''), 'call_id': event.get('call_id', '')})}\n\n"

                        elif etype == "tool_result":
                            yield f"event: tool_result\ndata: {json.dumps({'name': event.get('name', ''), 'content': event.get('content', ''), 'call_id': event.get('call_id', '')})}\n\n"

                        elif etype == "done":
                            if not full_response:
                                full_response = event.get("content", "") or "No response returned."

                    # [IMPROVE-16] Tokenizer-accurate count of the
                    # streamed response. Pre-IMPROVE-16 the loop
                    # accumulated ``token_count += max(1,
                    # len(text.split()))`` per chunk — undercounts
                    # English by ~25% and non-Latin scripts by far
                    # more. The helper prefers the agent's provider
                    # tokenizer (HF / LlamaCpp already cached one
                    # for this stream), tiktoken cl100k_base next,
                    # split as a last resort. Single encode of the
                    # full response is cheap enough for typical
                    # response sizes per the proposal.
                    token_count = count_tokens(
                        orchestrator.definitions[agent_name].provider,
                        orchestrator.definitions[agent_name].model_name,
                        full_response,
                        router=getattr(orchestrator, "router", None),
                    )
                    # Performance metrics
                    total_time = time.monotonic() - stream_start_time
                    ttft = (first_token_time - stream_start_time) if first_token_time else 0
                    tokens_per_sec = token_count / total_time if total_time > 0 else 0
                    perf_data = {
                        "tokens": token_count,
                        "total_sec": round(total_time, 2),
                        "tokens_per_sec": round(tokens_per_sec, 1),
                        "ttft_sec": round(ttft, 3),
                    }

                    add_message(
                        conv_id, "assistant", full_response,
                        agent=agent_name,
                        model=orchestrator.definitions[agent_name].model_name,
                        run_id=run_id,
                        perf=perf_data,
                    )
                    trace_data = recorder.finalize(success=True)
                    if trace_store:
                        trace_store.save(trace_data)

                    logger.info("Stream complete: %d tokens, %.1f tok/s, TTFT=%.3fs, total=%.2fs",
                                 token_count, tokens_per_sec, ttft, total_time)

                    ev.perf = {**perf_data, "response_length": len(full_response)}
                    # [IMPROVE-4] Streaming path is the cleanest source
                    # of gen_ai.usage.output_tokens — token_count is
                    # the number of completion tokens we actually
                    # streamed. [IMPROVE-16] now feeds it through the
                    # tokenizer-accurate ``count_tokens`` helper, so
                    # the OTel attribute and the perf_json column on
                    # ``messages`` both report tokenizer counts —
                    # cross-provider tok/s is now an honest absolute
                    # metric, not just a relative one.
                    ev.set_otel_attributes({
                        "gen_ai.usage.output_tokens": token_count,
                        "gen_ai.response.id": run_id,
                        "gen_ai.agent.name": agent_name,
                    })
                    yield f"event: end\ndata: {json.dumps({'conversation_id': conv_id, 'run_id': run_id, 'thread_id': thread_id, 'perf': perf_data})}\n\n"

                except Exception as exc:
                    trace_data = recorder.finalize(success=False, error=str(exc))
                    if trace_store:
                        trace_store.save(trace_data)
                    # [IMPROVE-16] Recount on the partial response so
                    # the failure-path perf isn't stuck at the
                    # initialization-zero. Pre-IMPROVE-16 the
                    # per-chunk accumulator captured partial counts
                    # for free; without it we'd lose mid-stream
                    # observability if we didn't recount here.
                    ev.perf = {"tokens": count_tokens(
                        orchestrator.definitions[agent_name].provider,
                        orchestrator.definitions[agent_name].model_name,
                        full_response,
                        router=getattr(orchestrator, "router", None),
                    )}
                    ev.mark_error(exc)
                    yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
                except BaseException as exc:
                    # Client disconnect (GeneratorExit / [IMPROVE-17]
                    # CancelledError("client_disconnected")) or process
                    # signal. Swallowing GeneratorExit is invalid per
                    # PEP 342 — re-raise.
                    #
                    # [IMPROVE-17] When the cancel originated from our
                    # deliberate ``raise CancelledError("client_
                    # disconnected")`` the args[0] string is the
                    # informative attribution; otherwise fall back to
                    # the exception's class name (matches the
                    # pre-IMPROVE-17 wording for GeneratorExit /
                    # KeyboardInterrupt etc).
                    cancel_reason = type(exc).__name__
                    try:
                        if isinstance(exc, asyncio.CancelledError) and exc.args:
                            cancel_reason = str(exc.args[0]) or cancel_reason
                    except Exception:
                        pass
                    # [IMPROVE-17] Pre-IMPROVE-17 this branch only
                    # finalized the recorder and never wrote the
                    # trace to disk — so cancelled streams were
                    # invisible on /runs. The bug was masked because
                    # the only real-world trigger before this commit
                    # was Starlette's GeneratorExit, rare enough that
                    # the missing save went unnoticed. With deliberate
                    # client-disconnect cancellation now firing
                    # routinely, persist the trace too — operators
                    # need to see "this run was cancelled at second 3"
                    # not "this run silently vanished".
                    try:
                        cancel_trace = recorder.finalize(
                            success=False, error=f"cancelled: {cancel_reason}"
                        )
                        if trace_store:
                            trace_store.save(cancel_trace)
                    except Exception:
                        pass
                    # [IMPROVE-16] Same recount on the cancelled
                    # path. Wrap in try/except — counting must not
                    # mask the original exception that triggered the
                    # cancel branch.
                    try:
                        partial_tokens = count_tokens(
                            orchestrator.definitions[agent_name].provider,
                            orchestrator.definitions[agent_name].model_name,
                            full_response,
                            router=getattr(orchestrator, "router", None),
                        )
                    except Exception:
                        partial_tokens = 0
                    ev.perf = {"tokens": partial_tokens}
                    ev.mark_cancelled(cancel_reason)
                    raise
        finally:
            # Restore original model if overridden
            if req.model:
                orchestrator.definitions[agent_name].model_name = original_model
                orchestrator.definitions[agent_name].provider = original_provider

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


# ── Supervisor Chat ───────────────────────────────────────────────


@router.post("/chat/supervisor/{supervisor_name}")
async def supervisor_chat(
    supervisor_name: str,
    req: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    result = orchestrator.chat_with_supervisor(supervisor_name, req.message)
    return result


# ── Chat Resume (after human-in-the-loop interrupt) ──────────────


@router.post("/chat/resume")
async def resume_chat(
    body: dict[str, Any],
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Resume an interrupted agent after human approval/rejection."""
    agent_name = body.get("agent", body.get("agent_name", ""))
    thread_id = body.get("thread_id", "")
    action = body.get("action", "approve")  # "approve" or "reject"
    conv_id = body.get("conversation_id", "")

    if not agent_name or not thread_id:
        raise HTTPException(400, "agent and thread_id required")

    async def stream_gen():
        yield f"event: start\ndata: {json.dumps({'thread_id': thread_id, 'action': action})}\n\n"

        full_response = ""
        try:
            async for event in orchestrator.astream_resume_after_interrupt(agent_name, thread_id, action):
                etype = event.get("type", "")
                if etype == "token":
                    text = event.get("text", "")
                    full_response += text
                    yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"
                elif etype == "tool_call":
                    yield f"event: tool_call\ndata: {json.dumps(event)}\n\n"
                elif etype == "tool_result":
                    yield f"event: tool_result\ndata: {json.dumps(event)}\n\n"
                elif etype == "done":
                    if not full_response:
                        full_response = event.get("content", "")

            if conv_id:
                add_message(conv_id, "assistant", full_response,
                            agent=agent_name,
                            model=orchestrator.definitions.get(agent_name, {}).model_name if agent_name in orchestrator.definitions else "")

            yield f"event: end\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(stream_gen(), media_type="text/event-stream")
