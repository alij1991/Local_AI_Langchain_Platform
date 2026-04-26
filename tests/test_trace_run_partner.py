"""[IMPROVE-68] Commit 4/5: partner under trace_run.

Pin the contract for ``POST /partner/chat`` and
``POST /partner/chat/stream``:
- Each request produces a TraceStore JSON with subsystem="partner",
  agent_name=<persona name>, model_provider="ollama",
  model_id=<request model>, conversation_id=None.
- Engine emits inside partner.chat / partner.astream_chat
  (``emit("partner", "chat.start" / "chat" / "emotion_detect" /
  "fact_extract", ...)``) flow into the trace's events list via the
  ``_active_recorder`` ContextVar set by Commit 1/5's trace_run.
- Stream path: trace_run lives INSIDE the async generator, not
  outside. The recorder stays active for the duration of the SSE
  iterator's actual run (Starlette consumes the generator AFTER the
  route returns); ContextVars survive yield/resume in async
  generators because asyncio preserves the active Context across
  the suspension.
- Failure on stream: re-raise after yielding the error event so
  trace_run sees the exception and saves success=False. Pre-IMPROVE-68
  the handler ate the exception (mark_error + yield, no re-raise),
  leaving failed streams at success=True. Re-raise puts the failed
  partner run in red on /runs.

Sources (2025-2026):
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- https://docs.python.org/3/library/contextvars.html — PEP 567
  (asyncio preserves Context across generator yield/resume)
- docs/features/09-observability.md §IMPROVE-68 (line 572)
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import api_server
from local_ai_platform.observability import emit


_client = TestClient(api_server.app)


@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    with _client:
        yield


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    """Point ``load_trace_config`` at a temp dir for isolation."""
    store_dir = tmp_path / "traces"
    monkeypatch.setenv("TRACE_STORE_DIR", str(store_dir))
    monkeypatch.setenv("TRACE_ENABLED", "true")
    from local_ai_platform.config import reset_settings_cache
    reset_settings_cache()
    yield store_dir
    reset_settings_cache()


class _FakeProfile:
    name = "test_persona"


class _FakePartner:
    """Drop-in for PartnerEngine — only the attrs the chat routes touch."""
    def __init__(self, *, fire_emit=True, raise_exc=None,
                 stream_events=None, stream_raise=None):
        self.profile = _FakeProfile()
        self.fire_emit = fire_emit
        self.raise_exc = raise_exc
        # Default stream events mimic a successful run: one emotion,
        # a few tokens, _metrics, done.
        self.stream_events = stream_events if stream_events is not None else [
            {"type": "emotion", "emotion": "happy"},
            {"type": "token", "text": "Hi"},
            {"type": "token", "text": " there"},
            {"type": "_metrics", "reply_length": 8, "token_count": 2,
             "emotion_detected": True},
            {"type": "done", "full_reply": "Hi there"},
        ]
        self.stream_raise = stream_raise

    def chat(self, message, model=None):
        if self.fire_emit:
            # Mirror engine.py:294/356 — chat.start + chat
            emit("partner", "chat.start", status="start",
                 context={"partner": self.profile.name})
            emit("partner", "chat", status="ok", duration_ms=42,
                 context={"partner": self.profile.name})
        if self.raise_exc:
            raise self.raise_exc
        return f"echo: {message}"

    async def astream_chat(self, message, model=None,
                           enable_thinking_pause=True):
        if self.fire_emit:
            emit("partner", "chat.start", status="start",
                 context={"partner": self.profile.name})
        for event in self.stream_events:
            if self.stream_raise and event.get("type") == "token":
                # Pretend the stream blew up midway
                raise self.stream_raise
            yield event
        if self.fire_emit:
            emit("partner", "chat", status="ok", duration_ms=100,
                 context={"partner": self.profile.name})


def _install_fake_partner(monkeypatch, **kwargs):
    """Install a fake partner engine on app.state. Same trick as the
    editor test: ``raising=False`` because the engine is lazy-init.
    """
    fake = _FakePartner(**kwargs)
    monkeypatch.setattr(
        api_server.app.state, "_partner_engine", fake, raising=False,
    )
    return fake


@pytest.fixture
def client(trace_dir):
    return _client


# ── /partner/chat (sync) ─────────────────────────────────────────────


def test_partner_chat_sync_writes_trace_with_subsystem_partner(
    client, trace_dir, monkeypatch
):
    """A successful /partner/chat produces a trace JSON with
    subsystem="partner" and the persona name as agent_name.
    """
    _install_fake_partner(monkeypatch)

    res = client.post("/partner/chat", json={
        "message": "hello partner",
        "model": "qwen3:8b",
    })
    assert res.status_code == 200
    assert res.json()["reply"] == "echo: hello partner"

    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["subsystem"] == "partner"
    assert payload["agent_name"] == "test_persona"
    assert payload["model_provider"] == "ollama"
    assert payload["model_id"] == "qwen3:8b"
    assert payload["conversation_id"] is None
    assert payload["success"] is True


def test_partner_chat_sync_engine_emits_flow_into_trace(
    client, trace_dir, monkeypatch
):
    """Engine emits inside partner.chat (chat.start, chat) land on
    the recorder via the ContextVar — sync handler runs in the event
    loop, so propagation is implicit (no executor crossing).
    """
    _install_fake_partner(monkeypatch, fire_emit=True)

    res = client.post("/partner/chat", json={"message": "x"})
    assert res.status_code == 200

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    event_names = [e["name"] for e in payload["events"]]
    assert "partner.chat.start" in event_names
    assert "partner.chat" in event_names


def test_partner_chat_sync_failure_saves_trace(
    client, trace_dir, monkeypatch
):
    """If partner.chat raises, the route returns 500 (re-raised by
    TestClient), and the trace lands with success=False.
    """
    _install_fake_partner(
        monkeypatch, fire_emit=True,
        raise_exc=RuntimeError("model not loaded"),
    )

    with pytest.raises(RuntimeError, match="model not loaded"):
        client.post("/partner/chat", json={"message": "x"})

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert "model not loaded" in (payload["error"] or "")


def test_partner_chat_sync_validation_no_trace(
    client, trace_dir, monkeypatch
):
    """Missing message → 400 BEFORE trace_run starts; no trace produced."""
    _install_fake_partner(monkeypatch)

    res = client.post("/partner/chat", json={})
    assert res.status_code == 400

    assert list(trace_dir.glob("*.json")) == []


# ── /partner/chat/stream (SSE) ───────────────────────────────────────


def test_partner_chat_stream_writes_trace_after_iterator_drains(
    client, trace_dir, monkeypatch
):
    """The stream path needs trace_run INSIDE the generator. Pin: a
    successful streaming request (full event sequence consumed by the
    client) produces a trace JSON. The trace is finalized AFTER the
    iterator drains, not when the route function returns.
    """
    _install_fake_partner(monkeypatch)

    with client.stream("POST", "/partner/chat/stream", json={
        "message": "hi",
        "model": "qwen3:4b",
    }) as res:
        assert res.status_code == 200
        chunks = list(res.iter_text())

    body = "".join(chunks)
    assert "event: start" in body
    assert "event: end" in body  # 'done' from engine maps to 'end' SSE event

    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["subsystem"] == "partner"
    assert payload["agent_name"] == "test_persona"
    assert payload["model_id"] == "qwen3:4b"
    assert payload["success"] is True


def test_partner_chat_stream_engine_emits_flow_into_trace(
    client, trace_dir, monkeypatch
):
    """ContextVars survive yield/resume across the async generator,
    so emit() calls fired between yields land on the recorder. Pin
    the chat.start + chat events make it into the trace.
    """
    _install_fake_partner(monkeypatch, fire_emit=True)

    with client.stream("POST", "/partner/chat/stream", json={
        "message": "hi",
    }) as res:
        list(res.iter_text())  # drain

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    event_names = [e["name"] for e in payload["events"]]
    assert "partner.chat.start" in event_names
    assert "partner.chat" in event_names


def test_partner_chat_stream_failure_saves_trace_success_false(
    client, trace_dir, monkeypatch
):
    """The load-bearing test for the re-raise change: when
    partner.astream_chat blows up mid-stream, the handler yields an
    error SSE event AND re-raises so trace_run records success=False.

    Pre-IMPROVE-68 this branch ate the exception (mark_error +
    yield, no re-raise), leaving failed streams at success=True
    in the Runs view. The re-raise is what flips it to red.
    """
    _install_fake_partner(
        monkeypatch, fire_emit=True,
        stream_raise=RuntimeError("LLM dropped connection"),
    )

    # The TestClient's iterator surfaces the re-raised exception
    # only when the body is fully consumed; pytest.raises catches
    # it cleanly at that point.
    with pytest.raises(RuntimeError, match="LLM dropped connection"):
        with client.stream("POST", "/partner/chat/stream", json={
            "message": "hi",
        }) as res:
            list(res.iter_text())

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert "LLM dropped connection" in (payload["error"] or "")


def test_partner_chat_stream_validation_no_trace(
    client, trace_dir, monkeypatch
):
    """Missing message → 400 BEFORE the generator is even constructed;
    no trace produced. The validation HTTPException is raised by the
    route function body, not from inside the generator.
    """
    _install_fake_partner(monkeypatch)

    res = client.post("/partner/chat/stream", json={})
    assert res.status_code == 400

    assert list(trace_dir.glob("*.json")) == []
