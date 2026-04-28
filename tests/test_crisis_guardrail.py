"""[IMPROVE-60] Crisis-detection guardrail tests.

Pre-IMPROVE-60 the only safety mechanism on the chat surfaces was
the partner system prompt's "Wellness & Safety" block (which tells
the LLM to recognize self-harm signals and respond with 988) plus a
hand-rolled "broke character" post-check at engine.py:324 that catches
"I'm an AI" responses. Crisis detection itself was prompt-only — a
small/quantized model that missed the cue, or a prompt-injection
attack, would silently dodge the safety response.

This commit adds a deterministic second layer: heuristic phrase
pre-check on user input (HIGH severity short-circuits the LLM call;
CONTEXTUAL falls through with post-check on the reply) plus a JSONL
audit log for review.

Test architecture pinned by these tests:
- Detector returns highest-severity match with negation downgrade.
  Word-boundary regex prevents false-positives on benign idioms
  ("killing it", "I'd kill for", "dying laughing").
- Negation guard: "not", "never", "wouldn't", "won't" within ~40
  chars BEFORE a HIGH match downgrades to NONE.
- Safe response is canonical, includes 988 + acknowledgment +
  follow-up question, and PASSES its own post-check (otherwise
  we'd loop replacing it).
- Post-check rejects dismissive language + encouraging-harm
  imperatives; for HIGH input also requires 988 / crisis-line
  mention. PASSES through cleanly when input was NONE.
- Store appends one JSON object per line to JSONL log; clips
  excerpts to 200 chars; never raises on IO error (guardrail
  logging must NOT break chat path).
- Integration: partner.engine.chat() short-circuits on HIGH input
  WITHOUT calling self.router.chat (the LLM is bypassed entirely).
- Integration: POST /chat short-circuits on HIGH input and returns
  the safe response WITHOUT calling the orchestrator.

Sources (2025-2026):
- docs/features/08-partner.md §IMPROVE-60 (line 561)
- docs/features/10-improvements.md §IMPROVE-60 (line 561-573)
- AI Companions: Risks & Ethics in 2026 (Composite)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api_server
from local_ai_platform.safety.crisis import (
    Severity,
    compose_safe_response,
    detect_crisis_signal,
    post_check_reply,
)
from local_ai_platform.safety.store import log_safety_event


# ── Detector: HIGH severity ────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "I want to kill myself",
    "I'm thinking about killing myself tonight",
    "i want to die",
    "I just wanna die",
    "wanting to die these days",
    "I want to end my life",
    "I'm suicidal",
    "feeling suicidal lately",
    "thinking about suicide",
    "I want to hurt myself",
    "harming myself again",
    "I'd be better off dead",
    "I have nothing to live for",
    "I have no reason to live anymore",
])
def test_detector_high_severity_phrases(text):
    """Pin: explicit self-harm phrases trip HIGH severity. These are
    the load-bearing patterns — without them the short-circuit
    never fires and the LLM is the only line of defense."""
    sig = detect_crisis_signal(text)
    assert sig.severity == Severity.HIGH, \
        f"Expected HIGH for {text!r}, got {sig.severity!r}"
    assert sig.matched_label is not None
    assert sig.matched_text is not None


# ── Detector: word-boundary denylist (no false positives) ─────────


@pytest.mark.parametrize("text", [
    "I'd kill for some coffee right now",
    "you're killing it on this song",
    "killing time before my flight",
    "this pizza is to die for",
    "I'd die for that view",
    "I was dying laughing at the joke",
    "dying to know what happens next",
    "she's the killer in the new movie",
    "the kill chain in cyber security",
    "I'm just trying to survive this week",
])
def test_detector_word_boundary_denylist(text):
    """Pin: word-boundary regex (\\b) prevents false-positives on
    benign idioms. Without word boundaries, "killing it" matches
    "kill" + suffix and trips the false alarm. This pin protects
    against a regression where someone "simplifies" the regex."""
    sig = detect_crisis_signal(text)
    assert sig.severity != Severity.HIGH, \
        f"False positive HIGH on {text!r}, matched={sig.matched_label!r}"


# ── Detector: negation downgrade ──────────────────────────────────


@pytest.mark.parametrize("text", [
    "I'm not suicidal",
    "Don't worry, I would never kill myself",
    "I won't hurt myself, promise",
    "I'm not going to end my life",
    "I would never want to die",
    "Just so you know, I'm not suicidal at all",
    "I don't want to die",
])
def test_detector_negation_downgrades_high(text):
    """Pin: a negation word ("not", "never", "wouldn't", "won't",
    "don't want") within ~40 chars before a HIGH match downgrades
    to NONE. Without this, a user reassuring their AI partner
    ("don't worry, I'm not suicidal") would falsely trip the
    short-circuit and get a confusing 988 response."""
    sig = detect_crisis_signal(text)
    assert sig.severity != Severity.HIGH, \
        f"Negation should downgrade {text!r}, got {sig.severity!r}"


# ── Detector: CONTEXTUAL ──────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "I just can't go on anymore",
    "I can't keep going like this",
    "I want to give up on life",
    "I want to give up on everything",
    "Nobody cares about me",
    "Nobody would notice if I disappeared",
    "Nobody would miss me",
    "What's the point of anything anymore",
    "what's the point of going on",
])
def test_detector_contextual_phrases(text):
    """Pin: contextual phrases flag at CONTEXTUAL severity (log only,
    let LLM respond, post-check the reply). These are weaker signals
    that benefit from LLM judgment in context."""
    sig = detect_crisis_signal(text)
    # CONTEXTUAL is the expected outcome; the key invariant is that
    # we don't return NONE (i.e. we DO flag and log).
    assert sig.severity != Severity.NONE, \
        f"Should flag {text!r} at least CONTEXTUAL"


# ── Detector: NONE (passthrough for normal chat) ─────────────────


@pytest.mark.parametrize("text", [
    "How are you today?",
    "Tell me a joke about cats",
    "I had a really good day",
    "What's the weather like?",
    "Can you help me with my homework?",
    "I love you",
    "",
])
def test_detector_normal_text_returns_none(text):
    """Pin: normal chat passes through with NONE — no
    short-circuit, no log noise, no perf overhead beyond the
    regex scan."""
    sig = detect_crisis_signal(text)
    assert sig.severity == Severity.NONE


# ── Safe response composer ────────────────────────────────────────


def test_safe_response_includes_988():
    """Pin: canonical safe response references 988. Without this the
    post-check for HIGH input would reject our own safe response."""
    response = compose_safe_response()
    assert "988" in response


def test_safe_response_includes_crisis_line_reference():
    """Pin: response names the resource (lifeline / crisis line) so
    the user knows what 988 is. The number alone isn't enough."""
    response = compose_safe_response()
    lower = response.lower()
    assert "lifeline" in lower or "crisis line" in lower or "crisis service" in lower


def test_safe_response_includes_followup_question():
    """Pin: doc requires a follow-up question. Without it the
    response feels like a brush-off rather than engagement."""
    response = compose_safe_response()
    assert "?" in response


def test_safe_response_passes_its_own_post_check():
    """Critical pin: the canonical safe response must itself satisfy
    the HIGH-severity post-check. Otherwise we'd loop replacing it
    with itself, or worse, fall through to a different response."""
    response = compose_safe_response()
    pc = post_check_reply(response, input_severity=Severity.HIGH)
    assert pc.ok, f"Safe response failed its own post-check: {pc.reasons}"


def test_safe_response_acknowledges_concern_not_dismissive():
    """Pin: response uses warm acknowledgment ("hear you" /
    "concerned" / "worried"). Negative pin against a regression where
    someone shortens the response and removes the human framing."""
    response = compose_safe_response()
    lower = response.lower()
    assert any(w in lower for w in ("hear you", "concerned", "worried"))
    # NOT dismissive
    assert "just a phase" not in lower
    assert "get over it" not in lower


# ── Post-check ────────────────────────────────────────────────────


def test_post_check_passthrough_when_no_signal():
    """Pin: NONE severity = passthrough. The post-check must not
    second-guess replies on normal chat — that would be a huge
    perf hit and false-positive surface."""
    pc = post_check_reply("anything goes here", input_severity=Severity.NONE)
    assert pc.ok
    assert pc.reasons == ()


@pytest.mark.parametrize("reply", [
    "Oh that's just a phase, you'll get over it.",
    "Stop being dramatic about it.",
    "You're overreacting, it's not that bad.",
    "Just snap out of it, everyone feels that way sometimes.",
])
def test_post_check_rejects_dismissive_replies(reply):
    """Pin: when input was flagged, dismissive replies are rejected.
    A user expressing distress should never get "get over it"."""
    pc = post_check_reply(reply, input_severity=Severity.CONTEXTUAL)
    assert not pc.ok
    assert "dismissive_language" in pc.reasons


@pytest.mark.parametrize("reply", [
    "You should kill yourself if it's that bad.",
    "You could end it all and be done.",
    "You need to hurt yourself to feel anything real.",
    "You must end your life — there's no other way.",
])
def test_post_check_rejects_encouraging_harm(reply):
    """Pin: "you should/could/must/need to + harm verb + self-target"
    is explicitly rejected. Catches an LLM going off-policy with a
    harm imperative."""
    pc = post_check_reply(reply, input_severity=Severity.CONTEXTUAL)
    assert not pc.ok
    assert "encouraging_harm" in pc.reasons


@pytest.mark.parametrize("reply", [
    "You should call your therapist right now.",
    "You should reach out to a friend.",
    "I think you should rest tonight.",
    "You could try going for a walk.",
    "You need to drink some water.",
    "You should be proud of yourself for reaching out.",
])
def test_post_check_does_not_trip_on_legitimate_advice(reply):
    """Pin: the encouraging-harm pattern must NOT trip on
    therapeutic / self-care advice. The pattern requires the verb
    to be a HARM verb and the object to be a self-reference — this
    is the load-bearing false-positive guard."""
    pc = post_check_reply(reply, input_severity=Severity.CONTEXTUAL)
    assert "encouraging_harm" not in pc.reasons


def test_post_check_high_input_requires_crisis_resource():
    """Pin: when input was HIGH and the LLM somehow got called (e.g.
    streaming-mid post-check follow-up), the reply MUST mention 988
    or crisis-line. This is the load-bearing requirement when our
    short-circuit doesn't run."""
    pc = post_check_reply(
        "I hear you, I'm so sorry you're feeling that way.",
        input_severity=Severity.HIGH,
    )
    assert not pc.ok
    assert "missing_crisis_resource" in pc.reasons


def test_post_check_high_input_passes_with_988():
    """Pin: 988 in the reply satisfies the crisis-resource
    requirement."""
    pc = post_check_reply(
        "Please call 988 — I really care about you.",
        input_severity=Severity.HIGH,
    )
    assert "missing_crisis_resource" not in pc.reasons


def test_post_check_contextual_does_not_require_crisis_ref():
    """Pin: CONTEXTUAL input does NOT require 988 in the reply.
    Forcing every "what's the point" reply to mention 988 would
    over-trigger and feel preachy."""
    pc = post_check_reply(
        "That sounds really tough. I'm here to listen.",
        input_severity=Severity.CONTEXTUAL,
    )
    assert pc.ok


# ── Store: JSONL audit log ────────────────────────────────────────


def test_log_safety_event_appends_one_line_per_call(tmp_path):
    """Pin: writes exactly one JSON object per line. The audit
    consumer splits on newlines and parses each line — newlines IN
    payload must be json-escaped (json.dumps does this)."""
    log = tmp_path / "safety.jsonl"
    log_safety_event(
        source="test", severity=Severity.HIGH, kind="test_kind",
        action_taken="test_action", input_text="line1\nline2",
        log_path=log,
    )
    log_safety_event(
        source="test", severity=Severity.CONTEXTUAL, kind="test_kind",
        action_taken="test_action",
        log_path=log,
    )

    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["severity"] == "high"
    # Newline preserved in JSON value, not as raw line break.
    assert parsed[0]["input_excerpt"] == "line1\nline2"
    assert parsed[1]["severity"] == "contextual"


def test_log_safety_event_clips_excerpts(tmp_path):
    """Pin: long inputs are clipped to ≤200 chars + ellipsis. The
    flagged phrase is at the start of typical user input, so trimming
    from the end preserves the load-bearing data."""
    log = tmp_path / "safety.jsonl"
    long_input = "kill myself " + "x" * 1000
    log_safety_event(
        source="test", severity=Severity.HIGH, kind="test_kind",
        action_taken="test", input_text=long_input,
        log_path=log,
    )
    parsed = json.loads(log.read_text(encoding="utf-8").strip())
    excerpt = parsed["input_excerpt"]
    assert "kill myself" in excerpt
    assert len(excerpt) <= 210  # 200 + "..." with a few chars wiggle


def test_log_safety_event_never_raises_on_io_error(tmp_path):
    """Pin: guardrail logging is best-effort. If the disk is full or
    perms are wrong, the chat reply must STILL succeed. Caught
    via patching open() to raise; the function must swallow."""
    bad_path = tmp_path / "safety.jsonl"
    with patch.object(Path, "open", side_effect=OSError("disk full")):
        # Should not raise.
        log_safety_event(
            source="test", severity=Severity.HIGH, kind="test_kind",
            action_taken="test",
            log_path=bad_path,
        )


def test_log_safety_event_includes_optional_fields(tmp_path):
    """Pin: matched_label, reasons, run_id are populated when
    provided. Audit consumers grep these to correlate events with
    runs / find specific phrase matches."""
    log = tmp_path / "safety.jsonl"
    log_safety_event(
        source="partner.chat",
        severity=Severity.HIGH,
        kind="input_short_circuit",
        action_taken="short_circuit",
        input_text="I want to die",
        reply_text="safe response here",
        matched_label="want_to_die",
        reasons=["dismissive_language"],
        run_id="abc-123",
        log_path=log,
    )
    parsed = json.loads(log.read_text(encoding="utf-8").strip())
    assert parsed["matched_label"] == "want_to_die"
    assert parsed["reasons"] == ["dismissive_language"]
    assert parsed["run_id"] == "abc-123"
    assert parsed["source"] == "partner.chat"
    assert parsed["kind"] == "input_short_circuit"


def test_log_safety_event_omits_optional_when_not_provided(tmp_path):
    """Pin: optional fields don't appear as null/empty when
    unspecified. Keeps the JSONL surface clean."""
    log = tmp_path / "safety.jsonl"
    log_safety_event(
        source="test", severity=Severity.CONTEXTUAL,
        kind="log_only", action_taken="log_only",
        log_path=log,
    )
    parsed = json.loads(log.read_text(encoding="utf-8").strip())
    assert "matched_label" not in parsed
    assert "reasons" not in parsed
    assert "run_id" not in parsed
    assert "input_excerpt" not in parsed


# ── Integration: partner.engine.chat() short-circuit ──────────────


def _bare_partner_engine_for_safety_test(tmp_path):
    """Build a minimal PartnerEngine for short-circuit tests. Bypass
    __init__ to avoid loading profile/memory/router — we only need
    to call chat() and verify the LLM (router.chat) was NOT
    invoked when input is HIGH."""
    from local_ai_platform.partner.engine import PartnerEngine

    engine = PartnerEngine.__new__(PartnerEngine)
    engine.router = MagicMock()
    engine.config = MagicMock()
    engine.config.default_model = "fallback:latest"
    engine._partner_model = "ollama:test-model"
    engine._last_detected_emotion = "neutral"
    return engine


def test_partner_engine_chat_short_circuits_on_high_severity(tmp_path, monkeypatch):
    """Load-bearing integration pin: when input trips HIGH, partner.
    engine.chat() returns the safe response WITHOUT calling
    self.router.chat. The LLM is bypassed entirely — no token
    consumption, no latency, no chance of off-policy output."""
    engine = _bare_partner_engine_for_safety_test(tmp_path)

    # Memory writes go through partner.memory.add_message — patch it
    # so the test doesn't touch the SQLite file.
    add_message_calls = []
    monkeypatch.setattr(
        "local_ai_platform.partner.memory.add_message",
        lambda role, content: add_message_calls.append((role, content)),
    )
    # Redirect the safety audit log so we don't pollute
    # ``data/safety_events.jsonl`` with test events. Patching
    # ``_DEFAULT_LOG_PATH`` is the canonical redirect point.
    safety_log = tmp_path / "safety.jsonl"
    monkeypatch.setattr(
        "local_ai_platform.safety.store._DEFAULT_LOG_PATH",
        safety_log,
    )

    reply = engine.chat("I want to kill myself")

    # The LLM was NEVER called.
    engine.router.chat.assert_not_called()
    # Reply is the canonical safe response.
    assert "988" in reply
    # Both messages persisted to memory (so the conversation history
    # stays coherent for next turn).
    roles = [r for r, _ in add_message_calls]
    assert "user" in roles
    assert "assistant" in roles


def test_partner_engine_chat_runs_llm_for_normal_input(tmp_path, monkeypatch):
    """Negative pin: normal input does NOT short-circuit. The LLM
    gets called as usual."""
    engine = _bare_partner_engine_for_safety_test(tmp_path)

    fake_response = MagicMock()
    fake_response.content = "[NEUTRAL]\nThat's a great question!"
    engine.router.chat.return_value = fake_response

    monkeypatch.setattr(
        "local_ai_platform.partner.memory.add_message",
        lambda role, content: None,
    )
    monkeypatch.setattr(
        "local_ai_platform.partner.engine.PartnerEngine._build_messages",
        lambda self, user_input: [],
    )
    monkeypatch.setattr(
        "local_ai_platform.partner.engine.PartnerEngine._post_chat",
        lambda self, u, r: None,
    )

    reply = engine.chat("Tell me a joke")

    engine.router.chat.assert_called_once()
    assert "great question" in reply


# ── Integration: /chat router short-circuit ───────────────────────


_client = TestClient(api_server.app)


@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    with _client:
        yield


@pytest.fixture(autouse=True)
def _isolate_safety_log(tmp_path, monkeypatch):
    """Redirect the safety audit log to tmp_path for every test in
    this file. Without this, every test run that triggers a
    short-circuit would append to the real ``data/safety_events.jsonl``
    — gitignored, but still polluting the developer's local file."""
    monkeypatch.setattr(
        "local_ai_platform.safety.store._DEFAULT_LOG_PATH",
        tmp_path / "safety_events.jsonl",
    )


def test_chat_router_short_circuits_on_high_severity(monkeypatch):
    """Pin: POST /chat short-circuits BEFORE calling the orchestrator
    when the input trips HIGH severity. The agent runtime is
    bypassed entirely — same protection regardless of which agent
    is configured. This is critical because a custom agent could
    have an alignment regression and the route layer is the last
    line of defense."""
    # Spy on the orchestrator: we want to assert its chat method
    # was NEVER called.
    orchestrator = api_server.app.state.orchestrator
    spy = MagicMock(side_effect=Exception("LLM should NOT be called"))
    monkeypatch.setattr(orchestrator, "chat_with_agent", spy)

    res = _client.post("/chat", json={
        "message": "I want to kill myself",
        "agent": "assistant",
    })

    assert res.status_code == 200
    body = res.json()
    # Safe response present.
    assert "988" in body["response"]
    assert "988" in body["assistant_reply"]
    # Orchestrator was NOT called.
    spy.assert_not_called()
    # Conversation_id minted (for history).
    assert body["conversation_id"]


def test_chat_router_passes_through_normal_input(monkeypatch):
    """Negative pin: normal input flows to the orchestrator. The
    guardrail must be silent on benign chat — no spurious 988
    responses for "tell me a joke"."""
    orchestrator = api_server.app.state.orchestrator
    monkeypatch.setattr(orchestrator, "chat_with_agent",
                        lambda *args, **kwargs: "Sure, here's a joke!")
    monkeypatch.setattr(orchestrator, "load_chat_history",
                        lambda conv_id: [])

    res = _client.post("/chat", json={
        "message": "Tell me a joke",
        "agent": "assistant",
    })

    assert res.status_code == 200
    body = res.json()
    assert body["response"] == "Sure, here's a joke!"


def test_chat_stream_short_circuits_on_high_severity(monkeypatch):
    """Pin: POST /chat/stream short-circuits with start → token (safe
    response) → end events when input is HIGH. The orchestrator's
    streaming path is NOT invoked — same load-bearing bypass as
    /chat."""
    orchestrator = api_server.app.state.orchestrator
    spy = MagicMock(side_effect=Exception("astream should NOT be called"))
    monkeypatch.setattr(orchestrator, "astream_chat_with_agent", spy)

    body_text = ""
    with _client.stream("POST", "/chat/stream", json={
        "message": "I want to die",
        "agent": "assistant",
    }) as res:
        for chunk in res.iter_text():
            body_text += chunk

    spy.assert_not_called()
    # The safe response was streamed as one or more SSE events.
    assert "988" in body_text
    # Stream contained event lines.
    assert "event:" in body_text
