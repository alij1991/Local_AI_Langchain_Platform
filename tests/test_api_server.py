import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import api_server


client = TestClient(api_server.app)


# Entering the TestClient as a context manager is what triggers FastAPI's
# lifespan — without it ``app.state.orchestrator`` / ``app.state.image_service``
# stay unset and nearly every test 503s or AttributeErrors on a NoneType.
#
# [IMPROVE-5] Commit 3 removed the module globals. Tests reach live
# singletons via ``api_server.app.state.*`` (set by lifespan). The
# existing tests patch attributes *on* those objects — that keeps
# working since ``app.state.X`` and the injected ``Depends(get_X)``
# value are the same reference. ``app.dependency_overrides`` is the
# alternative for tests that want to swap the dependency wholesale.
@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    with client:
        yield


# ── Notes on the 2026-04 triage pass ───────────────────────────────
#
# This file used to have ~50 tests, most of them failing on ``main``
# since well before Wave 2 work started. They fell into three buckets:
#
#   1. Endpoints or module-level helpers that were removed outright
#      (e.g. ``/chat_with_attachments``, ``/tools/help``,
#      ``/tools/mcp/*``, ``/prompt_drafts``, ``/images/doctor``,
#      ``_run_hf_download``, ``_hf_local_entries``,
#      ``_hf_discover_meta``, ``_discover_mcp_tools``,
#      ``api_server.controller``). Those tests were deleted — they
#      couldn't be made to pass without rebuilding the endpoint.
#
#   2. Response-shape drift (e.g. ``/tools`` items no longer carry a
#      ``tool_id`` field, ``/images/sessions`` returns ``id`` not
#      ``session_id``, ``/images/models/refresh`` returns
#      ``{status, items}`` instead of ``{refreshed, ...}``). Those
#      tests were updated to match the current payloads.
#
#   3. Three genuine behavioral regressions (Wave 2 residuals from the
#      [IMPROVE-1] router split). All three were restored in
#      [IMPROVE-71] (Wave 6); the four tests covering them now pass:
#      - ``/agents`` accepts agents with unknown tool_ids → now 400
#      - ``/agents/prompt-draft`` accepts empty body → now 422
#      - ``DELETE /agents/assistant`` succeeds → now 400 protected_agent
#      Boundary validation pinned by tests/test_agents_validation_audit.py.
#
# Post-[IMPROVE-5] Commit 3: the patch targets below use
# ``api_server.app.state.orchestrator`` / ``.image_service`` to reach
# the singletons that lifespan built. See the ``_run_lifespan``
# fixture comment above for the reasoning on keeping this pattern
# instead of ``app.dependency_overrides``.


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_agents_crud_and_validation(monkeypatch):
    # Pre-IMPROVE-71 this test was xfail'd because POST /agents accepted
    # unknown tool_ids without 400. With validation restored, the test
    # walks the current CRUD surface: bad-tool reject, create, read via
    # /definition (the flat ``GET /agents/{name}`` was retired during
    # the [IMPROVE-1] router split), update, delete. The retired
    # ``GET /agents/{name}/effective-config`` is not part of this audit.
    monkeypatch.setitem(api_server.app.state.orchestrator.definitions, 'assistant', object())

    bad = client.post('/agents', json={'name': 'a', 'provider': 'ollama', 'model_id': 'x', 'tool_ids': ['missing']})
    assert bad.status_code == 400
    assert bad.json()['detail']['error']['code'] == 'invalid_tool'

    create = client.post('/agents', json={
        'name': 'agent-a',
        'description': 'test',
        'provider': 'ollama',
        'model_id': 'gemma3:1b',
        'system_prompt': 'You are helpful',
        'tool_ids': [],
        'settings': {'temperature': 0.2},
        'resource_limits': {'max_context_messages': 20},
    })
    assert create.status_code == 200

    read = client.get('/agents/agent-a/definition')
    assert read.status_code == 200
    assert read.json()['agent_json']['model_id'] == 'gemma3:1b'

    update = client.put('/agents/agent-a', json={
        'name': 'agent-a',
        'description': 'test2',
        'provider': 'ollama',
        'model_id': 'gemma3:1b',
        'system_prompt': 'You are better',
        'tool_ids': [],
        'settings': {'temperature': 0.1},
        'resource_limits': {'max_context_messages': 10},
    })
    assert update.status_code == 200

    delete = client.delete('/agents/agent-a')
    assert delete.status_code == 200


def test_agent_tool_creation(monkeypatch):
    monkeypatch.setitem(api_server.app.state.orchestrator.definitions, 'assistant', object())
    response = client.post('/tools', json={
        'name': 'call_assistant',
        'type': 'agent_tool',
        'description': 'delegate',
        'config_json': {'target_agent': 'assistant', 'strict_output': True},
        'is_enabled': True,
    })
    assert response.status_code == 200
    assert response.json()['type'] == 'agent_tool'


def test_agents_list_includes_default_assistant_and_protects_delete():
    response = client.get('/agents')
    assert response.status_code == 200
    names = response.json()['agents']
    assert 'assistant' in names

    protected = client.delete('/agents/assistant')
    assert protected.status_code == 400
    assert protected.json()['detail']['error']['code'] == 'protected_agent'


def test_tools_list_includes_builtin_tools():
    # /tools items now expose {name, type, is_enabled}; the earlier
    # ``tool_id`` field was dropped when builtins stopped needing a
    # UUID. ``mcp_query`` is intentionally included as a builtin MCP
    # fallback tool (see docs/features/04-agents-tools.md §"Tier 1").
    response = client.get('/tools')
    assert response.status_code == 200
    items = response.json()['items']
    names = {item.get('name') for item in items}
    assert 'tavily_web_search' in names


def test_models_catalog_exposes_capability_flags():
    # The old ``GET /model-catalog`` root was dropped when catalog
    # routing consolidated under ``/models/*``. Items now use ``id``
    # as the stable identifier and expose ``supports_tools`` /
    # ``supports_vision`` / ``supports_streaming`` directly (the old
    # ``tool_calling`` alias is gone).
    response = client.get('/models/catalog')
    assert response.status_code == 200
    items = response.json()['items']
    assert items
    sample = items[0]
    assert 'id' in sample
    assert 'supports_tools' in sample
    assert 'supports_vision' in sample


def test_prompt_draft_minimal_payload_returns_200():
    # /agents/prompt-draft now returns {prompt_text, used_fallback}
    # only — the older {sections, draft_id} fields were dropped. See
    # regression tracking for the empty-body validation issue.
    response = client.post('/agents/prompt-draft', json={'goal': 'Build a bug triage agent'})
    assert response.status_code == 200
    payload = response.json()
    assert 'prompt_text' in payload
    assert 'used_fallback' in payload


def test_prompt_draft_missing_goal_returns_422():
    response = client.post('/agents/prompt-draft', json={})
    assert response.status_code == 422


def test_prompt_draft_fallback_path(monkeypatch):
    monkeypatch.setattr(
        api_server.app.state.orchestrator,
        'generate_system_prompt',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('offline')),
    )
    response = client.post('/agents/prompt-draft', json={'goal': 'Need deterministic fallback'})
    assert response.status_code == 200
    assert response.json()['used_fallback'] is True


def test_agent_create_with_valid_builtin_tool_id_succeeds(monkeypatch):
    monkeypatch.setattr(api_server.app.state.orchestrator, 'chat_with_agent', lambda *args, **kwargs: 'ok')
    response = client.post('/agents', json={
        'name': 'tool-agent',
        'description': 'test',
        'provider': 'ollama',
        'model_id': 'gemma3:1b',
        'system_prompt': 'You are helpful',
        'tool_ids': ['tavily_web_search'],
        'settings': {'temperature': 0.2},
        'resource_limits': {'max_context_messages': 20},
    })
    assert response.status_code == 200


def test_agent_create_with_unknown_tool_id_fails():
    response = client.post('/agents', json={
        'name': 'bad-tool-agent',
        'description': 'test',
        'provider': 'ollama',
        'model_id': 'gemma3:1b',
        'system_prompt': 'You are helpful',
        'tool_ids': ['not_real_tool'],
        'settings': {'temperature': 0.2},
        'resource_limits': {'max_context_messages': 20},
    })
    assert response.status_code == 400
    assert response.json()['detail']['error']['code'] == 'invalid_tool'


def test_tavily_status_endpoint(monkeypatch):
    # The endpoint now reads from AppSettings which is ``lru_cache``'d
    # at startup — env vars aren't reread mid-process. We can only
    # reliably exercise the "absent" branch here. The "present" branch
    # is covered by manual smoke testing with a populated ``.env``.
    monkeypatch.delenv('TAVILY_API_KEY', raising=False)
    response = client.get('/tools/tavily/status')
    assert response.status_code == 200
    assert 'present' in response.json()


def test_models_catalog_endpoint():
    response = client.get('/models/catalog?supports_streaming=true')
    assert response.status_code == 200
    payload = response.json()
    assert 'items' in payload
    if payload['items']:
        sample = payload['items'][0]
        assert 'id' in sample
        assert 'provider' in sample
        assert 'supports_streaming' in sample
        assert 'size_human' in sample


def test_images_models_endpoint():
    res = client.get('/images/models')
    assert res.status_code == 200
    body = res.json()
    assert 'items' in body


def test_images_models_refresh_endpoint(monkeypatch):
    # /images/models/refresh returns {status, items}; the older
    # ``refreshed`` boolean field was dropped.
    monkeypatch.setattr(
        api_server.app.state.image_service,
        'refresh_models',
        lambda: {'items': [{'model_id': 'local:test'}], 'local_text_models': []},
    )
    res = client.post('/images/models/refresh', json={})
    assert res.status_code == 200
    body = res.json()
    assert 'items' in body


def test_models_refresh_endpoint(monkeypatch):
    # /models/refresh shares the image_service refresh; same shape note
    # as test_images_models_refresh_endpoint.
    monkeypatch.setattr(
        api_server.app.state.image_service,
        'refresh_models',
        lambda: {'items': [], 'local_text_models': [{'model_id': 'local:text'}]},
    )
    res = client.post('/models/refresh', json={})
    assert res.status_code == 200


def test_images_runtime_endpoint(monkeypatch):
    # /images/runtime now returns a much richer payload (hardware
    # profile, attention backend, etc.). The old ``low_memory_mode``
    # field was dropped. We still verify device-status wiring via the
    # ``effective_device`` round-trip.
    monkeypatch.setattr(
        api_server.app.state.image_service,
        'get_device_status',
        lambda: {'cuda_available': False, 'effective_device': 'cpu', 'device_preference': 'auto'},
    )
    res = client.get('/images/runtime')
    assert res.status_code == 200
    assert res.json()['effective_device'] == 'cpu'


def test_images_session_and_generate_with_placeholder(monkeypatch):
    # /images/sessions POST returns ``id`` (not the older
    # ``session_id`` field). /images/generate now returns
    # ``{status, metadata, seed_used}`` — the ``image_id`` /
    # ``image_url`` / ``run_id`` fields the old client relied on are
    # gone; the session-detail payload is the canonical place to get
    # the persisted image entry.
    class _FakeResult:
        ok = True
        image_bytes = b"fake-png-bytes"
        metadata = {"runtime": "placeholder"}

    monkeypatch.setattr(api_server.app.state.image_service, 'generate', lambda **kwargs: _FakeResult())

    sess = client.post('/images/sessions', json={'title': 'img test'})
    assert sess.status_code == 200
    session_id = sess.json()['id']

    gen = client.post('/images/generate', json={
        'session_id': session_id,
        'model_id': 'Tongyi-MAI/Z-Image-Turbo',
        'prompt': 'a cat with hat',
    })
    assert gen.status_code == 200
    assert gen.json()['status'] == 'ok'

    detail = client.get(f"/images/sessions/{session_id}")
    assert detail.status_code == 200
    assert len(detail.json().get('images', [])) >= 1


def test_images_validate_model_endpoint(monkeypatch):
    monkeypatch.setattr(api_server.app.state.image_service, 'validate_model', lambda model_id: {
        'model_id': model_id,
        'resolved_path': '/tmp/models/x',
        'detected_type': 'diffusers_local',
        'loadable': False,
        'errors': ['missing_model_index_json'],
    })
    res = client.post('/images/validate-model', json={'model_id': 'local:x'})
    assert res.status_code == 200
    assert res.json()['loadable'] is False
    assert res.json()['detected_type'] == 'diffusers_local'


def test_images_recommendations_endpoint(monkeypatch):
    monkeypatch.setattr(api_server.app.state.image_service, 'recommended_settings', lambda model_id: {
        'recommended_width': 512,
        'recommended_height': 512,
        'recommended_steps': 16,
        'reason': 'Low-memory mode suggested for this hardware.',
    })
    res = client.get('/images/recommendations?model_id=local:test')
    assert res.status_code == 200
    assert res.json()['recommended_width'] == 512


def test_models_hf_discover_endpoint(monkeypatch):
    # /models/hf/discover no longer returns a top-level ``count``
    # field — callers iterate ``items`` directly.
    class _Model:
        id = 'sentence-transformers/all-MiniLM-L6-v2'
        pipeline_tag = 'feature-extraction'
        tags = ['sentence-transformers']
        downloads = 123
        likes = 45
        last_modified = '2024-01-01T00:00:00'

    class _Api:
        def __init__(self, token=None):
            pass

        def list_models(self, **kwargs):
            return [_Model()]

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'HfApi', _Api)

    res = client.get('/models/hf/discover?q=minilm&sort=downloads&limit=10')
    assert res.status_code == 200
    body = res.json()
    assert body['items']
    assert body['items'][0]['provider'] == 'huggingface'
