import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import api_server


client = TestClient(api_server.app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_model_catalog_provider_unavailable(monkeypatch):
    monkeypatch.setattr(api_server.controller, 'list_local_models_detailed', lambda: (False, [], 'offline'))
    response = client.get('/model-catalog?provider=ollama')
    assert response.status_code == 200
    items = response.json()['items']
    assert items
    assert items[0]['provider_unavailable'] is True


def test_agents_crud_and_validation(monkeypatch):
    monkeypatch.setitem(api_server.orchestrator.definitions, 'assistant', object())

    bad = client.post('/agents', json={'name': 'a', 'provider': 'ollama', 'model_id': 'x', 'tool_ids': ['missing']})
    assert bad.status_code == 400

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

    read = client.get('/agents/agent-a')
    assert read.status_code == 200
    assert read.json()['model_id'] == 'gemma3:1b'

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

    eff = client.get('/agents/agent-a/effective-config')
    assert eff.status_code == 200
    assert 'settings' in eff.json()

    delete = client.delete('/agents/agent-a')
    assert delete.status_code == 200


def test_tools_crud_and_tavily_disabled_state():
    create = client.post('/tools', json={
        'name': 'tavily_web_search',
        'type': 'tavily',
        'description': 'search',
        'config_json': {'max_results': 5},
        'is_enabled': True,
    })
    assert create.status_code == 200
    tid = create.json()['tool_id']

    status = client.get('/tools/status')
    assert status.status_code == 200
    reasons = [i['reason'] for i in status.json()['items'] if i['tool_id'] == tid]
    assert reasons

    read = client.get(f'/tools/{tid}')
    assert read.status_code == 200

    upd = client.put(f'/tools/{tid}', json={
        'name': 'tavily_web_search',
        'type': 'tavily',
        'description': 'search-updated',
        'config_json': {'max_results': 3},
        'is_enabled': False,
    })
    assert upd.status_code == 200

    delete = client.delete(f'/tools/{tid}')
    assert delete.status_code == 200


def test_agent_tool_creation(monkeypatch):
    monkeypatch.setitem(api_server.orchestrator.definitions, 'assistant', object())
    response = client.post('/tools', json={
        'name': 'call_assistant',
        'type': 'agent_tool',
        'description': 'delegate',
        'config_json': {'target_agent': 'assistant', 'strict_output': True},
        'is_enabled': True,
    })
    assert response.status_code == 200
    assert response.json()['type'] == 'agent_tool'


def test_mcp_server_crud_and_refresh():
    create = client.post('/tools/mcp/servers', json={'name': 'srv1', 'transport': 'http', 'endpoint': 'http://127.0.0.1:8001', 'enabled': True})
    assert create.status_code == 200
    sid = create.json()['id']

    refresh = client.post(f'/tools/mcp/servers/{sid}/refresh', json={})
    assert refresh.status_code == 200

    delete = client.delete(f'/tools/mcp/servers/{sid}')
    assert delete.status_code == 200


def test_agents_list_includes_default_assistant_and_protects_delete():
    response = client.get('/agents')
    assert response.status_code == 200
    names = response.json()['agents']
    assert 'assistant' in names

    protected = client.delete('/agents/assistant')
    assert protected.status_code == 400
    assert protected.json()['detail']['error']['code'] == 'protected_agent'


def test_tools_list_includes_builtin_tools_even_without_db_rows():
    response = client.get('/tools')
    assert response.status_code == 200
    items = response.json()['items']
    assert any(item['name'] == 'tavily_web_search' for item in items)
    assert not any(item['tool_id'] == 'mcp_query' for item in items)


def test_model_catalog_exposes_capability_flags():
    response = client.get('/model-catalog')
    assert response.status_code == 200
    items = response.json()['items']
    assert items
    sample = items[0]
    assert 'supports_tools' in sample
    assert 'tool_calling' in sample
    assert 'supports_vision' in sample


def test_chat_multipart_with_attachment(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'chat_with_agent', lambda *args, **kwargs: 'mocked reply')
    response = client.post(
        '/chat_with_attachments',
        data={'agent': 'assistant', 'message': 'Summarize attachment'},
        files=[('files', ('note.txt', b'Attachment says hello world', 'text/plain'))],
    )
    assert response.status_code == 200
    body = response.json()
    assert body['conversation_id']
    user_msgs = [m for m in body['messages'] if m['role'] == 'user']
    assert user_msgs
    assert 'note.txt' in user_msgs[-1]['attachments_json']


def test_prompt_draft_minimal_payload_returns_200():
    response = client.post('/agents/prompt-draft', json={'goal': 'Build a bug triage agent'})
    assert response.status_code == 200
    payload = response.json()
    assert 'prompt_text' in payload
    assert 'sections' in payload
    assert 'used_fallback' in payload


def test_prompt_draft_missing_goal_returns_422():
    response = client.post('/agents/prompt-draft', json={})
    assert response.status_code == 422


def test_prompt_draft_fallback_path(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'generate_system_prompt', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('offline')))
    response = client.post('/agents/prompt-draft', json={'goal': 'Need deterministic fallback'})
    assert response.status_code == 200
    assert response.json()['used_fallback'] is True


def test_agent_create_with_valid_builtin_tool_id_succeeds(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'chat_with_agent', lambda *args, **kwargs: 'ok')
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


def test_chat_stream_endpoint_emits_events(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'stream_chat_with_agent', lambda *args, **kwargs: iter(['Hi', 'Hi there']))
    response = client.post('/chat/stream', json={'agent': 'assistant', 'message': 'hello'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    body = response.text
    assert 'event: start' in body
    assert 'event: end' in body


def test_tools_help_endpoint():
    response = client.get('/tools/help')
    assert response.status_code == 200
    assert 'TAVILY_API_KEY' in response.json()['tavily']


def test_tools_returns_tavily_with_missing_key_status(monkeypatch):
    monkeypatch.delenv('TAVILY_API_KEY', raising=False)
    response = client.get('/tools')
    assert response.status_code == 200
    items = response.json()['items']
    tavily = next((i for i in items if i['tool_id'] == 'tavily_web_search'), None)
    assert tavily is not None
    assert tavily['status'] in {'missing_key', 'disabled'}


def test_mcp_import_accepts_json_config(monkeypatch):
    monkeypatch.setattr(api_server, '_discover_mcp_tools', lambda server: ([{
        'tool_id': f"mcp:{server['name']}:search",
        'name': f"{server['name']}:search",
        'description': 'Discovered',
        'type': 'mcp',
        'config_json': {'server_id': server['id'], 'tool_name': 'search'},
        'is_enabled': True,
    }], None))
    payload = {
        'description': 'import test',
        'config': {
            'mcpServers': {
                'amap-maps': {
                    'command': 'npx',
                    'args': ['-y', '@amap/amap-maps-mcp-server'],
                    'env': {'AMAP_MAPS_API_KEY': 'api_key'}
                }
            }
        }
    }
    response = client.post('/tools/mcp/import', json=payload)
    assert response.status_code == 200
    assert response.json()['imported_servers']


def test_tool_test_endpoint_returns_output(monkeypatch):
    monkeypatch.setenv('TAVILY_API_KEY', 'dummy')
    response = client.post('/tools/tavily_web_search/test', json={'input': {'query': 'hello'}})
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_mcp_alias_endpoints_work(monkeypatch):
    create = client.post('/mcp/servers', json={'name': 'alias-srv', 'transport': 'http', 'endpoint': 'http://127.0.0.1:9001', 'enabled': True})
    assert create.status_code == 200
    sid = create.json()['id']

    listed = client.get('/mcp/servers')
    assert listed.status_code == 200
    assert any(s['id'] == sid for s in listed.json()['servers'])

    monkeypatch.setattr(api_server, '_discover_mcp_tools', lambda server: ([{
        'tool_id': f"mcp:{server['name']}:x",
        'name': f"{server['name']}:x",
        'description': 'x',
        'type': 'mcp_tool',
        'config_json': {'server_id': server['id'], 'tool_name': 'x'},
        'is_enabled': True,
    }], None))

    discover = client.post(f'/mcp/servers/{sid}/discover', json={})
    assert discover.status_code == 200


def test_mcp_tools_select_persists_tools(monkeypatch):
    create = client.post('/mcp/servers', json={'name': 'select-srv', 'transport': 'http', 'endpoint': 'http://127.0.0.1:9002', 'enabled': True})
    sid = create.json()['id']

    response = client.post('/mcp/tools', json={
        'server_id': sid,
        'selected_tools': [
            {'tool_name': 'search', 'name': 'select-srv:search', 'description': 'search tool', 'schema': {'type': 'object'}, 'enabled': True}
        ]
    })
    assert response.status_code == 200
    assert response.json()['items']


def test_tavily_status_endpoint(monkeypatch):
    monkeypatch.delenv('TAVILY_API_KEY', raising=False)
    response = client.get('/tools/tavily/status')
    assert response.status_code == 200
    assert response.json()['present'] is False

    monkeypatch.setenv('TAVILY_API_KEY', 'abcd1234')
    response2 = client.get('/tools/tavily/status')
    assert response2.status_code == 200
    assert response2.json()['present'] is True
    assert response2.json()['masked_key'].endswith('1234')


def test_mcp_discover_does_not_create_top_level_mcp_tool(monkeypatch):
    create = client.post('/mcp/servers', json={'name': 'srv-hide', 'transport': 'http', 'endpoint': 'http://127.0.0.1:9191', 'enabled': True})
    sid = create.json()['id']
    monkeypatch.setattr(api_server, '_discover_mcp_tools', lambda _server: ([{
        'tool_id': 'mcp:srv-hide:search',
        'name': 'srv-hide:search',
        'description': 'search',
        'type': 'mcp_tool',
        'config_json': {'tool_name': 'search'},
        'is_enabled': True,
    }], None))
    resp = client.post(f'/mcp/servers/{sid}/discover', json={})
    assert resp.status_code == 200

    tools = client.get('/tools').json()['items']
    assert not any(t.get('tool_id') == 'mcp:srv-hide:search' for t in tools)
    server_item = next((t for t in tools if t.get('type') == 'mcp_server' and t.get('config_json', {}).get('server_id') == sid), None)
    assert server_item is not None
    assert server_item['config_json']['discovered_tools']


def test_chat_returns_run_id_and_trace_list(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'chat_with_agent', lambda *args, **kwargs: 'trace reply')
    response = client.post('/chat', json={'agent': 'assistant', 'message': 'trace me'})
    assert response.status_code == 200
    body = response.json()
    assert body.get('run_id')
    assert response.headers.get('x-run-id')

    traces = client.get(f"/traces?conversation_id={body['conversation_id']}&limit=5")
    assert traces.status_code == 200
    assert traces.json()['enabled'] in {True, False}


def test_agent_and_tool_definition_endpoints():
    tool_resp = client.get('/tools/tavily_web_search/definition')
    assert tool_resp.status_code == 200
    assert 'python_snippet' in tool_resp.json()

    agent_resp = client.get('/agents/assistant/definition')
    assert agent_resp.status_code == 200
    payload = agent_resp.json()
    assert 'agent_json' in payload
    assert 'python_snippet' in payload


def test_runs_endpoints_and_message_run_id(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'chat_with_agent', lambda *args, **kwargs: 'run detail reply')
    response = client.post('/chat', json={'agent': 'assistant', 'message': 'run endpoint check'})
    assert response.status_code == 200
    body = response.json()
    run_id = body['run_id']

    runs = client.get('/runs?limit=10')
    assert runs.status_code == 200
    assert any((r.get('run_id') == run_id) for r in runs.json().get('items', []))

    run = client.get(f'/runs/{run_id}')
    assert run.status_code == 200
    assert run.json().get('run_id') == run_id

    messages = client.get(f"/conversations/{body['conversation_id']}/messages").json()
    assistant = [m for m in messages if m.get('role') == 'assistant']
    assert assistant
    assert assistant[-1].get('run_id') == run_id


def test_prompt_draft_auto_saves_and_history_endpoints():
    response = client.post('/agents/prompt-draft', json={'goal': 'Build summarizer'})
    assert response.status_code == 200
    body = response.json()
    assert body.get('draft_id')

    listing = client.get('/prompt_drafts?limit=20')
    assert listing.status_code == 200
    items = listing.json()['items']
    assert any(i.get('id') == body['draft_id'] for i in items)

    item = client.get(f"/prompt_drafts/{body['draft_id']}")
    assert item.status_code == 200
    assert item.json()['id'] == body['draft_id']


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


def test_runs_view_endpoint_and_stream_summary(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'chat_with_agent', lambda *args, **kwargs: 'trace reply')
    response = client.post('/chat', json={'agent': 'assistant', 'message': 'aggregate me'})
    run_id = response.json()['run_id']

    view = client.get(f'/runs/{run_id}/view')
    assert view.status_code == 200
    body = view.json()
    assert 'summary' in body
    assert 'timeline' in body
    assert body['summary']['run_id'] == run_id


def test_systems_run_returns_run_id_and_creates_trace(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'run_agent_workflow', lambda *args, **kwargs: {'assistant': 'ok'})
    create = client.post('/systems', json={'name': 'sys-run', 'definition': {'nodes': [{'id': 'n1', 'type': 'agent', 'agent': 'assistant'}], 'edges': []}})
    assert create.status_code == 200

    run = client.post('/systems/sys-run/run', json={'prompt': 'hello'})
    assert run.status_code == 200
    run_id = run.json().get('run_id')
    assert run_id

    trace = client.get(f'/runs/{run_id}')
    assert trace.status_code == 200
    assert trace.json()['run_id'] == run_id


def test_systems_chat_returns_conversation_and_node_outputs(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'run_agent_workflow', lambda *args, **kwargs: {'assistant': 'hello from system'})
    create = client.post('/systems', json={'name': 'sys-chat', 'definition': {'nodes': [{'id': 'n1', 'type': 'agent', 'agent': 'assistant'}], 'edges': []}})
    assert create.status_code == 200

    res = client.post('/systems/sys-chat/chat', json={'message': 'hi'})
    assert res.status_code == 200
    body = res.json()
    assert body['conversation_id']
    assert body['run_id']
    assert body['final_text'] == 'hello from system'
    assert body['node_outputs'][0]['node'] == 'assistant'


def test_systems_chat_multipart_with_attachment(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'run_agent_workflow', lambda *args, **kwargs: {'assistant': 'system ok'})
    create = client.post('/systems', json={'name': 'sys-chat-file', 'definition': {'nodes': [{'id': 'n1', 'type': 'agent', 'agent': 'assistant'}], 'edges': []}})
    assert create.status_code == 200

    response = client.post(
        '/systems/sys-chat-file/chat',
        data={'message': 'use file'},
        files=[('files', ('note.txt', b'hello from file', 'text/plain'))],
    )
    assert response.status_code == 200
    body = response.json()
    assert body['conversation_id']
    assert body['attachments']

    msgs = client.get(f"/conversations/{body['conversation_id']}/messages").json()
    user_msgs = [m for m in msgs if m.get('role') == 'user']
    assert user_msgs
    assert 'note.txt' in user_msgs[-1]['attachments_json']


def test_images_session_and_generate_with_placeholder(monkeypatch):
    class _FakeResult:
        ok = True
        image_bytes = b"fake-png-bytes"
        metadata = {"runtime": "placeholder"}

    monkeypatch.setattr(api_server.image_service, 'generate', lambda **kwargs: _FakeResult())

    sess = client.post('/images/sessions', json={'title': 'img test'})
    assert sess.status_code == 200
    session_id = sess.json()['session_id']

    gen = client.post('/images/generate', json={
        'session_id': session_id,
        'model_id': 'Tongyi-MAI/Z-Image-Turbo',
        'prompt': 'a cat with hat',
    })
    assert gen.status_code == 200
    body = gen.json()
    assert body['image_id']
    assert body['image_url']
    assert body['run_id']

    detail = client.get(f"/images/sessions/{session_id}")
    assert detail.status_code == 200
    assert len(detail.json().get('images', [])) >= 1


def test_images_models_endpoint():
    res = client.get('/images/models')
    assert res.status_code == 200
    body = res.json()
    assert 'items' in body


def test_images_models_refresh_endpoint(monkeypatch):
    monkeypatch.setattr(api_server.image_service, 'refresh_models', lambda: {'items': [{'model_id': 'local:test'}], 'local_text_models': []})
    res = client.post('/images/models/refresh', json={})
    assert res.status_code == 200
    assert res.json()['refreshed'] is True


def test_models_refresh_endpoint(monkeypatch):
    monkeypatch.setattr(api_server.image_service, 'refresh_models', lambda: {'items': [], 'local_text_models': [{'model_id': 'local:text'}]})
    res = client.post('/models/refresh', json={})
    assert res.status_code == 200
    assert res.json()['refreshed'] is True
    assert res.json()['local_text_models'] == 1


def test_images_runtime_endpoint(monkeypatch):
    monkeypatch.setattr(api_server.image_service, 'get_device_status', lambda: {'cuda_available': False, 'effective_device': 'cpu', 'device_preference': 'auto'})
    res = client.get('/images/runtime')
    assert res.status_code == 200
    assert res.json()['effective_device'] == 'cpu'


def test_images_doctor_endpoint(monkeypatch):
    monkeypatch.setattr(api_server.image_service, 'doctor', lambda: {'ok': True, 'checks': [{'name': 'torch', 'ok': True}]})
    res = client.get('/images/doctor')
    assert res.status_code == 200
    assert res.json()['ok'] is True


def test_models_catalog_hf_local_scope(monkeypatch):
    monkeypatch.setattr(api_server, '_hf_local_entries', lambda search='': [{
        'provider': 'huggingface',
        'model_id': 'local:model-a',
        'display_name': 'model-a',
        'local_status': {'installed': True, 'location': '/tmp/models/model-a'},
        'supports': {'chat': True, 'tools': False, 'vision': False, 'embeddings': False, 'streaming': False},
        'metadata': {},
        'capabilities': {'supports_chat': True, 'supports_tools': False, 'supports_vision': False, 'supports_embeddings': False, 'supports_streaming': False},
        'local_path': '/tmp/models/model-a',
    }])
    res = client.get('/models/catalog?provider=huggingface&scope=local')
    assert res.status_code == 200
    body = res.json()
    assert body['count'] == 1
    assert body['items'][0]['installed'] is True
    assert body['items'][0]['local_path']


def test_models_hf_discover_endpoint(monkeypatch):
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
    assert body['count'] == 1
    assert body['items'][0]['provider'] == 'huggingface'
    assert body['items'][0]['capabilities']['supports_embeddings'] is True


def test_models_hf_download_job_and_status(monkeypatch):
    def fake_run(job, payload):
        api_server._set_job(job['download_id'], status='completed', progress_percent=100, local_path='/tmp/models/x')

    class _Thread:
        def __init__(self, target=None, args=(), daemon=None):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(api_server, '_run_hf_download', fake_run)
    monkeypatch.setattr(api_server.threading, 'Thread', _Thread)

    res = client.post('/models/hf/download', json={'model_id': 'sentence-transformers/all-MiniLM-L6-v2'})
    assert res.status_code == 200
    body = res.json()
    assert body['download_id']

    status = client.get(f"/models/hf/downloads/{body['download_id']}")
    assert status.status_code == 200
    assert status.json()['status'] == 'completed'


def test_models_catalog_hf_includes_source_url(monkeypatch):
    monkeypatch.setattr(api_server, '_hf_local_entries', lambda search='': [{
        'provider': 'huggingface',
        'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
        'display_name': 'all-MiniLM-L6-v2',
        'local_status': {'installed': True, 'location': '/tmp/models/minilm'},
        'supports': {'chat': False, 'tools': False, 'vision': False, 'embeddings': True, 'streaming': False},
        'metadata': {'source_url': 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2'},
        'capabilities': {'supports_chat': False, 'supports_tools': False, 'supports_vision': False, 'supports_embeddings': True, 'supports_streaming': False},
        'local_path': '/tmp/models/minilm',
        'task': 'feature-extraction',
    }])
    res = client.get('/models/catalog?provider=huggingface&scope=local')
    assert res.status_code == 200
    assert res.json()['items'][0]['source_url'].startswith('https://huggingface.co/')


def test_hf_local_entries_dedupes_by_local_path(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator.hf, 'configured_models', lambda: ['repo/model-a'])
    monkeypatch.setattr(api_server.orchestrator.hf, 'model_metadata', lambda model_id: {
        'installed': True,
        'supports': {'chat': True, 'tools': False, 'vision': False, 'embeddings': False, 'streaming': False},
        'location': '/tmp/models/shared',
        'runtime': 'transformers_local',
    })
    monkeypatch.setattr(api_server.image_service, 'list_models', lambda refresh=True: [{
        'provider': 'huggingface',
        'model_id': 'local:model-a',
        'display_name': 'model-a (local)',
        'local_status': {'location': '/tmp/models/shared'},
        'supported_features': {'img2img': True},
        'runtime': 'diffusers_local',
    }])

    items = api_server._hf_local_entries()
    assert len(items) == 1
